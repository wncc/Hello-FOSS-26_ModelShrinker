"""Find a pruning ratio for every layer with reinforcement learning.

A small policy network walks through the prunable layers in execution order and picks a ratio
for each one. Its plan is shifted up or down until it removes exactly the requested share of
FLOPs (or parameters), the model is pruned with ``compress(model, method, ratio=plan)``, and the
pruned model's validation accuracy is the reward. The policy is trained with REINFORCE, using
the other plans of the same update as the baseline (leave-one-out), and it starts from a greedy
plan built from ``analyze_sensitivity`` so the first episodes are not wasted.
"""

from __future__ import annotations

import inspect
import logging
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .core import compress
from .cost import PruningCost, kept_channels
from .registry import get_method
from .sensitivity import analyze_sensitivity

logger = logging.getLogger(__name__)

_AGENTS = ("reinforce", "greedy", "uniform")
_STEP = 0.05
_SENSITIVITY_RATIOS = (0.25, 0.5, 0.75)
_PLANS_PER_UPDATE = 4
_SHIFT_PENALTY = 0.2
_ENTROPY_BONUS = 0.01


def search_ratios(
    model: nn.Module,
    val_loader: Iterable[Any],
    target: float = 0.5,
    metric: str = "flops",
    episodes: int = 40,
    method: str = "l1_pruning",
    agent: str = "reinforce",
    max_ratio: float = 0.8,
    max_batches: Optional[int] = None,
    adapt_bn: bool = True,
    seed: int = 0,
    return_report: bool = False,
    **method_kwargs: Any,
):
    """Return ``{layer name: ratio}`` that removes ``target`` of the model's ``metric``.

    target:     share of the FLOPs (``metric="flops"``) or parameters (``metric="params"``)
                to remove, e.g. 0.5 halves the FLOPs.
    episodes:   how many plans the agent tries; each one prunes and evaluates the model once.
    method:     the pruning method the ratios are for. Use the result with
                ``compress(model, method, ratio=ratios)``.
    agent:      ``"reinforce"`` (default), ``"greedy"`` (sensitivity-based, no RL) or
                ``"uniform"`` (the same ratio everywhere).
    max_ratio:  the largest ratio any layer may get.
    max_batches: how many batches of ``val_loader`` to use (default: all). Half of them score
                the plans during the search, the other half pick the final plan.
    adapt_bn:   re-estimate BatchNorm statistics of every pruned model on a few batches before
                measuring its accuracy (labels are not used). It makes the accuracy of a pruned
                model a much better guide to how it does after finetuning.

    The final plan is checked on held-out batches against the uniform plan and the greedy
    plan, and the best of them is returned, so the search never returns something worse than
    uniform pruning on that data. With ``return_report=True`` the result is
    ``(ratios, report)``, where ``report`` has the accuracies of the candidates.
    The original model is not modified.
    """
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
    if agent not in _AGENTS:
        raise ValueError(f"agent must be one of {_AGENTS}, got {agent!r}")
    if not 0.0 <= target < 1.0:
        raise ValueError(f"target must be in [0, 1), got {target}")
    if not 0.0 < max_ratio < 1.0:
        raise ValueError(f"max_ratio must be in (0, 1), got {max_ratio}")
    if episodes < 0:
        raise ValueError(f"episodes must be >= 0, got {episodes}")
    if "ratio" not in inspect.signature(get_method(method).fn).parameters:
        raise ValueError(f"method {method!r} has no `ratio` argument, so it can't use per-layer ratios")

    batches = _labelled_batches(val_loader, max_batches)
    search_batches, holdout = (batches[0::2], batches[1::2]) if len(batches) > 1 else (batches, batches)
    bn_batches = search_batches[:2] if adapt_bn else []

    cost = PruningCost(model, batches[0][0], metric)
    if not cost.groups:
        raise ValueError("no layer of this model can be pruned (see shrinker.dependency.skipped_layers)")
    budget = (1.0 - target) * cost.full
    levels = [round(i * _STEP, 4) for i in range(int(round(max_ratio / _STEP)) + 1)]
    if levels[-1] != max_ratio:
        levels.append(max_ratio)
    most = {g: max_ratio for g in cost.groups}
    if cost(most) > budget:
        raise ValueError(
            f"target={target} can't be reached: pruning every layer at max_ratio={max_ratio} "
            f"removes only {1 - cost.fraction(most):.3f} of the {metric}"
        )

    def project(plan: Dict[str, float]) -> Tuple[Dict[str, float], float]:
        return _fit_budget(plan, cost, budget, max_ratio)

    def score(plan: Dict[str, float], data) -> float:
        pruned = compress(model, method, ratio=plan, **method_kwargs)
        if bn_batches:
            _recalibrate_bn(pruned, bn_batches)
        return _accuracy(pruned, data)

    uniform, _ = project({g: 0.0 for g in cost.groups})
    report: Dict[str, Any] = {"agent": agent, "metric": metric, "target": target}
    if agent == "uniform":
        return (uniform, report) if return_report else uniform

    sensitivity = analyze_sensitivity(model, search_batches, ratios=_SENSITIVITY_RATIOS)
    if sensitivity.baseline_accuracy is None:
        raise ValueError("val_loader must yield (inputs, labels) batches and the model must return class logits")
    drops = {g: _drop_curve(sensitivity.layers.get(g, {}), levels) for g in cost.groups}
    greedy, _ = project(_greedy_plan(cost, drops, levels, budget))
    if agent == "greedy":
        return (greedy, report) if return_report else greedy

    features = _layer_features(model, cost, drops, target, max_ratio)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        policy = nn.Sequential(nn.Linear(features.shape[1] + 2, 32), nn.Tanh(), nn.Linear(32, len(levels)))
    gen = torch.Generator().manual_seed(seed)
    walker = _Walker(policy, features, cost, levels)
    walker.imitate([levels.index(_nearest(greedy[g], levels)) for g in cost.groups])

    opt = torch.optim.Adam(policy.parameters(), lr=0.01)
    half = max(1, len(search_batches) // 2)
    tried: List[Tuple[float, Dict[str, float]]] = []
    rewards: List[float] = []
    for update in range(math.ceil(episodes / _PLANS_PER_UPDATE)):
        # plans of one update are scored on the same batches, so they are compared fairly
        start = (update % 2) * half
        data = search_batches[start:start + half] or search_batches
        n = min(_PLANS_PER_UPDATE, episodes - update * _PLANS_PER_UPDATE)
        log_probs, entropies, batch_rewards = [], [], []
        for _ in range(n):
            actions, log_prob, entropy = walker.sample(gen)
            plan, shift = project({g: levels[a] for g, a in zip(cost.groups, actions)})
            reward = score(plan, data) - _SHIFT_PENALTY * abs(shift)
            log_probs.append(log_prob)
            entropies.append(entropy)
            batch_rewards.append(reward)
            tried.append((reward, plan))
        rewards += batch_rewards
        if n < 2:
            continue
        r = torch.tensor(batch_rewards)
        advantage = r - (r.sum() - r) / (n - 1)
        loss = -(advantage * torch.stack(log_probs)).mean() - _ENTROPY_BONUS * torch.stack(entropies).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    learned, _ = project({g: levels[a] for g, a in zip(cost.groups, walker.greedy())})
    candidates = [("uniform", uniform), ("greedy", greedy), ("reinforce", learned)]
    candidates += [("reinforce", plan) for _, plan in sorted(tried, key=lambda t: -t[0])[:2]]
    scored: List[Tuple[str, Dict[str, float], float]] = []
    for name, plan in candidates:
        if all(plan != seen for _, seen, _ in scored):
            scored.append((name, plan, score(plan, holdout)))
    # max() keeps the first of equal scores, and uniform comes first
    chosen, best, accuracy = max(scored, key=lambda t: t[2])
    logger.info("search_ratios picked the %s plan (held-out accuracy %.4f)", chosen, accuracy)

    def accuracy_of(plan):
        return next(a for _, seen, a in scored if seen == plan)

    report.update(
        chosen=chosen,
        accuracy=accuracy,
        uniform_accuracy=accuracy_of(uniform),
        greedy_accuracy=accuracy_of(greedy),
        removed=1.0 - cost.fraction(best),
        rewards=rewards,
    )
    return (best, report) if return_report else best


class _Walker:
    """Runs the policy over the layers in order. Each step sees the layer's features plus the
    previous ratio and the share of the cost removed so far."""

    def __init__(self, policy: nn.Module, features: torch.Tensor, cost: PruningCost, levels: Sequence[float]):
        self.policy, self.features, self.cost, self.levels = policy, features, cost, levels

    def _state(self, i: int, actions: List[int]) -> torch.Tensor:
        done = {g: self.levels[a] for g, a in zip(self.cost.groups, actions)}
        previous = self.levels[actions[-1]] if actions else 0.0
        return torch.cat([self.features[i], torch.tensor([previous, 1.0 - self.cost.fraction(done)])])

    def sample(self, gen: torch.Generator):
        actions, log_prob, entropy = [], 0.0, 0.0
        for i in range(len(self.cost.groups)):
            dist = torch.distributions.Categorical(logits=self.policy(self._state(i, actions)))
            a = int(torch.multinomial(dist.probs.detach(), 1, generator=gen))
            log_prob = log_prob + dist.log_prob(torch.tensor(a))
            entropy = entropy + dist.entropy()
            actions.append(a)
        return actions, log_prob, entropy

    def greedy(self) -> List[int]:
        actions: List[int] = []
        with torch.no_grad():
            for i in range(len(self.cost.groups)):
                actions.append(int(self.policy(self._state(i, actions)).argmax()))
        return actions

    def imitate(self, actions: List[int], steps: int = 150) -> None:
        """Start the policy close to ``actions`` (still random enough to explore)."""
        states = torch.stack([self._state(i, actions[:i]) for i in range(len(actions))])
        labels = torch.tensor(actions)
        opt = torch.optim.Adam(self.policy.parameters(), lr=0.01)
        for _ in range(steps):
            loss = F.cross_entropy(self.policy(states), labels, label_smoothing=0.3)
            opt.zero_grad()
            loss.backward()
            opt.step()


def _fit_budget(plan: Dict[str, float], cost: PruningCost, budget: float, max_ratio: float):
    """Add the same amount to every ratio (clipped to [0, max_ratio]) so the plan just fits the
    budget. Returns the new plan and the amount added."""

    def shifted(t: float) -> Dict[str, float]:
        return {g: round(min(max(r + t, 0.0), max_ratio), 4) for g, r in plan.items()}

    lo, hi = -max_ratio, max_ratio  # cost(shifted(lo)) is the full cost, cost(shifted(hi)) the lowest
    if cost(shifted(0.0)) <= budget:
        hi = 0.0
    else:
        lo = 0.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if cost(shifted(mid)) <= budget:
            hi = mid
        else:
            lo = mid
    plan = shifted(hi)
    # 0.3 instead of 0.2969 when both keep the same number of channels
    for g, r in plan.items():
        n = cost.sizes[g]
        if kept_channels(n, round(r, 2)) == kept_channels(n, r):
            plan[g] = round(r, 2)
    return plan, hi


def _greedy_plan(cost: PruningCost, drops, levels: Sequence[float], budget: float) -> Dict[str, float]:
    """Raise one layer at a time by one step, always the one that loses the least accuracy
    for the cost it saves, until the budget is met."""
    step = {g: 0 for g in cost.groups}
    plan = {g: 0.0 for g in cost.groups}
    current = cost(plan)
    while current > budget:
        best, best_value = None, math.inf
        for g in cost.groups:
            if step[g] + 1 >= len(levels):
                continue
            saved = current - cost({**plan, g: levels[step[g] + 1]})
            if saved <= 0:
                continue
            value = (drops[g][step[g] + 1] - drops[g][step[g]] + 1e-4) / saved
            if value < best_value:
                best, best_value = g, value
        if best is None:
            break
        step[best] += 1
        plan[best] = levels[step[best]]
        current = cost(plan)
    return plan


def _drop_curve(points, levels: Sequence[float]) -> List[float]:
    """Accuracy drop at every level, read off the sensitivity measurements (0 at ratio 0),
    linear in between, and never decreasing."""
    xs = [0.0] + sorted(points)
    ys = [0.0] + [max(0.0, points[r].accuracy_drop) for r in sorted(points)]
    curve, worst = [], 0.0
    for level in levels:
        if level >= xs[-1]:
            slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2]) if len(xs) > 1 else 0.0
            value = ys[-1] + max(slope, 0.0) * (level - xs[-1])
        else:
            j = next(k for k in range(1, len(xs)) if level < xs[k])
            value = ys[j - 1] + (ys[j] - ys[j - 1]) * (level - xs[j - 1]) / (xs[j] - xs[j - 1])
        worst = max(worst, value)
        curve.append(worst)
    return curve


def _layer_features(model: nn.Module, cost: PruningCost, drops, target: float, max_ratio: float) -> torch.Tensor:
    rows = []
    n = len(cost.groups)
    sens_index = [int(round(r / _STEP)) for r in _SENSITIVITY_RATIOS]
    for i, g in enumerate(cost.groups):
        layer = model.get_submodule(g)
        is_conv = type(layer) is nn.Conv2d
        fan_in = layer.in_channels if is_conv else layer.in_features
        kernel = layer.kernel_size[0] * layer.kernel_size[1] if is_conv else 1
        share = 1.0 - cost.fraction({g: max_ratio})
        curve = drops[g]
        sens = [curve[min(k, len(curve) - 1)] for k in sens_index]
        rows.append([
            i / max(n - 1, 1), float(is_conv), math.log2(cost.sizes[g]) / 10, math.log2(fan_in) / 10,
            math.log2(kernel) / 4, share, target, *sens,
        ])
    return torch.tensor(rows, dtype=torch.float32)


def _nearest(value: float, levels: Sequence[float]) -> float:
    return min(levels, key=lambda level: abs(level - value))


def _labelled_batches(loader: Iterable[Any], max_batches: Optional[int]) -> list:
    if max_batches is not None and max_batches < 1:
        raise ValueError(f"max_batches must be at least 1, got {max_batches}")
    batches = []
    for batch in loader:
        if max_batches is not None and len(batches) >= max_batches:
            break
        if not isinstance(batch, (tuple, list)) or len(batch) != 2:
            raise ValueError("val_loader must yield (inputs, labels) batches")
        batches.append((batch[0], batch[1]))
    if not batches:
        raise ValueError("val_loader is empty")
    return batches


def _device(model: nn.Module) -> torch.device:
    p = next(model.parameters(), None)
    return p.device if p is not None else torch.device("cpu")


def _to(x: Any, device: torch.device) -> Any:
    if isinstance(x, (tuple, list)):
        return type(x)(_to(t, device) for t in x)
    return x.to(device) if isinstance(x, torch.Tensor) else x


def _run(model: nn.Module, x: Any):
    return model(*x) if isinstance(x, tuple) else model(x)


def _accuracy(model: nn.Module, batches) -> float:
    device = _device(model)
    correct = total = 0
    model.eval()
    with torch.no_grad():
        for x, y in batches:
            out = _run(model, _to(x, device))
            correct += (out.argmax(dim=1) == y.to(device).reshape(-1)).sum().item()
            total += y.numel()
    return correct / max(total, 1)


def _recalibrate_bn(model: nn.Module, batches) -> None:
    """Replace BatchNorm running statistics with ones measured on ``batches`` (EagleEye)."""
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.track_running_stats]
    if not bns:
        return
    device = _device(model)
    model.eval()
    momenta = [bn.momentum for bn in bns]
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None  # plain average over the batches
        bn.train()
    with torch.no_grad():
        for x, _ in batches:
            _run(model, _to(x, device))
    for bn, momentum in zip(bns, momenta):
        bn.momentum = momentum
        bn.eval()
