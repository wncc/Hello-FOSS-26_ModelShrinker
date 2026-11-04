import sys
import textwrap

import pytest
import torch

from examples.test_models import SmallVGG
from shrinker import compress, list_methods, register
from shrinker.registry import LOAD_ERRORS, REGISTRY, describe_methods, discover


def test_l1_pruning_registered():
    assert "l1_pruning" in list_methods()
    assert list_methods(kind="pruning") == [n for n in list_methods() if REGISTRY[n].kind == "pruning"]
    assert "l1_pruning" in describe_methods()


@pytest.mark.parametrize("name", ["l1_pruning", "fpgm", "gradient_based"])
def test_pruning_methods_registered_as_pruning(name):
    # test_methods_combine orders pairs by kind, so a wrong kind hides prune -> quantize pairs
    assert REGISTRY[name].kind == "pruning"
    assert REGISTRY[name].paper


def test_duplicate_name_rejected(restore_registry):
    with pytest.raises(ValueError, match="Two files register the method name 'l1_pruning'"):

        @register("l1_pruning")
        def apply(model):
            return model


def test_same_function_can_reregister(restore_registry):
    fn = REGISTRY["l1_pruning"].fn
    register("l1_pruning", kind="pruning", paper="x")(fn)


def test_name_defaults_to_file_name(restore_registry):
    @register
    def apply(model):
        return model

    assert REGISTRY["test_registry"].fn is apply
    assert REGISTRY["test_registry"].kind == "other"


def test_empty_brackets_also_use_file_name(restore_registry):
    @register()
    def apply(model):
        return model

    assert REGISTRY["test_registry"].fn is apply


def test_same_file_name_twice_clashes(restore_registry):
    @register
    def apply(model):
        return model

    with pytest.raises(ValueError, match="Two files register the method name 'test_registry'"):

        @register
        def apply2(model):
            return model


@pytest.mark.parametrize("name", ["", "   ", 5])
def test_empty_name_rejected(name):
    with pytest.raises(ValueError):
        register(name)


def test_labels_are_optional(restore_registry, caplog):
    @register("Some Method-v2", kind="prunning")
    def apply(model):
        return model

    assert REGISTRY["Some Method-v2"].kind == "other"
    assert "unknown kind" in caplog.text
    assert isinstance(compress(SmallVGG(), "Some Method-v2"), torch.nn.Module)


def test_discover_drop_in_file(tmp_path, monkeypatch, restore_registry):
    pkg = tmp_path / "fake_methods"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "member_method.py").write_text(textwrap.dedent('''
        from shrinker.registry import register
        from shrinker.utils import keep_indices, prunable_layers, prune_channels

        @register("member_method", kind="pruning", paper="https://example.org")
        def apply(model, ratio=0.5):
            for name, layer in prunable_layers(model):
                keep = keep_indices(layer.weight.detach().flatten(1).norm(dim=1), ratio)
                prune_channels(model, name, keep)
            return model
    '''))
    (pkg / "_draft.py").write_text("raise RuntimeError('should never be imported')")
    (pkg / "broken.py").write_text("import this_package_does_not_exist")

    monkeypatch.syspath_prepend(str(tmp_path))
    discover("fake_methods", [str(pkg)])

    assert "member_method" in list_methods()
    assert "fake_methods.broken" in LOAD_ERRORS
    assert not any("_draft" in k for k in LOAD_ERRORS)

    x = torch.randn(2, 3, 32, 32)
    small = compress(SmallVGG(), "member_method", example_input=x)
    assert sum(p.numel() for p in small.parameters()) < sum(p.numel() for p in SmallVGG().parameters())
    sys.modules.pop("fake_methods.member_method", None)
