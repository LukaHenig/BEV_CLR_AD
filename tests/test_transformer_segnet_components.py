import importlib.util
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def segnet_module():
    original_modules = {}

    def add_stub(name, module):
        if name in sys.modules:
            original_modules[name] = sys.modules[name]
        sys.modules[name] = module

    # Stub deformable attention dependencies that require compiled extensions
    ops_modules_stub = types.ModuleType("nets.ops.modules")

    class _DummyAttn(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, *args, **kwargs):
            query = args[0]
            return query

    ops_modules_stub.MSDeformAttn = _DummyAttn
    ops_modules_stub.MSDeformAttn3D = _DummyAttn
    add_stub("nets.ops.modules", ops_modules_stub)

    ops_functions_stub = types.ModuleType("nets.ops.functions")
    ops_functions_stub.MSDeformAttnFunction = object
    add_stub("nets.ops.functions", ops_functions_stub)

    module_path = Path(__file__).resolve().parent.parent / "nets" / "segnet_transformer_lift_fuse_new_decoders.py"
    spec = importlib.util.spec_from_file_location("segnet_transformer_lift_fuse_new_decoders", module_path)
    module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    assert loader is not None
    loader.exec_module(module)

    yield module

    for name, module in original_modules.items():
        sys.modules[name] = module
    for name in ["nets.ops.modules", "nets.ops.functions"]:
        if name not in original_modules:
            sys.modules.pop(name, None)


def test_set_bn_momentum_updates_instance_norm_layers(segnet_module):
    module = segnet_module

    class SmallModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Conv2d(3, 3, kernel_size=3, padding=1),
                torch.nn.InstanceNorm2d(3),
                torch.nn.ReLU(),
            )

    model = SmallModel()
    module.set_bn_momentum(model, momentum=0.2)

    for layer in model.modules():
        if isinstance(layer, torch.nn.InstanceNorm2d):
            assert layer.momentum == 0.2


def test_upsampling_concat_preserves_spatial_dimensions(segnet_module):
    module = segnet_module
    block = module.UpsamplingConcat(in_channels=6, out_channels=4, scale_factor=2)

    x_to_upsample = torch.randn(1, 6, 8, 8)
    skip = torch.randn(1, 6, 16, 16)
    output = block(x_to_upsample, skip)

    assert output.shape == (1, 4, 16, 16)


def test_upsampling_add_aligns_skip_connections(segnet_module):
    module = segnet_module
    block = module.UpsamplingAdd(in_channels=4, out_channels=2, scale_factor=2)

    x = torch.randn(1, 4, 8, 8)
    skip = torch.randn(1, 2, 16, 16)
    output = block(x, skip)

    assert output.shape == (1, 2, 16, 16)
    assert torch.allclose(output, block.upsample_layer(x) + skip)
