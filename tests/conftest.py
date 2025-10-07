import math
import sys
import types

import pytest

try:  # pragma: no cover
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


if torch is None:  # pragma: no cover

    def pytest_sessionstart(session):
        pytest.skip("PyTorch is required for the test suite", allow_module_level=True)

else:

    def _ensure_module(name: str, module: types.ModuleType) -> None:
        if name not in sys.modules:
            sys.modules[name] = module

    def _stub_tensorboardx() -> None:
        tensorboardx = types.ModuleType("tensorboardX")

        class _DummySummaryWriter:
            def __init__(self, *args, **kwargs):
                pass

            def add_scalar(self, *args, **kwargs):
                pass

            def close(self):
                pass

        tensorboardx.SummaryWriter = _DummySummaryWriter
        _ensure_module("tensorboardX", tensorboardx)

    def _stub_wandb() -> None:
        wandb = types.ModuleType("wandb")
        wandb.init = lambda *args, **kwargs: None
        wandb.log = lambda *args, **kwargs: None
        wandb.watch = lambda *args, **kwargs: None
        wandb.config = {}
        wandb.run = types.SimpleNamespace(name="stub")
        wandb.Image = object
        _ensure_module("wandb", wandb)

    def _stub_nuscenes() -> None:
        if "nuscenes" in sys.modules:
            return

        nuscenes_pkg = types.ModuleType("nuscenes")
        sys.modules["nuscenes"] = nuscenes_pkg

        map_expansion = types.ModuleType("nuscenes.map_expansion")
        sys.modules["nuscenes.map_expansion"] = map_expansion

        map_api = types.ModuleType("nuscenes.map_expansion.map_api")
        map_api.NuScenesMap = type("NuScenesMap", (), {})
        sys.modules["nuscenes.map_expansion.map_api"] = map_api

        nusc_core = types.ModuleType("nuscenes.nuscenes")
        nusc_core.NuScenes = type("NuScenes", (), {"__init__": lambda self, *args, **kwargs: None})
        sys.modules["nuscenes.nuscenes"] = nusc_core

        utils_pkg = types.ModuleType("nuscenes.utils")
        sys.modules["nuscenes.utils"] = utils_pkg

        data_classes = types.ModuleType("nuscenes.utils.data_classes")

        class _BasePointCloud:
            def __init__(self, points=None):
                self.points = torch.zeros((5, 0)) if points is None else points

            @classmethod
            def from_file(cls, *args, **kwargs):
                return cls()

            def remove_close(self, *args, **kwargs):
                return None

            def transform(self, *args, **kwargs):
                return None

            def nbr_points(self):
                return self.points.shape[1]

        data_classes.PointCloud = _BasePointCloud
        data_classes.RadarPointCloud = type(
            "RadarPointCloud",
            (_BasePointCloud,),
            {
                "disable_filters": staticmethod(lambda: None),
                "default_filters": staticmethod(lambda: None),
                "points": torch.zeros((19, 0)),
            },
        )
        data_classes.LidarPointCloud = type("LidarPointCloud", (_BasePointCloud,), {})
        data_classes.Box = type("Box", (), {})
        sys.modules["nuscenes.utils.data_classes"] = data_classes

        geometry_utils = types.ModuleType("nuscenes.utils.geometry_utils")
        geometry_utils.transform_matrix = lambda *args, **kwargs: torch.eye(4)
        sys.modules["nuscenes.utils.geometry_utils"] = geometry_utils

        splits = types.ModuleType("nuscenes.utils.splits")
        splits.create_splits_scenes = lambda: {"train": [], "val": [], "mini_train": [], "mini_val": [], "test": []}
        sys.modules["nuscenes.utils.splits"] = splits

        pyquaternion_module = types.ModuleType("pyquaternion")

        class _Quaternion:
            def __init__(self, values):
                self.rotation_matrix = torch.eye(3)

        pyquaternion_module.Quaternion = _Quaternion
        sys.modules["pyquaternion"] = pyquaternion_module

    class _FakeArray:
        def __init__(self, tensor: torch.Tensor):
            self.tensor = tensor

        def reshape(self, shape):
            return _FakeArray(self.tensor.reshape(*shape))

        def __getitem__(self, item):
            result = self.tensor.__getitem__(item)
            if isinstance(result, torch.Tensor):
                if result.dim() == 0:
                    return result.item()
                return _FakeArray(result.clone())
            return result

        def __setitem__(self, key, value):
            value_tensor = _to_tensor(value, dtype=self.tensor.dtype)
            self.tensor.__setitem__(key, value_tensor)

        def astype(self, dtype):
            return _FakeArray(self.tensor.to(_map_dtype(dtype)))

        def numpy(self):  # pragma: no cover
            raise NotImplementedError("NumPy is not available in this environment")

        def tolist(self):
            return self.tensor.detach().cpu().flatten().tolist()

    def _map_dtype(dtype):
        if dtype in (None, float):
            return torch.float32
        if dtype in ("float32", torch.float32):
            return torch.float32
        if dtype in ("float64", torch.float64):
            return torch.float64
        if dtype in ("int32", torch.int32):
            return torch.int32
        if isinstance(dtype, torch.dtype):
            return dtype
        return torch.float32

    def _to_tensor(value, dtype=torch.float32):
        if isinstance(value, _FakeArray):
            return value.tensor.to(dtype)
        if isinstance(value, torch.Tensor):
            return value.to(dtype)
        return torch.tensor(value, dtype=dtype)

    def _fake_array_from_data(data, dtype=None):
        tensor = _to_tensor(data, dtype=_map_dtype(dtype))
        return _FakeArray(tensor)

    def _stub_numpy() -> None:
        if "numpy" in sys.modules:
            return

        numpy_stub = types.ModuleType("numpy")

        numpy_stub.float32 = torch.float32
        numpy_stub.pi = math.pi

        def zeros(shape, dtype=None):
            return _fake_array_from_data(torch.zeros(*shape, dtype=_map_dtype(dtype)))

        def ones(shape, dtype=None):
            return _fake_array_from_data(torch.ones(*shape, dtype=_map_dtype(dtype)))

        def array(data, dtype=None):
            return _fake_array_from_data(data, dtype=dtype)

        def eye(n, dtype=None):
            return _fake_array_from_data(torch.eye(n, dtype=_map_dtype(dtype)))

        def concatenate(arrays, axis=0):
            tensors = [_to_tensor(arr.tensor if isinstance(arr, _FakeArray) else arr) for arr in arrays]
            return _fake_array_from_data(torch.cat(tensors, dim=axis))

        def allclose(a, b, atol=1e-8):
            return torch.allclose(_to_tensor(a), _to_tensor(b), atol=atol)

        def mod(a, b):
            return a % b

        class _Random:
            @staticmethod
            def seed(seed):
                torch.manual_seed(seed)

        numpy_stub.zeros = zeros
        numpy_stub.ones = ones
        numpy_stub.array = array
        numpy_stub.eye = eye
        numpy_stub.concatenate = concatenate
        numpy_stub.allclose = allclose
        numpy_stub.mod = mod
        numpy_stub.random = _Random()

        def cos(value):
            return math.cos(value)

        def sin(value):
            return math.sin(value)

        numpy_stub.cos = cos
        numpy_stub.sin = sin

        sys.modules["numpy"] = numpy_stub

        original_from_numpy = getattr(torch, "from_numpy", None)

        def _from_numpy_stub(array_like):
            if isinstance(array_like, _FakeArray):
                return array_like.tensor.clone()
            raise TypeError("torch.from_numpy requires numpy support")

        torch.from_numpy = _from_numpy_stub  # type: ignore[attr-defined]
        numpy_stub._torch_from_numpy_original = original_from_numpy

    def _stub_ops_modules() -> None:
        ops_modules = types.ModuleType("nets.ops.modules")

        class _DummyAttn(torch.nn.Module):  # type: ignore[name-defined]
            def __init__(self, *args, **kwargs):
                super().__init__()

            def forward(self, query, *args, **kwargs):
                return query

        ops_modules.MSDeformAttn = _DummyAttn
        ops_modules.MSDeformAttn3D = _DummyAttn
        _ensure_module("nets.ops.modules", ops_modules)

        ops_functions = types.ModuleType("nets.ops.functions")
        ops_functions.MSDeformAttnFunction = object
        _ensure_module("nets.ops.functions", ops_functions)

    def pytest_sessionstart(session):
        _stub_numpy()
        _stub_tensorboardx()
        _stub_wandb()
        _stub_nuscenes()
        _stub_ops_modules()
