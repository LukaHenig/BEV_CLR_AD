import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def nuscenes_data_module():
    created_modules: dict[str, types.ModuleType] = {}
    original_modules: dict[str, types.ModuleType] = {}

    def add_stub(name, module=None):
        if name in sys.modules:
            original_modules[name] = sys.modules[name]
            return sys.modules[name]
        if module is None:
            module = types.ModuleType(name)
        sys.modules[name] = module
        created_modules[name] = module
        return module

    # Stub the nested nuscenes modules that are imported inside nuscenes_data.py
    nuscenes_pkg = add_stub("nuscenes")
    map_expansion = add_stub("nuscenes.map_expansion")
    map_api = add_stub("nuscenes.map_expansion.map_api")
    map_api.NuScenesMap = type("NuScenesMap", (), {})

    nusc_core = add_stub("nuscenes.nuscenes")
    nusc_core.NuScenes = type("NuScenes", (), {"__init__": lambda self, *args, **kwargs: None})

    utils_pkg = add_stub("nuscenes.utils")
    data_classes = add_stub("nuscenes.utils.data_classes")
    geometry_utils = add_stub("nuscenes.utils.geometry_utils")
    splits = add_stub("nuscenes.utils.splits")
    pyquaternion_module = add_stub("pyquaternion")

    class _Quaternion:
        def __init__(self, values):
            self.rotation_matrix = np.eye(3, dtype=np.float32)

    pyquaternion_module.Quaternion = _Quaternion

    class _BasePointCloud:
        def __init__(self, points=None):
            self.points = np.zeros((5, 0)) if points is None else points

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
        (object,),
        {
            "__init__": lambda self: None,
            "from_file": classmethod(lambda cls, *args, **kwargs: cls()),
            "remove_close": lambda self, *args, **kwargs: None,
            "transform": lambda self, *args, **kwargs: None,
            "nbr_points": lambda self: 0,
            "disable_filters": staticmethod(lambda: None),
            "default_filters": staticmethod(lambda: None),
            "points": np.zeros((19, 0)),
        },
    )
    data_classes.LidarPointCloud = type(
        "LidarPointCloud",
        (_BasePointCloud,),
        {},
    )
    data_classes.Box = type("Box", (), {})

    geometry_utils.transform_matrix = lambda translation, rotation, inverse=False: np.eye(4, dtype=np.float32)
    splits.create_splits_scenes = lambda: {"train": [], "val": [], "mini_train": [], "mini_val": [], "test": []}

    # Load the nuscenes_data module with the stubs in place
    module_path = Path(__file__).resolve().parent.parent / "nuscenes_data.py"
    spec = importlib.util.spec_from_file_location("nuscenes_data", module_path)
    module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    assert loader is not None
    loader.exec_module(module)

    yield module

    # Clean up stubbed modules
    for name, module in created_modules.items():
        sys.modules.pop(name, None)
    for name, module in original_modules.items():
        sys.modules[name] = module


def test_convert_egopose_to_matrix_numpy_returns_valid_homogeneous_matrix(nuscenes_data_module):
    module = nuscenes_data_module
    egopose = {
        "rotation": [1.0, 0.0, 0.0, 0.0],
        "translation": [1.0, 2.0, 3.0],
    }
    matrix = module.convert_egopose_to_matrix_numpy(egopose)

    assert matrix.shape == (4, 4)
    assert np.allclose(matrix[3], np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
    assert np.allclose(matrix[:3, 3], np.array([1.0, 2.0, 3.0], dtype=np.float32))


def test_camera_coordinate_transforms_are_inverses(nuscenes_data_module):
    module = nuscenes_data_module

    # Points are expressed as 3 x N where columns are XYZ locations in the ego frame
    points = torch.tensor(
        [
            [0.0, 1.0, 2.0],
            [0.0, 1.0, 2.0],
            [5.0, 5.0, 5.0],
        ]
    )
    rot = torch.eye(3)
    trans = torch.zeros(3)
    intrins = torch.eye(3)

    cam_points = module.ego_to_cam(points.clone(), rot, trans, intrins)
    recovered = module.cam_to_ego(cam_points, rot, trans, intrins)

    assert torch.allclose(recovered, points, atol=1e-4)


def test_get_only_in_img_mask_filters_points_outside_image(nuscenes_data_module):
    module = nuscenes_data_module

    pts = torch.tensor([
        [10.0, 0.0, 100.0],  # x inside
        [10.0, 120.0, 100.0],  # y outside
        [50.0, 50.0, -5.0],  # behind camera
    ])

    mask = module.get_only_in_img_mask(pts, H=100, W=100)

    assert mask.dtype == torch.bool
    assert mask.sum() == 1


def test_img_transform_applies_resize_and_crop(nuscenes_data_module):
    module = nuscenes_data_module

    from PIL import Image

    image = Image.new("RGB", (20, 20), color=(255, 255, 255))
    resized = module.img_transform(image, resize_dims=(10, 10), crop=(0, 0, 5, 5))

    assert resized.size == (5, 5)
