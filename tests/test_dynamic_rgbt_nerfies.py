import importlib.util
import json
import math
import sys
import tempfile
import types
import unittest
from collections import namedtuple
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]


def _install_reader_stubs():
    scene_pkg = types.ModuleType("scene")
    scene_pkg.__path__ = [str(REPO_ROOT / "scene")]

    colmap_loader = types.ModuleType("scene.colmap_loader")
    for name in (
        "read_extrinsics_text",
        "read_intrinsics_text",
        "read_extrinsics_binary",
        "read_intrinsics_binary",
        "read_points3D_binary",
        "read_points3D_text",
        "read_points3D_obj",
    ):
        setattr(colmap_loader, name, lambda *args, **kwargs: None)
    colmap_loader.qvec2rotmat = lambda qvec: np.eye(3)

    gaussian_model = types.ModuleType("scene.gaussian_model")
    gaussian_model.BasicPointCloud = namedtuple("BasicPointCloud", "points colors normals")

    graphics_utils = types.ModuleType("utils.graphics_utils")

    def get_world_to_view(R, T):
        matrix = np.eye(4)
        matrix[:3, :3] = R.T
        matrix[:3, 3] = T
        return matrix

    graphics_utils.getWorld2View2 = get_world_to_view
    graphics_utils.focal2fov = lambda focal, pixels: 2.0 * math.atan(float(pixels) / (2.0 * float(focal)))
    graphics_utils.fov2focal = lambda fov, pixels: float(pixels) / (2.0 * math.tan(float(fov) / 2.0))

    sh_utils = types.ModuleType("utils.sh_utils")
    sh_utils.SH2RGB = lambda sh: sh

    plyfile = types.ModuleType("plyfile")

    class _PlyData:
        @staticmethod
        def read(_path):
            vertex = np.array(
                [(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 255, 128, 0)],
                dtype=[
                    ("x", "f4"),
                    ("y", "f4"),
                    ("z", "f4"),
                    ("nx", "f4"),
                    ("ny", "f4"),
                    ("nz", "f4"),
                    ("red", "u1"),
                    ("green", "u1"),
                    ("blue", "u1"),
                ],
            )
            return {"vertex": vertex}

    class _PlyElement:
        @staticmethod
        def describe(*args, **kwargs):
            return None

    plyfile.PlyData = _PlyData
    plyfile.PlyElement = _PlyElement

    sys.modules.update(
        {
            "scene": scene_pkg,
            "scene.colmap_loader": colmap_loader,
            "scene.gaussian_model": gaussian_model,
            "utils.graphics_utils": graphics_utils,
            "utils.sh_utils": sh_utils,
            "plyfile": plyfile,
        }
    )


def _load_module(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _create_dynamic_rgbt_nerfies_scene(root):
    ids = ["000", "001", "002"]
    for modality in ("rgb", "thermal"):
        modality_root = root / modality
        image_dir = modality_root / "rgb" / "2x"
        camera_dir = modality_root / "camera"
        image_dir.mkdir(parents=True)
        camera_dir.mkdir(parents=True)
        _write_json(modality_root / "dataset.json", {"ids": ids, "train_ids": ["001", "002"], "val_ids": ["000"]})
        for image_id in ids:
            _write_json(
                camera_dir / f"{image_id}.json",
                {
                    "orientation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "position": [float(image_id), 0.0, 0.0],
                    "focal_length": 100.0,
                    "image_size": [100, 50],
                },
            )
            color = (10, 20, 30) if modality == "rgb" else (80, 80, 80)
            Image.new("RGB", (50, 25), color).save(image_dir / f"{image_id}.png")

    (root / "rgb" / "points3D.ply").write_text("stub", encoding="utf-8")


class DynamicRgbtNerfiesTest(unittest.TestCase):
    def test_scene_defaults_match_dynamic_rgbt_table(self):
        metadata = _load_module(
            "scene.dynamic_rgbt_metadata",
            REPO_ROOT / "scene" / "dynamic_rgbt_metadata.py",
        )

        heating_table = metadata.get_dynamic_rgbt_scene_defaults(r"D:\DynamicRGBT\Lab\HeatingTable")
        self.assertEqual(heating_table["fps"], 10.0)
        self.assertEqual(heating_table["T_env"], 25.0)
        self.assertEqual(heating_table["min_value"], 10.0)
        self.assertEqual(heating_table["max_value"], 120.0)

        ice_packs = metadata.get_dynamic_rgbt_scene_defaults("/data/DynamicRGBT/MeetingRoom/IcePacks")
        self.assertEqual(ice_packs["fps"], 30.0)
        self.assertEqual(ice_packs["min_value"], -10.0)
        self.assertEqual(ice_packs["max_value"], 50.0)

        covers = metadata.get_dynamic_rgbt_scene_defaults("/data/DynamicRGBT/MeetingRoom/Covers")
        self.assertEqual(covers["fps"], 10.0)
        self.assertEqual(covers["min_value"], 20.0)
        self.assertEqual(covers["max_value"], 80.0)

    def test_reader_uses_thermal_branch_and_keeps_global_time_indices(self):
        _install_reader_stubs()
        reader = _load_module("scene.dataset_readers", REPO_ROOT / "scene" / "dataset_readers.py")

        with tempfile.TemporaryDirectory() as tmp:
            scene_root = Path(tmp) / "DynamicRGBT" / "MeetingRoom" / "IcePacks"
            _create_dynamic_rgbt_nerfies_scene(scene_root)

            scene_info = reader.readNerfiesSceneInfo(
                str(scene_root), images=None, depths="", eval=True, train_test_exp=False
            )

            self.assertIs(reader.sceneLoadTypeCallbacks["Nerfies"], reader.readNerfiesSceneInfo)
            self.assertEqual([cam.image_name for cam in scene_info.train_cameras], ["001", "002"])
            self.assertEqual([cam.idx for cam in scene_info.train_cameras], [1, 2])
            self.assertEqual([cam.image_name for cam in scene_info.test_cameras], ["000"])
            self.assertTrue(all("thermal" in cam.image_path for cam in scene_info.train_cameras))
            self.assertEqual(scene_info.ply_path, str(scene_root / "rgb" / "points3D.ply"))
            self.assertFalse(scene_info.is_nerf_synthetic)


if __name__ == "__main__":
    unittest.main()
