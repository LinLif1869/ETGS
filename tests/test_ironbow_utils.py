import importlib.util
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_ironbow_utils():
    spec = importlib.util.spec_from_file_location(
        "ironbow_utils_under_test",
        REPO_ROOT / "utils" / "ironbow_utils.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class IronbowUtilsTest(unittest.TestCase):
    def test_palette_has_expected_cold_and_hot_endpoints(self):
        ironbow = _load_ironbow_utils()
        np.testing.assert_array_equal(ironbow.IRONBOW_LUT[0], [0, 0, 10])
        np.testing.assert_array_equal(ironbow.IRONBOW_LUT[-1], [255, 255, 246])

    def test_palette_round_trip_recovers_gray_rgb_image(self):
        ironbow = _load_ironbow_utils()
        gray = np.array([[0, 32, 128, 255]], dtype=np.uint8)
        gray_rgb = np.repeat(gray[..., None], 3, axis=-1)

        pseudo = ironbow.gray_rgb_to_ironbow_pil(Image.fromarray(gray_rgb))
        recovered = np.asarray(ironbow.ironbow_to_gray_rgb_pil(pseudo))

        np.testing.assert_allclose(recovered, gray_rgb, atol=1)


if __name__ == "__main__":
    unittest.main()
