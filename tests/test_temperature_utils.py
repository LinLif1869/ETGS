import importlib.util
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_temperature_utils():
    spec = importlib.util.spec_from_file_location(
        "temperature_utils_under_test",
        REPO_ROOT / "utils" / "temperature_utils.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TemperatureUtilsTest(unittest.TestCase):
    def test_normalized_gray_to_temperature_uses_scene_bounds(self):
        temperature_utils = _load_temperature_utils()
        gray = np.asarray([0.0, 0.5, 1.0], dtype=np.float32)

        temps = temperature_utils.normalized_gray_to_temperature(gray, 10.0, 120.0)

        np.testing.assert_allclose(temps, [10.0, 65.0, 120.0], atol=1e-5)

    def test_temperature_mae_from_gray_arrays_reports_celsius_error(self):
        temperature_utils = _load_temperature_utils()
        render = np.asarray([[0.0, 0.5]], dtype=np.float32)
        gt = np.asarray([[0.1, 0.4]], dtype=np.float32)

        mae = temperature_utils.temperature_mae_from_gray_arrays(render, gt, 20.0, 80.0)

        self.assertAlmostEqual(mae, 6.0, places=5)


if __name__ == "__main__":
    unittest.main()
