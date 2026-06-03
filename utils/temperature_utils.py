import numpy as np


def normalized_gray_to_temperature(gray, min_temp, max_temp):
    gray = np.asarray(gray, dtype=np.float32)
    return np.clip(gray, 0.0, 1.0) * (float(max_temp) - float(min_temp)) + float(min_temp)


def temperature_mae_from_gray_arrays(render_gray, gt_gray, min_temp, max_temp):
    render_temp = normalized_gray_to_temperature(render_gray, min_temp, max_temp)
    gt_temp = normalized_gray_to_temperature(gt_gray, min_temp, max_temp)
    return float(np.mean(np.abs(render_temp - gt_temp)))
