import json
import os
import re


DYNAMIC_RGBT_T_ENV = 25.0


_DYNAMIC_RGBT_SCENE_DEFAULTS = {
    "heatingtable": {"fps": 10.0, "min_value": 10.0, "max_value": 120.0},
    "hottable": {"fps": 10.0, "min_value": 10.0, "max_value": 120.0},
    "hotbar": {"fps": 30.0, "min_value": 10.0, "max_value": 120.0},
    "heatgun": {"fps": 30.0, "min_value": 10.0, "max_value": 120.0},
    "bacon": {"fps": 30.0, "min_value": 20.0, "max_value": 100.0},
    "icepacks": {"fps": 30.0, "min_value": -10.0, "max_value": 50.0},
    "hairdryer": {"fps": 30.0, "min_value": 20.0, "max_value": 80.0},
    "ironcloth": {"fps": 30.0, "min_value": 20.0, "max_value": 80.0},
    "candles": {"fps": 30.0, "min_value": 20.0, "max_value": 80.0},
    "hotwater": {"fps": 30.0, "min_value": 20.0, "max_value": 80.0},
    "foam": {"fps": 30.0, "min_value": 20.0, "max_value": 70.0},
    "covers": {"fps": 10.0, "min_value": 20.0, "max_value": 80.0},
}


def _normalize_scene_name(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _path_parts(path):
    normalized = os.path.normpath(str(path))
    parts = []
    while True:
        head, tail = os.path.split(normalized)
        if tail:
            parts.append(tail)
            normalized = head
            continue
        if head:
            parts.append(head)
        break
    return parts


def get_dynamic_rgbt_scene_defaults(path):
    for part in _path_parts(path):
        key = _normalize_scene_name(part)
        for scene_key, defaults in _DYNAMIC_RGBT_SCENE_DEFAULTS.items():
            if key == scene_key or key.startswith(scene_key):
                result = dict(defaults)
                result["T_env"] = DYNAMIC_RGBT_T_ENV
                return result
    return None


def is_dynamic_rgbt_path(path):
    lower_parts = [_normalize_scene_name(part) for part in _path_parts(path)]
    return "dynamicrgbt" in lower_parts or get_dynamic_rgbt_scene_defaults(path) is not None


def find_nerfies_root(path):
    candidates = [str(path)]
    candidates.extend(os.path.join(str(path), child) for child in ("thermal", "rgb"))
    for candidate in candidates:
        if os.path.exists(os.path.join(candidate, "dataset.json")) and os.path.isdir(os.path.join(candidate, "camera")):
            return candidate
    return None


def is_nerfies_dataset(path):
    return find_nerfies_root(path) is not None


def load_nerfies_ids(path):
    root = find_nerfies_root(path)
    if root is None:
        return []
    dataset_path = os.path.join(root, "dataset.json")
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset_json = json.load(f)
    except Exception:
        return []
    return list(dataset_json.get("ids", []))


def build_frame_times(frame_count, fps):
    if frame_count <= 0:
        return []
    fps = float(fps)
    if fps <= 0:
        fps = 30.0
    return [idx / fps for idx in range(frame_count)]
