import os
import json
import torch
import GPUtil

def _real_id_from_visible(visible_idx: int) -> int:
    m = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not m:
        return visible_idx
    parts = [p.strip() for p in m.split(",") if p.strip() != ""]
    if 0 <= visible_idx < len(parts):
        try:
            return int(parts[visible_idx])
        except ValueError:
            pass
    return visible_idx

def _gpu_obj_by_real_id(real_id: int):
    gpus = GPUtil.getGPUs()
    for g in gpus:
        if int(g.id) == int(real_id):
            return g
    return gpus[0] if gpus else None

def save_efficiency_info(progress_bar, save_path):
    visible_idx = int(torch.cuda.current_device())
    real_id = _real_id_from_visible(visible_idx)
    gpu = _gpu_obj_by_real_id(real_id)
    used_gpu_memory = int(gpu.memoryUsed) if gpu is not None else -1  # MiB

    elapsed_time = float(f"{progress_bar.format_dict['elapsed']:.2f}")
    info = {
        "used_gpu_memory_MiB": used_gpu_memory,
        "elapsed_time": elapsed_time,
    }

    os.makedirs(save_path, exist_ok=True)
    json_path = os.path.join(save_path, "efficiency.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=4)
    print("saving efficiency information to", json_path)

def add_efficiency(model_path, speed):
    """
    Write the rendering speed to efficiency.json
    """
    json_path = os.path.join(model_path, "efficiency.json")
    try:
        with open(json_path, "r") as f:
            info = json.load(f)
    except Exception:
        info = {}
    info["render_speed"] = float(f"{speed:.2f}")
    with open(json_path, "w") as f:
        json.dump(info, f, ensure_ascii=False, indent=4)
        print("saving render speed", json_path)
