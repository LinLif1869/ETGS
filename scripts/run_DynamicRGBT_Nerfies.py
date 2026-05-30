import argparse
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    import GPUtil
except ImportError:
    GPUtil = None


DEFAULT_SCENES = [
    "HeatingTable",
    "HotBar",
    "HeatGun",
    "Bacon",
    "IcePacks",
    "HairDryer",
    "HairDryer2",
    "IronCloth",
    "Candles",
    "HotWater",
    "Foam",
    "Covers",
]


def _split_csv(value):
    if value is None or value == "":
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _split_extra_args(value):
    if value is None or value.strip() == "":
        return []
    return shlex.split(value)


def _scene_has_nerfies_files(scene_path):
    candidates = [scene_path, scene_path / "thermal", scene_path / "rgb"]
    return any((candidate / "dataset.json").exists() and (candidate / "camera").is_dir() for candidate in candidates)


def discover_scenes(data_root):
    root = Path(data_root)
    if _scene_has_nerfies_files(root):
        return [root.name]
    scenes = []
    for path in sorted([root] + [candidate for candidate in root.rglob("*") if candidate.is_dir()]):
        if path.name in ("rgb", "thermal"):
            continue
        if _scene_has_nerfies_files(path):
            scenes.append(str(path.relative_to(root)))
    if not scenes:
        raise RuntimeError(f"No Nerfies scenes found under {root}")
    return scenes


def resolve_scene_path(data_root, scene):
    root = Path(data_root)
    if _scene_has_nerfies_files(root) and root.name == scene:
        return root
    return root / scene


def run_command(command, gpu, cwd, dry_run):
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "4"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(" ".join(command))
    if dry_run:
        return
    subprocess.run(command, cwd=cwd, env=env, check=True)


def build_train_command(args, scene_path, model_path, gpu):
    command = [
        sys.executable,
        "train.py",
        "-s",
        str(scene_path),
        "-m",
        str(model_path),
        "--eval",
        "-r",
        str(args.resolution),
        "--iterations",
        str(args.iterations),
        "--checkpoint_iterations",
        str(args.iterations),
        "--port",
        str(args.port_base + int(gpu)),
    ]
    if args.disable_viewer:
        command.append("--disable_viewer")
    if args.images:
        command.extend(["--images", args.images])
    if args.extra_train_args:
        command.extend(_split_extra_args(args.extra_train_args))
    return command


def build_render_command(args, model_path):
    command = [
        sys.executable,
        "render.py",
        "-m",
        str(model_path),
        "--skip_train",
    ]
    if args.iteration is not None:
        command.extend(["--iteration", str(args.iteration)])
    if args.extra_render_args:
        command.extend(_split_extra_args(args.extra_render_args))
    return command


def build_metrics_command(model_path):
    return [
        sys.executable,
        "metrics.py",
        "-m",
        str(model_path),
    ]


def run_scene(args, gpu, scene):
    scene_path = resolve_scene_path(args.data_root, scene)
    model_path = Path(args.output_dir) / scene

    if not scene_path.exists():
        raise FileNotFoundError(f"Scene path does not exist: {scene_path}")

    print(f"Starting scene {scene} on GPU {gpu}")
    if args.stage in ("train", "all"):
        run_command(build_train_command(args, scene_path, model_path, gpu), gpu, args.cwd, args.dry_run)
    if args.stage in ("render", "all"):
        run_command(build_render_command(args, model_path), gpu, args.cwd, args.dry_run)
    if args.stage in ("metrics", "all"):
        run_command(build_metrics_command(model_path), gpu, args.cwd, args.dry_run)
    print(f"Finished scene {scene} on GPU {gpu}")


def available_gpus(excluded_gpus):
    if GPUtil is None:
        visible_devices = _split_csv(os.environ.get("CUDA_VISIBLE_DEVICES", ""))
        all_available = set(range(len(visible_devices))) if visible_devices else {0}
    else:
        all_available = set(GPUtil.getAvailable(order="first", limit=16, maxMemory=0.1))
    return sorted(all_available - excluded_gpus)


def dispatch_jobs(args, scenes):
    jobs = list(scenes)
    running = {}
    reserved_gpus = set()
    excluded_gpus = set(int(gpu) for gpu in _split_csv(args.exclude_gpus))

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        while jobs or running:
            free_gpus = [gpu for gpu in available_gpus(excluded_gpus) if gpu not in reserved_gpus]

            while free_gpus and jobs:
                gpu = free_gpus.pop(0)
                scene = jobs.pop(0)
                future = executor.submit(run_scene, args, gpu, scene)
                running[future] = (gpu, scene)
                reserved_gpus.add(gpu)

            done = [future for future in running if future.done()]
            for future in done:
                gpu, scene = running.pop(future)
                reserved_gpus.discard(gpu)
                future.result()
                print(f"Released GPU {gpu} after {scene}")

            time.sleep(args.poll_seconds)

    print("All jobs have been processed.")


def parse_args():
    parser = argparse.ArgumentParser(description="Run ETGS on DynamicRGBT Nerfies scenes.")
    parser.add_argument("--data_root", default="./datasets/DynamicRGBT", help="DynamicRGBT root or a single scene root.")
    parser.add_argument("--output_dir", default="dynamic_rgbt_nerfies", help="Output model root.")
    parser.add_argument("--scenes", nargs="+", default=None, help="Scene names. Defaults to auto discovery.")
    parser.add_argument("--stage", choices=["train", "render", "metrics", "all"], default="all")
    parser.add_argument("--resolution", "-r", default=1, type=int)
    parser.add_argument("--iterations", default=30000, type=int)
    parser.add_argument("--iteration", default=None, type=int, help="Render iteration. Default lets render.py load max iteration.")
    parser.add_argument("--images", default=None, help="Optional Nerfies image dir, e.g. rgb, rgb/2x, images/2x, thermal/rgb.")
    parser.add_argument("--exclude_gpus", default="", help="Comma-separated GPU ids to skip, e.g. 1,3.")
    parser.add_argument("--max_workers", default=8, type=int)
    parser.add_argument("--port_base", default=60019, type=int)
    parser.add_argument("--poll_seconds", default=5, type=int)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--disable_viewer", action="store_true", default=True)
    parser.add_argument("--viewer", dest="disable_viewer", action="store_false")
    parser.add_argument("--extra_train_args", default=None, help='Quoted extra args for train.py, e.g. "--lambda_dssim 0.2".')
    parser.add_argument("--extra_render_args", default=None, help='Quoted extra args for render.py.')
    args = parser.parse_args()
    args.cwd = Path(__file__).resolve().parents[1]
    return args


def main():
    args = parse_args()
    scenes = args.scenes if args.scenes is not None else discover_scenes(args.data_root)
    print("Scenes:", ", ".join(scenes))
    dispatch_jobs(args, scenes)


if __name__ == "__main__":
    main()
