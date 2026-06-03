#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
import re
import importlib.util
from PIL import Image, ImageChops
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
from utils.ironbow_utils import ironbow_to_gray_rgb_pil
from utils.temperature_utils import temperature_mae_from_gray_arrays, normalized_gray_to_temperature


def readImages(renders_dir, gt_dir, use_ironbow_inverse=False):
    renders = []
    gts = []
    gray_renders = []
    gray_gts = []
    image_names = []
    for fname in sorted(os.listdir(renders_dir)):
        render = Image.open(renders_dir / fname).convert("RGB")
        gt = Image.open(gt_dir / fname).convert("RGB")
        gray_render = image_to_temperature_gray(render, use_ironbow_inverse)
        gray_gt = image_to_temperature_gray(gt, use_ironbow_inverse)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        gray_renders.append(tf.to_tensor(gray_render).unsqueeze(0)[:, :3, :, :].cuda())
        gray_gts.append(tf.to_tensor(gray_gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, gray_renders, gray_gts, image_names

def get_metric_image_dirs(method_dir):
    pseudo_renders_dir = method_dir / "renders_pseudo"
    pseudo_gt_dir = method_dir / "gt_pseudo"
    if pseudo_renders_dir.is_dir() and pseudo_gt_dir.is_dir():
        return pseudo_renders_dir, pseudo_gt_dir, True
    return method_dir / "renders", method_dir / "gt", False


def is_rgb_gray(image):
    red, green, blue = image.split()
    return ImageChops.difference(red, green).getbbox() is None and ImageChops.difference(red, blue).getbbox() is None


def image_to_temperature_gray(image, force_ironbow_inverse=False):
    if force_ironbow_inverse or not is_rgb_gray(image):
        return ironbow_to_gray_rgb_pil(image)
    return image


def _load_dynamic_rgbt_defaults(path):
    metadata_path = Path(__file__).resolve().parent / "scene" / "dynamic_rgbt_metadata.py"
    if not metadata_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("dynamic_rgbt_metadata_for_metrics", metadata_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_dynamic_rgbt_scene_defaults(path)


def _read_source_path_from_cfg_args(scene_dir):
    cfg_path = Path(scene_dir) / "cfg_args"
    if not cfg_path.exists():
        return None
    try:
        text = cfg_path.read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(r"source_path=([\"'])(.*?)\1", text)
    if match:
        return match.group(2)
    return None


def load_temperature_bounds(scene_dir, temperature_bounds=None):
    if temperature_bounds is not None:
        return float(temperature_bounds[0]), float(temperature_bounds[1])

    metadata_path = Path(scene_dir) / "thermal_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
        return float(metadata.get("min_value", 10.0)), float(metadata.get("max_value", 50.0))

    for candidate in (_read_source_path_from_cfg_args(scene_dir), scene_dir):
        if candidate is None:
            continue
        defaults = _load_dynamic_rgbt_defaults(candidate)
        if defaults is not None:
            return float(defaults["min_value"]), float(defaults["max_value"])

    return 10.0, 50.0


def tensor_to_gray_numpy(image):
    gray = image[:, :3, :, :].mean(dim=1)
    return gray.squeeze(0).detach().cpu().numpy()


def save_temperature_comparison(image, gt_image, image_name, output_dir, min_temp, max_temp):
    import matplotlib.pyplot as plt

    image_temp = normalized_gray_to_temperature(tensor_to_gray_numpy(image), min_temp, max_temp)
    gt_temp = normalized_gray_to_temperature(tensor_to_gray_numpy(gt_image), min_temp, max_temp)
    abs_diff = abs(image_temp - gt_temp)
    mae_temp = float(abs_diff.mean())

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    im1 = axs[0].imshow(gt_temp, cmap="gray", vmin=min_temp, vmax=max_temp)
    axs[0].set_title("GT Temperature")
    fig.colorbar(im1, ax=axs[0], shrink=0.6, orientation="vertical", label="Temperature (C)")

    im2 = axs[1].imshow(abs_diff, cmap="coolwarm")
    axs[1].set_title("Absolute Temperature Difference")
    fig.colorbar(im2, ax=axs[1], shrink=0.6, orientation="vertical", label="Absolute Difference (C)")
    axs[1].text(0.5, -0.2, f"MAE: {mae_temp:.4f} C", transform=axs[1].transAxes, ha="center")

    im3 = axs[2].imshow(image_temp, cmap="gray", vmin=min_temp, vmax=max_temp)
    axs[2].set_title("Rendered Temperature")
    fig.colorbar(im3, ax=axs[2], shrink=0.6, orientation="vertical", label="Temperature (C)")

    plt.tight_layout()
    fig.savefig(output_dir / f"{Path(image_name).stem}_temperature_comparison.png")
    plt.close(fig)


def evaluate(model_paths, temperature_bounds=None, save_temp_viz=False):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"
            min_temp, max_temp = load_temperature_bounds(scene_dir, temperature_bounds)
            print("Temperature bounds: [{:.3f}, {:.3f}] C".format(min_temp, max_temp))

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                renders_dir, gt_dir, use_ironbow_inverse = get_metric_image_dirs(method_dir)
                renders, gts, gray_renders, gray_gts, image_names = readImages(
                    renders_dir, gt_dir, use_ironbow_inverse=use_ironbow_inverse
                )

                ssims = []
                psnrs = []
                lpipss = []
                mae_temps = []
                temp_viz_dir = method_dir / "temp_viz"

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    mae_temps.append(temperature_mae_from_gray_arrays(
                        tensor_to_gray_numpy(gray_renders[idx]),
                        tensor_to_gray_numpy(gray_gts[idx]),
                        min_temp,
                        max_temp,
                    ))
                    if save_temp_viz:
                        save_temperature_comparison(
                            gray_renders[idx],
                            gray_gts[idx],
                            image_names[idx],
                            temp_viz_dir,
                            min_temp,
                            max_temp,
                        )
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))

                print("  MAE_Temp : {:>12.7f}".format(torch.tensor(mae_temps).mean(), ".5"))
                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("")

                full_dict[scene_dir][method].update({"MAE_Temp": torch.tensor(mae_temps).mean().item(),
                                                        "Temperature_Min": min_temp,
                                                        "Temperature_Max": max_temp,
                                                        "SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "LPIPS": torch.tensor(lpipss).mean().item()})
                per_view_dict[scene_dir][method].update({"MAE_Temp": {name: mae for mae, name in zip(torch.tensor(mae_temps).tolist(), image_names)},
                                                            "SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except Exception as exc:
            print("Unable to compute metrics for model", scene_dir, ":", exc)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--temperature_bounds', nargs=2, type=float, default=None,
                        help="Override temperature bounds as: --temperature_bounds Tmin Tmax")
    parser.add_argument('--save_temp_viz', action="store_true",
                        help="Save per-view temperature comparison visualizations.")
    args = parser.parse_args()
    evaluate(args.model_paths, temperature_bounds=args.temperature_bounds, save_temp_viz=args.save_temp_viz)
