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

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text, read_points3D_obj
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
import re
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from scene.dynamic_rgbt_metadata import is_dynamic_rgbt_path

class CameraInfo(NamedTuple):
    uid: int
    idx: int 
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    depth_params: dict
    image_path: str
    image_name: str
    depth_path: str
    width: int
    height: int
    is_test: bool

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    is_nerf_synthetic: bool

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, depths_params, images_folder, depths_folder, test_cam_names_list):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        n_remove = len(extr.name.split('.')[-1]) + 1
        depth_params = None
        if depths_params is not None:
            try:
                depth_params = depths_params[extr.name[:-n_remove]]
            except:
                print("\n", key, "not found in depths_params")

        image_path = os.path.join(images_folder, extr.name)
        image_name = extr.name
        depth_path = os.path.join(depths_folder, f"{extr.name[:-n_remove]}.png") if depths_folder != "" else ""

        cam_info = CameraInfo(uid=uid, idx = 0,R=R, T=T, FovY=FovY, FovX=FovX, depth_params=depth_params,
                              image_path=image_path, image_name=image_name, depth_path=depth_path,
                              width=width, height=height, is_test=image_name in test_cam_names_list)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def _load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def _pick_nerfies_scale_dir(base_dir):
    for scale_dir in ("2x", "1x", "4x"):
        scale_path = os.path.join(base_dir, scale_dir)
        if os.path.isdir(scale_path):
            return scale_path
    return None

def _find_nerfies_modality_root(path, images):
    if images is not None:
        normalized = images.replace("\\", "/")
        modality, _, remainder = normalized.partition("/")
        if modality in ("thermal", "rgb"):
            modality_root = os.path.join(path, modality)
            if os.path.exists(os.path.join(modality_root, "dataset.json")):
                return modality_root, remainder if remainder else None

    for modality in ("thermal", "rgb"):
        modality_root = os.path.join(path, modality)
        if os.path.exists(os.path.join(modality_root, "dataset.json")):
            return modality_root, images

    return path, images

def _get_nerfies_image_dir(modality_root, images):
    reading_dir = "rgb" if images is None else images
    candidate_dirs = [os.path.join(modality_root, reading_dir)]
    if reading_dir.startswith("images/") or reading_dir.startswith("rgb/"):
        candidate_dirs.append(os.path.join(modality_root, reading_dir.split("/", 1)[1]))

    for candidate_dir in candidate_dirs:
        if os.path.isdir(candidate_dir):
            scale_path = _pick_nerfies_scale_dir(candidate_dir)
            return scale_path if scale_path is not None else candidate_dir

    for fallback_dir in ("rgb", "images"):
        candidate_dir = os.path.join(modality_root, fallback_dir)
        if os.path.isdir(candidate_dir):
            scale_path = _pick_nerfies_scale_dir(candidate_dir)
            return scale_path if scale_path is not None else candidate_dir

    scale_path = _pick_nerfies_scale_dir(modality_root)
    if scale_path is not None:
        return scale_path

    raise FileNotFoundError(f"Nerfies image directory does not exist under {modality_root}")

def _find_image_path(image_dir, image_id):
    for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
        image_path = os.path.join(image_dir, image_id + ext)
        if os.path.exists(image_path):
            return image_path
    return None

def _camera_params_from_nerfies(camera_json, width, height):
    orientation = np.asarray(camera_json["orientation"], dtype=np.float32)
    position = np.asarray(camera_json["position"], dtype=np.float32)

    R = orientation.T
    T = -position @ R

    focal = camera_json["focal_length"]
    if isinstance(focal, (list, tuple, np.ndarray)):
        focal_x = float(focal[0])
        focal_y = float(focal[1]) if len(focal) >= 2 else focal_x
    else:
        focal_x = float(focal)
        focal_y = focal_x

    image_size = camera_json.get("image_size", None)
    if image_size is not None and len(image_size) >= 2:
        base_w = float(image_size[0])
        base_h = float(image_size[1])
        if base_w > 0 and base_h > 0:
            focal_x *= float(width) / base_w
            focal_y *= float(height) / base_h

    focal_y *= float(camera_json.get("pixel_aspect_ratio", 1.0))
    return R, T, focal2fov(focal_y, height), focal2fov(focal_x, width)

def _get_nerfies_split_ids(all_ids, dataset_json, eval, llffhold=8):
    val_ids = set(dataset_json.get("val_ids", []))
    train_ids_cfg = set(dataset_json.get("train_ids", []))

    if eval:
        train_ids_set = train_ids_cfg if train_ids_cfg else set([image_id for image_id in all_ids if image_id not in val_ids])
        if val_ids:
            test_ids_set = val_ids
        else:
            test_ids_set = set([image_id for idx, image_id in enumerate(all_ids) if idx % llffhold == 0])
            train_ids_set = set(all_ids) - test_ids_set
    else:
        train_ids_set = set(all_ids)
        test_ids_set = set()

    train_ids = [image_id for image_id in all_ids if image_id in train_ids_set]
    test_ids = [image_id for image_id in all_ids if image_id in test_ids_set]
    return train_ids, test_ids

def _resolve_nerfies_depth_path(path, modality_root, depths, image_id):
    if depths == "":
        return ""

    for base_dir in (os.path.join(path, depths), os.path.join(modality_root, depths)):
        if os.path.isdir(base_dir):
            depth_path = _find_image_path(base_dir, image_id)
            return depth_path if depth_path is not None else os.path.join(base_dir, f"{image_id}.png")
    return os.path.join(path, depths, f"{image_id}.png")

def _load_nerfies_point_cloud(path, modality_root):
    candidates = [modality_root]
    for modality in ("rgb", "thermal"):
        root = os.path.join(path, modality)
        if root not in candidates:
            candidates.append(root)
    candidates.append(path)

    for root in candidates:
        ply_path = os.path.join(root, "points3D.ply")
        bin_path = os.path.join(root, "points3D.bin")
        txt_path = os.path.join(root, "points3D.txt")
        npy_path = os.path.join(root, "points.npy")

        if os.path.exists(ply_path):
            return fetchPly(ply_path), ply_path

        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
            storePly(ply_path, xyz, rgb)
            return fetchPly(ply_path), ply_path
        except Exception:
            pass

        try:
            xyz, rgb, _ = read_points3D_text(txt_path)
            storePly(ply_path, xyz, rgb)
            return fetchPly(ply_path), ply_path
        except Exception:
            pass

        if os.path.exists(npy_path):
            points = np.load(npy_path)
            if points.ndim != 2 or points.shape[1] < 3:
                raise ValueError(f"Invalid points.npy shape: {points.shape}, expected (N, >=3)")
            xyz = np.asarray(points[:, :3], dtype=np.float32)
            if points.shape[1] >= 6:
                rgb = np.asarray(points[:, 3:6], dtype=np.float32)
                if rgb.max() <= 1.0:
                    rgb = (rgb * 255.0).clip(0, 255)
                rgb = rgb.astype(np.uint8)
            else:
                rgb = np.full((xyz.shape[0], 3), 127, dtype=np.uint8)
            storePly(ply_path, xyz, rgb)
            return fetchPly(ply_path), ply_path

    raise FileNotFoundError(f"Nerfies point cloud not found in {path} (.ply/.bin/.txt/.npy)")

def readNerfiesSceneInfo(path, images, depths, eval, train_test_exp, llffhold=8):
    modality_root, images = _find_nerfies_modality_root(path, images)
    dataset_path = os.path.join(modality_root, "dataset.json")
    camera_dir = os.path.join(modality_root, "camera")
    if not os.path.exists(dataset_path) or not os.path.isdir(camera_dir):
        raise FileNotFoundError("Nerfies folder must contain dataset.json and camera/")

    dataset_json = _load_json(dataset_path)
    image_ids = list(dataset_json.get("ids", []))
    if len(image_ids) == 0:
        raise ValueError(f"No ids found in {dataset_path}")

    train_ids, test_ids = _get_nerfies_split_ids(image_ids, dataset_json, eval, llffhold=llffhold)
    train_ids_set = set(train_ids)
    test_ids_set = set(test_ids)
    image_dir = _get_nerfies_image_dir(modality_root, images)

    cam_infos = []
    for idx, image_id in enumerate(image_ids):
        camera_path = os.path.join(camera_dir, image_id + ".json")
        if not os.path.exists(camera_path):
            raise FileNotFoundError(f"Camera json not found: {camera_path}")
        camera_json = _load_json(camera_path)

        image_path = _find_image_path(image_dir, image_id)
        if image_path is None:
            raise FileNotFoundError(f"Image not found for id '{image_id}' in {image_dir}")

        with Image.open(image_path) as image:
            width, height = image.size

        R, T, FovY, FovX = _camera_params_from_nerfies(camera_json, width, height)
        cam_infos.append(
            CameraInfo(
                uid=idx,
                idx=idx,
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                depth_params=None,
                image_path=image_path,
                image_name=Path(image_path).stem,
                depth_path=_resolve_nerfies_depth_path(path, modality_root, depths, image_id),
                width=width,
                height=height,
                is_test=image_id in test_ids_set,
            )
        )

    train_cam_infos = [c for c, image_id in zip(cam_infos, image_ids) if train_test_exp or image_id in train_ids_set]
    test_cam_infos = [c for c, image_id in zip(cam_infos, image_ids) if image_id in test_ids_set]
    norm_cams = train_cam_infos if len(train_cam_infos) > 0 else test_cam_infos
    nerf_normalization = getNerfppNorm(norm_cams)
    pcd, ply_path = _load_nerfies_point_cloud(path, modality_root)

    if is_dynamic_rgbt_path(path):
        print("Detected DynamicRGBT Nerfies scene; using assumed FPS and thermal bounds.")

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=False)
    return scene_info

def readColmapSceneInfo(path, images, depths, eval, train_test_exp, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    ## if depth_params_file isnt there AND depths file is here -> throw error
    depths_params = None
    if depths != "":
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)

    if eval:
        if "360" in path:
            llffhold = 8
        if llffhold:
            print("------------LLFF HOLD-------------")
            cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]
            cam_names = sorted(cam_names)
            test_cam_names_list = [name for idx, name in enumerate(cam_names) if idx % llffhold == 0]
        else:
            with open(os.path.join(path, "sparse/0", "test.txt"), 'r') as file:
                test_cam_names_list = [line.strip() for line in file]
    else:
        test_cam_names_list = []

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, depths_params=depths_params,
        images_folder=os.path.join(path, reading_dir), 
        depths_folder=os.path.join(path, depths) if depths != "" else "", test_cam_names_list=test_cam_names_list)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=False)
    return scene_info

def readColmapSceneInfo_thermal(path, images, depths, eval, train_test_exp, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")

    for cam in cam_extrinsics:
        updated_name = cam_extrinsics[cam].name.replace("color", "red")
        cam_extrinsics[cam] = cam_extrinsics[cam]._replace(name = updated_name)
        
    ## if depth_params_file isnt there AND depths file is here -> throw error
    depths_params = None
    if depths != "":
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)

    if eval:
        if "360" in path:
            llffhold = 8
        if llffhold:
            print("------------LLFF HOLD-------------")
            cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]
            part = 4 
            cam_names = sorted(cam_names, key=lambda x: int(x[1][:9]))

            total_count = len(cam_names)
            part_size = total_count // part
            test_cam_names_list = []
            
            for i in range(part):
                start_idx = i * part_size
                end_idx = (i + 1) * part_size if i < part - 1 else total_count
                part_cam_names = cam_names[start_idx:end_idx]
                
                part_length = len(part_cam_names)
                test_count = max(1, part_length // llffhold)
                
                middle_start = (part_length - test_count) // 2
                middle_end = middle_start + test_count
                
                test_cam_names_list.extend(part_cam_names[middle_start:middle_end])

        else:
            with open(os.path.join(path, "sparse/0", "test.txt"), 'r') as file:
                test_cam_names_list = [line.strip() for line in file]
    else:
        test_cam_names_list = []

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, depths_params=depths_params,
        images_folder=os.path.join(path, reading_dir), 
        depths_folder=os.path.join(path, depths) if depths != "" else "", test_cam_names_list=test_cam_names_list)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    order = sorted(
        range(len(cam_infos)),
        key=lambda i: (int(cam_infos[i].image_name[:9]), i)
    )
    for rank, pos in enumerate(order):
        cam_infos[pos] = cam_infos[pos]._replace(idx=rank)

    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=False)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, depths_folder, white_background, is_test, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            depth_path = os.path.join(depths_folder, f"{image_name}.png") if depths_folder != "" else ""

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                            image_path=image_path, image_name=image_name,
                            width=image.size[0], height=image.size[1], depth_path=depth_path, depth_params=None, is_test=is_test))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, depths, eval, extension=".png"):

    depths_folder=os.path.join(path, depths) if depths != "" else ""
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", depths_folder, white_background, False, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", depths_folder, white_background, True, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=True)
    return scene_info

def readCamerasFromTransforms_NTR(path, transformsfile, depths_folder, white_background, is_test, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            # c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            depth_path = os.path.join(depths_folder, f"{image_name}.png") if depths_folder != "" else ""

            cam_infos.append(CameraInfo(uid=idx, idx=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                            image_path=image_path, image_name=image_name,
                            width=image.size[0], height=image.size[1], depth_path=depth_path, depth_params=None, is_test=is_test))
            
    return cam_infos



sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Colmap_thermal": readColmapSceneInfo_thermal,
    "Nerfies": readNerfiesSceneInfo,
    "Blender" : readNerfSyntheticInfo
}
