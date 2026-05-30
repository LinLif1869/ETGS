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

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os, re, glob
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.dynamic_rgbt_metadata import build_frame_times, get_dynamic_rgbt_scene_defaults, load_nerfies_ids

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, dataset, optimizer_type="default"):
        self.optimizer_type = optimizer_type 
        self._xyz = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        # thermal dynamics attributes
        self.temperature = torch.empty(0) # gaussian temperature
        self.t_env = torch.empty(0) # environment temperature
        self._T0   = torch.empty(0) # initial temperature
        self._tau  = torch.empty(0) # time constant
        self._gamma= torch.empty(0) # heat source scaling
        # Periodic excitation coefficients A_i,k / B_i,k
        self._A = torch.empty(0)
        self._B = torch.empty(0)
        # Frequency dictionary
        self.omega_k = None
        self.K = 0
        # time buffers
        self.times_np = None
        self.times = torch.empty(0)
        self.dts = torch.empty(0)
        self.U_sin = torch.empty(0)
        self.U_cos = torch.empty(0)
        # gray bounds 
        self.gray_min = torch.empty(0)
        self.gray_max = torch.empty(0)

        # init functions
        self.setup_functions()
        self.get_t_env(dataset)
        self.init_frequency_grid(dataset)
        self.build_time_buffers(dataset)
        self.load_gray_bounds(dataset)

    def capture(self):
        return (
            self._xyz,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (
        self._xyz, 
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    def get_t_env(self, dataset):
            """Get the environment temperature T_env from the info.json."""
            info_path = os.path.join(dataset.source_path, "info.json")
            if os.path.exists(info_path):
                try:
                    with open(info_path, 'r') as f:
                        info_data = json.load(f)
                    t_env_value = info_data.get("T_env", 26.0)
                    self.t_env = torch.tensor(t_env_value, device="cuda", dtype=torch.float32)
                    print(f"T_env loaded: {t_env_value}")
                except Exception as e:
                    print(f"Reading info.json failed: {e}, using default value 26.0")
                    self.t_env = torch.tensor(26.0, device="cuda", dtype=torch.float32)  # default value
            else:
                defaults = get_dynamic_rgbt_scene_defaults(dataset.source_path)
                if defaults is not None:
                    t_env_value = defaults["T_env"]
                    print(f"DynamicRGBT metadata inferred: T_env={t_env_value}")
                    self.t_env = torch.tensor(t_env_value, device="cuda", dtype=torch.float32)
                else:
                    print(f"File not found: {info_path}, using default value 26.0")
                    self.t_env = torch.tensor(26.0, device="cuda", dtype=torch.float32)  # default value

    def init_frequency_grid(self, dataset, alpha=0.25, K_default=24):
        """
        A globally shared frequency dictionary {omega_k} is constructed based on sampling time and thermal priors.
        Here, timestamps are parsed from filenames:
        e.g., "205502308.png" -> 20:55:02.308
        """
        # ---------- 1) Parse the timestamp (in seconds) from the filename. ----------
        times = None
        try:
            img_dir = os.path.join(str(dataset.source_path), "images")
            pat = re.compile(r"(\d{9})(?:\D|$)")  # 9-bit timecode
            def _timecode_key(path):
                m = pat.search(os.path.basename(path))
                return int(m.group(1)) if m else 10**9
            files = sorted(glob.glob(os.path.join(img_dir, "*")), key=_timecode_key)

            times_sod = []  # second-of-day
            for f in files:
                name = os.path.basename(f)
                m = pat.search(name)
                if not m:
                    continue
                t9 = m.group(1)  # e.g. "205502308"
                hh, mm, ss, ms = int(t9[0:2]), int(t9[2:4]), int(t9[4:6]), int(t9[6:9])
                sod = hh * 3600.0 + mm * 60.0 + ss + ms / 1000.0
                times_sod.append(sod)

            if len(times_sod) >= 2:
                t = np.asarray(times_sod, dtype=np.float64)
                unwrapped = np.empty_like(t)
                offset = 0.0
                unwrapped[0] = t[0]
                for i in range(1, len(t)):
                    if t[i] + 1e-9 < t[i - 1]:
                        offset += 86400.0
                        print("Entanglement occurs")
                    unwrapped[i] = t[i] + offset
                times = unwrapped
        except Exception as e:
            print(f"[freq-grid] filename parsing failed: {e}")

        if times is None or len(times) < 2:
            if hasattr(dataset, "timestamps") and dataset.timestamps is not None and len(dataset.timestamps) >= 2:
                times = np.asarray(dataset.timestamps, dtype=np.float64)
            elif hasattr(dataset, "frame_times") and dataset.frame_times is not None and len(dataset.frame_times) >= 2:
                times = np.asarray(dataset.frame_times, dtype=np.float64)
            else:
                defaults = get_dynamic_rgbt_scene_defaults(dataset.source_path)
                nerfies_ids = load_nerfies_ids(dataset.source_path)
                if defaults is not None and len(nerfies_ids) >= 2:
                    times = np.asarray(build_frame_times(len(nerfies_ids), defaults["fps"]), dtype=np.float64)
                    print(f"[freq-grid] DynamicRGBT Nerfies times inferred from fps={defaults['fps']:.1f}, frames={len(nerfies_ids)}")

        if times is None or len(times) < 2:
            T_span = 3600.0
            dt_min = 20.0
            times = np.asarray([0.0, dt_min], dtype=np.float64)
        else:
            T_span = float(times[-1] - times[0])
            diffs = np.diff(times)
            dt_min = float(np.clip(np.min(diffs), 1e-3, None))

        # ---------- 2) frequency band ----------
        tau_min = float(getattr(dataset, "tau_min", 60.0))  
        f_min = max(1.0 / max(T_span, 1e-3), 1e-5)
        f_nyq = 1.0 / (2.0 * dt_min)                        
        # Thermal bandwidth
        f_th = (1.0 / (2.0 * np.pi * tau_min)) * np.sqrt(max(1.0 / (alpha * alpha) - 1.0, 0.0))
        f_max = max(min(f_nyq, f_th), f_min * 1.2) 

        # ---------- 3) Generate a logarithmic frequency grid ----------
        K = int(getattr(dataset, "K_freq", K_default))
        f_grid = np.logspace(np.log10(f_min), np.log10(f_max), num=max(K, 2), base=10.0, dtype=np.float32)
        f_sep = 1.0 / max(2.0 * T_span, 1e-3)
        keep = [0]
        for i in range(1, len(f_grid)):
            if (f_grid[i] - f_grid[keep[-1]]) >= f_sep:
                keep.append(i)
        f_grid = f_grid[keep]
        omega = (2.0 * np.pi * f_grid).astype(np.float32)

        if omega.size < 2:
            f_bump = min(f_max * 1.5, f_nyq) if f_max > 0 else 1e-3
            omega = np.array([2*np.pi*f_min, 2*np.pi*max(f_bump, f_min*1.1)], dtype=np.float32)

        # ---------- 4) write buffer ----------
        self.K = int(len(omega))
        self.omega_k = torch.tensor(omega, device="cuda", dtype=torch.float32)
        self.times_np = times.astype(np.float32)
        print(f"[freq-grid] K={self.K}, f in [{(omega[0]/(2*np.pi)):.4e}, {(omega[-1]/(2*np.pi)):.4e}] Hz, "
            f"dt_min={dt_min:.3f}s, T_span={T_span:.1f}s, tau_min={tau_min:.1f}s")

    def build_time_buffers(self, dataset):
        """
        Pre-compute times[t_n], dts[n] = t_{n+1}-t_n, and U_sin/U_cos
        """
        # 1) Get times
        if not hasattr(self, "times_np") or self.times_np is None:
            print("[time-buf] times not found, re-parse via init_frequency_grid ...")
            self.init_frequency_grid(dataset)
        t = self.times_np.astype(np.float64)
        if t.shape[0] < 2:
            raise RuntimeError("Not enough timestamps to build time buffers.")
        # 2) Compute Δt_n
        dts = np.diff(t).astype(np.float32)
        # 3) Compute sin/cos bases
        omega = self.omega_k.detach().cpu().numpy().astype(np.float64)
        T, W = np.meshgrid(t, omega, indexing="ij")
        U_sin = np.sin(W * T).astype(np.float32)
        U_cos = np.cos(W * T).astype(np.float32)

        # 4) Save as buffer
        self.times = torch.from_numpy(t.astype(np.float32)).cuda()
        self.dts   = torch.from_numpy(dts).cuda()
        self.U_sin = torch.from_numpy(U_sin).cuda()
        self.U_cos = torch.from_numpy(U_cos).cuda()
        print(f"[time-buf] built: N={t.shape[0]}, K={omega.shape[0]}")
    
    def load_gray_bounds(self, dataset):
        """
        Read the min/max mapping boundaries from info.json;
        """
        info_path = os.path.join(str(dataset.source_path), "info.json")
        t_env_fallback = float(getattr(self, "t_env", 295.15).item() if hasattr(self, "t_env") else 295.15)

        tmin, tmax = None, None
        try:
            with open(info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
            tmin = float(info["min_value"])
            tmax = float(info["max_value"])
        except Exception as e:
            defaults = get_dynamic_rgbt_scene_defaults(dataset.source_path)
            if defaults is not None:
                tmin = float(defaults["min_value"])
                tmax = float(defaults["max_value"])
                print(f"[gray-bounds] DynamicRGBT metadata inferred from scene path: {dataset.source_path}")
            else:
                print(f"[gray-bounds] fail to read {info_path}: {e}")
                tmin = float(getattr(dataset, "tmin_fallback", t_env_fallback - 5.0))
                tmax = float(getattr(dataset, "tmax_fallback", t_env_fallback + 5.0))

        if not (tmax > tmin):
            tmax = tmin + 1.0

        # Save as GPU constant
        self.gray_min = torch.tensor(tmin, device="cuda", dtype=torch.float32)
        self.gray_max = torch.tensor(tmax, device="cuda", dtype=torch.float32)
        print(f"[gray-bounds] Tmin={tmin:.3f}, Tmax={tmax:.3f} (Celsius Temperature)")
    
    def temperature_to_gray(self, T: torch.Tensor) -> torch.Tensor:
        """
        Linear normalization using min/max from info.json: 
        gray = clamp((T - Tmin)/(Tmax - Tmin), 0, 1)
        """
        tmin = self.gray_min.to(T.device, T.dtype)
        tmax = self.gray_max.to(T.device, T.dtype)
        gray = (T - tmin) / torch.clamp(tmax - tmin, min=1e-6)
        gray = torch.clamp(gray, 0.0, 1.0)
        override_color = gray.expand(-1, 3)
        return override_color
    
    def temperature_closed_form(self, idxs: torch.Tensor) -> torch.Tensor:
        """
        Closed-form calculation;
        Formula:
        T(t)=T_env + (T0-T_env)e^{-Δt/τ}
                + Σ_k [ τ/(C(1+ω^2τ^2)) * { A[sin(ωt)-ωτ cos(ωt)+ωτ e^{-Δt/τ}]
                                            + B[cos(ωt)+ωτ sin(ωt)-e^{-Δt/τ}] } ]
        Where γ=τ/C ⇒ τ/(C(1+ω^2τ^2)) = γ/(1+ω^2τ^2)。
        """
        device = self._tau.device
        t = self.times[idxs].to(device=device, dtype=self._tau.dtype)             
        t0 = self.times[0].to(device=device, dtype=self._tau.dtype)
        dt = (t - t0)                                                             

        N = self._tau.shape[0]
        K = self.K
        M = t.shape[0]

        # Pre-calculated trigonometric basis
        if hasattr(self, "U_sin") and self.U_sin is not None:
            S = self.U_sin[idxs, :]
            C = self.U_cos[idxs, :]
        else:
            w = self.omega_k.view(1, K).to(device=device, dtype=self._tau.dtype)
            S = torch.sin(t.view(M, 1) * w)
            C = torch.cos(t.view(M, 1) * w)

        # Parameters
        tau   = torch.clamp(self._tau,   min=1e-3)
        gamma = torch.clamp(self._gamma, min=1e-8)
        Acoef = self._A
        Bcoef = self._B
        T0    = self._T0
        Te    = self.t_env

        a = torch.exp(-dt.view(1, M) / tau)

        # ωτ, (1+ω^2τ^2), Gain coefficient: g = γ / (1+ω^2τ^2)
        w = self.omega_k.view(1, K).to(device=device, dtype=tau.dtype)
        wt = w * tau
        denom = 1.0 + wt * wt
        g = gamma / denom

        # Part1:  Σ g*A*sin + g*B*cos
        part1 = (Acoef * g) @ S.t() + (Bcoef * g) @ C.t()

        # Part2:  Σ ( -g*A*ωτ*cos + g*B*ωτ*sin )
        Agwt = (Acoef * g) * wt
        Bgwt = (Bcoef * g) * wt
        part2 = (-Agwt) @ C.t() + (Bgwt) @ S.t()

        # Part3:  e^{-Δt/τ} * Σ ( g*A*ωτ - g*B )
        vA = Agwt.sum(dim=1, keepdim=True)
        vB = (Bcoef * g).sum(dim=1, keepdim=True)
        part3 = a * (vA - vB)

        # Final T
        T = Te + (T0 - Te) * a + part1 + part2 + part3
        temperature = T.permute(1, 0).unsqueeze(-1)
        self.temperature = temperature[0]
        return self.temperature


    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

        # ================== Thermal params init ==================
        N = fused_point_cloud.shape[0]
        T0_init = torch.full((N, 1), float(self.t_env.item()), device="cuda")
        tau_default = float(getattr(cam_infos, "tau_init", getattr(self, "tau_init", 60.0)))
        tau_init = torch.full((N, 1), tau_default, device="cuda")
        gamma_default = float(getattr(cam_infos, "gamma_init", getattr(self, "gamma_init", 1e-3)))
        gamma_init = torch.full((N, 1), gamma_default, device="cuda")
        K = int(self.K)
        if K == 0:
            self.omega_k = torch.tensor([2*np.pi/max(3600.0,1.0)], device="cuda", dtype=torch.float32)
            K = 1
            self.K = 1
            print("[freq-grid] Fallback to K=1 at ~1 hour period.")
        A_init = torch.zeros((N, K), device="cuda", dtype=torch.float32)
        B_init = torch.zeros((N, K), device="cuda", dtype=torch.float32)

        # Register as learnable parameters
        self._T0    = nn.Parameter(T0_init.requires_grad_(True))
        self._tau   = nn.Parameter(tau_init.requires_grad_(True))
        self._gamma = nn.Parameter(gamma_init.requires_grad_(True))
        self._A     = nn.Parameter(A_init.requires_grad_(True))
        self._B     = nn.Parameter(B_init.requires_grad_(True))
        # =========================================================

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._T0], 'lr': training_args.T0, "name": "T0"},
            {'params': [self._tau], 'lr': training_args.tau_lr, "name": "tau"},
            {'params': [self._gamma], 'lr': training_args.gamma_lr, "name": "gamma"},
            {'params': [self._A], 'lr': training_args.ab_lr, "name": "A"},
            {'params': [self._B], 'lr': training_args.ab_lr, "name": "B"},
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        # ---- thermal per-Gaussian ----
        l += ['T0', 'tau', 'gamma']
        for k in range(int(self.K)):
            l.append(f'A_{k}')
        for k in range(int(self.K)):
            l.append(f'B_{k}')
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        # ---------- vertex (per-Gaussian) ----------
        xyz = self._xyz.detach().cpu().numpy()
        N = xyz.shape[0]
        normals = np.zeros_like(xyz)

        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        # thermal per-Gaussian
        T0 = self._T0.detach().cpu().numpy()
        tau = self._tau.detach().cpu().numpy()
        gamma = self._gamma.detach().cpu().numpy()
        A = self._A.detach().cpu().numpy()
        B = self._B.detach().cpu().numpy()

        # dtype & pack
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        elements = np.empty(N, dtype=dtype_full)

        attrs = [
            xyz, normals,
            opacities, scale, rotation,
            T0, tau, gamma,
            A, B
        ]
        attributes = np.concatenate(attrs, axis=1)
        elements[:] = list(map(tuple, attributes))
        el_vertex = PlyElement.describe(elements, 'vertex')

        # ---------- meta (global, single row) ----------
        K = int(self.K)
        omega = (self.omega_k.detach().cpu().numpy().astype(np.float32) if self.omega_k is not None else np.zeros((0,), np.float32))
        t_env = float(self.t_env.detach().cpu().numpy()) if torch.is_tensor(self.t_env) else float(self.t_env)
        gmin = float(self.gray_min.detach().cpu().numpy()) if torch.is_tensor(self.gray_min) else float(self.gray_min)
        gmax = float(self.gray_max.detach().cpu().numpy()) if torch.is_tensor(self.gray_max) else float(self.gray_max)

        meta_dtype = [('K', 'i4'), ('t_env', 'f4'), ('gray_min', 'f4'), ('gray_max', 'f4')] + [(f'omega_{i}', 'f4') for i in range(K)]
        meta = np.empty(1, dtype=meta_dtype)
        meta['K'][0] = K
        meta['t_env'][0] = t_env
        meta['gray_min'][0] = gmin
        meta['gray_max'][0] = gmax
        for i in range(K):
            meta[f'omega_{i}'][0] = omega[i]
        el_meta = PlyElement.describe(meta, 'meta')

        # ---------- write ----------
        PlyData([el_vertex, el_meta]).write(path)
        print(f"[save_ply] Saved {N} vertices + meta(K={K}) to: {path}")

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)

        # exposures
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        # -------- find elements --------
        el_vertex = None
        el_meta = None
        for el in plydata.elements:
            if el.name == 'vertex':
                el_vertex = el
            if el.name == 'meta':
                el_meta = el

        if el_vertex is None:
            raise RuntimeError("PLY missing 'vertex' element.")
        if el_meta is None:
            raise RuntimeError("PLY missing 'meta' element.")
        
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        if "T0" not in el_vertex.data.dtype.names or "tau" not in el_vertex.data.dtype.names or "gamma" not in el_vertex.data.dtype.names:
            raise RuntimeError("PLY missing thermal per-vertex fields (T0/tau/gamma).")
        T0 = np.asarray(el_vertex["T0"])[..., np.newaxis].astype(np.float32)
        tau = np.asarray(el_vertex["tau"])[..., np.newaxis].astype(np.float32)
        gamma = np.asarray(el_vertex["gamma"])[..., np.newaxis].astype(np.float32)

        A_names = [p.name for p in el_vertex.properties if p.name.startswith("A_")]
        B_names = [p.name for p in el_vertex.properties if p.name.startswith("B_")]
        A_names = sorted(A_names, key=lambda x: int(x.split('_')[-1]))
        B_names = sorted(B_names, key=lambda x: int(x.split('_')[-1]))
        A = np.zeros((xyz.shape[0], len(A_names)), dtype=np.float32)
        B = np.zeros((xyz.shape[0], len(B_names)), dtype=np.float32)
        for idx, attr_name in enumerate(A_names):
            A[:, idx] = np.asarray(el_vertex[attr_name])
        for idx, attr_name in enumerate(B_names):
            B[:, idx] = np.asarray(el_vertex[attr_name])

        # -------- meta --------
        if el_meta is not None:
            K_meta = int(np.asarray(el_meta["K"])[0])
            t_env = float(np.asarray(el_meta["t_env"])[0])
            gmin = float(np.asarray(el_meta["gray_min"])[0])
            gmax = float(np.asarray(el_meta["gray_max"])[0])
            omega = []
            for i in range(K_meta):
                omega.append(float(np.asarray(el_meta[f'omega_{i}'])[0]))
            omega = np.asarray(omega, dtype=np.float32)

            self.K = K_meta
            self.omega_k = torch.tensor(omega, device="cuda", dtype=torch.float32)
            self.t_env = torch.tensor(t_env, device="cuda", dtype=torch.float32)
            self.gray_min = torch.tensor(gmin, device="cuda", dtype=torch.float32)
            self.gray_max = torch.tensor(gmax, device="cuda", dtype=torch.float32)
            print(f"[load_ply] meta loaded: K={self.K}, Tmin={gmin:.3f}, Tmax={gmax:.3f}, T_env={t_env:.3f}")
        else:
            self.K = int(min(len(A_names), len(B_names)))
            if self.omega_k is None or self.K != int(self.omega_k.numel()):
                print("[load_ply] WARNING: 'meta' missing; K inferred from A/B, omega_k unknown. "
                    "Set self.omega_k manually for closed-form evaluation.")
                
        # -------- Write back to nn.Parameter --------
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        # thermal
        self._T0    = nn.Parameter(torch.tensor(T0,    dtype=torch.float32, device="cuda").requires_grad_(True))
        self._tau   = nn.Parameter(torch.tensor(tau,   dtype=torch.float32, device="cuda").requires_grad_(True))
        self._gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32, device="cuda").requires_grad_(True))
        if self.K > 0:
            self._A = nn.Parameter(torch.tensor(A[:, :self.K], dtype=torch.float32, device="cuda").requires_grad_(True))
            self._B = nn.Parameter(torch.tensor(B[:, :self.K], dtype=torch.float32, device="cuda").requires_grad_(True))
        else:
            self._A = nn.Parameter(torch.zeros((xyz.shape[0], 0), dtype=torch.float32, device="cuda").requires_grad_(True))
            self._B = nn.Parameter(torch.zeros((xyz.shape[0], 0), dtype=torch.float32, device="cuda").requires_grad_(True))

        print(f"[load_ply] Loaded {xyz.shape[0]} vertices with thermal params. ")

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self._T0 = optimizable_tensors["T0"]
        self._tau = optimizable_tensors["tau"]
        self._gamma = optimizable_tensors["gamma"]
        self._A = optimizable_tensors["A"]
        self._B = optimizable_tensors["B"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_opacities, new_scaling, new_rotation, new_tmp_radii, new_T0, new_tau, new_gamma, new_A, new_B):
        d = {"xyz": new_xyz,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "T0" : new_T0,
        "tau" : new_tau,
        "gamma" : new_gamma,
        "A" : new_A,
        "B" : new_B
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._T0 = optimizable_tensors["T0"]
        self._tau = optimizable_tensors["tau"]
        self._gamma = optimizable_tensors["gamma"]
        self._A = optimizable_tensors["A"]
        self._B = optimizable_tensors["B"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)
        new_T0 = self._T0[selected_pts_mask].repeat(N, 1)
        new_tau = self._tau[selected_pts_mask].repeat(N, 1)
        new_gamma = self._gamma[selected_pts_mask].repeat(N, 1)
        new_A = self._A[selected_pts_mask].repeat(N, 1)
        new_B = self._B[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_opacity, new_scaling, new_rotation, new_tmp_radii, new_T0, new_tau, new_gamma, new_A, new_B)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_T0 = self._T0[selected_pts_mask]
        new_tau = self._tau[selected_pts_mask]
        new_gamma = self._gamma[selected_pts_mask]
        new_A = self._A[selected_pts_mask]
        new_B = self._B[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        self.densification_postfix(new_xyz, new_opacities, new_scaling, new_rotation, new_tmp_radii, new_T0, new_tau, new_gamma, new_A, new_B)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
