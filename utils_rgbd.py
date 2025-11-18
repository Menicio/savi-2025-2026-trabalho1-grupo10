# utils_rgbd.py
# Funções utilitárias para carregamento, conversão e preparação de clouds RGB-D (TUM + Open3D)

# João Menício - 93300
# Pascoal Sumbo - 123190

from pathlib import Path
import numpy as np
import open3d as o3d


# --------------------------
# Carregar imagens TUM
# --------------------------
def load_tum_pair(rgb_path, depth_path):
    rgb = o3d.io.read_image(str(rgb_path))
    depth = o3d.io.read_image(str(depth_path))
    rgbd = o3d.geometry.RGBDImage.create_from_tum_format(rgb, depth)
    return rgbd


# --------------------------
# Converter RGBD → PointCloud
# --------------------------
def rgbd_to_pcd(rgbd, intrinsics):
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)
    return pcd


# --------------------------
# Flip para Z ficar para a frente
# --------------------------
def flip_pointcloud(pcd):
    flip_T = np.array([
        [1, 0, 0, 0],
        [0,-1, 0, 0],
        [0, 0,-1, 0],
        [0, 0, 0, 1],
    ])
    pcd.transform(flip_T)
    return pcd


# --------------------------
# Downsample + Normais
# --------------------------
def prepare_pointcloud(pcd, voxel_size):
    pcd = pcd.voxel_down_sample(voxel_size)
    radius = voxel_size * 2.0
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius,
            max_nn=30
        )
    )
    return pcd


# --------------------------
# Função de alto nível:
# Carrega RGBD → PCD → Flip → Downsample → Normais
# --------------------------
def load_and_prepare_cloud(rgb_path, depth_path, intrinsics, voxel_size):
    rgbd = load_tum_pair(rgb_path, depth_path)
    pcd = rgbd_to_pcd(rgbd, intrinsics)
    pcd = flip_pointcloud(pcd)
    pcd = prepare_pointcloud(pcd, voxel_size)
    return pcd
