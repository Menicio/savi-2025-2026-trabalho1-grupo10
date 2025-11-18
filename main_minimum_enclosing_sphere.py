#!/usr/bin/env python3
# SAVI - Trabalho 1
# Tarefa 3 – Esfera Englobante Mínima
# João Menício - 93300

import numpy as np
import open3d as o3d
from scipy.optimize import least_squares
from copy import deepcopy
from pathlib import Path
from utils_rgbd import load_and_prepare_cloud


VOXEL_SIZE = 0.025  # manter consistente com as outras tarefas

# ============================================================
# Função de resíduos da esfera:
# residual_i = max(0, ||p_i - centro|| - r)
# penaliza apenas pontos fora da esfera
# ============================================================

def sphere_residuals(params, points):
    """
    params = [xc, yc, zc, r]
    residual_i = max(0, ||p_i - c|| - r)
    """
    xc, yc, zc, r = params
    c = np.array([xc, yc, zc])

    d = np.linalg.norm(points - c, axis=1)   # distâncias ao centro
    res = np.maximum(0, d - r)                # só pontos fora contribuem

    return res


# ============================================================
# Otimização do centro e raio da esfera
# ============================================================

def compute_min_enclosing_sphere(points, init_center=None, init_r=1.0):

    if init_center is None:
        init_center = np.mean(points, axis=0)

    params0 = np.array([init_center[0], init_center[1], init_center[2], init_r])

    result = least_squares(
        fun=sphere_residuals,
        x0=params0,
        args=(points,),
        loss="soft_l1",
        f_scale=0.1,
        max_nfev=400
    )

    xc, yc, zc, r = result.x
    return np.array([xc, yc, zc]), abs(r), result


# ============================================================
# Visualização Open3D
# ============================================================

def o3d_draw_sphere(center, radius, clouds=[]):
    # esfera (mesh)
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.paint_uniform_color([1, 0, 0])        # vermelho
    sphere.compute_vertex_normals()
    sphere.translate(center)

    # clouds são point clouds → usam estimate_normals()
    for c in clouds:
        c.estimate_normals()

    o3d.visualization.draw_geometries([sphere] + clouds)


# ============================================================
# MAIN
# ============================================================

def main():

    voxel_size = 0.025
                 # Diretório base (mesma pasta do .py e das imagens)
    root = Path(__file__).resolve().parent
                 # Ficheiros RGB-D na mesma pasta

    rgb1_path   = root / "1.png"
    depth1_path = root / "depth1.png"
    rgb2_path   = root / "2.png"
    depth2_path = root / "depth2.png"

    
    
    # ---------------------------------------------------------
    # 1) Carregar imagens TUM (como nas tarefas anteriores)
    # ---------------------------------------------------------
    

    #rgb1   = o3d.io.read_image("/home/menicio/savi_25-26/Parte08/tum_dataset/rgb/1.png")
    #depth1 = o3d.io.read_image("/home/menicio/savi_25-26/Parte08/tum_dataset/depth/1.png")

    #rgb2   = o3d.io.read_image("/home/menicio/savi_25-26/Parte08/tum_dataset/rgb/2.png")
    #depth2 = o3d.io.read_image("/home/menicio/savi_25-26/Parte08/tum_dataset/depth/2.png")

    # RGBD
    #rgbd1 = o3d.geometry.RGBDImage.create_from_tum_format(rgb1, depth1)
    #rgbd2 = o3d.geometry.RGBDImage.create_from_tum_format(rgb2, depth2)

    # Intrinsecos
    intr = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
    )

      # ---------------------------------------------------------
    # 1) Carregar e preparar as clouds (flip + voxel + normais)
    # ---------------------------------------------------------
    pcd1 = load_and_prepare_cloud(rgb1_path, depth1_path, intr, VOXEL_SIZE)
    pcd2 = load_and_prepare_cloud(rgb2_path, depth2_path, intr, VOXEL_SIZE)


  
    # Pontos combinados
    pts1 = np.asarray(pcd1.points)
    pts2 = np.asarray(pcd2.points)
    pts = np.vstack((pts1, pts2))

    print("Total de pontos (PC1 + PC2):", pts.shape[0])

    # ---------------------------------------------------------
    # 2) Otimização da Esfera Mínima
    # ---------------------------------------------------------

    center, radius, result = compute_min_enclosing_sphere(pts)

    print("\n===== RESULTADO FINAL =====")
    print("Centro:", center)
    print("Raio  :", radius)
    print("Custo final:", result.cost)
    print("Número de iterações:", result.nfev)

    # ---------------------------------------------------------
    # 3) Visualização
    # ---------------------------------------------------------

    p1 = deepcopy(pcd1); p1.paint_uniform_color([0,1,0])   # verde
    p2 = deepcopy(pcd2); p2.paint_uniform_color([0,0,1])   # azul

    o3d_draw_sphere(center, radius, [p1, p2])


if __name__ == "__main__":
    main()
