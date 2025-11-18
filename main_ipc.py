#!/usr/bin/env python3
# SAVI - Trabalho 1
# Tarefa 1 — ICP com Open3D
# João Menício - 93300
# Pascoal Sumbo - 123190

"""
Registo de duas nuvens de pontos RGB-D usando ICP (Open3D).
As imagens usadas são 1.png e 2.png do dataset TUM.
"""

from copy import deepcopy
from pathlib import Path
import time

import numpy as np
import open3d as o3d

from utils_rgbd import load_and_prepare_cloud  # módulo utilitário

# ==========================
# PARÂMETROS EDITÁVEIS
# ==========================
VOXEL_SIZE = 0.025               # Tamanho do voxel (m), deve ser coerente com a cena
MAX_CORR_DIST = 0.08             # Distância máx. para correspondências (~3 x VOXEL_SIZE)
ICP_METHOD = "point_to_plane"    # "point_to_plane" ou "point_to_point"
MAX_ITERS = 100                  # Iterações máximas do ICP
SHOW_VIEWER = True               # Mostrar janela 3D no final
# ==========================


def main():
    # Diretório base do projeto (pasta onde está este ficheiro)
    root = Path(__file__).resolve().parent


    filename_rgb1   = root / "1.png"
    filename_depth1 = root / "depth1.png"
    filename_rgb2   = root / "2.png"
    filename_depth2 = root / "depth2.png"


    # Caminhos relativos para as imagens TUM
    #rgb_dir = root / "tum_dataset" / "rgb"
    #depth_dir = root / "tum_dataset" / "depth"

    #filename_rgb1 = rgb_dir / "1.png"
    #filename_depth1 = depth_dir / "1.png"
    #filename_rgb2 = rgb_dir / "2.png"
    #filename_depth2 = depth_dir / "2.png"

    # -----------------------------
    # Intrínsecos da câmara
    # -----------------------------
    intr = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
    )
    # Alternativa TUM explícita (ex: fr1) se necessário:
    # width  = np.asarray(o3d.io.read_image(str(filename_rgb1))).shape[1]
    # height = np.asarray(o3d.io.read_image(str(filename_rgb1))).shape[0]
    # intr = o3d.camera.PinholeCameraIntrinsic(width, height, 517.3, 516.5, 318.6, 255.3)

    # -----------------------------
    # Construção e preparação das point clouds
    # (leitura RGB-D, criação de cloud, flip, voxel + normais)
    # -----------------------------
    pcd1_ds = load_and_prepare_cloud(filename_rgb1, filename_depth1, intr, VOXEL_SIZE)
    pcd2_ds = load_and_prepare_cloud(filename_rgb2, filename_depth2, intr, VOXEL_SIZE)

    # ------------------------------------
    # Registo com ICP
    # ------------------------------------

    # Escolhe qual é a fonte e o alvo (a fonte é movida)
    source = deepcopy(pcd1_ds)   # fonte
    target = deepcopy(pcd2_ds)   # alvo

    # Método de estimação
    if ICP_METHOD.lower() == "point_to_plane":
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    elif ICP_METHOD.lower() == "point_to_point":
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    else:
        raise ValueError('ICP_METHOD deve ser "point_to_plane" ou "point_to_point"')

    # Transformação inicial (identidade)
    trans_init = np.eye(4)

    # Critério de convergência do ICP
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=MAX_ITERS,
        relative_fitness=1e-6,
        relative_rmse=1e-6
    )

    # Executa o ICP
    t0 = time.time()
    reg_icp = o3d.pipelines.registration.registration_icp(
        source, target,
        MAX_CORR_DIST,
        trans_init,
        estimation,
        criteria
    )
    t_icp = time.time() - t0

    print("\n=== Resultados do ICP ===")
    print(f"method = {ICP_METHOD}")
    print(f"max_correspondence_distance = {MAX_CORR_DIST:.3f} m")
    print(f"max_iterations = {MAX_ITERS}")
    print(f"Tempo ICP: {t_icp:.3f} s")
    print("Fitness:", reg_icp.fitness)
    print("Inlier RMSE:", reg_icp.inlier_rmse)
    print("Transformação estimada:")
    print(reg_icp.transformation)

    # Aplica a transformação à nuvem fonte
    source.transform(reg_icp.transformation)

    # Visualização (verde = alvo, vermelho = fonte alinhada)
    if SHOW_VIEWER:
        source.paint_uniform_color([1, 0, 0])  # vermelho
        target.paint_uniform_color([0, 1, 0])  # verde
        o3d.visualization.draw_geometries([target, source])

    # Debugging
    print("\n--- Debug ---")
    print("Número de pontos em pcd1 downsampled:", np.asarray(pcd1_ds.points).shape[0])
    print("Número de pontos em pcd2 downsampled:", np.asarray(pcd2_ds.points).shape[0])
    print("pcd1_down normals computed:", len(pcd1_ds.normals))


if __name__ == '__main__':
    main()

