#!/usr/bin/env python3
# SAVI - Trabalho 1
# Tarefa 1
# João Menício - 93300

# Dúvidas:
# como importar os parâmetros intrinsecos da câmara?
# -> Por defeito uso PrimeSenseDefault. Se quiseres TUM fr1:
#    intr = o3d.camera.PinholeCameraIntrinsic(
#        width, height, 517.3, 516.5, 318.6, 255.3
#    )
#    e depois usa esse 'intr' em create_from_rgbd_image.

from copy import deepcopy
from functools import partial
import glob
from random import randint
from matplotlib import pyplot as plt
import numpy as np
import argparse
import open3d as o3d

# ==========================
# PARÂMETROS EDITÁVEIS
# ==========================
VOXEL_SIZE = 0.025              # mantém coerente com a cena
MAX_CORR_DIST = 0.6            # max_correspondence_distance (m) ~ 1.0–2.0 * VOXEL_SIZE
ICP_METHOD = "point_to_plane"   # "point_to_plane" ou "point_to_point"
MAX_ITERS = 2400                  # iterações máximas do ICP
# ==========================


def main():

    # Carregamento de Imagens e Filtragem de Profundidade
    voxel_size = VOXEL_SIZE

    # Upload files
    # Point Cloud 1
    filename_rgb1 = '/home/menicio/savi_25-26/Parte08/tum_dataset/rgb/1.png'
    rgb1 = o3d.io.read_image(filename_rgb1)

    filename_depth1 = '/home/menicio/savi_25-26/Parte08/tum_dataset/depth/1.png'
    depth1 = o3d.io.read_image(filename_depth1)

    # Point Cloud 2
    filename_rgb2 = '/home/menicio/savi_25-26/Parte08/tum_dataset/rgb/2.png'
    rgb2 = o3d.io.read_image(filename_rgb2)

    filename_depth2 = '/home/menicio/savi_25-26/Parte08/tum_dataset/depth/2.png'
    depth2 = o3d.io.read_image(filename_depth2)

    # Convert do RGB-D (TUM: esta função já trata da escala/encoding)
    rgbd1 = o3d.geometry.RGBDImage.create_from_tum_format(rgb1, depth1)
    rgbd2 = o3d.geometry.RGBDImage.create_from_tum_format(rgb2, depth2)

    # Criar pointclouds (intrínsecos: PrimeSenseDefault)
    intr = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
    )
    # Se quiseres TUM fr1 explícito, usa (descomenta e ajusta width/height se necessário):
    # width = np.asarray(rgbd1.color).shape[1]
    # height = np.asarray(rgbd1.color).shape[0]
    # intr = o3d.camera.PinholeCameraIntrinsic(width, height, 517.3, 516.5, 318.6, 255.3)

    pcd1 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd1, intr)
    pcd2 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd2, intr)

    # Orientar (flip para ter Z para a frente)
    flip_T = [[1, 0, 0, 0],
              [0, -1, 0, 0],
              [0, 0, -1, 0],
              [0, 0, 0, 1]]
    
    pcd1.transform(flip_T)
    pcd2.transform(flip_T)

    # Downsampling
    pcd1_ds = pcd1.voxel_down_sample(voxel_size=voxel_size)
    pcd2_ds = pcd2.voxel_down_sample(voxel_size=voxel_size)

    # Estimar normais
    pcd1_ds.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    pcd2_ds.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))

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
    trans_init = np.identity(4)

    # Executa o ICP
    reg_icp = o3d.pipelines.registration.registration_icp(
        source, target,
        MAX_CORR_DIST,
        trans_init,
        estimation,
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=MAX_ITERS)
    )

    print("\n=== Resultados do ICP ===")
    print(f"method={ICP_METHOD}  mcd={MAX_CORR_DIST:.3f}  iters={MAX_ITERS}")
    print("Fitness:", reg_icp.fitness)
    print("Inlier RMSE:", reg_icp.inlier_rmse)
    print("Transformação estimada:")
    print(reg_icp.transformation)

    # Aplica a transformação à nuvem fonte
    source.transform(reg_icp.transformation)

    # Visualização (verde = alvo, vermelho = fonte alinhada)
    source.paint_uniform_color([1, 0, 0])  # vermelho
    target.paint_uniform_color([0, 1, 0])  # verde
    o3d.visualization.draw_geometries([target, source])

    # Debugging
    print("Número de pontos em pcd1:", np.asarray(pcd1.points).shape)
    print("Número de pontos em pcd1 downsampled:", np.asarray(pcd1_ds.points).shape)
    print("Número de pontos em pcd2:", np.asarray(pcd2.points).shape)
    print("Número de pontos em pcd2 downsampled:", np.asarray(pcd2_ds.points).shape)
    print("pcd1_down normals computed:", len(pcd1_ds.normals))


if __name__ == '__main__':
    main()
