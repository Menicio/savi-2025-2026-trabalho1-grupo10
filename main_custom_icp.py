#!/usr/bin/env python3
# SAVI - Trabalho 1
# Tarefa 2 — ICP Personalizado com Least-Squares e Inicialização Manual

# João Menício - 93300
# Pascoal Sumbo - 123190

from copy import deepcopy
from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.optimize import least_squares
from utils_rgbd import load_and_prepare_cloud  # módulo comum RGB-D

# ==========================
# Parâmetros de otimização
# ==========================

VOXEL_SIZE       = 0.025               # Tamanho do voxel para downsampling
MAX_ITERS        = 500                 # iterações ICP externas
MAX_CORR_DIST    = 0.01                # max_correspondence_distance, distância máxima para aceitar correspondência
LS_MAX_ITERS     = 50                  # iterações internas do least_squares
LOSS_FUNC        = "huber"             # "linear", "soft_l1", "huber", "cauchy"
LOSS_SCL         = 1.0                 # parâmetro de escala da loss robusta
EPS_XI_NORM      = 1e-10               # critério de paragem no incremento
SHOW_PROGRESS    = True                # print por iteração


# ==========================
# Transformação inicial manual
# ==========================

T_INIT = np.eye(4)
T_INIT = np.array([[ 0.99126294, 0.05318669, -0.12070192, -0.76307333],
         [-0.05498046, 0.99842031, -0.01157746, -0.08140979],
        [ 0.11989548, 0.01811255, 0.99262128, -0.11052373],
        [ 0, 0, 0, 1]] )

# ==========================

def rodrigues(r):
    """Converte vetor rotação (axis-angle) r (3,) em matriz 3x3 (Rodrigues)."""
    theta = np.linalg.norm(r)
    if theta < 1e-12:
        return np.eye(3)
    k = r / theta
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R


def se3_to_T(xi):
    """
    Constrói T a partir de um vetor de 6 parâmetros:
      xi = [rx, ry, rz, tx, ty, tz]
    onde r* é axis-angle pequeno.
    """
    r = xi[:3]
    t = xi[3:]
    R = rodrigues(r)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def compose_T(T1, T2):
    """Composição SE(3): retorna T1 @ T2."""
    return T1 @ T2


def build_kdtree(pcd):
    """KDTreeFlann do Open3D para a nuvem alvo."""
    return o3d.geometry.KDTreeFlann(pcd)


def find_correspondences(src_pts, tgt_pcd, kdtree, max_dist):
    """
    Para cada ponto da fonte (em Nx3), encontra vizinho mais próximo na alvo.
    Retorna índices válidos (i na src, j na tgt) e distâncias.
    """
    tgt_pts = np.asarray(tgt_pcd.points)
    valid_src_idx = []
    valid_tgt_idx = []
    dists = []

    for i, p in enumerate(src_pts):
        # knn_search devolve (k, idx, dist2)
        k, idx, dist2 = kdtree.search_knn_vector_3d(p, 1)
        if k == 1:
            d = np.sqrt(dist2[0])
            if d < max_dist:
                valid_src_idx.append(i)
                valid_tgt_idx.append(idx[0])
                dists.append(d)
    return np.array(valid_src_idx, dtype=int), np.array(valid_tgt_idx, dtype=int), np.array(dists)


def point_to_plane_residuals(xi, src_pts, tgt_pts, tgt_normals, T_curr):
    """
    Residuals point-to-plane: n^T ( R*(T_curr*p_src) + t - p_tgt ).
    Nota: aplicamos incremento à esquerda: T_new = exp(xi^) @ T_curr
    """
    # incremento
    dT = se3_to_T(xi)
    T = compose_T(dT, T_curr)
    R = T[:3, :3]
    t = T[:3, 3]

    # transformar pontos fonte
    src_tr = (R @ src_pts.T).T + t  # Nx3

    # distâncias assinadas ao plano
    diff = src_tr - tgt_pts         # Nx3
    res = np.sum(tgt_normals * diff, axis=1)  # N
    return res


def icp_custom_point_to_plane(source_pcd, target_pcd, T_init,
                              max_corr_dist=0.07,
                              max_iters=15,
                              ls_max_iters=50,
                              loss="huber",
                              loss_scale=1.0,
                              eps_xi=1e-5,
                              show=True):
    """
    ICP personalizado:
      - correspondências: KDTree (NN) com corte por distância
      - custo: point-to-plane
      - solver: scipy.optimize.least_squares
      - atualização: T_{k+1} = exp(xi^) @ T_k
    """
    # arrays numpy
    tgt_pts = np.asarray(target_pcd.points)
    tgt_nrm = np.asarray(target_pcd.normals)
    if tgt_nrm.shape[0] != tgt_pts.shape[0]:
        raise RuntimeError("Target sem normais. Estima primeiro as normais do alvo.")

    # KDTree alvo
    kdtree = build_kdtree(target_pcd)

    # T corrente
    T_curr = T_init.copy()

    for it in range(1, max_iters + 1):
        # pontos da fonte transformados pela T_curr (para corresponder)
        src_pts = np.asarray(source_pcd.points)
        src_tr = (T_curr[:3, :3] @ src_pts.T).T + T_curr[:3, 3]

        # encontrar NN no alvo
        src_idx, tgt_idx, dists = find_correspondences(src_tr, target_pcd, kdtree, max_corr_dist)
        if len(src_idx) < 10:
            if show:
                print(f"[ICP {it}] poucas correspondências ({len(src_idx)}). A terminar.")
            break

        # preparar dados para LS
        src_sel = src_pts[src_idx]                      # N x 3 (não-transformados: aplicamos T dentro do residual)
        tgt_sel = tgt_pts[tgt_idx]                      # N x 3
        nrm_sel = tgt_nrm[tgt_idx]                      # N x 3

        # função residual que o LS vai minimizar
        fun = lambda x: point_to_plane_residuals(x, src_sel, tgt_sel, nrm_sel, T_curr)

        # resolver incremento xi
        res = least_squares(fun, x0=np.zeros(6),
                            loss=loss, f_scale=loss_scale,
                            max_nfev=ls_max_iters, verbose=0)

        xi = res.x
        # atualizar T
        dT = se3_to_T(xi)
        T_next = compose_T(dT, T_curr)

        # métricas simples
        res_vals = fun(np.zeros(6))   # residual com T_curr
        res_vals_next = point_to_plane_residuals(np.zeros(6), src_sel, tgt_sel, nrm_sel, T_next)
        rmse_curr = np.sqrt(np.mean(res_vals**2))
        rmse_next = np.sqrt(np.mean(res_vals_next**2))

        if show:
            print(f"[ICP {it:02d}] corr={len(src_idx):5d}  rmse -> {rmse_curr:.5f} -> {rmse_next:.5f}  |xi|={np.linalg.norm(xi):.3e}")

        T_curr = T_next

        # critério de paragem no incremento
        if np.linalg.norm(xi) < eps_xi:
            if show:
                print(f"[ICP {it}] incremento pequeno (|xi|<{eps_xi}). Parar.")
            break

    return T_curr


def main():
    voxel_size = VOXEL_SIZE

    root = Path(__file__).resolve().parent
    
    filename_rgb1   = root / "1.png"
    filename_depth1 = root / "depth1.png"
    filename_rgb2   = root / "2.png"
    filename_depth2 = root / "depth2.png"

    # Intrínsecos (PrimeSenseDefault)
    intr = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
    )
    
    
   # Construir e preparar nuvens (leitura TUM + flip + voxel + normais)
    pcd1_ds = load_and_prepare_cloud(filename_rgb1, filename_depth1, intr, voxel_size)
    pcd2_ds = load_and_prepare_cloud(filename_rgb2, filename_depth2, intr, voxel_size)


    # --- Visualização antes ---
    b_tgt = deepcopy(pcd1_ds); b_tgt.paint_uniform_color([0,1,0])
    b_src = deepcopy(pcd2_ds); b_src.paint_uniform_color([1,0,0])
    o3d.visualization.draw_geometries([b_tgt, b_src])

    # --- ICP personalizado (Point-to-Plane + LS) ---
    T_final = icp_custom_point_to_plane(
        source_pcd=pcd2_ds,
        target_pcd=pcd1_ds,
        T_init=T_INIT,
        max_corr_dist=MAX_CORR_DIST,
        max_iters=MAX_ITERS,
        ls_max_iters=LS_MAX_ITERS,
        loss=LOSS_FUNC,
        loss_scale=LOSS_SCL,
        eps_xi=EPS_XI_NORM,
        show=SHOW_PROGRESS
    )

    print("\nTransformação final (fonte -> alvo):\n", T_final)

    # --- Aplicar e visualizar resultado --- cinzento???
    src_aligned = deepcopy(pcd2_ds).transform(T_final)

    f_tgt = deepcopy(pcd1_ds); f_tgt.paint_uniform_color([0,1,0])
    f_src = deepcopy(pcd2_ds); f_src.paint_uniform_color([1,0,0])
    f_src_al = deepcopy(src_aligned); f_src_al.paint_uniform_color([0,0,1])
    o3d.visualization.draw_geometries([f_tgt, f_src,f_src_al])


if __name__ == "__main__":
    main()
