# -*- coding: utf-8 -*-
"""
batch_sort_ddpg.py  —  批量评估 DDPG‑MPC checkpoint，按平均横向误差排序
"""

import os
import numpy as np
import torch
from stable_baselines3 import DDPG               # ★ 改成 DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from scipy.io import loadmat

# === 导入与你训练时一致的环境 ==========================
# 若 FourWSEnv 已单独放在 rl_ddpg_fast_spawn.py 中，则这样引入
from rl_ddpg import FourWSEnv, VehicleModelPublic

# ------------------------------------------------------
def evaluate_single(checkpoint, dll, path_mat,
                    vx_kmh=70/3.6, Ts=0.05, dt=0.01):
    """
    评估单个 DDPG checkpoint，返回平均横向误差
    """
    ref_path = np.asarray(loadmat(path_mat)["path_ref"], float)

    env = DummyVecEnv([lambda: FourWSEnv(ref_path, dll,
                                         vx=vx_kmh, Ts=Ts, dt=dt)])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model: DDPG = DDPG.load(checkpoint, env=env, device=device,
                            print_system_info=False)

    obs = env.reset()        # (n_envs, obs_dim)
    done = [False]

    lat_err_hist = []
    while not done[0]:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)  # done 为列表
        lat_err_hist.append(float(obs[0][0]))  # obs[0] = [lat_err, yaw_err, β]

    mean_lat_err = np.mean(np.abs(lat_err_hist))
    env.close()
    return mean_lat_err
# ------------------------------------------------------


def evaluate_all(checkpoints_dir, dll, path_mat):
    """
    批量评估目录下所有 DDPG checkpoint（*.zip），输出排序结果
    """
    checkpoints = sorted(
        [os.path.join(checkpoints_dir, f)
         for f in os.listdir(checkpoints_dir)
         if f.endswith(".zip")]
    )

    results = []
    for ckpt in checkpoints:
        print(f"\n📌 正在评估 {ckpt} ...")
        err = evaluate_single(ckpt, dll, path_mat)
        print(f"➡️ 平均横向误差: {err:.4f} m")
        results.append((ckpt, err))

    # 升序排序（误差越小越好）
    results.sort(key=lambda x: x[1])

    print("\n====== 📊 全部模型横向误差排序 ======")
    for ckpt, err in results:
        print(f"{os.path.basename(ckpt):>25}  ->  {err:.4f} m")

    best_ckpt, best_err = results[0]
    print("\n✅ 最佳模型:")
    print(f"{os.path.basename(best_ckpt)}  ->  {best_err:.4f} m")

    return best_ckpt, best_err


# =========================== MAIN =============================
if __name__ == "__main__":
    CHECKPOINT_DIR = "checkpoints/ddpg_70"   # ★ DDPG 模型目录
    DLL            = r"D:\chrome-download\5\5.4\vehiclemodel_public_0326_win64.dll"
    PATH_MAT       = "mat/lanechange_double_hold20m_40_40_shift20.mat"

    evaluate_all(CHECKPOINT_DIR, DLL, PATH_MAT)
