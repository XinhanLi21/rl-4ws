# -*- coding: utf-8 -*-
"""
eval_ddpg_checkpoint.py
评估 ❶ DDPG-MPC 联合控制模型 ❷ 生成 XY 轨迹 & 舵角曲线（三联图）
    并计算关键实验结果：横向误差与航向误差的均值与最大值
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from scipy.io import loadmat

plt.rcParams["font.family"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False

# ------- 引入与训练一致的环境 ---------------------------------
# 如果 FourWSEnv 定义在 rl_ddpg.py 内，直接这样导入
from rl_ddpg import FourWSEnv, VehicleModelPublic
# --------------------------------------------------------------

def evaluate(checkpoint, dll, path_mat, out_dir="eval/ddpg_70",vx_kmh=60/3.6, Ts=0.05, dt=0.01):
    os.makedirs(out_dir, exist_ok=True)

    # 1) 参考路径
    ref_path = np.asarray(loadmat(path_mat)["path_ref"], float)

    # 2) 单环境
    env = DummyVecEnv([lambda: FourWSEnv(ref_path, dll,vx=vx_kmh, Ts=Ts, dt=dt)])

    # 3) 加载 DDPG 模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model: DDPG = DDPG.load(checkpoint, env=env, device=device,
                            print_system_info=True)

    # 4) 回放
    obs = env.reset()
    done = [False]

    xs, ys = [], []
    delta_f_hist, delta_r_hist = [], []
    lat_err_hist, yaw_err_hist = [], []

    fenv: FourWSEnv = env.envs[0]

    while not done[0]:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)

        # 记录位置
        X, Y, *_ = fenv.veh._get_current_observation()
        xs.append(X)
        ys.append(Y)

        # 记录转角
        df = fenv.delta_f
        dr = fenv.beta * fenv.delta_f
        delta_f_hist.append(df)
        delta_r_hist.append(dr)

        # 记录误差：obs 返回 [lat_error, yaw_error, ...]
        lat_err_hist.append(float(obs[0][0]))
        yaw_err_hist.append(float(obs[0][1]))

    env.close()

    # === 结果处理 ===============================================
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    delta_f_hist = np.asarray(delta_f_hist)
    delta_r_hist = np.asarray(delta_r_hist)
    lat_err_hist = np.asarray(lat_err_hist)
    yaw_err_hist = np.asarray(yaw_err_hist)

    time_axis = np.arange(len(lat_err_hist)) * Ts

    # 计算统计量
    mean_lat = np.abs(lat_err_hist).mean()
    max_lat = np.abs(lat_err_hist).max()
    mean_yaw = np.abs(yaw_err_hist).mean()
    max_yaw = np.abs(yaw_err_hist).max()

    print(f"横向误差：均值={mean_lat:.4f}m,最大值 = {max_lat:.4f}m|")
    print(f"航向误差：均值={np.degrees(mean_yaw):.4f}°,最大值={np.degrees(max_yaw):.4f}°")

    # 5) 保存 CSV
    csv_path = os.path.join(out_dir, "traj_full.csv")
    np.savetxt(csv_path,
               np.column_stack([
                   time_axis, xs, ys,
                   delta_f_hist, delta_r_hist,
                   lat_err_hist, yaw_err_hist
               ]),
               delimiter=",",
               header="t,X,Y,delta_f(rad),delta_r(rad),lat_err(m),yaw_err(rad)",
               comments="")
    print(f"✓ 轨迹与误差数据已保存到 {csv_path}")

    # 6) 三联图：XY、δf/δr、β
    square = 6
    fig = plt.figure(figsize=(square, square + 4))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 0.45, 0.4])

    # (1) XY 轨迹
    ax_xy = fig.add_subplot(gs[0])
    ax_xy.plot(ref_path[:, 0], ref_path[:, 1],
               "k--", lw=2, label="参考轨迹")
    mask = ys > -1
    ax_xy.plot(xs[mask], ys[mask], "r-", lw=2, label="实际轨迹")
    ax_xy.set_xlim(ref_path[:, 0].min(), ref_path[:, 0].max())
    ax_xy.set_ylim(-1, 6)
    ax_xy.yaxis.set_major_locator(MultipleLocator(0.25))
    ax_xy.set_aspect("auto")
    ax_xy.grid(ls="--", alpha=0.5)
    ax_xy.set_xlabel("X (m)")
    ax_xy.set_ylabel("Y (m)")
    ax_xy.set_title(
        f"DDPG+MPC – 平均横向误差 {mean_lat:.3f} m, 最大 {max_lat:.3f} m"
    )
    ax_xy.legend()

    # (2) δf / δr
    ax_ang = fig.add_subplot(gs[1])
    df_deg = np.degrees(delta_f_hist)
    dr_deg = np.degrees(delta_r_hist)
    ax_ang.plot(time_axis, df_deg, lw=1.8, label="δf (前轮)")
    ax_ang.plot(time_axis, dr_deg, lw=1.8, label="δr (后轮)")
    ang_max = max(np.abs(np.concatenate([df_deg, dr_deg])).max(), 1e-2)
    ax_ang.set_ylim(-1.05 * ang_max, 1.05 * ang_max)
    ax_ang.grid(ls="--", alpha=0.5)
    ax_ang.set_xlabel("Time (s)")
    ax_ang.set_ylabel("Steer Angle (°)")
    ax_ang.legend()

    # (3) β 曲线
    ax_beta = fig.add_subplot(gs[2])
    beta_hist = dr_deg / np.where(np.abs(df_deg) < 1e-3, 1e-3, df_deg)
    ax_beta.plot(time_axis, beta_hist, lw=1.6)
    b_abs = np.abs(beta_hist).max()
    ax_beta.set_ylim(-1.05 * b_abs if b_abs > 0 else -1,
                     1.05 * b_abs if b_abs > 0 else 1)
    ax_beta.grid(ls="--", alpha=0.5)
    ax_beta.set_xlabel("Time (s)")
    ax_beta.set_ylabel("β")

    fig.tight_layout()
    img_path = os.path.join(out_dir, "traj_3panel.png")
    fig.savefig(img_path, dpi=150)
    plt.close(fig)
    print(f"✓ 三联图已保存到 {img_path}")


    # ── 7) 绘制并保存 XY 轨迹动画 ───────────────────────────────
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig_anim, ax_anim = plt.subplots(figsize=(6, 6))
    # 先画参考轨迹
    ax_anim.plot(ref_path[:, 0], ref_path[:, 1], 'k--', lw=2, label='reference')
    # 初始化实际轨迹线
    line_actual, = ax_anim.plot([], [], 'r-', lw=2, label='actual')
    ax_anim.set_xlim(ref_path[:, 0].min(), 140.0)
    ax_anim.set_ylim(-1, 5)
    ax_anim.set_xlabel("X (m)")
    ax_anim.set_ylabel("Y (m)")
    ax_anim.set_title("DDPG算法+70km/h")
    ax_anim.legend()
    ax_anim.grid(ls="--", alpha=0.5)

    print(f"横向误差：均值={mean_lat:.4f} m,最大={max_lat:.4f}m/n")
    print(f"航向误差：均值={np.degrees(mean_yaw):.4f}°, 最大 = {np.degrees(max_yaw):.4f}°")

    def init():
        line_actual.set_data([], [])
        return (line_actual,)

    def update(frame):
        # 只画到第 frame 点
        line_actual.set_data(xs[:frame], ys[:frame])
        return (line_actual,)

    # frames=len(xs) 帧数，interval=50 毫秒/帧
    ani = FuncAnimation(fig_anim, update, frames=len(xs), init_func=init,
                        blit=True, interval=50)
    gif_path = os.path.join(out_dir, "xy_traj.gif")
    ani.save(gif_path, writer=PillowWriter(fps=20))
    plt.close(fig_anim)
    print(f"✓ XY 轨迹动画已保存到 {gif_path}")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate DDPG model")
    parser.add_argument(
        "--dll",
        default="vehiclemodel_public_0326_win64.dll",
        help="Path to vehicle model DLL",
    )
    parser.add_argument(
        "--checkpoint",
        default=r"checkpoints/ddpg_70/ddpg_step_2516000_steps.zip",
        help="Path to DDPG checkpoint",
    )
    parser.add_argument(
        "--path-mat",
        default="mat/lanechange_double_hold20m_40_40_shift20.mat",
        help="MAT file containing reference path",
    )
    parser.add_argument(
        "--out-dir",
        default="eval/ddpg_70",
        help="Output directory for evaluation results",
    )
    args = parser.parse_args()

    evaluate(args.checkpoint, args.dll, args.path_mat, args.out_dir)
