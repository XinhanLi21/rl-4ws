# -*- coding: utf-8 -*-
"""
run_closed_loop.py

闭环仿真：MPC4WS + vehiclemodel_public_0326
输出：δ_f、横向误差、航向误差、速度曲线、X-Y 轨迹
并计算平均/最大横向误差与航向误差，最后生成 XY 轨迹动画
"""

import os
import pathlib
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.io import loadmat
from matplotlib.animation import FuncAnimation, PillowWriter

from mpc_v2 import MPC4WS
from wrapper import VehicleModelPublic

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False


def main(dll: str):
    # ── 1. 读取路径 ───────────────────────────────────────
    mat = loadmat('mat/path5_mpc4ws.mat')
    path = np.asarray(mat['path5'], dtype=float)

    # ── 2. 初始化控制器与车辆模型 ───────────────────────
    ctrl = MPC4WS(path_ref=path, Ts=0.05, vx=60/3.6)
    veh = VehicleModelPublic(str(pathlib.Path(dll).absolute()))

    obs = veh.initial(delta_f=0.0, V_ini=60.0/3.6, delta_r=0.0)
    print('Vehicle model initial state:', obs)

    # ── 3. 仿真设置 ───────────────────────────────────────
    t_end = 10.0
    dt_vehicle = 0.01
    dt_mpc = 0.05
    steps_tot = int(t_end / dt_vehicle)
    mpc_ratio = int(dt_mpc / dt_vehicle)

    # ── 4. 数据记录 ───────────────────────────────────────
    T = []
    delta_f_hist = []
    lat_err_hist = []
    yaw_err_hist = []
    X_hist, Y_hist = [], []
    Vx_hist = []

    # ── 5. 主循环 ───────────────────────────────────────
    t = 0.0
    delta_f_cmd = 0.0

    for k in range(steps_tot):
        obs = veh.step(delta_f=delta_f_cmd, delta_r=0.0)
        X, Y, yaw, Vx, Vy, r = obs

        if k % mpc_ratio == 0:
            u = np.array([
                Vy * 3.6, Vx * 3.6,
                np.rad2deg(yaw), np.rad2deg(r),
                Y, X, 0.0, 0.0
            ])
            mpc_out = ctrl.step(t, u)
            delta_f_cmd = float(mpc_out[0])

            T.append(t)
            delta_f_hist.append(delta_f_cmd)
            lat_err_hist.append(float(mpc_out[2]))
            yaw_err_hist.append(float(mpc_out[3]))
            X_hist.append(X)
            Y_hist.append(Y)
            Vx_hist.append(Vx)

        t += dt_vehicle

    veh.terminate()

    # ── 6. 分析结果 ───────────────────────────────────────
    lat_arr = np.asarray(lat_err_hist)
    yaw_arr = np.asarray(yaw_err_hist)

    avg_lat_err = np.mean(np.abs(lat_arr))
    max_lat_err = np.max(np.abs(lat_arr))
    avg_yaw_err = np.mean(np.abs(yaw_arr))
    max_yaw_err = np.max(np.abs(yaw_arr))

    print(f"✔ 平均横向误差：{avg_lat_err:.4f} m")
    print(f"✔ 最大横向误差：{max_lat_err:.4f} m")
    print(f"✔ 平均航向误差：{np.degrees(avg_yaw_err):.4f} °")
    print(f"✔ 最大航向误差：{np.degrees(max_yaw_err):.4f} °")

# ── 7. 保存数据 ───────────────────────────────────────
    out_dir = "eval/mpc_60"
    os.makedirs(out_dir, exist_ok=True)

    time_axis = np.asarray(T)
    delta_f_hist = np.asarray(delta_f_hist)
    X_hist = np.asarray(X_hist)
    Y_hist = np.asarray(Y_hist)
    Vx_hist = np.asarray(Vx_hist)

    np.savetxt(
        os.path.join(out_dir, "mpc_traj_full.csv"),
        np.column_stack([
            time_axis,
            X_hist,
            Y_hist,
            delta_f_hist,
            lat_arr,
            yaw_arr,
            Vx_hist,
        ]),
        delimiter=",",
        header="t,X,Y,delta_f(rad),lat_err(m),yaw_err(rad),Vx(m/s)",
        comments="",
    )

# ── 8. 绘图 & 结果保存（三联：轨迹, δ_f, 速度） ─────────────────
    fig = plt.figure(figsize=(6, 10))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 0.45, 0.45])

    # (1) XY 轨迹
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(path[:, 0], path[:, 1], "k--", lw=2, label="参考轨迹")
    mask = Y_hist > -1
    ax1.plot(X_hist[mask], Y_hist[mask], "r-", lw=2, label="实际轨迹")
    ax1.set_xlim(path[:, 0].min(), path[:, 0].max())
    ax1.set_ylim(-1, 6)
    ax1.yaxis.set_major_locator(MultipleLocator(0.5))
    ax1.set_aspect("auto")
    ax1.grid(ls="--", alpha=0.5)
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    # ax1.set_title(f"MPC–横向误差={avg_lat_err:.3f}m,max={max_lat_err:.3f}m | 横摆误差={np.degrees(avg_yaw_err):.2f}°, max={np.degrees(max_yaw_err):.2f}°")
    ax1.legend()

    # (2) 前轮转角 δ_f
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(time_axis, np.degrees(delta_f_hist), color="royalblue", lw=1.8)
    ax2.set_ylabel("δ_f (deg)")
    ax2.grid(ls="--", alpha=0.5)
    ax2.set_xlabel("Time (s)")

    # (3) 纵向速度 Vx
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(time_axis, Vx_hist, color="green", lw=1.8)
    ax3.set_ylabel("Vx (m/s)")
    ax3.grid(ls="--", alpha=0.5)
    ax3.set_xlabel("Time (s)")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "traj_3panel.png"), dpi=150)
    plt.close(fig)

    print(f"✓ 结果与图像已保存到 {out_dir}")

    # ── 9. 绘制并保存 XY 轨迹动画 ───────────────────────────────
    fig_anim, ax_anim = plt.subplots(figsize=(6, 6))
    ax_anim.plot(path[:, 0], path[:, 1], 'k--', lw=2, label='参考轨迹')
    line_actual, = ax_anim.plot([], [], 'r-', lw=2, label='实际轨迹')
    ax_anim.set_xlim(path[:, 0].min(), path[:, 0].max())
    ax_anim.set_ylim(-1, 6)
    ax_anim.yaxis.set_major_locator(MultipleLocator(0.25))
    ax_anim.set_xlabel("X (m)")
    ax_anim.set_ylabel("Y (m)")
    ax_anim.set_title("MPC 60 km/h 闭环轨迹动画")
    ax_anim.legend()
    ax_anim.grid(ls="--", alpha=0.5)

    def init():
        line_actual.set_data([], [])
        return (line_actual,)

    def update(frame):
        line_actual.set_data(X_hist[:frame], Y_hist[:frame])
        return (line_actual,)

    ani = FuncAnimation(
        fig_anim,
        update,
        frames=len(X_hist),
        init_func=init,
        blit=True,
        interval=50,
    )
    gif_path = os.path.join(out_dir, "xy_traj.gif")
    ani.save(gif_path, writer=PillowWriter(fps=20))
    plt.close(fig_anim)

    print(f"✓ XY 轨迹动画已保存到 {gif_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MPC closed loop test")
    parser.add_argument(
        "--dll",
        default="vehiclemodel_public_0326_win64.dll",
        help="Path to vehicle model DLL",
    )
    args = parser.parse_args()

    main(args.dll)




