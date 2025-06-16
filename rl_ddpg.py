# -*- coding: utf-8 -*-
"""
rl_ddpg.py – DDPG + 三联图 (XY, δfδr, β) + TensorBoard Q‑value
XY 子图为正方形；Y 轴固定 [-1, 6] 且 0.25 m 主刻度。
"""
# -------------------------------------------------------------
import multiprocessing as mp
mp.set_start_method("spawn", force=True)

import os, math, pathlib, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import gymnasium as gym
from gymnasium.spaces import Box
from scipy.io import loadmat
import torch

from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, CallbackList
)
from stable_baselines3.common.utils import set_random_seed

from rl_mpc_v22 import MPC4WS
from wrapper      import VehicleModelPublic


# ---------------- 环境封装 ------------------------------------
class FourWSEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, path, dll,
                 vx=70/3.6, Ts=0.05, dt=0.01, beta_lim=0.9):
        super().__init__()
        self.path, self.vx = path, vx
        self.Ts, self.dt   = Ts, dt
        self.ratio         = int(Ts / dt)
        self.beta_lim, self.act_lim = beta_lim, 0.3

        self.action_space      = Box(-self.act_lim, self.act_lim, (1,), np.float32)
        self.observation_space = Box(-np.inf, np.inf, (3,),   np.float32)

        self.ctrl = MPC4WS(path, Ts=Ts, vx=vx)
        self.veh  = VehicleModelPublic(str(pathlib.Path(dll)))

    # ----------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.ctrl.reset(); self.veh.terminate()
        self.veh.initial(0.0, self.vx, 0.0)

        self.t = self.beta = self.delta_f = 0.0
        self.traj = []
        self.delta_f_hist, self.delta_r_hist, self.beta_hist = [], [], []

        return self._obs(), {}

    # ----------------------------------------------------------
    def step(self, action):
        # β 累积控制
        self.beta = float(np.clip(
            self.beta + action[0], -self.beta_lim, self.beta_lim))

        # 车辆前向积分 dt
        for _ in range(self.ratio):
            ov = self.veh.step(self.delta_f, self.beta * self.delta_f)
            self.traj.append((ov[0], ov[1]))
            self.t += self.dt

        # MPC 预测
        y = self.ctrl.step(
            self.t,
            self._assemble_u(ov, self.beta * self.delta_f)
        )
        self.delta_f = float(y[0])

        # 记录
        self.delta_f_hist.append(self.delta_f)
        self.delta_r_hist.append(self.beta * self.delta_f)
        self.beta_hist.append(self.beta)

        lat_err, yaw_err = float(y[2]), float(y[3])
        reward = -20.0 * lat_err**2
        done   = self.t >= 10.0

        info = {}
        if done:
            info = dict(
                traj         = np.asarray(self.traj,        np.float32),
                delta_f_hist = np.asarray(self.delta_f_hist,np.float32),
                delta_r_hist = np.asarray(self.delta_r_hist,np.float32),
                beta_hist    = np.asarray(self.beta_hist,   np.float32),
                Ts           = self.Ts
            )
        return np.array([lat_err, yaw_err, self.beta], np.float32), reward, done, False, info

    # ----------------------------------------------------------
    def _obs(self):
        ov  = self.veh._get_current_observation()
        lat, yaw = self.ctrl.step(self.t, self._assemble_u(ov, 0.0))[2:4]
        return np.array([lat, yaw, self.beta], np.float32)

    def _assemble_u(self, ov, delta_r):
        X, Y, yaw, Vx, Vy, r = ov
        return np.array([Vy * 3.6, Vx * 3.6,
                         math.degrees(yaw), math.degrees(r),
                         Y, X, math.degrees(delta_r), 0.0], np.float32)

    def close(self): self.veh.terminate()


# ---------------- Trajectory Callback -------------------------
class TrajCallback(BaseCallback):
    """
    每回合输出 3‑联图：
        (1) XY 轨迹 – 正方形；Y∈[-1,6]；0.25 m 主刻度
        (2) 舵角曲线 – δf / δr（deg）
        (3) β 曲线      – β = δr / δf
    """
    def __init__(self, ref_path, out_dir="trajectories/ddpg_70"):
        super().__init__()
        self.ref = ref_path
        self.out = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.i = 0

    def _on_step(self): return True

    def _on_rollout_end(self):
        for info in self.locals["infos"]:
            if "traj" not in info:
                continue

            traj = info["traj"]
            df   = np.degrees(info["delta_f_hist"])
            dr   = np.degrees(info["delta_r_hist"])
            beta_hist = info.get(
                "beta_hist", dr / np.maximum(df, 1e-6)
            )
            Ts = info["Ts"]; t_ax = np.arange(len(df)) * Ts

            square = 6
            fig = plt.figure(figsize=(square, square + 4))
            gs  = fig.add_gridspec(3, 1, height_ratios=[1, 0.45, 0.4])

            ax_xy   = fig.add_subplot(gs[0])
            ax_ang  = fig.add_subplot(gs[1])
            ax_beta = fig.add_subplot(gs[2])

            # —— XY
            ax_xy.plot(self.ref[:, 0], self.ref[:, 1],
                       "k--", lw=2, label="reference")
            ax_xy.plot(traj[:, 0], traj[:, 1],
                       "r-",  lw=2, label="actual")
            ax_xy.set_xlim(self.ref[:, 0].min(), self.ref[:, 0].max())
            ax_xy.set_ylim(-1, 6)
            ax_xy.yaxis.set_major_locator(MultipleLocator(0.25))
            ax_xy.set_aspect("auto")
            ax_xy.grid(ls="--", alpha=0.5)
            ax_xy.set_xlabel("X (m)")
            ax_xy.set_ylabel("Y (m)")
            ax_xy.legend()

            # —— δf / δr
            ax_ang.plot(t_ax, df, color="royalblue", lw=1.8, label="δf (front)")
            ax_ang.plot(t_ax, dr, color="orange",   lw=1.8, label="δr (rear)")
            ang_max = max(np.abs(np.concatenate([df, dr])).max(), 1e-2)
            ax_ang.set_ylim(-1.05 * ang_max, 1.05 * ang_max)
            ax_ang.grid(ls="--", alpha=0.5)
            ax_ang.set_xlabel("Time (s)")
            ax_ang.set_ylabel("Steer Angle (deg)")
            ax_ang.legend()

            # —— β
            ax_beta.plot(t_ax, beta_hist, color="purple", lw=1.6)
            b_abs = np.abs(beta_hist).max()
            ax_beta.set_ylim(-1.05 * b_abs if b_abs > 0 else -1,
                             1.05 * b_abs if b_abs > 0 else 1)
            ax_beta.grid(ls="--", alpha=0.5)
            ax_beta.set_xlabel("Time (s)")
            ax_beta.set_ylabel("β")

            fig.tight_layout()
            fig.savefig(f"{self.out}/ep{self.i:04d}.png", dpi=150)
            plt.close(fig)
            np.savetxt(f"{self.out}/ep{self.i:04d}.csv", traj, delimiter=",")
            self.i += 1
        return True


# ---------------- TensorBoard Q‑value Callback ----------------
# ---------------- TensorBoard Q‑value Callback ----------------
class TensorboardQCallback(BaseCallback):
    def __init__(self, sample_size=256, log_freq=1000):
        super().__init__()
        self.n = sample_size
        self.f = log_freq

    def _on_step(self):
        if self.n_calls % self.f or self.model.replay_buffer.size() < self.n:
            return True

        batch = self.model.replay_buffer.sample(self.n)
        obs_t = torch.as_tensor(batch.observations, device=self.model.device)
        act_t = torch.as_tensor(batch.actions,      device=self.model.device)

        with torch.no_grad():
            q_out = self.model.critic(obs_t, act_t)
        # 取第一个张量
        q_tensor = q_out[0] if isinstance(q_out, tuple) else q_out
        self.logger.record("custom/mean_q", q_tensor.mean().item())
        return True


# ---------------- VecEnv 工厂 -------------------------
def make_env(rank, path, dll, seed=0):
    def _init():
        env = FourWSEnv(path, dll)
        env.reset(seed=seed + rank)
        return env
    return _init


# =========================== MAIN =============================
if __name__ == "__main__":
    DLL  = r"D:\chrome-download\5\5.4\vehiclemodel_public_0326_win64.dll"
    PATH = np.asarray(
        loadmat("mat/lanechange_double_hold20m_40_40_shift20.mat")["path_ref"],
        float
    )

    n_envs = 4
    set_random_seed(42)

    # --- VecEnv ---
    env = SubprocVecEnv(
        [make_env(i, PATH, DLL, seed=42) for i in range(n_envs)],
        start_method="spawn"
    )
    env = VecMonitor(env)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, "| envs:", n_envs)

    # ---------- DDPG ----------
    ddpg = DDPG(
        policy="MlpPolicy",
        env=env,
        device=device,
        policy_kwargs=dict(net_arch=[400, 300]),
        buffer_size=300_000,
        batch_size=512,
        learning_rate=1e-4,          # 较小 LR 避免震荡
        gamma=0.98,
        tau=0.005,
        train_freq=(1, "step"),
        gradient_steps=1,
        action_noise=NormalActionNoise(
            mean=np.zeros(1),
            sigma=0.1 * np.ones(1)   # 连续探索
        ),
        verbose=1,
        tensorboard_log="runs_ddpg"
    )

    # ---------- 回调 ----------
    ckpt = CheckpointCallback(
        save_freq=1000,
        save_path="checkpoints/ddpg_70",
        name_prefix="ddpg_step"
    )
    callbacks = CallbackList([
        TrajCallback(PATH),
        TensorboardQCallback(256, 1000),
        ckpt
    ])

    # ---------- 训练 ----------
    ddpg.learn(
        total_timesteps=8_000_000,
        callback=callbacks,
        progress_bar=True
    )

    ddpg.save("ddpg_mpc_big_final_spawn_v2.zip")
    print("✔ training finished — final DDPG model saved")

