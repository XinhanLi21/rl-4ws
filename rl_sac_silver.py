# -*- coding: utf-8 -*-
"""
rl_td3_fast_spawn.py — TD3 训练
• 轨迹/舵角/β 三联图
• TensorBoard 记录 Q1/Q2、Loss
"""
import multiprocessing as mp
mp.set_start_method("spawn", force=True)

import os, math, pathlib, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium.spaces import Box
from scipy.io import loadmat
import torch

from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, CallbackList
)
from stable_baselines3.common.utils import set_random_seed

from rl_mpc_silver import MPC4WS
from wrapper import VehicleModelPublic


# ---------------- 环境封装 -------------------------------------------------
class FourWSEnv(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self, path, dll, vx=60/3.6,
                 Ts=0.05, dt=0.01, beta_lim=0.4):
        super().__init__()
        self.path, self.vx = path, vx
        self.Ts, self.dt   = Ts, dt
        self.ratio         = int(Ts / dt)
        self.beta_lim, self.act_lim = beta_lim, 0.2
        self.action_space      = Box(-self.act_lim, self.act_lim, (1,), np.float32)
        self.observation_space = Box(-np.inf, np.inf, (3,), np.float32)
        self.ctrl = MPC4WS(path, Ts=Ts, vx=vx)
        self.veh  = VehicleModelPublic(str(pathlib.Path(dll)))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.ctrl.reset(); self.veh.terminate()
        self.veh.initial(0.0, self.vx, 0.0)
        self.t = self.beta = self.delta_f = 0.0
        self.traj = []
        self.delta_f_hist = []; self.delta_r_hist = []; self.beta_hist = []
        return self._obs(), {}

    def step(self, action):
        self.beta = float(np.clip(self.beta + action[0], -self.beta_lim, self.beta_lim))
        for _ in range(self.ratio):
            ov = self.veh.step(self.delta_f, self.beta * self.delta_f)
            self.traj.append((ov[0], ov[1]))
            self.t += self.dt
        y = self.ctrl.step(self.t, self._assemble_u(ov, self.beta * self.delta_f))
        self.delta_f = float(y[0])
        self.delta_f_hist.append(self.delta_f)
        self.delta_r_hist.append(self.beta * self.delta_f)
        self.beta_hist.append(self.beta)
        lat_err, yaw_err = float(y[2]), float(y[3])
        reward = -20.0 * lat_err**2 - 10 * yaw_err**2
        done   = self.t >= 40.0
        info = {}
        if done:
            info = dict(
                traj=np.asarray(self.traj, np.float32),
                delta_f_hist=np.asarray(self.delta_f_hist, np.float32),
                delta_r_hist=np.asarray(self.delta_r_hist, np.float32),
                beta_hist   =np.asarray(self.beta_hist,    np.float32),
                Ts=self.Ts
            )
        return np.array([lat_err, yaw_err, self.beta], np.float32), reward, done, False, info

    def _obs(self):
        ov  = self.veh._get_current_observation()
        lat, yaw = self.ctrl.step(self.t, self._assemble_u(ov, 0.0))[2:4]
        return np.array([lat, yaw, self.beta], np.float32)

    def _assemble_u(self, ov, delta_r):
        X, Y, yaw, Vx, Vy, r = ov
        return np.array([Vy*3.6, Vx*3.6,
                         math.degrees(yaw), math.degrees(r),
                         Y, X, math.degrees(delta_r), 0.0], np.float32)

    def close(self):
        self.veh.terminate()


# ---------------- 绘图回调 -------------------------------------------------
class TrajCallback(BaseCallback):
    def __init__(self, ref_path, out_dir="trajectories/trajectories_silver"):
        super().__init__()
        self.ref = ref_path; self.out = out_dir
        os.makedirs(out_dir, exist_ok=True); self.i = 0
    def _on_step(self): return True
    def _on_rollout_end(self):
        for info in self.locals["infos"]:
            if "traj" not in info: continue
            traj, df, dr, beta = info["traj"], info["delta_f_hist"], info["delta_r_hist"], info["beta_hist"]
            Ts = info["Ts"]; t = np.arange(len(df)) * Ts
            # --- 画幅大小 ---
            x_min = min(self.ref[:,0].min(), traj[:,0].min()); x_max = max(self.ref[:,0].max(), traj[:,0].max())
            y_min = min(self.ref[:,1].min(), traj[:,1].min()); y_max = max(self.ref[:,1].max(), traj[:,1].max())
            pad_x = 0.05*(x_max-x_min); pad_y = 0.05*(y_max-y_min)
            x_min-=pad_x; x_max+=pad_x; y_min-=pad_y; y_max+=pad_y
            W,H = x_max-x_min, y_max-y_min
            fig,axs = plt.subplots(3,1,figsize=(6,6*H/W+3.5),
                                   gridspec_kw={"height_ratios":[H/W,0.35,0.25]})
            # XY
            axs[0].plot(self.ref[:,0], self.ref[:,1],"k--",label="Ref")
            axs[0].plot(traj[:,0],traj[:,1],"r-",label="Actual")
            axs[0].set_xlim(x_min,x_max); axs[0].set_ylim(y_min,y_max)
            axs[0].set_aspect("equal"); axs[0].legend(); axs[0].grid(ls="--")
            # 舵角
            axs[1].plot(t,np.degrees(df),label="δf"); axs[1].plot(t,np.degrees(dr),label="δr")
            axs[1].set_ylabel("deg"); axs[1].legend(); axs[1].grid(ls="--")
            # beta
            axs[2].plot(t,beta,color="purple"); axs[2].set_xlabel("t(s)"); axs[2].set_ylabel("β"); axs[2].grid(ls="--")
            fig.tight_layout(); fig.savefig(f"{self.out}/ep{self.i:04d}.png",dpi=150); plt.close(fig)
            np.savetxt(f"{self.out}/ep{self.i:04d}.csv",traj,delimiter=","); self.i+=1
        return True


# ---------------- TensorBoard Q‑value 回调 ---------------------
class TensorboardQCallback(BaseCallback):
    def __init__(self, sample_size=256, log_freq=1000):
        super().__init__()
        self.sample_size = sample_size; self.log_freq = log_freq
    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq: return True
        if self.model.replay_buffer.size() < self.sample_size: return True
        # ★ FIX 1: 不传 env，避免 normalize_obs 报错
        batch = self.model.replay_buffer.sample(self.sample_size)  # ★
        obs_t = torch.as_tensor(batch.observations, device=self.model.device)
        act_t = torch.as_tensor(batch.actions,      device=self.model.device)
        with torch.no_grad():
            q1,q2 = self.model.critic(obs_t, act_t)
        # 记录
        self.logger.record("custom/mean_q1", q1.mean().item())
        self.logger.record("custom/mean_q2", q2.mean().item())
        return True


# ---------------- VecEnv 工厂 ---------------------------------
def make_env(rank, path, dll, seed=0):
    def _init():
        env = FourWSEnv(path, dll)
        env.reset(seed=seed+rank); return env
    return _init


# =========================== MAIN =============================
if __name__ == "__main__":
    DLL  = r"D:\chrome-download\5\5.4\vehiclemodel_public_0326_win64.dll"
    PATH = np.asarray(loadmat("mat/smooth_closed_track_path_corrected.mat")["path_ref"], float)
    n_envs = 4; set_random_seed(42)

    env = SubprocVecEnv([make_env(i, PATH, DLL, 42) for i in range(n_envs)],
                        start_method="spawn")
    env = VecMonitor(env)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, "| envs:", n_envs)

    td3 = TD3(
        "MlpPolicy", env, device=device,
        policy_kwargs=dict(net_arch=[400,300]),
        buffer_size=300_000, batch_size=256,
        learning_rate=1e-3, gradient_steps=8,
        action_noise=NormalActionNoise(np.zeros(1), 0.1*np.ones(1)),
        gamma=0.98, tau=0.005, train_freq=(1,"step"), verbose=1,
        tensorboard_log="runs_td3"                 # ★ FIX 2
    )

    ckpt = CheckpointCallback(save_freq=80000,
                              save_path="checkpoints/checkpoints_silver",
                              name_prefix="td3_step")
    callbacks = CallbackList([
        TrajCallback(PATH),
        TensorboardQCallback(sample_size=256, log_freq=1000),
        ckpt
    ])

    td3.learn(total_timesteps=8_000_000,
              callback=callbacks,
              progress_bar=True)
    td3.save("td3_mpc_big_final_spawn_v2.zip")