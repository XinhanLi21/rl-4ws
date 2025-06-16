# -*- coding: utf-8 -*-
"""
mpc_4ws.py —— 4‑wheel‑steer MPC (δr as measured input)
----------------------------------------------------------------
* 输入 u 至少含 7 项: [Vy(km/h), Vx(km/h), ψ(deg), r(deg/s), Y, X, δr(deg), …]
* MPC 只优化前轮 δf；测得的 δr 直接进入预测模型 (常值保持 Np 步)
* 返回: [δ_f(rad), 0.0, lat_err(m), yaw_err(rad)]
"""

import numpy as np
from cvxopt import matrix, solvers
import sympy as sp

# ───────────────────────── 车辆 & MPC 参数 ─────────────────────────
class VehiclePara:
    def __init__(self):
        self.m, self.g = 1350.0, 9.8
        self.Lf, self.Lr = 1.4, 1.6
        self.Iz = 1536.0
        self.Ccf, self.Ccr = 6.69e4, 6.27e4          # N/rad


class MPCParameters:
    def __init__(self, Ts=0.05, Np=50, Nc=16):
        self.Ts, self.Np, self.Nc = Ts, Np, Nc


# ──────────────────────────── MPC 控制器 ───────────────────────────
class MPC4WS:
    """
    前轮单变量优化；后轮角为已知输入并参与模型。
    """
    def __init__(self, path_ref, Ts=0.05, vx=70/3.6):
        self.par = VehiclePara()
        self.mpc = MPCParameters(Ts, Np=50, Nc=16)
        self.vx  = float(vx)
        self.U   = 0.0                               # δf(k‑1)

        self.path = np.asarray(path_ref, float)
        if self.path.ndim != 2 or self.path.shape[1] != 3:
            raise ValueError("path_ref 需为 (N,3) = [X, Y, ψ(rad)]")

        self._build_symbolic_jac()

    # --------------------------------------------------------------
    def reset(self):
        self.U = 0.0

    # --------------------------------------------------------------
    def step(self, t, u):
        # ① 读入实测量 ------------------------------------------------
        Vy  = u[0] / 3.6
        Vx  = u[1] / 3.6 + 1e-4
        phi = np.deg2rad(u[2])
        r   = np.deg2rad(u[3])
        Yp, Xp = u[4], u[5]
        delta_r = np.deg2rad(u[6]) if len(u) > 6 else 0.0

        # ② 线性化 (含 δr) -------------------------------------------
        A, Bf, Br = self._jac(phi, Vy, r)            # 4×4 4×1 4×1

        # ③ 增广系统 (把 δf(k‑1) 视作状态) ---------------------------
        Nx, Nu = 4, 1
        Aaug = np.block([[A,               Bf.reshape(-1, 1)],
                         [np.zeros((Nu, Nx)), np.eye(Nu)]])     # 5×5
        Bf_aug = np.vstack([Bf.reshape(-1, 1), [[1.0]]])        # 5×1
        Br_aug = np.vstack([Br.reshape(-1, 1), [[0.0]]])        # 5×1

        C = np.array([[1, 0, 0, 0, 0],                          # yaw
                      [0, 0, 1, 0, 0]])                         # Y

        # ④ 预测矩阵 -------------------------------------------------
        Np, Nc, Ts = self.mpc.Np, self.mpc.Nc, self.mpc.Ts
        PSI, THETA, OMEGA = [], [], []

        Apow = np.eye(5)
        for p in range(1, Np + 1):
            Apow = Apow @ Aaug
            PSI.append(C @ Apow)

            # — THETA (δf 未来增量) —
            rowT = []
            Ap_k = np.eye(5)
            for k in range(1, Nc + 1):
                if k <= p:
                    Ap_k = np.linalg.matrix_power(Aaug, p - k)
                    rowT.append(C @ Ap_k @ Bf_aug)
                else:
                    rowT.append(np.zeros((2, 1)))
            THETA.append(np.hstack(rowT))

            # — OMEGA (δr 常量影响) —
            S = np.zeros_like(Bf_aug)
            Ak_i = np.eye(5)
            for i in range(p):
                if i > 0:
                    Ak_i = Ak_i @ Aaug
                S += Ak_i @ Br_aug
            OMEGA.append(C @ S)

        PSI, THETA = np.vstack(PSI), np.vstack(THETA)           # 2Np×5  2Np×Nc
        OMEGA = np.vstack(OMEGA)                                # 2Np×1

        # ⑤ 参考轨迹 -------------------------------------------------
        vx_inert = Vx * np.cos(phi) - Vy * np.sin(phi)
        X_pred   = Xp + Ts * np.arange(1, Np + 1) * vx_inert
        Y_ref    = np.interp(X_pred, self.path[:, 0], self.path[:, 1])
        psi_ref  = np.interp(X_pred, self.path[:, 0], self.path[:, 2])
        ref_vec  = np.vstack([psi_ref, Y_ref]).T.reshape(-1, 1)

        # ⑥ 增广状态 & 误差 -----------------------------------------
        kesi_aug = np.array([phi, Vy, Yp, r, self.U]).reshape(-1, 1)
        err = ref_vec - (PSI @ kesi_aug + OMEGA * delta_r)

        # ⑦ QP -------------------------------------------------------
        Qblk = np.kron(np.eye(Np), np.diag([4000., 7000.]))
        Rblk = 2e4 * np.eye(Nc)

        H = 2 * (THETA.T @ Qblk @ THETA + Rblk)
        f = (-2 * err.T @ Qblk @ THETA).flatten()

        A_t = np.tril(np.ones((Nc, Nc)))
        A_I = np.kron(A_t, np.eye(1))
        Ut  = self.U * np.ones((Nc, 1))
        umin, umax = -0.64, 0.64
        dumin, dumax = -0.194, 0.194
        A_cons = np.vstack([A_I, -A_I])
        b_cons = np.vstack([umax - Ut, -umin + Ut])
        lb, ub = dumin * np.ones((Nc, 1)), dumax * np.ones((Nc, 1))

        du = self._solve_qp(H, f, A_cons, b_cons, lb, ub)

        # ⑧ 状态更新 & 返回 -----------------------------------------
        self.U += du
        lat_err = Yp  - Y_ref[0]
        yaw_err = phi - psi_ref[0]
        return np.array([self.U, 0.0, lat_err, yaw_err])

    # ──────────────────── 建模 & 工具函数 ────────────────────
    def _build_symbolic_jac(self):
        φ, y, r, δf, δr = sp.symbols("phi y_dot r delta_f delta_r")
        p, Ts, vx = self.par, self.mpc.Ts, self.vx

        ydd = (-(p.Ccf + p.Ccr) * y - (p.Lf * p.Ccf - p.Lr * p.Ccr) * r) / (p.m * vx) \
              + vx * r + p.Ccf * δf / p.m + p.Ccr * δr / p.m
        rd  = (-(p.Lf * p.Ccf - p.Lr * p.Ccr) * y
               - (p.Lf**2 * p.Ccf + p.Lr**2 * p.Ccr) * r) / (p.Iz * vx) \
              + p.Lf * p.Ccf * δf / p.Iz - p.Lr * p.Ccr * δr / p.Iz
        Yd  = vx * φ + y
        φd  = r

        f_vec = sp.Matrix([φd, ydd, Yd, rd])
        A_c   = f_vec.jacobian([φ, y, sp.symbols("Ypos"), r])
        B_c   = f_vec.jacobian([δf, δr])                      # 4×2

        self._Afun = sp.lambdify((φ, y, r), (sp.eye(4) + Ts * A_c), 'numpy')
        self._Bfun = sp.lambdify((φ, y, r), (Ts * B_c), 'numpy')

    def _jac(self, phi, y_dot, r):
        A = np.asarray(self._Afun(phi, y_dot, r), float)
        B = np.asarray(self._Bfun(phi, y_dot, r), float)
        return A, B[:, :1], B[:, 1:2]                        # Bf, Br

    # ---------- QP 求解（CVXOPT） ----------
    @staticmethod
    def _solve_qp(H, f, A, b, lb, ub):
        P = matrix(0.5 * (H + H.T))
        q = matrix(f)
        G_box = np.vstack([np.eye(len(f)), -np.eye(len(f))])
        h_box = np.vstack([ub, -lb])
        G = matrix(np.vstack([A, G_box]))
        h = matrix(np.vstack([b, h_box]))
        solvers.options["show_progress"] = False
        try:
            sol = solvers.qp(P, q, G, h)
            return float(sol['x'][0])
        except Exception:
            return 0.0
