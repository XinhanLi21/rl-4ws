# -*- coding: utf-8 -*-
"""
mpc_front_only.py ── 前轮单变量 MPC（δr 恒 0）
接口与 mpc_v2.py 相同：step(t,u) -> [delta_f, delta_r=0, e_lat, e_yaw]
"""

import numpy as np
from cvxopt import matrix, solvers
import sympy as sp

# ─────────── 车辆与 MPC 参数 ───────────
class VehiclePara:
    def __init__(self):
        self.m, self.g = 1350.0, 9.8
        self.Lf, self.Lr = 1.4, 1.6
        self.Iz = 1536.0
        self.Ccf, self.Ccr = 6.69e4, 6.27e4   # N / rad


class MPCParameters:
    def __init__(self, Ts=0.05, Np=50, Nc=16):
        self.Ts, self.Np, self.Nc = Ts, Np, Nc


# ─────────── MPC 控制器 ───────────
class MPC4WS:
    """
    对照组：仅优化前轮 δf，后轮 δr 恒 0
    u = [Vy(km/h), Vx(km/h), psi(deg), r(deg/s), Y, X]
    """

    def __init__(self, path_ref, Ts=0.05, vx=70 / 3.6):
        self.param = VehiclePara()
        self.mpc = MPCParameters(Ts=Ts)
        self.vx = float(vx)
        self.U = 0.0                                  # 上一时刻 δf (rad)

        self.path = np.asarray(path_ref, float)
        if self.path.ndim != 2 or self.path.shape[1] != 3:
            raise ValueError("path_ref 需为 (N,3) [X,Y,psi]")

        self._build_symbolic_jacobians()

    # ----------------------------------------------------------
    def reset(self):
        self.U = 0.0

    # ----------------------------------------------------------
    def step(self, t, u):
        # 1. 实时状态转换
        y_dot = u[0] / 3.6
        x_dot = u[1] / 3.6 + 1e-4
        phi = np.deg2rad(u[2])
        r = np.deg2rad(u[3])
        Y_pos, X_pos = u[4], u[5]

        # 2. 线性化模型
        A, b = self._model_jacobian(phi, y_dot, r)

        # 3. 增广系统
        Nu, Nx, Ny = 1, 4, 2
        Ts, Np, Nc = self.mpc.Ts, self.mpc.Np, self.mpc.Nc
        Aaug = np.block([[A, b.reshape(-1, 1)],
                         [np.zeros((Nu, Nx)), np.eye(Nu)]])
        Baug = np.vstack([b.reshape(-1, 1), [[1.0]]])
        Cmat = np.array([[1, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0]])

        # 4. PSI、THETA
        PSI, THETA = [], []
        for p in range(1, Np + 1):
            Ap = np.linalg.matrix_power(Aaug, p)
            PSI.append(Cmat @ Ap)
            row = []
            for k in range(1, Nc + 1):
                row.append(Cmat @ np.linalg.matrix_power(Aaug, p - k) @ Baug
                           if k <= p else np.zeros((Ny, Nu)))
            THETA.append(np.hstack(row))
        PSI, THETA = np.vstack(PSI), np.vstack(THETA)

        # 5. 参考轨迹
        vx_body = x_dot * np.cos(phi) - y_dot * np.sin(phi)
        X_pred = X_pos + Ts * np.arange(1, Np + 1) * vx_body
        Y_ref = np.interp(X_pred, self.path[:, 0], self.path[:, 1])
        psi_ref = np.interp(X_pred, self.path[:, 0], self.path[:, 2])
        ref_vec = np.vstack([psi_ref, Y_ref]).T.reshape(-1, 1)

        # 6. 误差
        kesi_aug = np.array([phi, y_dot, Y_pos, r, self.U]).reshape(-1, 1)
        err = ref_vec - PSI @ kesi_aug

        # 7. 自适应权重
        lat_err0 = Y_pos - Y_ref[0]
        yaw_err0 = phi - psi_ref[0]
        w_lat = 8000.0 * (1 + min(abs(lat_err0), 1.0) * 4.0)
        w_yaw = 4000.0 * (1 + min(abs(yaw_err0), 0.5) * 4.0)
        Qblk = np.kron(np.eye(Np), np.diag([w_yaw, w_lat]))
        Rblk = 1.2e4 * np.eye(Nu * Nc)

        H = 2 * (THETA.T @ Qblk @ THETA + Rblk)
        f = (-2 * (err.T @ Qblk @ THETA)).flatten()

        # 8. 约束
        A_t = np.tril(np.ones((Nc, Nc)))
        A_I = np.kron(A_t, np.eye(Nu))
        Ut_vec = self.U * np.ones((Nc, 1))
        u_min, u_max = -0.3744, 0.3744
        dumin, dumax = -0.174, 0.174
        A_cons = np.vstack([A_I, -A_I])
        b_cons = np.vstack([u_max - Ut_vec, -u_min + Ut_vec])
        lb, ub = dumin * np.ones((Nc, 1)), dumax * np.ones((Nc, 1))

        # 9. QP
        P = matrix(0.5 * (H + H.T))
        q = matrix(f)
        G = matrix(np.vstack([A_cons, np.eye(Nc), -np.eye(Nc)]))
        h = matrix(np.vstack([b_cons, ub, -lb]))
        solvers.options['show_progress'] = False
        try:
            du = float(solvers.qp(P, q, G, h)['x'][0])
        except Exception:
            du = 0.0

        # 10. 舵角更新
        self.U += du
        delta_f = np.clip(self.U, u_min, u_max)
        delta_r = 0.0                         # 后轮固定 0°

        return np.array([delta_f, delta_r, float(lat_err0), float(yaw_err0)])

    # ─────────── 雅可比函数 ───────────
    def _build_symbolic_jacobians(self):
        phi, y_dot, r, delta_f = sp.symbols('phi y_dot r delta_f')
        p, Ts, vx = self.param, self.mpc.Ts, self.vx

        dy = (-(p.Ccf + p.Ccr) * y_dot - (p.Lf * p.Ccf - p.Lr * p.Ccr) * r) / (p.m * vx) \
             + vx * r + p.Ccf * delta_f / p.m
        dr = (-(p.Lf * p.Ccf - p.Lr * p.Ccr) * y_dot
              - (p.Lf**2 * p.Ccf + p.Lr**2 * p.Ccr) * r) / (p.Iz * vx) \
             + p.Lf * p.Ccf * delta_f / p.Iz
        Yd = vx * phi + y_dot
        fvec = sp.Matrix([r, dy, Yd, dr])
        Ac = fvec.jacobian([phi, y_dot, sp.symbols('Ypos'), r])
        Bc = fvec.jacobian([delta_f])

        self._A = sp.lambdify((phi, y_dot, r),
                              sp.eye(4) + Ts * Ac, 'numpy')
        self._B = sp.lambdify((phi, y_dot, r),
                              Ts * Bc, 'numpy')

    def _model_jacobian(self, phi, y_dot, r):
        A = np.asarray(self._A(phi, y_dot, r), float)
        B = np.asarray(self._B(phi, y_dot, r), float).flatten()
        return A, B



