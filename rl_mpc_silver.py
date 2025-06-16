# -*- coding: utf-8 -*-
"""
mpc_front_only.py —— 单前轮 MPC (δr 作为已知输入) · Frenet 预瞄版
接口:
    __init__(path_ref, Ts=…, vx=…)
    reset()
    step(t, u)  # u=[Vy,Vx,ψ,r,Y,X,δr,…], 单位同 mpc_4ws
返回: [δf(rad), 0.0, lat_err(m), yaw_err(rad)]
"""

import numpy as np
import sympy as sp
from cvxopt import matrix, solvers


# ────────────── 参数 ────────────── #
class VehiclePara:
    def __init__(self):
        self.m, self.g = 1350.0, 9.8
        self.Lf, self.Lr = 1.4, 1.6
        self.Iz = 1536.0
        self.Ccf, self.Ccr = 6.69e4, 6.27e4     # N / rad


class MPCParameters:
    def __init__(self, Ts=0.05, Np=35, Nc=8):
        self.Ts, self.Np, self.Nc = Ts, Np, Nc


# ────────────── 工具函数 ──────────── #
def wrap(a):                       # 把角度包到 [-π,π]
    return (a + np.pi) % (2*np.pi) - np.pi


def global2frenet(px, py, vxg, vyg, phi, xr, yr, psir):
    """全局坐标 → Frenet 局部坐标"""
    c, s = np.cos(psir), np.sin(psir)
    dx, dy = px - xr, py - yr
    x_l  =  c*dx + s*dy
    y_l  = -s*dx + c*dy
    vx_l =  c*vxg + s*vyg
    vy_l = -s*vxg + c*vyg
    phi_l = wrap(phi - psir)
    return x_l, y_l, vx_l, vy_l, phi_l


# ────────────── MPC 控制器 ──────────── #
class MPC4WS:
    """只优化前轮 δf，δr 作为已知扰动进入模型"""

    def __init__(self, path_ref, Ts=0.05, vx=60/3.6, look_ahead=300):
        # 基本参数
        self.par  = VehiclePara()
        self.mpc  = MPCParameters(Ts, 35, 8)
        self.vx   = float(vx)          # 线性化参考纵向速度
        self.Nu, self.Nx = 1, 4
        self.U    = 0.0
        self.idx_prev = 0
        self.LOOK = int(look_ahead)

        # 路径预处理
        self.path = np.asarray(path_ref, float)
        if self.path.ndim != 2 or self.path.shape[1] != 3:
            raise ValueError("path_ref 必须 (N,3)=[X,Y,ψ(rad)]")
        self.path[:, 2] = np.unwrap(self.path[:, 2])

        self._build_symbolic_jac()

    # ---------- 接口 ----------
    def reset(self):
        self.U = 0.0
        self.idx_prev = 0

    def step(self, t, u):
        # ① 车辆当前状态
        Vy  = u[0] / 3.6
        Vx  = max(u[1] / 3.6, 0.1)
        phi = np.deg2rad(u[2]);   r = np.deg2rad(u[3])
        Yp, Xp = u[4], u[5]
        delta_r = np.deg2rad(u[6]) if len(u) > 6 else 0.0

        # ② 最近参考点（2D 距离 + 窗口）
        N = len(self.path)
        win = (self.idx_prev + np.arange(-self.LOOK, self.LOOK+1)) % N
        d   = np.linalg.norm(self.path[win, :2] - [Xp, Yp], axis=1)
        self.idx_prev = int(win[np.argmin(d)])
        Xr, Yr, psir  = self.path[self.idx_prev]

        # ③ Frenet 误差
        _, y_l, vx_l, vy_l, phi_l = global2frenet(
            Xp, Yp, Vx, Vy, phi, Xr, Yr, psir)

        # ④ 线性化
        A, Bf, Br = self._jac(phi_l, vy_l, r)
        Nx, Nu = self.Nx, self.Nu
        Aaug = np.block([[A,               Bf],
                         [np.zeros((Nu, Nx)), np.eye(Nu)]])
        Bf_aug = np.vstack([Bf, [[1.0]]])
        Br_aug = np.vstack([Br, [[0.0]]])
        C = np.array([[1,0,0,0,0], [0,0,1,0,0]], float)  # ψ,y

        # ⑤ 预测矩阵
        Np, Nc = self.mpc.Np, self.mpc.Nc
        PSI, THETA, OMEGA = [], [], []
        Ap = np.eye(Nx + Nu)
        for p in range(1, Np+1):
            Ap = Ap @ Aaug
            PSI.append(C @ Ap)

            row = [C @ np.linalg.matrix_power(Aaug, p-k) @ Bf_aug if k<=p else
                   np.zeros((2,1)) for k in range(1, Nc+1)]
            THETA.append(np.hstack(row))

            S = np.zeros_like(Bf_aug)
            Ak = np.eye(Nx+Nu)
            for i in range(p):
                if i: Ak = Ak @ Aaug
                S += Ak @ Br_aug
            OMEGA.append(C @ S)

        PSI, THETA = np.vstack(PSI), np.vstack(THETA)
        OMEGA      = np.vstack(OMEGA)

        # ⑥ 参考序列（Np 个点）
        ref_idx = (self.idx_prev + np.arange(0, Np)) % N
        psi_ref = self.path[ref_idx, 2]
        y_ref   = np.zeros(Np)
        ref_vec = np.vstack([psi_ref, y_ref]).T.reshape(-1,1)

        # ⑦ 当前增广状态 & 误差
        kesi = np.array([phi_l, vy_l, y_l, r, self.U]).reshape(-1,1)
        err  = ref_vec - (PSI @ kesi + OMEGA * delta_r)
        err[0::2,0] = wrap(err[0::2,0])

        # ⑧ QP
        Qblk = np.kron(np.eye(Np), np.diag([4e3, 8e3]))
        Rblk = 2e4 * np.eye(Nc)
        H = 2*(THETA.T @ Qblk @ THETA + Rblk)
        f = (-2 * err.T @ Qblk @ THETA).flatten()

        # 约束 Δδf
        A_t = np.tril(np.ones((Nc, Nc)))
        A_I = np.kron(A_t, np.eye(Nu))
        Ut  = self.U*np.ones((Nc,1))
        umin, umax = -0.68, 0.68
        dumin, dumax = -0.1, 0.1
        A_cons = np.vstack([A_I, -A_I])
        b_cons = np.vstack([umax-Ut, -umin+Ut])
        lb, ub = dumin*np.ones((Nc,1)), dumax*np.ones((Nc,1))

        du = self._solve_qp(H, f, A_cons, b_cons, lb, ub)

        # ⑨ 输出
        self.U += du
        return np.array([self.U, 0.0, y_l, phi_l])

    # ────────────── 雅可比 ────────────── #
    def _build_symbolic_jac(self):
        φ, y, r, δf, δr = sp.symbols("phi y_dot r delta_f delta_r")
        p, Ts, vx = self.par, self.mpc.Ts, self.vx

        ydd = (-(p.Ccf+p.Ccr)*y - (p.Lf*p.Ccf-p.Lr*p.Ccr)*r)/(p.m*vx) \
              + vx*r + p.Ccf*δf/p.m + p.Ccr*δr/p.m
        rd  = (-(p.Lf*p.Ccf-p.Lr*p.Ccr)*y - (p.Lf**2*p.Ccf+p.Lr**2*p.Ccr)*r)/(p.Iz*vx) \
              + p.Lf*p.Ccf*δf/p.Iz - p.Lr*p.Ccr*δr/p.Iz
        Yd, φd = vx*φ + y, r
        f = sp.Matrix([φd, ydd, Yd, rd])
        Ac = f.jacobian([φ, y, sp.symbols("Ypos"), r])
        Bc = f.jacobian([δf, δr])

        self._Afun = sp.lambdify((φ, y, r), np.eye(4)+Ts*Ac, 'numpy')
        self._Bfun = sp.lambdify((φ, y, r), Ts*Bc, 'numpy')

    def _jac(self, phi, ydot, r):
        A = np.asarray(self._Afun(phi, ydot, r), float)
        B = np.asarray(self._Bfun(phi, ydot, r), float)
        return A, B[:, :1], B[:, 1:]     # Bf, Br

    # ────────────── QP 求解 ───────────── #
    @staticmethod
    def _solve_qp(H, f, A, b, lb, ub):
        P, q = matrix(0.5*(H+H.T)), matrix(f)
        G_box = np.vstack([np.eye(len(f)), -np.eye(len(f))])
        h_box = np.vstack([ub, -lb])
        G = matrix(np.vstack([A, G_box]))
        h = matrix(np.vstack([b, h_box]))
        solvers.options['show_progress'] = False
        try:
            sol = solvers.qp(P, q, G, h)
            return float(sol['x'][0])
        except Exception:
            return 0.0
