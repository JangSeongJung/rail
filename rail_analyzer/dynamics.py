"""매달린 사람(구면진자)의 동역학 시간이력.

트롤리(동력체)가 경로를 강제 주행할 때, 지지점 가속도 a_trolley(t)가
길이 L 줄에 매달린 사람(구면진자)을 가진한다. 트롤리 좌표계(비관성)에서
유효중력 g_eff = -g*ẑ - a_trolley 방향으로 거동한다.

줄 단위벡터 n (트롤리→사람), 강체 줄(길이 L 일정):
    n̈ = (1/L)[g_eff - (g_eff·n)n] - (ṅ·ṅ)n - 2ζωₙ ṅ
    T  = m_p[(g_eff·n) + L(ṅ·ṅ)]                     (줄 장력)
    F_node = -m_t a_trolley - m_t g ẑ + T n           (레일 노드 하중)

검증: a_trolley=0 → 자유진동 주기 2π√(L/g);
      수평 등가속 a_h → 정상상태 기울기 atan(a_h/g), 장력 m_p√(g²+a_h²).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GRAVITY = 9.80665
_Z = np.array([0.0, 0.0, 1.0])


@dataclass
class Pendulum:
    """매달린 사람 + 트롤리 제원."""
    m_person: float                 # 사람+하네스 질량 [kg]
    m_trolley: float                # 트롤리(동력체) 질량 [kg]
    length: float                   # 줄 길이 [m]
    damping: float = 0.02           # 진자 감쇠비 ζ (공기저항 등)
    g: float = GRAVITY

    @property
    def omega_n(self) -> float:
        return np.sqrt(self.g / self.length)

    @property
    def period(self) -> float:
        return 2.0 * np.pi / self.omega_n


def _g_eff(a_trolley: np.ndarray, pen: Pendulum) -> np.ndarray:
    return -pen.g * _Z - a_trolley


def _accel(n: np.ndarray, ndot: np.ndarray,
           a_trolley: np.ndarray, pen: Pendulum) -> np.ndarray:
    """n̈ (구속면 접선 + 구심 + 감쇠)."""
    ge = _g_eff(a_trolley, pen)
    tang = ge - np.dot(ge, n) * n            # g_eff의 접선성분
    return (tang / pen.length
            - np.dot(ndot, ndot) * n
            - 2.0 * pen.damping * pen.omega_n * ndot)


def tension(n: np.ndarray, ndot: np.ndarray,
            a_trolley: np.ndarray, pen: Pendulum) -> float:
    ge = _g_eff(a_trolley, pen)
    return float(pen.m_person * (np.dot(ge, n) + pen.length * np.dot(ndot, ndot)))


def node_force(n: np.ndarray, a_trolley: np.ndarray,
               T: float, pen: Pendulum) -> np.ndarray:
    """레일 부착점(노드)에 전달되는 3성분 힘 [N].

    = 트롤리 관성 반작용 + 트롤리 자중 + 줄 장력.
    """
    return (-pen.m_trolley * a_trolley
            - pen.m_trolley * pen.g * _Z
            + T * n)


def _renorm(n: np.ndarray, ndot: np.ndarray):
    """수치 드리프트 보정: |n|=1, ndot⊥n 유지."""
    n = n / np.linalg.norm(n)
    ndot = ndot - np.dot(ndot, n) * n
    return n, ndot


def simulate(a_func, t_end: float, dt: float, pen: Pendulum,
             n0: np.ndarray | None = None,
             ndot0: np.ndarray | None = None) -> dict:
    """구면진자 시간이력 (RK4).

    Args:
        a_func: t[s] -> 트롤리 가속도 벡터 (3,) [m/s²] (전역좌표).
        t_end, dt: 적분 종료시간/스텝 [s].
        pen: 진자 제원.
        n0, ndot0: 초기 줄방향/각속도. 기본은 연직 매달림 정지.
    Returns:
        dict(t, n, theta, T, F_node) — 시간배열과 응답이력.
    """
    if n0 is None:
        n0 = np.array([0.0, 0.0, -1.0])
    if ndot0 is None:
        ndot0 = np.zeros(3)
    n0, ndot0 = _renorm(np.asarray(n0, float), np.asarray(ndot0, float))

    steps = int(round(t_end / dt))
    ts = np.zeros(steps + 1)
    ns = np.zeros((steps + 1, 3))
    Ts = np.zeros(steps + 1)
    Fs = np.zeros((steps + 1, 3))
    thetas = np.zeros(steps + 1)  # 연직(-ẑ)으로부터 줄 기울기 [rad]

    n, ndot = n0.copy(), ndot0.copy()
    for k in range(steps + 1):
        t = k * dt
        a = np.asarray(a_func(t), float)
        T = tension(n, ndot, a, pen)
        ts[k] = t
        ns[k] = n
        Ts[k] = T
        Fs[k] = node_force(n, a, T, pen)
        thetas[k] = np.arccos(np.clip(np.dot(-n, _Z), -1.0, 1.0))
        if k == steps:
            break
        # RK4 on state (n, ndot)
        def f(state, tt):
            nn, nd = state[:3], state[3:]
            aa = np.asarray(a_func(tt), float)
            return np.concatenate([nd, _accel(nn, nd, aa, pen)])
        y = np.concatenate([n, ndot])
        k1 = f(y, t)
        k2 = f(y + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = f(y + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = f(y + dt * k3, t + dt)
        y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        n, ndot = _renorm(y[:3], y[3:])

    return {"t": ts, "n": ns, "theta": thetas, "T": Ts, "F_node": Fs}
