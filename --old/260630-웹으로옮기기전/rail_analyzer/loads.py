"""이동하중(1점 집중) 산정.

레일 위 위치 s에 있는 기기에 대해 전역좌표 힘 벡터를 계산한다.
    - 연직 자중:        W = m * g                (전역 -Z, 충격계수 적용)
    - 접선 관성(가속/제동): F_t = m * a            (경로 접선 방향)
    - 곡선 원심력:        F_c = m * v^2 / R         (주법선 = 곡률 중심 방향)
충격계수(DAF) = 1 + impact_factor 는 기본적으로 연직 자중에 적용한다.

접선 가속도는 추진체 출력(마력)으로부터 산정할 수 있다. 등출력 구동에서
기저속도 v_b 이하는 일정 견인력(최대 토크) 영역으로 보고, 그때의 최대
견인력 F_max = η·P/v_b 가 내는 가속도를 worst-case 최대 가속도로 본다.
    a_max = η · P / (m · v_b)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import PathGeometry

GRAVITY = 9.80665  # m/s^2

# 마력 단위 → 와트
POWER_UNITS = {
    "ps": 735.49875,   # 미터마력(PS)
    "hp": 745.69987,   # 영국마력(HP)
    "kw": 1000.0,      # 킬로와트
    "w": 1.0,
}


def traction_accel(mass: float, power_w: float, base_speed: float,
                   efficiency: float = 1.0) -> tuple[float, float]:
    """등출력 추진체의 최대 견인력/최대 가속도 산정.

    기저속도(base speed) 이하에서는 일정 견인력 영역으로 보고,
    그 최대 견인력 F_max = η·P/v_b 가 내는 가속도를 최대 가속도로 본다.
    주행저항(롤링/구배)은 가속을 깎으므로 무시하는 쪽이 보수적이다.

    Args:
        mass: 이동체 질량 [kg].
        power_w: 추진 출력 [W].
        base_speed: 기저속도 [m/s] (이 속도까지 일정 견인력).
        efficiency: 구동효율 η (P_바퀴 = η·P_출력).
    Returns:
        (F_max [N], a_max [m/s^2])
    """
    if base_speed <= 0:
        raise ValueError("기저속도는 0보다 커야 합니다.")
    if mass <= 0:
        raise ValueError("질량은 0보다 커야 합니다.")
    f_max = efficiency * power_w / base_speed
    return f_max, f_max / mass


@dataclass
class MovingDevice:
    """이동 기기 제원.

    가속도(accel)는 두 방식 중 하나로 정한다.
        1) accel 직접 입력
        2) power(+base_speed)로부터 자동 산정 → accel을 덮어씀
    power가 주어지면 항상 마력 기반 산정값이 우선한다.
    """

    mass: float                      # 질량 [kg]
    speed: float                     # 최고속도 v_max [m/s] (원심력에 사용)
    accel: float = 0.0               # 접선 가속/제동 [m/s^2] (부호=방향)
    impact_factor: float = 0.0       # 충격계수 phi (DAF = 1 + phi)
    g: float = GRAVITY
    gravity_dir: tuple[float, float, float] = (0.0, 0.0, -1.0)
    daf_on_all: bool = False         # True면 동적성분 전체에 DAF 적용
    apply_tangential: bool = True
    apply_centrifugal: bool = True

    # --- 마력 기반 가속도 산정(선택) ---
    power: float | None = None       # 추진 출력 [W]
    base_speed: float | None = None  # 기저속도 [m/s] (None이면 0.3*speed)
    efficiency: float = 1.0          # 구동효율 η
    traction: float = field(default=0.0, init=False)  # 산출 최대 견인력 [N]

    def __post_init__(self):
        if self.power is not None:
            vb = self.base_speed if self.base_speed else 0.3 * self.speed
            self.base_speed = vb
            self.traction, self.accel = traction_accel(
                self.mass, self.power, vb, self.efficiency)

    @property
    def daf(self) -> float:
        return 1.0 + self.impact_factor


def device_force_at(device: MovingDevice, geom: PathGeometry, i: int) -> np.ndarray:
    """경로 노드 i에 위치한 기기의 전역 힘 벡터 [N] (3,)."""
    m = device.mass
    g_dir = np.asarray(device.gravity_dir, dtype=float)
    g_dir = g_dir / (np.linalg.norm(g_dir) or 1.0)

    # 연직 자중 (충격계수 적용)
    f_gravity = m * device.g * device.daf * g_dir

    f_dyn = np.zeros(3)
    # 접선 관성력 (가속/제동)
    if device.apply_tangential and device.accel != 0.0:
        f_dyn += m * device.accel * geom.tangent[i]

    # 원심력 (곡선부)
    if device.apply_centrifugal:
        R = geom.radius[i]
        if np.isfinite(R) and R > 1e-9:
            f_dyn += m * (device.speed ** 2) / R * geom.normal[i]

    if device.daf_on_all:
        f_dyn *= device.daf

    return f_gravity + f_dyn


def self_weight_nodal(geom: PathGeometry, section) -> np.ndarray:
    """레일 자중을 노드별 전역 연직하중 [N]으로 환산. (선택적 추가하중)

    Returns:
        (N, 3) 각 노드에 작용하는 자중 벡터(전역 -Z).
    """
    n = geom.n_nodes
    seg = np.diff(geom.s)
    w_per_len = section.A * section.density * GRAVITY  # [N/m]
    nodal = np.zeros((n, 3))
    # 각 요소 자중을 양 끝 노드에 절반씩 분배
    for e in range(n - 1):
        half = 0.5 * w_per_len * seg[e]
        nodal[e, 2] -= half
        nodal[e + 1, 2] -= half
    return nodal
