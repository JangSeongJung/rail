"""경로 주행 3케이스 + 진자 동역학 연동 → 노드별 하중.

트롤리를 경로 위에서 세 가지로 주행시키고, 각 노드에서 진자(사람) 동역학을
풀어 부착점(레일 노드) 하중을 산정한다.

주행 케이스:
    ACC (최대가속) : 정지에서 출발해 a_acc = ηP/((m_t+m_p)·v_b) 로 가속.
                     노드 속도 v=min(√(2 a_acc s), v_max), 진자는 정지(연직)에서 과도.
    CON (최고등속) : v_max 등속. 접선가속 0, 곡선 원심만. (정상상태)
    DEC (최대감속) : v_max 진입 후 비상제동 -a_brake. 등속 정상상태에서 과도.

각 노드에서 그 주행상태로 짧게 시간적분(수 진자주기)하여 과도까지 포함한
부착점 하중의 포락(최대)을 그 노드 하중으로 쓴다. 비상정지는 임의 위치에서
발생할 수 있으므로 노드별 독립 평가가 보수적이고 타당하다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import PathGeometry
from .dynamics import Pendulum, simulate, _Z

_BIG = 1e8


def _steady_dir(a_trolley: np.ndarray, g: float) -> np.ndarray:
    """주어진 트롤리 가속에서 진자 정상상태 줄방향(트롤리→사람)."""
    g_eff = -g * _Z - a_trolley
    return g_eff / np.linalg.norm(g_eff)


def _centripetal(geom: PathGeometry, i: int, v: float) -> np.ndarray:
    """노드 i, 속도 v에서 원심(구심) 가속도 벡터 [m/s²]."""
    R = geom.radius[i]
    if R >= _BIG:
        return np.zeros(3)
    return (v * v / R) * geom.normal[i]   # normal = 구심(곡률중심) 방향


@dataclass
class CaseResult:
    name: str
    node_force: np.ndarray   # (N,3) 부착점 하중 [N]
    tension: np.ndarray      # (N,) 줄 장력 최대 [N]
    daf: np.ndarray          # (N,) 동적증폭 = T_max / (m_p g)
    v_node: np.ndarray       # (N,) 통과 속도 [m/s]
    a_tan: np.ndarray        # (N,) 접선가속 [m/s²]


def run_case(geom: PathGeometry, pen: Pendulum, mode: str,
             v_max: float, a_acc: float, a_brake: float,
             n_periods: float = 3.0, dt: float = 2e-3,
             progress=None) -> CaseResult:
    """한 주행 케이스의 노드별 하중 산정.

    progress: 호출 가능 객체. 노드 1개 끝날 때마다 progress() 호출(진행률용).
    """
    N = geom.n_nodes
    F = np.zeros((N, 3))
    Tn = np.zeros(N)
    daf = np.zeros(N)
    vv = np.zeros(N)
    aa = np.zeros(N)
    t_end = n_periods * pen.period
    Wp = pen.m_person * pen.g

    for i in range(N):
        if mode == "acc":
            v = min(np.sqrt(max(0.0, 2.0 * a_acc * geom.s[i])), v_max)
            a_tan = a_acc if v < v_max - 1e-9 else 0.0
            a_prev = np.zeros(3)            # 정지/연직에서 출발
            n0 = np.array([0.0, 0.0, -1.0])
        elif mode == "con":
            v = v_max
            a_tan = 0.0
            a_now = _centripetal(geom, i, v)
            a_prev = a_now                 # 이미 정상상태(원심)
            n0 = _steady_dir(a_now, pen.g)
        elif mode == "dec":
            v = v_max
            a_tan = -a_brake
            a_prev = _centripetal(geom, i, v)   # 제동 직전 등속 정상상태
            n0 = _steady_dir(a_prev, pen.g)
        else:
            raise ValueError(mode)

        a_now = a_tan * geom.tangent[i] + _centripetal(geom, i, v)
        res = simulate(lambda t: a_now, t_end, dt, pen, n0=n0)
        Fmag = np.linalg.norm(res["F_node"], axis=1)
        k = int(np.argmax(Fmag))
        F[i] = res["F_node"][k]
        Tn[i] = res["T"].max()
        daf[i] = Tn[i] / Wp
        vv[i] = v
        aa[i] = a_tan
        if progress is not None:
            progress()

    return CaseResult(mode.upper(), F, Tn, daf, vv, aa)


def run_all_cases(geom: PathGeometry, pen: Pendulum, device,
                  a_brake: float, accel_limit: float = 2.0,
                  progress=None, **kw) -> dict[str, CaseResult]:
    """ACC/CON/DEC 세 케이스 모두 산정.

    device: MovingDevice (v_max=device.speed).
    accel_limit: 가속도 상한 [m/s²]. 사람운반은 접지마찰·안락도로 제한되므로
        마력 기반 견인 가속이 이를 넘으면 클램프한다(기본 2.0 m/s² ≈ 0.2g).
    progress: 노드 1개 처리마다 호출되는 콜백(진행률 표시용).
    """
    v_max = device.speed
    m_tot = pen.m_person + pen.m_trolley
    a_raw = device.traction / m_tot if device.traction > 0 else device.accel
    a_acc = min(a_raw, accel_limit)
    return {
        "acc": run_case(geom, pen, "acc", v_max, a_acc, a_brake, progress=progress, **kw),
        "con": run_case(geom, pen, "con", v_max, a_acc, a_brake, progress=progress, **kw),
        "dec": run_case(geom, pen, "dec", v_max, a_acc, a_brake, progress=progress, **kw),
    }
