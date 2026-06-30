"""이동하중 포락선 해석.

기기를 경로 노드를 따라 한 위치씩 이동시키며 매 위치마다 정적 해석을 수행하고,
각 지지점 반력과 부재 단면력의 최대값(포락선) 및 발생 위치를 집계한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import PathGeometry
from .loads import MovingDevice, device_force_at, self_weight_nodal
from .model import StructuralModel
from .solver import FrameSolver


@dataclass
class EnvelopeResult:
    s_positions: np.ndarray                  # (P,) 하중 위치(호장)

    support_nodes: list[int]                 # 지지 노드 인덱스
    support_xyz: np.ndarray                  # (M, 3)
    reaction_max_resultant: np.ndarray       # (M,) 지지점별 합력 최대
    reaction_max_at_s: np.ndarray            # (M,) 그 최대가 생기는 하중 위치
    reaction_comp_at_max: np.ndarray         # (M, 3) 최대시 Rx,Ry,Rz
    reaction_history: np.ndarray             # (P, M) 영향선용 합력 이력

    gov_support_index: int                   # 지배 지지점(전체 최대) 인덱스
    gov_reaction: float                      # 최대 반력 합력 [N]
    gov_s: float                             # 발생 하중 위치 [m]

    elem_max_axial: np.ndarray               # (E,) [N]
    elem_max_shear: np.ndarray               # (E,) [N]
    elem_max_moment: np.ndarray              # (E,) [N·m]
    elem_max_torsion: np.ndarray             # (E,) [N·m]
    elem_moment_at_s: np.ndarray             # (E,) 최대 휨 발생 위치

    gov_moment: float = 0.0                  # 최대 휨모멘트 [N·m]
    gov_moment_elem: int = 0
    meta: dict = field(default_factory=dict)


def _assemble_force(model: StructuralModel, geom: PathGeometry,
                    device: MovingDevice, i: int,
                    selfweight: np.ndarray | None) -> np.ndarray:
    F = np.zeros(model.n_dof)
    if selfweight is not None:
        # 노드별 연직 자중을 병진 DOF에 적용
        for n in range(model.n_nodes):
            F[6 * n:6 * n + 3] += selfweight[n]
    f = device_force_at(device, geom, i)
    F[6 * i:6 * i + 3] += f
    return F


def run_analysis(model: StructuralModel, geom: PathGeometry,
                 device: MovingDevice, include_selfweight: bool = False,
                 step: int = 1) -> EnvelopeResult:
    """이동하중 포락선 해석 실행.

    Args:
        model: 해석 모델.
        geom: 경로 기하.
        device: 이동 기기.
        include_selfweight: 레일 자중 동시 고려 여부.
        step: 하중 위치 간격(노드 단위). 1이면 모든 노드.
    """
    solver = FrameSolver(model)
    n_nodes = model.n_nodes
    n_elem = len(model.elements)
    selfweight = self_weight_nodal(geom, model.section) if include_selfweight else None

    support_nodes = sorted(model.supports.keys())
    M = len(support_nodes)
    positions = list(range(0, n_nodes, max(1, step)))
    P = len(positions)

    reaction_history = np.zeros((P, M))
    reaction_comp = np.zeros((P, M, 3))

    elem_max_axial = np.zeros(n_elem)
    elem_max_shear = np.zeros(n_elem)
    elem_max_moment = np.zeros(n_elem)
    elem_max_torsion = np.zeros(n_elem)
    elem_moment_at_s = np.zeros(n_elem)

    s_positions = geom.s[positions]

    for p, i in enumerate(positions):
        F = _assemble_force(model, geom, device, i, selfweight)
        res = solver.solve(F)

        # 지지점 반력 합력
        for m, node in enumerate(support_nodes):
            rx = res.reactions.get(6 * node + 0, 0.0)
            ry = res.reactions.get(6 * node + 1, 0.0)
            rz = res.reactions.get(6 * node + 2, 0.0)
            reaction_comp[p, m] = (rx, ry, rz)
            reaction_history[p, m] = np.sqrt(rx * rx + ry * ry + rz * rz)

        # 요소 단면력 포락
        ef = res.elem_forces
        for e in range(n_elem):
            f = ef[e]
            axial = max(abs(f[0]), abs(f[6]))
            shear = max(np.hypot(f[1], f[2]), np.hypot(f[7], f[8]))
            mom = max(np.hypot(f[4], f[5]), np.hypot(f[10], f[11]))
            tor = max(abs(f[3]), abs(f[9]))
            if axial > elem_max_axial[e]:
                elem_max_axial[e] = axial
            if shear > elem_max_shear[e]:
                elem_max_shear[e] = shear
            if tor > elem_max_torsion[e]:
                elem_max_torsion[e] = tor
            if mom > elem_max_moment[e]:
                elem_max_moment[e] = mom
                elem_moment_at_s[e] = s_positions[p]

    # 지지점별 최대 반력 및 발생 위치
    reaction_max_resultant = reaction_history.max(axis=0)
    argmax_p = reaction_history.argmax(axis=0)
    reaction_max_at_s = s_positions[argmax_p]
    reaction_comp_at_max = np.array(
        [reaction_comp[argmax_p[m], m] for m in range(M)]
    ) if M else np.empty((0, 3))

    gov_m = int(reaction_max_resultant.argmax()) if M else 0
    gov_reaction = float(reaction_max_resultant[gov_m]) if M else 0.0
    gov_s = float(reaction_max_at_s[gov_m]) if M else 0.0

    gov_e = int(elem_max_moment.argmax()) if n_elem else 0
    gov_moment = float(elem_max_moment[gov_e]) if n_elem else 0.0

    return EnvelopeResult(
        s_positions=s_positions,
        support_nodes=support_nodes,
        support_xyz=model.support_xyz,
        reaction_max_resultant=reaction_max_resultant,
        reaction_max_at_s=reaction_max_at_s,
        reaction_comp_at_max=reaction_comp_at_max,
        reaction_history=reaction_history,
        gov_support_index=gov_m,
        gov_reaction=gov_reaction,
        gov_s=gov_s,
        elem_max_axial=elem_max_axial,
        elem_max_shear=elem_max_shear,
        elem_max_moment=elem_max_moment,
        elem_max_torsion=elem_max_torsion,
        elem_moment_at_s=elem_moment_at_s,
        gov_moment=gov_moment,
        gov_moment_elem=gov_e,
        meta={"n_positions": P, "n_nodes": n_nodes, "n_elements": n_elem,
              "include_selfweight": include_selfweight},
    )
