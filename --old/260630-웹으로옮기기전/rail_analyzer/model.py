"""해석 모델: 노드/요소/지지점/단면 정의 및 DXF 지지점 매핑.

레일을 3D 빔요소 연속체로 모델링한다. 각 요소는 양 끝 노드(노드당 6 DOF:
변위 3 + 회전 3)를 갖는 표준 3D 프레임 요소다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import PathGeometry


@dataclass
class Section:
    """레일 단면/재료 제원 (SI: N, m, Pa)."""

    E: float = 200e9          # 탄성계수 [Pa] (강재 기본값)
    G: float = 77e9           # 전단탄성계수 [Pa]
    A: float = 6.0e-3         # 단면적 [m^2]
    Iy: float = 1.0e-5        # 약축 단면2차모멘트 [m^4]
    Iz: float = 3.0e-5        # 강축 단면2차모멘트 [m^4]
    J: float = 5.0e-7         # 비틀림 상수 [m^4]
    density: float = 7850.0   # 밀도 [kg/m^3] (자중 고려 시)

    # 강도 검토용(선택). KDS 검토는 후속 단계.
    Sy: float | None = None   # 약축 단면계수 [m^3]
    Sz: float | None = None   # 강축 단면계수 [m^3]
    Fy: float | None = None   # 항복강도 [Pa]


# 지지 조건 프리셋: (UX, UY, UZ, RX, RY, RZ) True=구속
SUPPORT_PRESETS = {
    "pinned": (True, True, True, False, False, False),   # 3방향 병진 구속
    "vertical": (False, False, True, False, False, False),  # 연직만
    "fixed": (True, True, True, True, True, True),        # 완전고정
}


@dataclass
class StructuralModel:
    """3D 프레임 해석 모델."""

    nodes: np.ndarray                      # (N, 3)
    elements: np.ndarray                   # (E, 2) 노드 인덱스 쌍
    section: Section
    supports: dict[int, tuple[bool, ...]]  # node_index -> 6 DOF 구속 여부
    support_xyz: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_dof(self) -> int:
        return 6 * self.n_nodes


def nearest_node(nodes: np.ndarray, point: np.ndarray) -> int:
    """주어진 점에 가장 가까운 노드 인덱스."""
    d = np.linalg.norm(nodes - point[None, :], axis=1)
    return int(np.argmin(d))


def make_interval_supports(geom: PathGeometry, spacing: float) -> np.ndarray:
    """DXF에 지지점이 없을 때, 경로 호장 기준 일정 간격으로 지지점 좌표 생성.

    시작/끝점을 포함하며, 끝단 돌출부(캔틸레버)를 피하기 위해 경로 끝점도 지지점에
    포함한다.

    Args:
        geom: 경로 기하.
        spacing: 지지 간격 [m].
    Returns:
        (M, 3) 지지점 좌표.
    """
    total = geom.length
    s_targets = list(np.arange(0.0, total + 1e-9, spacing))
    if abs(s_targets[-1] - total) > 1e-6:
        s_targets.append(total)
    pts = []
    for s in s_targets:
        xyz = [np.interp(s, geom.s, geom.nodes[:, k]) for k in range(3)]
        pts.append(xyz)
    return np.asarray(pts, dtype=float)


def build_model(geom: PathGeometry,
                support_points: np.ndarray,
                section: Section,
                support_type: str = "pinned",
                stabilize_axial: bool = True) -> StructuralModel:
    """경로 기하 + 지지점에서 해석 모델 생성.

    Args:
        geom: 재분할된 경로 기하.
        support_points: (M, 3) DXF에서 읽은 지지점 좌표.
        section: 단면/재료.
        support_type: 'pinned' | 'vertical' | 'fixed'.
        stabilize_axial: 연직/측방만 구속할 때 강체이동 방지를 위해
            첫 지지점에 길이방향(UX) 구속을 추가.
    """
    nodes = geom.nodes
    n = len(nodes)
    elements = np.column_stack([np.arange(n - 1), np.arange(1, n)])

    preset = SUPPORT_PRESETS.get(support_type, SUPPORT_PRESETS["pinned"])

    supports: dict[int, tuple[bool, ...]] = {}
    used_xyz = []
    if len(support_points) == 0:
        raise ValueError("지지점이 없습니다. DXF에 POINT 지지점을 추가하세요.")

    for idx, p in enumerate(support_points):
        ni = nearest_node(nodes, p)
        dof = list(preset)
        if stabilize_axial and idx == 0 and not dof[0]:
            dof[0] = True
        supports[ni] = tuple(dof)
        used_xyz.append(nodes[ni])

    return StructuralModel(
        nodes=nodes,
        elements=elements,
        section=section,
        supports=supports,
        support_xyz=np.asarray(used_xyz, dtype=float).reshape(-1, 3),
    )
