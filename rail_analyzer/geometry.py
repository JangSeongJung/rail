"""경로 기하: 호장(arc-length) 매개변수화, 접선/주법선, 곡률반경 계산.

레일 경로를 3D 폴리라인으로 받아서
    - 일정 간격으로 재분할(meshing)
    - 각 노드의 단위 접선 t, 주법선 n(곡률 중심 방향), 곡률반경 R
을 제공한다. 원심력(법선)과 접선 관성력 산정에 쓰인다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.zeros_like(v)


@dataclass
class PathGeometry:
    """재분할된 경로의 노드별 기하 정보."""

    nodes: np.ndarray        # (N, 3) 노드 좌표
    s: np.ndarray            # (N,)  각 노드의 누적 호장
    tangent: np.ndarray      # (N, 3) 단위 접선
    normal: np.ndarray       # (N, 3) 단위 주법선(곡률 중심 방향). 직선부는 0벡터
    radius: np.ndarray       # (N,)  곡률반경(직선부는 inf)

    @property
    def length(self) -> float:
        return float(self.s[-1])

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)


def resample_path(vertices: np.ndarray, seg_len: float) -> np.ndarray:
    """폴리라인을 대략 seg_len 간격으로 재분할(정점/끝점 보존은 하지 않고 균일 분할).

    원래 정점의 위치는 호장 기준 선형보간으로 유지되므로 형상은 보존된다.
    """
    vertices = np.asarray(vertices, dtype=float)
    seg = np.diff(vertices, axis=0)
    seglen = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    total = cum[-1]
    if total <= 0:
        raise ValueError("경로 길이가 0입니다.")

    n = max(2, int(np.ceil(total / seg_len)) + 1)
    s_new = np.linspace(0.0, total, n)
    # 각 축을 호장에 대해 선형보간
    out = np.column_stack([np.interp(s_new, cum, vertices[:, k]) for k in range(3)])
    return out


def build_geometry(vertices: np.ndarray, seg_len: float) -> PathGeometry:
    """경로 정점에서 재분할 + 접선/주법선/곡률반경 계산."""
    nodes = resample_path(vertices, seg_len)
    n = len(nodes)

    seg = np.diff(nodes, axis=0)                 # (n-1, 3)
    seglen = np.linalg.norm(seg, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seglen)])
    u = np.array([_unit(d) for d in seg])        # (n-1, 3) 세그먼트 단위벡터

    tangent = np.zeros((n, 3))
    normal = np.zeros((n, 3))
    radius = np.full(n, np.inf)

    for i in range(n):
        if i == 0:
            tangent[i] = u[0]
        elif i == n - 1:
            tangent[i] = u[-1]
        else:
            tangent[i] = _unit(u[i - 1] + u[i])

        # 곡률: 인접 세그먼트 접선 변화량 기준
        if 0 < i < n - 1:
            du = u[i] - u[i - 1]
            ds = 0.5 * (seglen[i - 1] + seglen[i])
            # 두 단위벡터 사이 각도
            cos_a = np.clip(np.dot(u[i - 1], u[i]), -1.0, 1.0)
            theta = np.arccos(cos_a)
            if ds > 1e-12 and theta > 1e-9:
                kappa = theta / ds
                radius[i] = 1.0 / kappa
                normal[i] = _unit(du)
    return PathGeometry(nodes=nodes, s=s, tangent=tangent,
                        normal=normal, radius=radius)
