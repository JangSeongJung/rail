"""DXF 파서: 레일 경로(3D 폴리라인)와 지지점(POINT) 추출.

지원 엔티티:
    - 경로: POLYLINE(3D), LWPOLYLINE, LINE(여러 개를 끝점 기준으로 연결), SPLINE(평탄화)
    - 지지점: 지정한 레이어의 POINT 엔티티
"""
from __future__ import annotations

from dataclasses import dataclass, field

import ezdxf
import numpy as np


@dataclass
class DxfData:
    """DXF에서 읽어들인 원시 기하 정보."""

    path: np.ndarray                       # (N, 3) 경로 정점
    supports: np.ndarray                   # (M, 3) 지지점 좌표
    path_layer: str = ""
    support_layer: str = ""
    meta: dict = field(default_factory=dict)


def _spline_points(entity, n: int = 50) -> list[tuple[float, float, float]]:
    """SPLINE을 n개 점으로 평탄화."""
    try:
        return [tuple(p) for p in entity.flattening(distance=0.01, segments=n)]
    except Exception:
        return [tuple(p[:3]) for p in entity.control_points]


def _polyline_points(entity) -> list[tuple[float, float, float]]:
    dxftype = entity.dxftype()
    if dxftype == "LWPOLYLINE":
        z = float(getattr(entity.dxf, "elevation", 0.0) or 0.0)
        return [(float(x), float(y), z) for x, y, *_ in entity.get_points()]
    if dxftype == "POLYLINE":
        pts = []
        for v in entity.vertices:
            loc = v.dxf.location
            pts.append((float(loc.x), float(loc.y), float(loc.z)))
        return pts
    if dxftype == "SPLINE":
        return _spline_points(entity)
    return []


def _chain_lines(lines: list[tuple[np.ndarray, np.ndarray]], tol: float = 1e-6
                 ) -> list[tuple[float, float, float]]:
    """떨어진 LINE 세그먼트들을 끝점 매칭으로 하나의 경로로 연결."""
    if not lines:
        return []
    remaining = list(lines)
    chain = list(remaining.pop(0))
    changed = True
    while remaining and changed:
        changed = False
        for i, (a, b) in enumerate(remaining):
            if np.linalg.norm(chain[-1] - a) <= tol:
                chain.append(b)
            elif np.linalg.norm(chain[-1] - b) <= tol:
                chain.append(a)
            elif np.linalg.norm(chain[0] - b) <= tol:
                chain.insert(0, a)
            elif np.linalg.norm(chain[0] - a) <= tol:
                chain.insert(0, b)
            else:
                continue
            remaining.pop(i)
            changed = True
            break
    return [tuple(float(c) for c in p) for p in chain]


def list_layers(dxf_path: str) -> list[str]:
    """DXF에 존재하는 레이어 이름 목록."""
    doc = ezdxf.readfile(dxf_path)
    return sorted(layer.dxf.name for layer in doc.layers)


def parse_dxf(dxf_path: str,
              path_layer: str | None = None,
              support_layer: str | None = None) -> DxfData:
    """DXF에서 레일 경로와 지지점을 추출한다.

    Args:
        dxf_path: DXF 파일 경로.
        path_layer: 경로 엔티티가 있는 레이어. None이면 모든 레이어에서
            가장 긴(정점이 많은) 경로를 채택.
        support_layer: 지지점 POINT가 있는 레이어. None이면 모든 POINT 사용.

    Returns:
        DxfData
    """
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    # --- 경로 후보 수집 ---
    candidates: list[list[tuple[float, float, float]]] = []
    line_segments: list[tuple[np.ndarray, np.ndarray]] = []

    for e in msp:
        if path_layer and e.dxf.layer != path_layer:
            continue
        dxftype = e.dxftype()
        if dxftype in ("POLYLINE", "LWPOLYLINE", "SPLINE"):
            pts = _polyline_points(e)
            if len(pts) >= 2:
                candidates.append(pts)
        elif dxftype == "LINE":
            a = np.array([e.dxf.start.x, e.dxf.start.y, e.dxf.start.z], float)
            b = np.array([e.dxf.end.x, e.dxf.end.y, e.dxf.end.z], float)
            line_segments.append((a, b))

    chained = _chain_lines(line_segments)
    if len(chained) >= 2:
        candidates.append(chained)

    if not candidates:
        raise ValueError(
            "경로로 쓸 폴리라인/라인/스플라인을 찾지 못했습니다."
            + (f" (레이어='{path_layer}')" if path_layer else "")
        )

    # 가장 긴(정점 많은) 경로 채택
    path_pts = max(candidates, key=len)
    path = _dedupe(np.asarray(path_pts, dtype=float))

    # --- 지지점 수집 ---
    supports = []
    for e in msp.query("POINT"):
        if support_layer and e.dxf.layer != support_layer:
            continue
        loc = e.dxf.location
        supports.append((float(loc.x), float(loc.y), float(loc.z)))
    supports_arr = np.asarray(supports, dtype=float).reshape(-1, 3)

    return DxfData(
        path=path,
        supports=supports_arr,
        path_layer=path_layer or "(auto)",
        support_layer=support_layer or "(all)",
        meta={"n_path_vertices": len(path), "n_supports": len(supports_arr)},
    )


def _dedupe(pts: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """연속 중복 정점 제거."""
    if len(pts) == 0:
        return pts
    keep = [pts[0]]
    for p in pts[1:]:
        if np.linalg.norm(p - keep[-1]) > tol:
            keep.append(p)
    return np.asarray(keep, dtype=float)
