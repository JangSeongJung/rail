"""검증용 샘플 DXF 생성.

직선 → 곡선 → 직선 형태의 3D 레일 경로(POLYLINE)와
일정 간격 지지점(POINT, 'SUPPORT' 레이어)을 만든다.
"""
from __future__ import annotations

import numpy as np
import ezdxf


def make_sample(path_out: str = "sample_rail.dxf",
                support_spacing: float = 3.0) -> str:
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    doc.layers.add("RAIL", color=1)
    doc.layers.add("SUPPORT", color=3)

    # 경로: 직선 6m -> 반경 5m 90도 곡선 -> 직선 6m, 약간의 높이 변화
    pts = []
    # 직선부 1
    for x in np.linspace(0, 6, 13):
        pts.append((x, 0.0, 0.0))
    # 곡선부 (반경 5, 중심 (6,5))
    R = 5.0
    cx, cy = 6.0, 5.0
    for a in np.linspace(-np.pi / 2, 0.0, 18)[1:]:
        x = cx + R * np.cos(a)
        y = cy + R * np.sin(a)
        pts.append((x, y, 0.0))
    # 직선부 2 (+y 방향, 살짝 상승)
    x_end = cx + R
    for k, y in enumerate(np.linspace(5.0, 11.0, 13)[1:]):
        pts.append((x_end, y, 0.02 * k))

    msp.add_polyline3d(pts, dxfattribs={"layer": "RAIL"})

    # 지지점: 경로 호장 기준 일정 간격
    arr = np.array(pts)
    seg = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    s_sup = np.arange(0.0, total + 1e-9, support_spacing)
    for s in s_sup:
        p = [np.interp(s, cum, arr[:, k]) for k in range(3)]
        msp.add_point(p, dxfattribs={"layer": "SUPPORT"})

    doc.saveas(path_out)
    print(f"생성: {path_out}  (경로길이={total:.2f} m, 지지점={len(s_sup)}개)")
    return path_out


if __name__ == "__main__":
    make_sample()
