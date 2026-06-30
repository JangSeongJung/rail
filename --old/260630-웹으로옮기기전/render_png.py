"""해석 결과를 PNG로 렌더링 (GUI 없이 시각화 확인용)."""
from __future__ import annotations

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

from rail_analyzer.dxf_parser import parse_dxf
from rail_analyzer.geometry import build_geometry
from rail_analyzer.model import Section, build_model, make_interval_supports
from rail_analyzer.loads import MovingDevice
from rail_analyzer.analysis import run_analysis


def main(dxf, out, path_layer, spacing, mass, speed, power_kw, base_speed,
         impact, seg, eff=0.9):
    data = parse_dxf(dxf, path_layer, "SUPPORT")
    geom = build_geometry(data.path, seg)
    supports = data.supports if len(data.supports) else make_interval_supports(geom, spacing)
    model = build_model(geom, supports, Section())
    device = MovingDevice(mass=mass, speed=speed, impact_factor=impact,
                          power=power_kw * 1000.0, base_speed=base_speed,
                          efficiency=eff)
    res = run_analysis(model, geom, device)
    print(f"[추진] 최대 견인력 {device.traction/1000:.2f} kN, "
          f"최대 가속도 {device.accel:.3f} m/s²")

    fig = plt.figure(figsize=(11, 9), dpi=110)
    ax = fig.add_subplot(2, 1, 1, projection="3d")
    nd = geom.nodes
    ax.plot(nd[:, 0], nd[:, 1], nd[:, 2], "-", color="0.4", lw=1.3, label="레일 경로")
    sx = res.support_xyz
    rk = res.reaction_max_resultant / 1000.0
    sc = ax.scatter(sx[:, 0], sx[:, 1], sx[:, 2], c=rk, cmap="jet", s=70,
                    depthshade=False, edgecolors="k", label="지지점")
    fig.colorbar(sc, ax=ax, shrink=0.6, label="최대반력 [kN]")
    g = res.gov_support_index
    ax.scatter([sx[g, 0]], [sx[g, 1]], [sx[g, 2]], s=200, facecolors="none",
               edgecolors="red", lw=2.5)
    gi = int(np.argmin(np.abs(geom.s - res.gov_s)))
    ax.scatter([nd[gi, 0]], [nd[gi, 1]], [nd[gi, 2]], marker="v", s=110,
               color="red", label="지배 하중위치")
    ax.set_title(f"레일 경로 / 지지점 최대반력  (지배 #{g}: {res.gov_reaction/1000:.1f} kN)")
    ax.legend(loc="upper left", fontsize=8)
    try:
        ax.set_box_aspect(np.ptp(nd, axis=0) + 1e-6)
    except Exception:
        pass

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(res.s_positions, res.reaction_history[:, g] / 1000.0, "-b", lw=1.6)
    ax2.axhline(res.gov_reaction / 1000.0, color="r", ls="--", lw=1,
                label=f"최대 {res.gov_reaction/1000:.1f} kN")
    ax2.axvline(res.gov_s, color="r", ls=":", lw=1)
    ax2.set_title(f"지배 지지점 #{g} 반력 영향선")
    ax2.set_xlabel("하중 위치 s [m]")
    ax2.set_ylabel("반력 [kN]")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print("저장:", out)


if __name__ == "__main__":
    # 인자: dxf out  (나머지는 기본값; path_layer=None → 자동 채택)
    main(sys.argv[1], sys.argv[2], None, 2.0, 2000, 2.0, 10, 0.6, 0.3, 0.3)
