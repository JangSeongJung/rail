"""커맨드라인 검증/실행 스크립트.

사용:
    python run_cli.py sample_rail.dxf --mass 2000 --speed 1.5 --accel 0.5 \
        --impact 0.3 --seg 0.3 --support-layer SUPPORT
"""
from __future__ import annotations

import argparse

from rail_analyzer.dxf_parser import parse_dxf
from rail_analyzer.geometry import build_geometry
from rail_analyzer.model import Section, build_model, make_interval_supports
from rail_analyzer.loads import MovingDevice
from rail_analyzer.analysis import run_analysis


def main():
    ap = argparse.ArgumentParser(description="레일 이동하중 구조검토")
    ap.add_argument("dxf")
    ap.add_argument("--path-layer", default=None)
    ap.add_argument("--support-layer", default="SUPPORT")
    ap.add_argument("--mass", type=float, default=2000.0, help="기기 질량 [kg]")
    ap.add_argument("--speed", type=float, default=1.5, help="최고속도 v_max [m/s]")
    ap.add_argument("--accel", type=float, default=0.0,
                    help="접선 가속/제동 [m/s^2] (마력 미지정 시 사용)")
    # --- 마력 기반 가속도 산정 ---
    ap.add_argument("--power", type=float, default=None,
                    help="추진 출력(마력). 지정 시 accel을 자동 산정")
    ap.add_argument("--power-unit", default="kw",
                    choices=["ps", "hp", "kw", "w"], help="마력 단위")
    ap.add_argument("--base-speed", type=float, default=None,
                    help="기저속도 [m/s] (미지정 시 0.3*speed)")
    ap.add_argument("--eff", type=float, default=1.0, help="구동효율 η")
    ap.add_argument("--impact", type=float, default=0.3, help="충격계수 phi")
    ap.add_argument("--seg", type=float, default=0.3, help="요소 분할 길이 [m]")
    ap.add_argument("--support-type", default="pinned",
                    choices=["pinned", "vertical", "fixed"])
    ap.add_argument("--support-spacing", type=float, default=None,
                    help="DXF에 지지점이 없을 때 자동 생성 간격 [m]")
    ap.add_argument("--selfweight", action="store_true")
    args = ap.parse_args()

    data = parse_dxf(args.dxf, args.path_layer, args.support_layer)
    print(f"[DXF] 경로 정점 {len(data.path)}개, 지지점 {len(data.supports)}개")

    geom = build_geometry(data.path, args.seg)
    print(f"[GEOM] 길이 {geom.length:.2f} m, 해석노드 {geom.n_nodes}개, "
          f"최소곡률반경 {geom.radius[geom.radius < 1e8].min() if (geom.radius<1e8).any() else float('inf'):.2f} m")

    supports = data.supports
    if len(supports) == 0:
        if args.support_spacing is None:
            raise SystemExit(
                "DXF에 지지점(POINT)이 없습니다. --support-spacing 으로 간격을 지정하세요.")
        supports = make_interval_supports(geom, args.support_spacing)
        print(f"[지지점 자동생성] 간격 {args.support_spacing} m → {len(supports)}개")

    section = Section()
    model = build_model(geom, supports, section, args.support_type)
    print(f"[MODEL] 요소 {len(model.elements)}개, 지지노드 {len(model.supports)}개")

    from rail_analyzer.loads import POWER_UNITS
    power_w = args.power * POWER_UNITS[args.power_unit] if args.power else None
    device = MovingDevice(mass=args.mass, speed=args.speed, accel=args.accel,
                          impact_factor=args.impact, power=power_w,
                          base_speed=args.base_speed, efficiency=args.eff)
    if power_w is not None:
        print(f"[추진] 출력 {args.power} {args.power_unit} = {power_w/1000:.1f} kW, "
              f"기저속도 {device.base_speed:.2f} m/s, η={args.eff}")
        print(f"[추진] → 최대 견인력 {device.traction/1000:.2f} kN, "
              f"최대 가속도 {device.accel:.3f} m/s²")
    res = run_analysis(model, geom, device, include_selfweight=args.selfweight)

    print("\n=== 결과 요약 ===")
    print(f"지배 지지점: #{res.gov_support_index} "
          f"(좌표 {res.support_xyz[res.gov_support_index].round(2)})")
    print(f"최대 지지점 반력: {res.gov_reaction/1000:.2f} kN  "
          f"(하중 위치 s={res.gov_s:.2f} m)")
    print(f"최대 휨모멘트: {res.gov_moment/1000:.2f} kN·m "
          f"(요소 #{res.gov_moment_elem}, s={res.elem_moment_at_s[res.gov_moment_elem]:.2f} m)")
    print(f"최대 전단: {res.elem_max_shear.max()/1000:.2f} kN")
    print(f"최대 축력: {res.elem_max_axial.max()/1000:.2f} kN")

    print("\n지지점별 최대 반력 [kN]:")
    for m, node in enumerate(res.support_nodes):
        xyz = res.support_xyz[m].round(2)
        print(f"  지지#{m} (노드{node}, {xyz}): "
              f"{res.reaction_max_resultant[m]/1000:7.2f}  @s={res.reaction_max_at_s[m]:.2f}m")


if __name__ == "__main__":
    main()
