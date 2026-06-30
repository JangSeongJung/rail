"""레일 이동하중 종합 검토 - HTML 리포트 + 3케이스 MGT 생성.

사용:
    python run_report.py sample_rail.dxf --html report.html --mgt rail_3case.mgt \
        --L 3.0 --m-person 80 --m-trolley 120 --speed 2.0 \
        --power 10 --base-speed 0.6 --brake 4.0 --start-node 1001 --start-elem 5001
"""
from __future__ import annotations

import argparse

from rail_analyzer.dxf_parser import parse_dxf
from rail_analyzer.geometry import build_geometry
from rail_analyzer.dynamics import Pendulum
from rail_analyzer.loads import MovingDevice, POWER_UNITS
from rail_analyzer.driving import run_all_cases
from rail_analyzer.report import build_report
from rail_analyzer.mgt_export import write_mgt_cases


def main():
    ap = argparse.ArgumentParser(description="레일 이동하중 종합 검토(동·정역학)")
    ap.add_argument("dxf")
    ap.add_argument("--path-layer", default=None)
    ap.add_argument("--seg", type=float, default=0.3)
    # 진자/기기
    ap.add_argument("--L", type=float, default=3.0, help="줄 길이 [m]")
    ap.add_argument("--m-person", type=float, default=80.0, help="사람+하네스 [kg]")
    ap.add_argument("--m-trolley", type=float, default=120.0, help="트롤리 [kg]")
    ap.add_argument("--damping", type=float, default=0.03, help="진자 감쇠비")
    ap.add_argument("--speed", type=float, default=2.0, help="최고속도 [m/s]")
    ap.add_argument("--power", type=float, default=10.0, help="추진출력")
    ap.add_argument("--power-unit", default="kw", choices=["ps", "hp", "kw", "w"])
    ap.add_argument("--base-speed", type=float, default=0.6, help="기저속도 [m/s]")
    ap.add_argument("--eff", type=float, default=0.85, help="구동효율")
    ap.add_argument("--brake", type=float, default=4.0, help="비상감속 [m/s²]")
    ap.add_argument("--accel-limit", type=float, default=2.0, help="가속 상한 [m/s²]")
    # 출력
    ap.add_argument("--html", default="report.html")
    ap.add_argument("--mgt", default=None)
    ap.add_argument("--start-node", type=int, default=1001)
    ap.add_argument("--start-elem", type=int, default=5001)
    ap.add_argument("--imat", type=int, default=1)
    ap.add_argument("--ipro", type=int, default=1)
    ap.add_argument("--mgt-force", default="N")
    ap.add_argument("--mgt-len", default="MM")
    args = ap.parse_args()

    data = parse_dxf(args.dxf, args.path_layer, "SUPPORT")
    geom = build_geometry(data.path, args.seg)
    pen = Pendulum(args.m_person, args.m_trolley, args.L, args.damping)
    power_w = args.power * POWER_UNITS[args.power_unit]
    device = MovingDevice(mass=args.m_person + args.m_trolley, speed=args.speed,
                          power=power_w, base_speed=args.base_speed, efficiency=args.eff)
    print(f"[기기] 견인력 {device.traction/1000:.2f} kN, 진자주기 {pen.period:.2f} s")

    cases = run_all_cases(geom, pen, device, a_brake=args.brake,
                          accel_limit=args.accel_limit)
    import numpy as np
    for nm, c in cases.items():
        fmag = np.linalg.norm(c.node_force, axis=1)
        print(f"  [{c.name}] |F|최대 {fmag.max():.0f} N, DAF최대 {c.daf.max():.2f}")

    html = build_report(geom, cases, pen, device, args.start_node,
                        args.brake, args.accel_limit)
    with open(args.html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[HTML] 저장: {args.html}")

    if args.mgt:
        case_forces = {nm.upper(): c.node_force for nm, c in cases.items()}
        write_mgt_cases(args.mgt, geom, case_forces,
                        start_node=args.start_node, start_elem=args.start_elem,
                        imat=args.imat, ipro=args.ipro,
                        unit_force=args.mgt_force, unit_len=args.mgt_len)
        ncase = 3 * geom.n_nodes
        print(f"[MGT] 저장: {args.mgt}  (케이스 {ncase}개 = 3×{geom.n_nodes}, "
              f"노드 {args.start_node}~{args.start_node + geom.n_nodes - 1})")


if __name__ == "__main__":
    main()
