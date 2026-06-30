"""해석 모델/하중을 midas Gen MGT 텍스트로 내보내기.

'위치별 케이스 + 포락(envelope)' 방식:
    이동하중은 1점 집중하중이므로, 기기가 노드 i에 있는 '한 순간'을
    하나의 정적 하중케이스(MVxxx)로 만든다. 각 케이스에는 그 노드 하나에만
    device_force_at(자중×충격 + 접선관성 + 원심력)를 건다.
    모든 케이스를 LOADCOMB(Envelope)로 묶으면 Midas에서 그 조합 하나로
    각 지지점·부재의 이동하중 최대/최소(포락)를 자동 산출한다.

번호 매핑:
    - 노드: start_node 부터 1씩 (경로 형상, 1벌만 생성)
    - 요소: start_elem 부터 1씩, i번째 = (start_node+i, start_node+i+1)
    - 케이스: lcprefix+일련번호(MV001…). step으로 솎을 수 있음(기본 전체).

단위:
    내부 계산은 SI(m, N). MGT UNIT(force/length)에 맞춰 환산.
    예) UNIT=N, MM 이면 좌표 m→mm(×1000), 힘 N 그대로.
"""
from __future__ import annotations

import numpy as np

from .geometry import PathGeometry
from .loads import MovingDevice, device_force_at

# 길이 환산: m → 목표단위
_LEN = {"MM": 1000.0, "CM": 100.0, "M": 1.0}
# 힘 환산: N → 목표단위
_FORCE = {"N": 1.0, "KN": 1e-3, "KGF": 1.0 / 9.80665, "TONF": 1.0 / 9806.65}

# midas Gen LOADCOMB iTYPE: 0=Add, 1=Envelope, 2=ABS, 3=SRSS
# (실제 Midas export로 1=Envelope 확인됨)
LCOMB_ENVELOPE = 1
# LOADCOMB 라인2: 한 줄에 넣을 항목 수 (Midas export 관례: 4)
LCOMB_TERMS_PER_LINE = 4


def build_mgt(geom: PathGeometry,
              device: MovingDevice,
              start_node: int,
              start_elem: int,
              imat: int = 1,
              ipro: int = 1,
              lcprefix: str = "MV",
              lctype: str = "L",
              env_name: str = "MV_ENV",
              unit_force: str = "N",
              unit_len: str = "MM",
              angle: float = 0.0,
              step: int = 1,
              make_envelope: bool = True) -> str:
    """위치별 케이스 + envelope MGT 텍스트 생성.

    Args:
        geom: 경로 기하(노드 = 레일 형상).
        device: 이동 기기(하중 산정).
        start_node: 시작 노드 번호(이후 1씩 증가).
        start_elem: 시작 요소 번호(이후 1씩 증가).
        imat, ipro: Midas 모델의 재질/단면 번호(레일 단면).
        lcprefix: 하중케이스 이름 접두(MV → MV001…).
        lctype: 케이스 타입(D/L/W/E/S…). 이동하중은 보통 L.
        env_name: 포락 조합 이름.
        unit_force, unit_len: MGT 단위.
        angle: 요소 베타각.
        step: 케이스 생성 노드 간격(1=모든 노드).
        make_envelope: True면 LOADCOMB(Envelope)도 생성.
    """
    fl = _LEN[unit_len.upper()]
    ff = _FORCE[unit_force.upper()]

    nodes = geom.nodes
    n = len(nodes)
    positions = list(range(0, n, max(1, step)))
    npad = max(3, len(str(len(positions))))  # 케이스 일련번호 자리수

    out: list[str] = []
    out.append("; midas Gen Text(MGT) - rail moving load (per-position cases)")
    out.append(f"; nodes {start_node}..{start_node + n - 1}, "
               f"elems {start_elem}..{start_elem + n - 2}, "
               f"cases {len(positions)}")
    out.append("")
    out.append("*UNIT")
    out.append("; FORCE, LENGTH, HEAT, TEMPER")
    out.append(f"   {unit_force.upper()}, {unit_len.upper()}, KCAL, C")
    out.append("")

    # --- NODE ---
    out.append("*NODE")
    out.append("; iNO, X, Y, Z")
    for i in range(n):
        x, y, z = nodes[i] * fl
        out.append(f"   {start_node + i}, {x:.6f}, {y:.6f}, {z:.6f}")
    out.append("")

    # --- ELEMENT ---
    out.append("*ELEMENT")
    out.append("; iEL, TYPE, iMAT, iPRO, iN1, iN2, ANGLE, iSUB")
    for e in range(n - 1):
        n1 = start_node + e
        n2 = start_node + e + 1
        out.append(f"   {start_elem + e}, BEAM, {imat}, {ipro}, "
                   f"{n1}, {n2}, {angle:g}, 0")
    out.append("")

    # --- STLDCASE: 위치 수만큼 케이스 정의 ---
    case_names = [f"{lcprefix}{k + 1:0{npad}d}" for k in range(len(positions))]
    out.append("*STLDCASE")
    out.append("; LCNAME, LCTYPE, DESC")
    for name, pos in zip(case_names, positions):
        out.append(f"   {name}, {lctype}, device at node {start_node + pos} "
                   f"(s={geom.s[pos]:.3f}m)")
    out.append("")

    # --- 케이스별 CONLOAD (해당 노드 1점) ---
    for name, pos in zip(case_names, positions):
        f = device_force_at(device, geom, pos) * ff
        out.append(f"*USE-STLD, {name}")
        out.append("")
        out.append("*CONLOAD")
        out.append("; NODE_LIST, FX, FY, FZ, MX, MY, MZ, GROUP, STRTYPENAME")
        out.append(f"   {start_node + pos}, {f[0]:.4f}, {f[1]:.4f}, "
                   f"{f[2]:.4f}, 0, 0, 0, ,")
        out.append("")

    # --- LOADCOMB: 전 케이스 포락 ---
    if make_envelope:
        out.append("*LOADCOMB")
        out.append("; NAME=NAME, KIND, ACTIVE, bES, iTYPE, DESC, iSERV-TYPE, "
                   "nLCOMTYPE, nSEISTYPE, LcomFactor ; line 1")
        out.append(";      ANAL1, LCNAME1, FACT1, ... ; from line 2")
        out.append(f"   NAME={env_name}, GEN, ACTIVE, 0, {LCOMB_ENVELOPE}, "
                   f"rail moving load envelope, 0, 0, 0, 1")
        # 라인2: 정적케이스(ST)를 계수 1로, 줄당 N개씩 분할
        terms = [f"ST, {name}, 1" for name in case_names]
        for j in range(0, len(terms), LCOMB_TERMS_PER_LINE):
            out.append("        " + ", ".join(terms[j:j + LCOMB_TERMS_PER_LINE]))
        out.append("")

    out.append("*ENDDATA")
    return "\r\n".join(out)


def write_mgt(path: str, *args, **kwargs) -> str:
    """build_mgt 결과를 파일로 저장(midas 관례에 따라 CRLF)."""
    text = build_mgt(*args, **kwargs)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    return path


def build_mgt_cases(geom: PathGeometry,
                    case_forces: dict,
                    start_node: int,
                    start_elem: int,
                    imat: int = 1,
                    ipro: int = 1,
                    lctype: str = "L",
                    env_name: str = "MV_ENV",
                    unit_force: str = "N",
                    unit_len: str = "MM",
                    angle: float = 0.0) -> str:
    """3주행 케이스 × 위치별 CONLOAD + 전체 Envelope MGT.

    case_forces: {"ACC": (N,3), "CON": (N,3), "DEC": (N,3)} 부착점 하중 [N].
    케이스명: <케이스>001… (예 ACC001, CON001, DEC001), 각 노드 1점 하중.
    모든 케이스를 단일 LOADCOMB(Envelope)로 묶는다.
    """
    fl = _LEN[unit_len.upper()]
    ff = _FORCE[unit_force.upper()]
    nodes = geom.nodes
    n = len(nodes)
    npad = max(3, len(str(n)))

    out: list[str] = []
    out.append("; midas Gen Text(MGT) - rail moving load (3 cases + envelope)")
    out.append(f"; nodes {start_node}..{start_node + n - 1}, "
               f"elems {start_elem}..{start_elem + n - 2}, "
               f"cases {', '.join(case_forces)} x {n} nodes")
    out.append("")
    out.append("*UNIT")
    out.append("; FORCE, LENGTH, HEAT, TEMPER")
    out.append(f"   {unit_force.upper()}, {unit_len.upper()}, KCAL, C")
    out.append("")

    out.append("*NODE")
    out.append("; iNO, X, Y, Z")
    for i in range(n):
        x, y, z = nodes[i] * fl
        out.append(f"   {start_node + i}, {x:.6f}, {y:.6f}, {z:.6f}")
    out.append("")

    out.append("*ELEMENT")
    out.append("; iEL, TYPE, iMAT, iPRO, iN1, iN2, ANGLE, iSUB")
    for e in range(n - 1):
        out.append(f"   {start_elem + e}, BEAM, {imat}, {ipro}, "
                   f"{start_node + e}, {start_node + e + 1}, {angle:g}, 0")
    out.append("")

    # STLDCASE: 모든 케이스 정의
    all_names = []
    out.append("*STLDCASE")
    out.append("; LCNAME, LCTYPE, DESC")
    for cname in case_forces:
        for i in range(n):
            nm = f"{cname}{i + 1:0{npad}d}"
            all_names.append(nm)
            out.append(f"   {nm}, {lctype}, {cname} at node {start_node + i}")
    out.append("")

    # 케이스별 CONLOAD
    for cname, F in case_forces.items():
        Fc = np.asarray(F) * ff
        for i in range(n):
            nm = f"{cname}{i + 1:0{npad}d}"
            fx, fy, fz = Fc[i]
            out.append(f"*USE-STLD, {nm}")
            out.append("")
            out.append("*CONLOAD")
            out.append("; NODE_LIST, FX, FY, FZ, MX, MY, MZ, GROUP, STRTYPENAME")
            out.append(f"   {start_node + i}, {fx:.4f}, {fy:.4f}, {fz:.4f}, "
                       f"0, 0, 0, ,")
            out.append("")

    # LOADCOMB Envelope: 전체 케이스
    out.append("*LOADCOMB")
    out.append("; NAME=NAME, KIND, ACTIVE, bES, iTYPE, DESC, iSERV-TYPE, "
               "nLCOMTYPE, nSEISTYPE, LcomFactor ; line 1")
    out.append(";      ANAL1, LCNAME1, FACT1, ... ; from line 2")
    out.append(f"   NAME={env_name}, GEN, ACTIVE, 0, {LCOMB_ENVELOPE}, "
               f"rail moving load 3-case envelope, 0, 0, 0, 1")
    terms = [f"ST, {nm}, 1" for nm in all_names]
    for j in range(0, len(terms), LCOMB_TERMS_PER_LINE):
        out.append("        " + ", ".join(terms[j:j + LCOMB_TERMS_PER_LINE]))
    out.append("")
    out.append("*ENDDATA")
    return "\r\n".join(out)


def write_mgt_cases(path: str, *args, **kwargs) -> str:
    text = build_mgt_cases(*args, **kwargs)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    return path
