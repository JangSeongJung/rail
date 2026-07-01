"""레일 이동하중 검토 - 웹(Streamlit) 버전.

레이아웃:
  ┌ col1 ────────┬ col2 ───┬ col3 ───┬ col4 ────┐
  │ 1. DXF        │ 2. 기본 │ 3. MGT  │ 4. 고급  │
  │ [해석 실행]   │   제원  │  시작   │   설정   │
  │ 결과(진행률   │ (좌우로 │  번호   │ (좌우로  │
  │  →MGT·PDF)    │  좁게)  │         │  좁게)   │
  └──────────────┴─────────┴─────────┴──────────┘
  (그 아래) 해석 결과 미리보기

계산 코드(solver/dynamics/driving/report/mgt_export)는 그대로 호출하고,
PDF는 rail_analyzer/pdf_report.py 로 새로 생성한다.
"""
from __future__ import annotations

import base64
import os
import tempfile
import traceback

import numpy as np
import streamlit as st

from rail_analyzer.dxf_parser import parse_dxf
from rail_analyzer.geometry import build_geometry
from rail_analyzer.dynamics import Pendulum
from rail_analyzer.loads import MovingDevice, POWER_UNITS
from rail_analyzer.driving import run_all_cases
from rail_analyzer.report import build_report
from rail_analyzer.mgt_export import write_mgt_cases
from rail_analyzer.pdf_report import build_pdf

SPEED_UNITS = {"m/s": 1.0, "km/h": 1.0 / 3.6}

st.set_page_config(page_title="레일 이동하중 검토", page_icon="🛤️", layout="wide")
st.title("🛤️ 레일 이동하중 검토 (동·정역학)")
st.caption("DXF 경로를 올리고 제원을 입력하면 ACC/CON/DEC 3케이스 하중 리포트와 Midas Gen MGT를 생성합니다.")

col1, col2, col3, col4 = st.columns([1.1, 1.1, 0.85, 1.35], gap="medium")

# ── col1 : DXF · 해석실행 버튼 · 결과 ──
with col1:
    st.markdown("**1. DXF 경로 파일**")
    dxf_file = st.file_uploader("DXF 파일", type=["dxf"], label_visibility="collapsed")
    run = st.button("해석 실행", type="primary", use_container_width=True)
    st.markdown("**결과**")
    result_slot = st.empty()   # 진행률 게이지 → (끝나면) MGT·PDF 다운 버튼

# ── col2 : 2. 기본 제원 (세로로 좁게) ──
with col2:
    st.markdown("**2. 기본 제원**")
    m_person = st.number_input("사람(운전자) 무게 [kg]", value=150.0, step=1.0)
    m_trolley = st.number_input("트롤리(동력체) 무게 [kg]", value=30.0, step=1.0)
    L = st.number_input("거리(줄 길이) [m]", value=1.5, step=0.1, format="%.2f")
    s1, s2 = st.columns([3, 2])
    speed = s1.number_input("최고속도", value=2.0, step=0.1, format="%.2f")
    speed_unit = s2.selectbox("단위", ["m/s", "km/h"], key="speed_unit")
    p1, p2 = st.columns([3, 2])
    power = p1.number_input("추진출력", value=10.0, step=1.0)
    power_unit = p2.selectbox("단위", ["kw", "w", "ps", "hp"], key="power_unit")

# ── col3 : 3. MGT 시작 번호 (세로로 좁게) ──
with col3:
    st.markdown("**3. MGT 시작 번호**")
    start_node = st.number_input("시작 노드번호", value=1001, step=1)
    start_elem = st.number_input("시작 요소번호", value=5001, step=1)
    imat = st.number_input("재질번호 iMAT", value=1, step=1)
    ipro = st.number_input("단면번호 iPRO", value=1, step=1)

# ── col4 : 4. 고급 설정 (세로로 좁게) ──
with col4:
    st.markdown("**4. 고급 설정 (기본값 사용 가능)**")
    g = st.columns([3, 2])
    base_speed = g[0].number_input("기저속도", value=0.6, step=0.1, format="%.2f")
    base_speed_unit = g[1].selectbox("단위", ["m/s", "km/h"], key="base_unit")
    g = st.columns(2)
    eff = g[0].number_input("구동효율 η", value=0.85, step=0.01, format="%.2f")
    brake = g[1].number_input("비상감속 [m/s²]", value=4.0, step=0.1, format="%.2f")
    g = st.columns(2)
    accel_limit = g[0].number_input("가속 상한 [m/s²]", value=2.0, step=0.1, format="%.2f")
    damping = g[1].number_input("진자 감쇠비 ζ", value=0.03, step=0.01, format="%.2f")
    g = st.columns(2)
    seg = g[0].number_input("요소분할 [m]", value=0.3, step=0.05, format="%.2f")

# ── 결과 미리보기 자리 (빨간 네모 아래, 전체 폭) ──
preview_slot = st.container()

# ─────────────────────────── 실행 ───────────────────────────
if run:
    if dxf_file is None:
        result_slot.warning("먼저 DXF 파일을 올려주세요.")
        st.stop()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".dxf")
    tmp.write(dxf_file.getvalue())
    tmp.close()
    dxf_path = tmp.name

    prog = result_slot.progress(0, text="해석 준비 중…")
    try:
        data = parse_dxf(dxf_path, None, "SUPPORT")
        geom = build_geometry(data.path, seg)
        pen = Pendulum(m_person, m_trolley, L, damping)

        speed_ms = speed * SPEED_UNITS[speed_unit]
        base_ms = base_speed * SPEED_UNITS[base_speed_unit]
        power_w = power * POWER_UNITS[power_unit]
        device = MovingDevice(mass=m_person + m_trolley, speed=speed_ms,
                              power=power_w, base_speed=base_ms, efficiency=eff)

        total = 3 * geom.n_nodes
        counter = {"n": 0}

        def on_step():
            counter["n"] += 1
            prog.progress(min(counter["n"] / total, 1.0),
                          text=f"진자 동역학 적분 중…  {counter['n']}/{total}")

        cases = run_all_cases(geom, pen, device, a_brake=brake,
                              accel_limit=accel_limit, progress=on_step)

        # HTML(미리보기) 안에 표 PDF를 심어, 미리보기의 'PDF로 저장' 버튼이
        # [화면 이미지 + 표 PDF]를 한 개로 합쳐 내려받게 한다.
        mgt_path = os.path.join(tempfile.gettempdir(), "rail_3case.mgt")
        case_forces = {c.name: c.node_force for c in cases.values()}
        write_mgt_cases(mgt_path, geom, case_forces,
                        start_node=int(start_node), start_elem=int(start_elem),
                        imat=int(imat), ipro=int(ipro))
        with open(mgt_path, "r", encoding="utf-8") as f:
            mgt_text = f.read()

        pdf_inputs = dict(
            m_person=m_person, m_trolley=m_trolley, L=L,
            speed_disp=f"{speed:.2f} {speed_unit}",
            power_disp=f"{power:.1f} {power_unit}",
            start_node=int(start_node))
        pdf_bytes = build_pdf(geom, cases, pdf_inputs)
        pdf_b64 = base64.b64encode(pdf_bytes).decode("ascii")

        html = build_report(geom, cases, pen, device, int(start_node),
                            brake, accel_limit, pdf_b64=pdf_b64)

        env = max(np.linalg.norm(c.node_force, axis=1).max() for c in cases.values())

        st.session_state["result"] = {
            "html": html, "mgt": mgt_text,
            "n_nodes": geom.n_nodes, "speed_ms": speed_ms, "env": env,
            "stem": os.path.splitext(dxf_file.name)[0],
        }
    except Exception as e:
        result_slot.error(f"실행 오류: {e}")
        with preview_slot:
            with st.expander("상세 오류 보기"):
                st.code(traceback.format_exc())
        st.stop()
    finally:
        try:
            os.unlink(dxf_path)
        except OSError:
            pass

# ─────────────── 결과 표시 (진행률 자리에 MGT·PDF 버튼만) ───────────────
res = st.session_state.get("result")
if res:
    box = result_slot.container()
    box.download_button("⬇️ MGT 다운", data=res["mgt"],
                        file_name=f"{res['stem']}_3case.mgt", mime="text/plain",
                        use_container_width=True)
    box.caption("PDF는 아래 미리보기의 '📄 PDF로 저장' 버튼")

    with preview_slot:
        st.markdown("---")
        st.success(
            f"완료!  노드 {res['n_nodes']}개 · 속도 {res['speed_ms']:.2f} m/s · "
            f"Envelope 최대 |F| {res['env']:,.0f} N")
        st.subheader("해석 결과 미리보기")
        st.components.v1.html(res["html"], height=900, scrolling=True)
