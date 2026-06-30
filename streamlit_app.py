"""레일 이동하중 검토 - 웹(Streamlit) 버전.

기존 데스크톱 GUI(rail_analyzer/report_gui.py)의 '껍데기'만 웹으로 바꾼 것.
계산 코드(solver/dynamics/driving/report/mgt_export)는 한 줄도 건드리지 않고
그대로 호출한다.

배포:
    1) 이 파일(streamlit_app.py)과 requirements.txt를 저장소 루트에 둔다
       (rail_analyzer/ 폴더, app.py 와 같은 위치)
    2) GitHub에 push
    3) share.streamlit.io 에서 이 저장소 + streamlit_app.py 선택해 Deploy
로컬 실행:
    pip install -r requirements.txt
    streamlit run streamlit_app.py
"""
from __future__ import annotations

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

# 속도 단위 → m/s  (report_gui.py 와 동일)
SPEED_UNITS = {"m/s": 1.0, "km/h": 1.0 / 3.6}

st.set_page_config(page_title="레일 이동하중 검토", page_icon="🛤️", layout="wide")
st.title("🛤️ 레일 이동하중 검토 (동·정역학)")
st.caption(
    "DXF 경로를 올리고 제원을 입력하면 ACC/CON/DEC 3케이스 하중 리포트(HTML)와 "
    "Midas Gen MGT를 생성합니다."
)

# ─────────────────────────── 입력 폼 ───────────────────────────
# 폼으로 묶으면 '생성' 버튼을 누를 때만 한 번에 실행된다(중간 입력마다 재실행 X).
with st.form("input_form"):
    dxf_file = st.file_uploader("1. DXF 경로 파일", type=["dxf"])

    st.markdown("**2. 기본 제원**")
    c1, c2, c3 = st.columns(3)
    m_person = c1.number_input("사람(운전자) 무게 [kg]", value=150.0, step=1.0)
    m_trolley = c2.number_input("트롤리(동력체) 무게 [kg]", value=30.0, step=1.0)
    L = c3.number_input("거리(줄 길이) [m]", value=1.5, step=0.1, format="%.2f")

    c4, c5 = st.columns(2)
    with c4:
        sc1, sc2 = st.columns([2, 1])
        speed = sc1.number_input("최고속도", value=2.0, step=0.1, format="%.2f")
        speed_unit = sc2.selectbox("단위", ["m/s", "km/h"], key="speed_unit")
    with c5:
        pc1, pc2 = st.columns([2, 1])
        power = pc1.number_input("추진출력", value=10.0, step=1.0)
        power_unit = pc2.selectbox("단위", ["kw", "w", "ps", "hp"], key="power_unit")

    st.markdown("**3. MGT 시작 번호**")
    m1, m2, m3, m4 = st.columns(4)
    start_node = m1.number_input("시작 노드번호", value=1001, step=1)
    start_elem = m2.number_input("시작 요소번호", value=5001, step=1)
    imat = m3.number_input("재질번호 iMAT", value=1, step=1)
    ipro = m4.number_input("단면번호 iPRO", value=1, step=1)

    with st.expander("4. 고급 설정 (기본값 사용 가능)"):
        a1, a2, a3 = st.columns(3)
        with a1:
            bc1, bc2 = st.columns([2, 1])
            base_speed = bc1.number_input("기저속도", value=0.6, step=0.1, format="%.2f")
            base_speed_unit = bc2.selectbox("단위", ["m/s", "km/h"], key="base_unit")
        eff = a2.number_input("구동효율 η", value=0.85, step=0.01, format="%.2f")
        brake = a3.number_input("비상감속 [m/s²]", value=4.0, step=0.1, format="%.2f")
        b1, b2, b3 = st.columns(3)
        accel_limit = b1.number_input("가속 상한 [m/s²]", value=2.0, step=0.1, format="%.2f")
        damping = b2.number_input("진자 감쇠비 ζ", value=0.03, step=0.01, format="%.2f")
        seg = b3.number_input("요소분할 [m]", value=0.3, step=0.05, format="%.2f")

    submitted = st.form_submit_button("리포트 + MGT 생성", type="primary", use_container_width=True)

# ─────────────────────────── 실행 ───────────────────────────
if submitted:
    if dxf_file is None:
        st.warning("먼저 DXF 파일을 올려주세요.")
        st.stop()

    # 업로드된 파일은 메모리에 있으므로, ezdxf가 읽을 수 있게 임시 파일로 저장
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".dxf")
    tmp.write(dxf_file.getvalue())
    tmp.close()
    dxf_path = tmp.name

    progress = st.progress(0, text="해석 준비 중…")
    try:
        data = parse_dxf(dxf_path, None, "SUPPORT")
        geom = build_geometry(data.path, seg)
        pen = Pendulum(m_person, m_trolley, L, damping)

        speed_ms = speed * SPEED_UNITS[speed_unit]
        base_ms = base_speed * SPEED_UNITS[base_speed_unit]
        power_w = power * POWER_UNITS[power_unit]
        device = MovingDevice(
            mass=m_person + m_trolley, speed=speed_ms, power=power_w,
            base_speed=base_ms, efficiency=eff)

        # 진행률: 3케이스 × 노드 수 (report_gui와 동일)
        total = 3 * geom.n_nodes
        counter = {"n": 0}

        def on_step():
            counter["n"] += 1
            progress.progress(min(counter["n"] / total, 1.0),
                              text=f"진자 동역학 적분 중…  {counter['n']}/{total}")

        cases = run_all_cases(geom, pen, device, a_brake=brake,
                              accel_limit=accel_limit, progress=on_step)

        html = build_report(geom, cases, pen, device, int(start_node), brake, accel_limit)

        mgt_path = os.path.join(tempfile.gettempdir(), "rail_3case.mgt")
        case_forces = {c.name: c.node_force for c in cases.values()}
        write_mgt_cases(mgt_path, geom, case_forces,
                        start_node=int(start_node), start_elem=int(start_elem),
                        imat=int(imat), ipro=int(ipro))
        with open(mgt_path, "r", encoding="utf-8") as f:
            mgt_text = f.read()

        env = max(np.linalg.norm(c.node_force, axis=1).max() for c in cases.values())

        progress.progress(1.0, text="완료")

        # 결과를 세션에 저장 → 다운로드 버튼을 눌러도 다시 계산하지 않음
        st.session_state["result"] = {
            "html": html,
            "mgt": mgt_text,
            "n_nodes": geom.n_nodes,
            "speed_ms": speed_ms,
            "env": env,
            "stem": os.path.splitext(dxf_file.name)[0],
        }
    except Exception as e:
        progress.empty()
        st.error(f"실행 오류: {e}")
        with st.expander("상세 오류 보기"):
            st.code(traceback.format_exc())
    finally:
        try:
            os.unlink(dxf_path)
        except OSError:
            pass

# ─────────────────────────── 결과 표시 ───────────────────────────
res = st.session_state.get("result")
if res:
    st.success(
        f"완료!  노드 {res['n_nodes']}개 · 속도 {res['speed_ms']:.2f} m/s · "
        f"Envelope 최대 |F| {res['env']:,.0f} N"
    )
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ 리포트 HTML 내려받기", data=res["html"],
                       file_name=f"{res['stem']}_report.html",
                       mime="text/html", use_container_width=True)
    d2.download_button("⬇️ MGT 내려받기", data=res["mgt"],
                       file_name=f"{res['stem']}_3case.mgt",
                       mime="text/plain", use_container_width=True)

    st.markdown("---")
    st.subheader("리포트 미리보기")
    st.components.v1.html(res["html"], height=900, scrolling=True)
