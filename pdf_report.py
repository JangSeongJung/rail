"""해석 결과 PDF 리포트 생성 (reportlab).

한글은 reportlab 내장 CID 폰트(HYGothic-Medium)를 써서 TTF 파일 없이 출력한다.
→ Streamlit Cloud 등 어떤 환경에서도 폰트 설치 없이 그대로 동작.

기존 계산 모듈은 건드리지 않고, 결과 객체(cases, geom)만 받아 표로 정리한다.
"""
from __future__ import annotations

import io
import os

import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle,
                                Paragraph, Spacer)

# 한글 폰트: 실제 TTF를 PDF에 임베드 → 뷰어 종류와 무관하게 글자가 보인다.
# NanumGothic-Regular.ttf 파일이 이 모듈과 같은 폴더(rail_analyzer/)에 있어야 함.
_FONT = "NanumGothic"
_FONT_PATH = os.path.join(os.path.dirname(__file__), "NanumGothic-Regular.ttf")
pdfmetrics.registerFont(TTFont(_FONT, _FONT_PATH))
pdfmetrics.registerFontFamily(_FONT, normal=_FONT, bold=_FONT,
                              italic=_FONT, boldItalic=_FONT)

CASE_LABEL = {"acc": "가속(ACC)", "con": "등속(CON)", "dec": "비상감속(DEC)"}


def _styles():
    ss = getSampleStyleSheet()
    title = ParagraphStyle("t", parent=ss["Title"], fontName=_FONT, fontSize=16)
    h = ParagraphStyle("h", parent=ss["Heading2"], fontName=_FONT, fontSize=11,
                       spaceBefore=8, spaceAfter=4)
    body = ParagraphStyle("b", parent=ss["Normal"], fontName=_FONT, fontSize=9)
    return title, h, body


def _table(data, col_widths=None, header=True):
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    style = [
        ("FONTNAME", (0, 0), (-1, -1), _FONT),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#888888")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f2f4f7")]),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ]
    t.setStyle(TableStyle(style))
    return t


def build_pdf(geom, cases: dict, inputs: dict) -> bytes:
    """해석 결과를 A4 PDF(bytes)로 만든다.

    geom   : PathGeometry (nodes, n_nodes, length)
    cases  : {name: CaseResult(name, node_force(N,3), tension, daf, ...)}
    inputs : 화면 입력값 요약 dict
    """
    title, h, body = _styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=15 * mm, rightMargin=15 * mm,
                            topMargin=15 * mm, bottomMargin=15 * mm)
    el = []

    el.append(Paragraph("레일 이동하중 검토 결과", title))
    el.append(Spacer(1, 6))

    # ── 입력 요약 ──
    el.append(Paragraph("1. 입력 요약", h))
    info = [
        ["항목", "값", "항목", "값"],
        ["사람(운전자) 무게", f"{inputs['m_person']:.1f} kg",
         "트롤리 무게", f"{inputs['m_trolley']:.1f} kg"],
        ["거리(줄 길이)", f"{inputs['L']:.2f} m",
         "최고속도", inputs["speed_disp"]],
        ["추진출력", inputs["power_disp"],
         "MGT 시작노드", str(inputs["start_node"])],
        ["경로 길이", f"{geom.length:.2f} m",
         "노드 수", f"{geom.n_nodes} 개"],
    ]
    el.append(_table(info, col_widths=[38 * mm, 42 * mm, 38 * mm, 42 * mm]))
    el.append(Spacer(1, 8))

    # ── 케이스별 최대값 ──
    el.append(Paragraph("2. 케이스별 최대값", h))
    head = ["케이스", "최대 |F| [N]", "발생 노드", "최대 장력 [N]", "최대 DAF"]
    rows = [head]
    env_mag = None
    for name, c in cases.items():
        mag = np.linalg.norm(c.node_force, axis=1)
        imax = int(mag.argmax())
        rows.append([
            CASE_LABEL.get(name, name),
            f"{mag.max():,.0f}",
            str(inputs["start_node"] + imax),
            f"{c.tension.max():,.0f}",
            f"{c.daf.max():.2f}",
        ])
        env_mag = mag if env_mag is None else np.maximum(env_mag, mag)
    el.append(_table(rows, col_widths=[34 * mm, 32 * mm, 28 * mm, 34 * mm, 24 * mm]))
    el.append(Spacer(1, 4))
    el.append(Paragraph(
        f"<b>Envelope 최대 |F| = {env_mag.max():,.0f} N</b> "
        f"(노드 {inputs['start_node'] + int(env_mag.argmax())})", body))
    el.append(Spacer(1, 8))

    # ── Envelope 노드별 하중표 ──
    el.append(Paragraph("3. Envelope 노드별 하중 (전 케이스 최대 절대성분)", h))
    # 각 노드에서 케이스별 |Fx|,|Fy|,|Fz| 최대를 취함
    fx = np.max([np.abs(c.node_force[:, 0]) for c in cases.values()], axis=0)
    fy = np.max([np.abs(c.node_force[:, 1]) for c in cases.values()], axis=0)
    fz = np.max([np.abs(c.node_force[:, 2]) for c in cases.values()], axis=0)
    nodes = geom.nodes
    thead = ["노드", "X [m]", "Y [m]", "Z [m]", "|Fx| [N]", "|Fy| [N]", "|Fz| [N]"]
    nrows = [thead]
    for i in range(geom.n_nodes):
        nrows.append([
            str(inputs["start_node"] + i),
            f"{nodes[i, 0]:.2f}", f"{nodes[i, 1]:.2f}", f"{nodes[i, 2]:.2f}",
            f"{fx[i]:,.0f}", f"{fy[i]:,.0f}", f"{fz[i]:,.0f}",
        ])
    el.append(_table(nrows, col_widths=[18 * mm, 22 * mm, 22 * mm, 22 * mm,
                                        24 * mm, 24 * mm, 24 * mm]))

    doc.build(el)
    return buf.getvalue()
