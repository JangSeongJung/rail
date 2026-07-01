"""3케이스 동·정역학 하중 HTML 리포트 생성.

구성: ①제원 요약 ②사용 공식(HTML 텍스트) ③케이스별 노드 하중 표
      ④3D 경로 + 노드번호 + 하중 화살표(plotly, 케이스 토글)
      ⑤위치별 하중 그래프(plotly).
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .geometry import PathGeometry
from .dynamics import Pendulum
from .driving import CaseResult

_CASE_COLOR = {"ACC": "#2a7fff", "CON": "#22a06b", "DEC": "#e0563b"}
_CASE_KO = {"ACC": "최대가속", "CON": "최고등속", "DEC": "최대감속"}


def _formula_block() -> str:
    return """
<h2>사용 공식</h2>
<div class="grid2">
<div class="card fml">
<b>주행 · 정역학</b><br><br>
견인력 / 가속도 (마력 기반, 접지·안락 상한):<br>
&nbsp;&nbsp;F<sub>tr</sub> = η·P / v<sub>b</sub> &nbsp;,&nbsp;
a<sub>acc</sub> = min( F<sub>tr</sub> / (m<sub>t</sub>+m<sub>p</sub>) , a<sub>lim</sub> )<br><br>
원심(구심) 가속도 (곡률반경 R):<br>
&nbsp;&nbsp;<b>a</b><sub>n</sub> = (v² / R)·<b>n̂</b><sub>c</sub> &nbsp;,&nbsp;
<b>a</b><sub>tr</sub> = a<sub>tan</sub>·<b>t̂</b> + <b>a</b><sub>n</sub><br><br>
정적 자중:&nbsp; W = (m<sub>t</sub> + m<sub>p</sub>)·g
</div>
<div class="card fml">
<b>진자 동역학 (구면진자, 줄길이 L)</b><br><br>
유효중력:&nbsp; <b>g</b><sub>eff</sub> = −g·<b>ẑ</b> − <b>a</b><sub>tr</sub> &nbsp;(줄방향 <b>n</b>)<br><br>
운동방정식:<br>
&nbsp;&nbsp;<b>n̈</b> = (1/L)·[ <b>g</b><sub>eff</sub> − (<b>g</b><sub>eff</sub>·<b>n</b>)·<b>n</b> ]
− (<b>ṅ</b>·<b>ṅ</b>)·<b>n</b> − 2ζω<sub>n</sub>·<b>ṅ</b><br><br>
줄 장력:&nbsp; T = m<sub>p</sub>·[ (<b>g</b><sub>eff</sub>·<b>n</b>) + L·(<b>ṅ</b>·<b>ṅ</b>) ]<br><br>
부착점 하중:&nbsp; <b>F</b><sub>node</sub> = −m<sub>t</sub>·<b>a</b><sub>tr</sub> − m<sub>t</sub>·g·<b>ẑ</b> + T·<b>n</b><br><br>
동적증폭:&nbsp; DAF = T<sub>max</sub> / (m<sub>p</sub>·g) &nbsp;,&nbsp; ω<sub>n</sub> = √(g / L)
</div>
<div class="card" style="grid-column:1/-1">
<b>※ 출처 / 근거</b>
<p style="font-size:13px;margin:4px 0">위 식은 <b>고전역학 일반 원리</b>이며 특정 설계기준의 조항이 아님:</p>
<ul style="font-size:13px;margin:4px 0 6px 18px">
<li>견인력·가속도 (P=F·v, a=F/m) — 뉴턴 운동법칙</li>
<li>원심가속도 (v²/R) — 원운동 운동학</li>
<li>구면진자 운동방정식·줄장력 T — 해석역학(구면진자), 강체 줄 가정</li>
<li>동적증폭계수 DAF — 구조동역학 시간이력 해석(RK4 적분)</li>
</ul>
<p style="font-size:13px;color:#a8381f;margin:4px 0"><b>KDS 근거 없음.</b> 구조설계 적용 시 하중조합·충격계수·안전율은 적용 기준(KDS 또는 삭도/궤도·유원시설 안전기준 등)의 해당 조항을 별도 명기 요망.</p>
</div>
</div>
"""


def _table(geom, cases, start_node, m_tot, g=9.80665) -> str:
    N = geom.n_nodes
    R = geom.radius

    def rstr(i):
        return "∞ (직선)" if R[i] >= 1e8 else f"{R[i]:.2f}"

    head = ("<tr><th>노드</th><th>s [m]</th><th>R [m]</th><th>v [m/s]</th>"
            "<th>a<sub>tan</sub> [m/s²]</th><th>원심력 [N]</th>"
            "<th>Fx [N]</th><th>Fy [N]</th><th>Fz [N]</th>"
            "<th>|F| [N]</th><th>T [N]</th><th>DAF</th></tr>")
    blocks = []
    for nm in ("ACC", "CON", "DEC"):
        c = cases[nm]
        rows = []
        for i in range(N):
            v = c.v_node[i]
            fc = (m_tot * v * v / R[i]) if R[i] < 1e8 else 0.0
            fx, fy, fz = c.node_force[i]
            fmag = float(np.sqrt(fx * fx + fy * fy + fz * fz))
            rows.append(
                "<tr>"
                f"<td>{start_node + i}</td><td>{geom.s[i]:.2f}</td><td>{rstr(i)}</td>"
                f"<td>{v:.2f}</td><td>{c.a_tan[i]:.2f}</td><td>{fc:.0f}</td>"
                f"<td>{fx:.0f}</td><td>{fy:.0f}</td><td>{fz:.0f}</td>"
                f"<td><b>{fmag:.0f}</b></td><td>{c.tension[i]:.0f}</td>"
                f"<td>{c.daf[i]:.2f}</td></tr>")
        blocks.append(
            f'<h3 class="c{nm}cap">{nm} ({_CASE_KO[nm]})</h3>'
            f'<div class="tablewrap c{nm}"><table>'
            f'<thead>{head}</thead><tbody>{"".join(rows)}</tbody></table></div>')
    return ('<h2>노드별 하중 상세 (노드 '
            f'{start_node}~{start_node + N - 1})</h2>'
            '<p style="font-size:13px;color:#444">R=곡률반경(∞=직선), '
            '원심력=m·v²/R(명목 참고값), Fx·Fy·Fz=부착점 합력 3성분(자중·관성·원심·진자 장력 포함), '
            'T=줄 장력, DAF=동적증폭계수. 부호: Z 음수=아래(중력방향).</p>'
            + "".join(blocks))


def _fig3d(geom, cases, start_node) -> go.Figure:
    nd = geom.nodes
    N = geom.n_nodes
    # 표시 간격 자동: 약 12개만 보이도록 (노드 많을수록 stride 커짐)
    stride = max(1, round(N / 12))
    idx = list(range(0, N, stride))
    if idx[-1] != N - 1:
        idx.append(N - 1)        # 끝 노드는 항상 표시
    idx = np.array(idx)

    fig = go.Figure()
    # 경로 선: 전체
    fig.add_trace(go.Scatter3d(
        x=nd[:, 0], y=nd[:, 1], z=nd[:, 2], mode="lines",
        line=dict(color="#888", width=4), name="레일 경로", hoverinfo="skip"))
    # 노드 마커 + 번호: 솎은 것만
    fig.add_trace(go.Scatter3d(
        x=nd[idx, 0], y=nd[idx, 1], z=nd[idx, 2], mode="markers+text",
        marker=dict(size=4, color="#333"),
        text=[str(start_node + i) for i in idx],
        textfont=dict(size=11, color="#333"), textposition="top center",
        name="노드 번호",
        hovertext=[f"노드 {start_node + i}" for i in idx]))
    # 하중 화살표: 솎은 노드만 (스케일은 전체 하중 기준으로 통일)
    allF = np.concatenate([c.node_force for c in cases.values()])
    fmax = np.linalg.norm(allF, axis=1).max()
    span = float(np.linalg.norm(np.ptp(nd, axis=0)))
    scale = (0.16 * span) / fmax if fmax > 0 else 1.0
    for nm, c in cases.items():
        F = c.node_force[idx]
        fig.add_trace(go.Cone(
            x=nd[idx, 0], y=nd[idx, 1], z=nd[idx, 2],
            u=F[:, 0] * scale, v=F[:, 1] * scale, w=F[:, 2] * scale,
            anchor="tail", sizemode="absolute", sizeref=0.4,
            showscale=False, colorscale=[[0, _CASE_COLOR[nm]], [1, _CASE_COLOR[nm]]],
            name=f"{nm} 하중", visible=(True if nm == "DEC" else "legendonly")))
    fig.update_layout(
        scene=dict(aspectmode="data", xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        margin=dict(l=0, r=0, t=10, b=0), height=560,
        legend=dict(orientation="h", y=0))
    return fig


def _fig_graph(geom, cases) -> go.Figure:
    s = geom.s
    fig = go.Figure()
    for nm, c in cases.items():
        fmag = np.linalg.norm(c.node_force, axis=1)
        fig.add_trace(go.Scatter(x=s, y=fmag, mode="lines",
                      line=dict(color=_CASE_COLOR[nm], width=2),
                      name=f"{nm} |F|"))
    env = np.max([np.linalg.norm(c.node_force, axis=1) for c in cases.values()], axis=0)
    fig.add_trace(go.Scatter(x=s, y=env, mode="lines",
                  line=dict(color="#111", width=1, dash="dash"), name="Envelope"))
    fig.update_layout(xaxis_title="경로 위치 s [m]", yaxis_title="부착점 합력 |F| [N]",
                      height=480, margin=dict(l=60, r=20, t=20, b=40),
                      legend=dict(orientation="h", y=1.12))
    return fig


_PDF_JS = r"""
<script>
function _img(url){return new Promise(function(r){var im=new Image();im.onload=function(){r(im);};im.src=url;});}
function _page(W){var c=document.createElement('canvas');c.width=W;c.height=Math.round(W*1.41421);
  var x=c.getContext('2d');x.imageSmoothingEnabled=true;x.imageSmoothingQuality='high';
  x.fillStyle='#ffffff';x.fillRect(0,0,c.width,c.height);return c;}
function _bytes(b64){var bin=atob(b64);var n=bin.length;var a=new Uint8Array(n);for(var i=0;i<n;i++){a[i]=bin.charCodeAt(i);}return a;}
async function savePDF(){
  var btn=document.getElementById('pdfbtn'); var old=btn.textContent;
  btn.textContent='PDF 만드는 중…'; btn.disabled=true;
  try{
    var W=2480, M=70, iw=W-2*M;
    var u3=await Plotly.toImage('plot3d',{format:'png',width:1100,height:680,scale:3});
    var ug=await Plotly.toImage('graph2d',{format:'png',width:1100,height:560,scale:3});
    var im3=await _img(u3), img=await _img(ug);
    var cf=await html2canvas(document.getElementById('cap-formula'),{scale:3,backgroundColor:'#fff'});
    var p1=_page(W); var x=p1.getContext('2d');
    var titleH=92, gap=38;
    var h3=iw*im3.height/im3.width, hg=iw*img.height/img.width, hf=iw*cf.height/cf.width;
    var avail=p1.height-2*M-titleH-2*gap;
    var s=Math.min(1, avail/(h3+hg+hf));
    x.fillStyle='#16407a'; x.font='bold 50px sans-serif';
    x.fillText('레일 형태 · 하중 분포 · 사용 공식', M, M+48);
    var y=M+titleH; function _ctr(w){return M+(iw-w)/2;}
    x.drawImage(im3,_ctr(iw*s),y,iw*s,h3*s); y+=h3*s+gap;
    x.drawImage(img,_ctr(iw*s),y,iw*s,hg*s); y+=hg*s+gap;
    x.drawImage(cf,_ctr(iw*s),y,iw*s,hf*s);
    var jpgBytes=_bytes(p1.toDataURL('image/jpeg',0.95).split(',')[1]);
    var out=await PDFLib.PDFDocument.create();
    var jpg=await out.embedJpg(jpgBytes);
    var A4W=595.28, A4H=841.89;
    out.addPage([A4W,A4H]).drawImage(jpg,{x:0,y:0,width:A4W,height:A4H});
    var src=await PDFLib.PDFDocument.load(_bytes("__PDF_B64__"));
    var cp=await out.copyPages(src, src.getPageIndices());
    cp.forEach(function(p){out.addPage(p);});
    var bytes=await out.save();
    var blob=new Blob([bytes],{type:'application/pdf'});
    var a=document.createElement('a'); a.href=URL.createObjectURL(blob);
    a.download='레일_이동하중_검토.pdf'; a.click();
  }catch(e){alert('PDF 저장 오류: '+e);}
  btn.textContent=old; btn.disabled=false;
}
</script>
"""


def build_report(geom: PathGeometry, cases: dict[str, CaseResult],
                 pen: Pendulum, device, start_node: int,
                 a_brake: float, accel_limit: float, pdf_b64: str = "",
                 title: str = "레일 이동하중 검토") -> str:
    cases = {c.name: c for c in cases.values()}   # 키 대문자 정규화
    spec = (f"줄길이 L={pen.length} m · 사람 {pen.m_person} kg · 트롤리 {pen.m_trolley} kg · "
            f"감쇠 ζ={pen.damping} · 최고속도 {device.speed} m/s · "
            f"비상감속 {a_brake} m/s² · 가속상한 {accel_limit} m/s² · 진자주기 {pen.period:.2f} s")
    f3d = _fig3d(geom, cases, start_node).to_html(
        full_html=False, include_plotlyjs="cdn", div_id="plot3d",
        config={"responsive": True})
    fg = _fig_graph(geom, cases).to_html(
        full_html=False, include_plotlyjs=False, div_id="graph2d",
        config={"responsive": True})
    css = """
    body{font-family:'Malgun Gothic',sans-serif;margin:24px;color:#222;line-height:1.5;background:#ffffff}
    h1{border-bottom:3px solid #2a7fff;padding-bottom:6px;color:#000000}
    h2{margin-top:28px;color:#2a7fff}
    .spec{background:#f3f7ff;padding:10px 14px;border-radius:8px;font-size:13px}
    .grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
    .lr{display:grid;grid-template-columns:1fr 1fr;gap:16px;align-items:center}
    @media(max-width:900px){.lr{grid-template-columns:1fr}}
    .card{border:1px solid #ddd;border-radius:8px;padding:12px;background:#fafafa;font-size:14px}
    .fml{font-family:'Cambria','Times New Roman',serif;font-size:15px;line-height:1.5}
    .fml sub{font-size:71%}
    .tablewrap{max-height:420px;overflow:auto;border:1px solid #ddd;border-radius:6px}
    table{border-collapse:collapse;width:100%;font-size:16px}
    th,td{border:1px solid #e3e3e3;padding:5px 6px;text-align:center;white-space:nowrap}
    h3{margin:16px 0 4px;font-size:15px}
    .cACCcap{color:#16407a}.cCONcap{color:#155f3c}.cDECcap{color:#9c3318}
    thead th{position:sticky;top:0;background:#2a7fff;color:#fff}
    thead th.cACC{background:#cfe0ff;color:#16407a}
    thead th.cCON{background:#c9ecd9;color:#155f3c}
    thead th.cDEC{background:#f7d2c4;color:#9c3318}
    .cACC{background:#eaf2ff}.cCON{background:#eafaf2}.cDEC{background:#fdeee9}
    .toolbar{position:sticky;top:0;z-index:50;background:#fff;padding:8px 0;border-bottom:1px solid #eee}
    .btn{background:#2a7fff;color:#fff;border:0;border-radius:6px;padding:9px 16px;font-size:14px;cursor:pointer}
    .btn:hover{background:#1769e0}
    """
    if pdf_b64:
        h2c = ('<script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>'
               '<script src="https://cdnjs.cloudflare.com/ajax/libs/pdf-lib/1.17.1/pdf-lib.min.js"></script>')
        toolbar = ('<div class="toolbar">'
                   '<button id="pdfbtn" class="btn" onclick="savePDF()">📄 PDF로 저장</button>'
                   '<span style="font-size:12px;color:#666;margin-left:10px">'
                   '3D·그래프·공식을 앞장에, 결과표를 뒷장에 붙여 PDF 한 개로 저장합니다.</span></div>')
        js = _PDF_JS.replace("__PDF_B64__", pdf_b64)
    else:
        h2c = ""
        toolbar = ""
        js = ""
    return f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<title>{title}</title>{h2c}<style>{css}</style></head><body>
{toolbar}
<h1>{title}</h1>
<div class="lr">
<div>
<h2>레일 형태 및 노드번호</h2>
{f3d}
</div>
<div>
<h2>위치별 하중 분포</h2>
{fg}
</div>
</div>
<div id="cap-formula">
<div class="spec">{spec}</div>
{_formula_block()}
</div>
<div id="cap-table">
{_table(geom, cases, start_node, pen.m_person + pen.m_trolley)}
</div>
{js}
</body></html>"""
