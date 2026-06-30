"""레일 이동하중 리포트 GUI.

DXF를 고르고 사람/트롤리 무게·거리(줄 길이)·최고속도(m/s 또는 km/h)·
출력(W/kW/PS/HP)을 입력받아 동·정역학 3케이스 하중 리포트(HTML)와 MGT를
생성한다. MGT 시작 노드/요소 번호도 입력받는다.
"""
from __future__ import annotations

import os
import threading
import traceback
import webbrowser

import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from .dxf_parser import parse_dxf
from .geometry import build_geometry
from .dynamics import Pendulum
from .loads import MovingDevice, POWER_UNITS
from .driving import run_all_cases
from .report import build_report
from .mgt_export import write_mgt_cases

# 속도 단위 → m/s
SPEED_UNITS = {"m/s": 1.0, "km/h": 1.0 / 3.6}


class ReportApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("레일 이동하중 검토 (동·정역학)")
        root.geometry("450x680")
        self.vars: dict[str, tk.StringVar] = {}
        pad = dict(padx=8, pady=3)

        # 1. DXF 선택
        f_dxf = ttk.LabelFrame(root, text="1. DXF 경로 파일", padding=6)
        f_dxf.pack(fill="x", **pad)
        self.dxf_path = tk.StringVar(value="(선택하세요)")
        ttk.Entry(f_dxf, textvariable=self.dxf_path, state="readonly").pack(
            side="left", fill="x", expand=True, padx=(0, 6))
        ttk.Button(f_dxf, text="찾아보기…", command=self._browse).pack(side="left")

        # 2. 기본 제원
        f_main = ttk.LabelFrame(root, text="2. 기본 제원", padding=6)
        f_main.pack(fill="x", **pad)
        self._row(f_main, 0, "사람(운전자) 무게 [kg]", "m_person", "150")
        self._row(f_main, 1, "트롤리(동력체) 무게 [kg]", "m_trolley", "30")
        self._row(f_main, 2, "거리(줄 길이) [m]", "L", "1.5")
        self._row_unit(f_main, 3, "최고속도", "speed", "2.0",
                       "speed_unit", ["m/s", "km/h"], "m/s")
        self._row_unit(f_main, 4, "추진출력", "power", "10",
                       "power_unit", ["kw", "w", "ps", "hp"], "kw")

        # 3. MGT 시작 번호
        f_mgt = ttk.LabelFrame(root, text="3. MGT 시작 번호", padding=6)
        f_mgt.pack(fill="x", **pad)
        self._row(f_mgt, 0, "시작 노드번호", "start_node", "1001")
        self._row(f_mgt, 1, "시작 요소번호", "start_elem", "5001")
        self._row(f_mgt, 2, "재질번호 iMAT", "imat", "1")
        self._row(f_mgt, 3, "단면번호 iPRO", "ipro", "1")

        # 4. 고급 설정
        f_adv = ttk.LabelFrame(root, text="4. 고급 설정 (기본값 사용 가능)", padding=6)
        f_adv.pack(fill="x", **pad)
        self._row_unit(f_adv, 0, "기저속도", "base_speed", "0.6",
                       "base_speed_unit", ["m/s", "km/h"], "m/s")
        self._row(f_adv, 1, "구동효율 η", "eff", "0.85")
        self._row(f_adv, 2, "비상감속 [m/s²]", "brake", "4.0")
        self._row(f_adv, 3, "가속 상한 [m/s²]", "accel_limit", "2.0")
        self._row(f_adv, 4, "진자 감쇠비 ζ", "damping", "0.03")
        self._row(f_adv, 5, "요소분할 [m]", "seg", "0.3")

        ttk.Button(root, text="리포트 + MGT 생성", command=self._run).pack(
            fill="x", padx=8, pady=(8, 4))
        self.progress = ttk.Progressbar(root, mode="determinate", maximum=100)
        self.progress.pack(fill="x", padx=8, pady=(0, 4))
        self.status = tk.StringVar(value="대기 중")
        ttk.Label(root, textvariable=self.status, foreground="#2a7fff",
                  wraplength=430, justify="left").pack(fill="x", padx=10)

    def _row(self, parent, r, label, key, default):
        ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w")
        self.vars[key] = tk.StringVar(value=default)
        ttk.Entry(parent, textvariable=self.vars[key], width=14).grid(
            row=r, column=1, sticky="we", pady=2)
        parent.columnconfigure(1, weight=1)

    def _row_unit(self, parent, r, label, key, default, ukey, uvals, udef):
        """값 입력 + 단위 콤보 한 행."""
        ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w")
        cell = ttk.Frame(parent)
        cell.grid(row=r, column=1, sticky="we", pady=2)
        self.vars[key] = tk.StringVar(value=default)
        ttk.Entry(cell, textvariable=self.vars[key], width=9).pack(side="left")
        self.vars[ukey] = tk.StringVar(value=udef)
        ttk.Combobox(cell, textvariable=self.vars[ukey], values=uvals,
                     width=6, state="readonly").pack(side="left", padx=4)
        parent.columnconfigure(1, weight=1)

    def _browse(self):
        p = filedialog.askopenfilename(
            title="DXF 경로 파일 선택",
            filetypes=[("DXF 파일", "*.dxf"), ("모든 파일", "*.*")])
        if p:
            self.dxf_path.set(p)

    def _f(self, key):
        return float(self.vars[key].get())

    def _i(self, key):
        return int(float(self.vars[key].get()))

    def _run(self):
        dxf = self.dxf_path.get()
        if not os.path.isfile(dxf):
            messagebox.showwarning("DXF 없음", "먼저 DXF 파일을 선택하세요.")
            return
        self.status.set("해석 중… (진자 동역학 적분에 잠시 걸립니다)")
        threading.Thread(target=self._work, args=(dxf,), daemon=True).start()

    def _work(self, dxf):
        try:
            seg = self._f("seg")
            data = parse_dxf(dxf, None, "SUPPORT")
            geom = build_geometry(data.path, seg)
            pen = Pendulum(self._f("m_person"), self._f("m_trolley"),
                           self._f("L"), self._f("damping"))
            speed_ms = self._f("speed") * SPEED_UNITS[self.vars["speed_unit"].get()]
            base_ms = self._f("base_speed") * SPEED_UNITS[self.vars["base_speed_unit"].get()]
            power_w = self._f("power") * POWER_UNITS[self.vars["power_unit"].get()]
            device = MovingDevice(
                mass=self._f("m_person") + self._f("m_trolley"),
                speed=speed_ms, power=power_w,
                base_speed=base_ms, efficiency=self._f("eff"))
            brake = self._f("brake")
            alim = self._f("accel_limit")

            # 진행률: 3케이스 × 노드 수
            total = 3 * geom.n_nodes
            self._pcount = 0
            self.root.after(0, lambda: self.progress.config(maximum=total, value=0))

            def on_step():
                self._pcount += 1
                c = self._pcount
                self.root.after(0, lambda: self.progress.config(value=c))

            cases = run_all_cases(geom, pen, device, a_brake=brake,
                                  accel_limit=alim, progress=on_step)

            base = os.path.splitext(dxf)[0]
            html_path = base + "_report.html"
            mgt_path = base + "_3case.mgt"
            start_node = self._i("start_node")

            html = build_report(geom, cases, pen, device, start_node, brake, alim)
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html)
            case_forces = {c.name: c.node_force for c in cases.values()}
            write_mgt_cases(mgt_path, geom, case_forces,
                            start_node=start_node, start_elem=self._i("start_elem"),
                            imat=self._i("imat"), ipro=self._i("ipro"))

            env = max(np.linalg.norm(c.node_force, axis=1).max() for c in cases.values())
            msg = (f"완료!  노드 {geom.n_nodes}개 (속도 {speed_ms:.2f} m/s), "
                   f"Envelope 최대 |F| {env:.0f} N\n"
                   f"리포트: {os.path.basename(html_path)}\n"
                   f"MGT: {os.path.basename(mgt_path)} (3×{geom.n_nodes} 케이스)")
            self.root.after(0, lambda: self._done(msg, html_path))
        except Exception as e:
            tb = traceback.format_exc()
            self.root.after(0, lambda: self._fail(e, tb))

    def _done(self, msg, html_path):
        self.progress.config(value=self.progress["maximum"])
        self.status.set(msg)
        webbrowser.open("file://" + os.path.abspath(html_path))

    def _fail(self, e, tb):
        self.status.set(f"오류: {e}")
        messagebox.showerror("실행 오류", tb)


def main():
    root = tk.Tk()
    ReportApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
