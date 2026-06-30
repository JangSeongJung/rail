"""tkinter GUI: 입력 + 실행 + matplotlib 3D 시각화.

단위(입력 편의):
    E [GPa], A [cm^2], I/J [cm^4], 질량 [kg], 속도 [m/s], 가속 [m/s^2]
내부 계산은 SI(N, m, Pa)로 변환하여 수행하고 결과는 kN, kN·m로 표시한다.
"""
from __future__ import annotations

import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)

from .dxf_parser import parse_dxf, list_layers
from .geometry import build_geometry
from .model import Section, build_model
from .loads import MovingDevice, POWER_UNITS
from .analysis import run_analysis, EnvelopeResult


# 단위 환산 계수 → SI
GPA = 1e9
CM2 = 1e-4
CM4 = 1e-8


class RailApp(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=8)
        self.pack(fill="both", expand=True)
        self.result: EnvelopeResult | None = None
        self.geom = None
        self._build_ui()

    # ---------------- UI ----------------
    def _build_ui(self):
        left = ttk.Frame(self)
        left.pack(side="left", fill="y", padx=(0, 8))
        right = ttk.Frame(self)
        right.pack(side="right", fill="both", expand=True)

        self.vars: dict[str, tk.Variable] = {}

        # 파일
        f_file = ttk.LabelFrame(left, text="DXF 입력", padding=6)
        f_file.pack(fill="x", pady=4)
        self.vars["dxf"] = tk.StringVar()
        ttk.Entry(f_file, textvariable=self.vars["dxf"], width=30).grid(
            row=0, column=0, columnspan=2, sticky="we")
        ttk.Button(f_file, text="찾아보기", command=self._browse).grid(
            row=0, column=2, padx=2)
        self.vars["path_layer"] = tk.StringVar()
        self.vars["support_layer"] = tk.StringVar(value="SUPPORT")
        ttk.Label(f_file, text="경로 레이어").grid(row=1, column=0, sticky="w")
        self.cb_path = ttk.Combobox(f_file, textvariable=self.vars["path_layer"], width=14)
        self.cb_path.grid(row=1, column=1, columnspan=2, sticky="we", pady=1)
        ttk.Label(f_file, text="지지 레이어").grid(row=2, column=0, sticky="w")
        self.cb_sup = ttk.Combobox(f_file, textvariable=self.vars["support_layer"], width=14)
        self.cb_sup.grid(row=2, column=1, columnspan=2, sticky="we", pady=1)

        # 단면/재료
        f_sec = ttk.LabelFrame(left, text="레일 단면/재료", padding=6)
        f_sec.pack(fill="x", pady=4)
        self._row(f_sec, 0, "E [GPa]", "E", "200")
        self._row(f_sec, 1, "A [cm²]", "A", "60")
        self._row(f_sec, 2, "Iy(약축) [cm⁴]", "Iy", "1000")
        self._row(f_sec, 3, "Iz(강축) [cm⁴]", "Iz", "3000")
        self._row(f_sec, 4, "J [cm⁴]", "J", "50")
        self._row(f_sec, 5, "밀도 [kg/m³]", "density", "7850")

        # 기기/추진
        f_dev = ttk.LabelFrame(left, text="이동 기기 / 추진", padding=6)
        f_dev.pack(fill="x", pady=4)
        self._row(f_dev, 0, "질량 [kg]", "mass", "2000")
        self._row(f_dev, 1, "최고속도 [m/s]", "speed", "2.0")
        self._row(f_dev, 2, "추진출력", "power", "10")
        ttk.Label(f_dev, text="출력 단위").grid(row=3, column=0, sticky="w")
        self.vars["power_unit"] = tk.StringVar(value="kw")
        ttk.Combobox(f_dev, textvariable=self.vars["power_unit"],
                     values=["kw", "ps", "hp"], width=10,
                     state="readonly").grid(row=3, column=1, sticky="we")
        self._row(f_dev, 4, "기저속도 [m/s]", "base_speed", "0.6")
        self._row(f_dev, 5, "구동효율 η", "eff", "0.9")
        self._row(f_dev, 6, "충격계수 φ", "impact", "0.3")
        ttk.Label(f_dev, text="(기저속도 비우면 0.3×최고속도)",
                  foreground="#666").grid(row=7, column=0, columnspan=2,
                                          sticky="w", pady=(2, 0))

        # 해석 옵션
        f_opt = ttk.LabelFrame(left, text="해석 옵션", padding=6)
        f_opt.pack(fill="x", pady=4)
        self._row(f_opt, 0, "요소분할 [m]", "seg", "0.3")
        ttk.Label(f_opt, text="지지조건").grid(row=1, column=0, sticky="w")
        self.vars["support_type"] = tk.StringVar(value="pinned")
        ttk.Combobox(f_opt, textvariable=self.vars["support_type"],
                     values=["pinned", "vertical", "fixed"], width=12,
                     state="readonly").grid(row=1, column=1, sticky="we")
        self.vars["selfweight"] = tk.BooleanVar(value=False)
        ttk.Checkbutton(f_opt, text="레일 자중 포함",
                        variable=self.vars["selfweight"]).grid(
            row=2, column=0, columnspan=2, sticky="w")

        ttk.Button(left, text="해석 실행", command=self._run).pack(
            fill="x", pady=6)

        # 결과 텍스트
        self.txt = tk.Text(left, width=40, height=14, font=("Consolas", 9))
        self.txt.pack(fill="both", expand=True)

        # 그래프
        self.fig = Figure(figsize=(7, 6), dpi=100)
        self.ax3d = self.fig.add_subplot(2, 1, 1, projection="3d")
        self.ax2d = self.fig.add_subplot(2, 1, 2)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas, right)
        self._init_plots()

    def _row(self, parent, r, label, key, default):
        ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w")
        self.vars[key] = tk.StringVar(value=default)
        ttk.Entry(parent, textvariable=self.vars[key], width=12).grid(
            row=r, column=1, sticky="we", pady=1)

    def _init_plots(self):
        self.ax3d.set_title("레일 경로 / 지지점 반력")
        self.ax2d.set_title("지배 지지점 반력 영향선")
        self.ax2d.set_xlabel("하중 위치 s [m]")
        self.ax2d.set_ylabel("반력 [kN]")
        self.canvas.draw()

    # ------------- 동작 -------------
    def _browse(self):
        fn = filedialog.askopenfilename(
            filetypes=[("DXF", "*.dxf"), ("모든 파일", "*.*")])
        if fn:
            self.vars["dxf"].set(fn)
            try:
                layers = list_layers(fn)
                self.cb_path["values"] = layers
                self.cb_sup["values"] = layers
            except Exception:
                pass

    def _getf(self, key) -> float:
        return float(self.vars[key].get())

    def _run(self):
        try:
            self._run_inner()
        except Exception as exc:
            messagebox.showerror("오류", f"{exc}\n\n{traceback.format_exc()}")

    def _run_inner(self):
        dxf = self.vars["dxf"].get().strip()
        if not dxf:
            messagebox.showwarning("입력 필요", "DXF 파일을 선택하세요.")
            return
        path_layer = self.vars["path_layer"].get().strip() or None
        sup_layer = self.vars["support_layer"].get().strip() or None

        data = parse_dxf(dxf, path_layer, sup_layer)
        seg = self._getf("seg")
        geom = build_geometry(data.path, seg)
        self.geom = geom

        section = Section(
            E=self._getf("E") * GPA,
            G=self._getf("E") * GPA / 2.6,   # 강재 근사 G≈E/2.6
            A=self._getf("A") * CM2,
            Iy=self._getf("Iy") * CM4,
            Iz=self._getf("Iz") * CM4,
            J=self._getf("J") * CM4,
            density=self._getf("density"),
        )
        model = build_model(geom, data.supports, section,
                            self.vars["support_type"].get())
        power_w = self._getf("power") * POWER_UNITS[self.vars["power_unit"].get()]
        bs = self.vars["base_speed"].get().strip()
        device = MovingDevice(
            mass=self._getf("mass"), speed=self._getf("speed"),
            impact_factor=self._getf("impact"),
            power=power_w,
            base_speed=float(bs) if bs else None,
            efficiency=self._getf("eff"))
        self.device = device

        res = run_analysis(model, geom, device,
                           include_selfweight=self.vars["selfweight"].get())
        self.result = res
        self._show_results(data, geom, res)
        self._draw(geom, res)

    def _show_results(self, data, geom, res: EnvelopeResult):
        finite_R = geom.radius[geom.radius < 1e8]
        minR = finite_R.min() if len(finite_R) else float("inf")
        dev = getattr(self, "device", None)
        lines = [
            f"경로 길이      : {geom.length:8.2f} m",
            f"해석 노드/요소 : {geom.n_nodes} / {len(res.elem_max_moment)}",
            f"최소 곡률반경  : {minR:8.2f} m",
            f"지지점 수      : {len(res.support_nodes)}",
        ]
        if dev is not None and dev.power is not None:
            lines += [
                "─" * 34,
                f"기저속도       : {dev.base_speed:8.2f} m/s",
                f"최대 견인력    : {dev.traction/1000:8.2f} kN",
                f"최대 가속도    : {dev.accel:8.3f} m/s²",
            ]
        lines += [
            "─" * 34,
            f"[지배] 지지점 #{res.gov_support_index}",
            f"  최대 반력   : {res.gov_reaction/1000:8.2f} kN",
            f"  발생 위치 s : {res.gov_s:8.2f} m",
            f"최대 휨모멘트  : {res.gov_moment/1000:8.2f} kN·m",
            f"최대 전단      : {res.elem_max_shear.max()/1000:8.2f} kN",
            f"최대 축력      : {res.elem_max_axial.max()/1000:8.2f} kN",
            f"최대 비틀림    : {res.elem_max_torsion.max()/1000:8.2f} kN·m",
            "─" * 34,
            "지지점별 최대반력 [kN]:",
        ]
        for m, node in enumerate(res.support_nodes):
            lines.append(f"  #{m:<2d} {res.reaction_max_resultant[m]/1000:7.2f}"
                         f"  @s={res.reaction_max_at_s[m]:5.2f}")
        self.txt.delete("1.0", "end")
        self.txt.insert("1.0", "\n".join(lines))

    def _draw(self, geom, res: EnvelopeResult):
        self.ax3d.clear()
        self.ax2d.clear()

        nodes = geom.nodes
        self.ax3d.plot(nodes[:, 0], nodes[:, 1], nodes[:, 2],
                       "-", color="0.4", lw=1.2, label="레일 경로")

        # 지지점: 반력 크기로 색/크기
        sx = res.support_xyz
        rk = res.reaction_max_resultant / 1000.0
        sc = self.ax3d.scatter(sx[:, 0], sx[:, 1], sx[:, 2], c=rk,
                               cmap="jet", s=60, depthshade=False,
                               edgecolors="k", label="지지점")
        self.fig.colorbar(sc, ax=self.ax3d, shrink=0.6, label="최대반력 [kN]")

        # 지배 지지점 강조
        g = res.gov_support_index
        self.ax3d.scatter([sx[g, 0]], [sx[g, 1]], [sx[g, 2]],
                          s=160, facecolors="none", edgecolors="red", lw=2)
        # 지배 하중 위치 마커
        gi = int(np.argmin(np.abs(geom.s - res.gov_s)))
        self.ax3d.scatter([nodes[gi, 0]], [nodes[gi, 1]], [nodes[gi, 2]],
                          marker="v", s=80, color="red", label="지배 하중위치")
        self.ax3d.set_title("레일 경로 / 지지점 최대반력")
        self.ax3d.legend(loc="upper left", fontsize=8)
        try:
            self.ax3d.set_box_aspect(
                np.ptp(nodes, axis=0) + 1e-6)
        except Exception:
            pass

        # 영향선: 지배 지지점 반력 vs 하중위치
        self.ax2d.plot(res.s_positions, res.reaction_history[:, g] / 1000.0,
                       "-b", lw=1.5)
        self.ax2d.axhline(res.gov_reaction / 1000.0, color="r", ls="--", lw=1,
                          label=f"최대 {res.gov_reaction/1000:.1f} kN")
        self.ax2d.axvline(res.gov_s, color="r", ls=":", lw=1)
        self.ax2d.set_title(f"지배 지지점 #{g} 반력 영향선")
        self.ax2d.set_xlabel("하중 위치 s [m]")
        self.ax2d.set_ylabel("반력 [kN]")
        self.ax2d.grid(True, alpha=0.3)
        self.ax2d.legend(fontsize=8)

        self.fig.tight_layout()
        self.canvas.draw()


def main():
    root = tk.Tk()
    root.title("레일 이동하중 구조검토")
    root.geometry("1200x780")
    # 한글 폰트
    try:
        matplotlib.rcParams["font.family"] = "Malgun Gothic"
        matplotlib.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass
    RailApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
