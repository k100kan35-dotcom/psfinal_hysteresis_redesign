"""
Data Input Tab for Persson Friction Model
==========================================
다중 컴파운드 데이터 입력 및 일괄 계산 탭

Usage in main.py:
    from data_input_tab import bind_data_input_tab
    # __init__ 내에서, _create_main_layout() 호출 전에:
    bind_data_input_tab(self)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import os
import threading

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from persson_model.core.psd_models import MeasuredPSD
from persson_model.core.g_calculator import GCalculator
from persson_model.core.friction import FrictionCalculator
from persson_model.core.viscoelastic import ViscoelasticMaterial
from persson_model.utils.data_loader import (
    load_psd_from_file,
    load_dma_from_file,
    create_material_from_dma,
    create_psd_from_data,
    smooth_dma_data,
    load_strain_sweep_file,
    compute_fg_from_strain_sweep,
    create_fg_interpolator,
    average_fg_curves,
    load_fg_curve_file,
)


def bind_data_input_tab(app):
    """Data Input 탭을 app에 바인딩한다."""
    dit = DataInputTab(app)
    app._data_input_tab = dit


class CompoundData:
    """단일 컴파운드의 데이터 컨테이너."""
    def __init__(self, name='Compound'):
        self.name = name
        self.material = None           # ViscoelasticMaterial
        self.dma_file = None           # DMA 파일 경로

        # aT shift factor
        self.aT_data = None            # dict: {T, aT, log_aT, bT, T_ref, has_bT}
        self.aT_interp = None          # interp1d(T -> log10(aT))
        self.bT_interp = None          # interp1d(T -> bT) (optional)
        self.aT_file = None            # aT 파일 경로

        # Strain sweep / f,g curves (two input modes)
        self.strain_mode = None        # 'raw' (온도별) or 'fg' (완성된 f,g 파일)
        self.strain_data = None        # strain sweep raw data {T: [...]} (raw mode)
        self.fg_by_T = None            # f,g by temperature (raw mode)
        self.fg_raw = None             # dict: {strain, f, g} (fg mode)
        self.f_interpolator = None     # f(strain) callable
        self.g_interpolator = None     # g(strain) callable
        self.strain_file = None        # strain/f,g 파일 경로

        # mu_dry
        self.mu_dry_data = None        # (log10_v, mu_dry) arrays
        self.mu_dry_file = None        # mu_dry 파일 경로

        # Results
        self.results = None            # calculation results dict


class DataInputTab:
    """데이터 입력 탭 (Tab 0): 다중 컴파운드 입력 + 일괄 계산."""

    MAX_COMPOUNDS = 8
    COMPOUND_COLORS = [
        '#2563EB', '#059669', '#D97706', '#DC2626',
        '#7C3AED', '#DB2777', '#0891B2', '#65A30D',
    ]

    def __init__(self, app):
        self.app = app
        app._create_data_input_tab = self._create_tab

        # State
        self.compounds = []          # list of CompoundData
        self.psd_model = None        # MeasuredPSD (shared)
        self.psd_file = None         # PSD file path
        self.psd_info = None         # info string
        self._calculating = False

        # UI references
        self._compound_frames = []
        self._compound_widgets = {}  # idx -> dict of widget references
        self._status_labels = {}     # idx -> status label

    # ================================================================
    #  Tab creation
    # ================================================================
    def _create_tab(self, parent):
        """Build the Data Input tab."""
        C = self.app.COLORS
        F = self.app.FONTS

        main = ttk.Frame(parent)
        main.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        # ── Left panel (scrollable controls) ──
        left = ttk.Frame(main, width=380)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 4))
        left.pack_propagate(False)

        scroll_canvas = tk.Canvas(left, highlightthickness=0, bg=C['bg'])
        scrollbar = ttk.Scrollbar(left, orient='vertical', command=scroll_canvas.yview)
        content = ttk.Frame(scroll_canvas)
        content.bind('<Configure>',
                     lambda e: scroll_canvas.configure(scrollregion=scroll_canvas.bbox('all')))
        cw_id = scroll_canvas.create_window((0, 0), window=content, anchor='nw', width=362)
        scroll_canvas.configure(yscrollcommand=scrollbar.set)
        scroll_canvas.bind('<Configure>',
                           lambda e, _c=scroll_canvas, _id=cw_id:
                               _c.itemconfigure(_id, width=e.width))
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Mousewheel
        def _on_mw(event):
            if event.delta:
                scroll_canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')
            elif event.num == 4:
                scroll_canvas.yview_scroll(-1, 'units')
            elif event.num == 5:
                scroll_canvas.yview_scroll(1, 'units')
        scroll_canvas.bind('<Enter>',
                           lambda e: (scroll_canvas.bind_all('<MouseWheel>', _on_mw),
                                      scroll_canvas.bind_all('<Button-4>', _on_mw),
                                      scroll_canvas.bind_all('<Button-5>', _on_mw)))
        scroll_canvas.bind('<Leave>',
                           lambda e: (scroll_canvas.unbind_all('<MouseWheel>'),
                                      scroll_canvas.unbind_all('<Button-4>'),
                                      scroll_canvas.unbind_all('<Button-5>')))

        self._left_content = content

        # ── Section 1: Compound count ──
        sec1 = self._make_section(content, "1) 컴파운드 설정")
        row1 = ttk.Frame(sec1)
        row1.pack(fill=tk.X, pady=2)
        ttk.Label(row1, text="컴파운드 수:", font=F['body']).pack(side=tk.LEFT)
        self._compound_count_var = tk.IntVar(value=1)
        spin = ttk.Spinbox(row1, from_=1, to=self.MAX_COMPOUNDS,
                           textvariable=self._compound_count_var, width=5)
        spin.pack(side=tk.LEFT, padx=4)
        ttk.Button(row1, text="확인", command=self._on_compound_count_confirm).pack(side=tk.LEFT, padx=4)

        # ── Section 2: Shared PSD ──
        sec2 = self._make_section(content, "2) 공통 PSD")
        ttk.Button(sec2, text="PSD 파일 로드 (CSV/TXT)",
                   command=self._load_psd_file).pack(fill=tk.X, pady=2)
        ttk.Button(sec2, text="노면 프로파일 로드",
                   command=self._load_profile_file).pack(fill=tk.X, pady=2)
        if hasattr(self.app, 'psd_model') and self.app.psd_model is not None:
            ttk.Button(sec2, text="기존 PSD 사용 (Tab 0)",
                       command=self._use_existing_psd).pack(fill=tk.X, pady=2)
        self._psd_status_var = tk.StringVar(value="PSD 미로드")
        ttk.Label(sec2, textvariable=self._psd_status_var, font=F['small'],
                  foreground=C['text_secondary']).pack(fill=tk.X, pady=2)

        # ── Section 3: Compounds (dynamic) ──
        self._compounds_container = ttk.Frame(content)
        self._compounds_container.pack(fill=tk.X, pady=4)

        # ── Section 4: Model options ──
        sec4 = self._make_section(content, "3) 계산 설정")

        # σ₀
        row_s0 = ttk.Frame(sec4)
        row_s0.pack(fill=tk.X, pady=2)
        ttk.Label(row_s0, text="σ₀ (MPa):", font=F['body']).pack(side=tk.LEFT)
        self._sigma0_var = tk.StringVar(value="0.3")
        ttk.Entry(row_s0, textvariable=self._sigma0_var, width=8).pack(side=tk.RIGHT, padx=2)

        # Temperature
        row_t = ttk.Frame(sec4)
        row_t.pack(fill=tk.X, pady=2)
        ttk.Label(row_t, text="T (°C):", font=F['body']).pack(side=tk.LEFT)
        self._temp_var = tk.StringVar(value="20")
        ttk.Entry(row_t, textvariable=self._temp_var, width=8).pack(side=tk.RIGHT, padx=2)

        # Velocity range
        row_v = ttk.Frame(sec4)
        row_v.pack(fill=tk.X, pady=2)
        ttk.Label(row_v, text="v 범위:", font=F['body']).pack(side=tk.LEFT)
        self._v_min_var = tk.StringVar(value="1e-4")
        self._v_max_var = tk.StringVar(value="10")
        ttk.Entry(row_v, textvariable=self._v_min_var, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(row_v, text="~", font=F['body']).pack(side=tk.LEFT)
        ttk.Entry(row_v, textvariable=self._v_max_var, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(row_v, text="m/s", font=F['body']).pack(side=tk.LEFT)

        # n_v, n_q
        row_n = ttk.Frame(sec4)
        row_n.pack(fill=tk.X, pady=2)
        ttk.Label(row_n, text="n_v:", font=F['body']).pack(side=tk.LEFT)
        self._n_v_var = tk.StringVar(value="30")
        ttk.Entry(row_n, textvariable=self._n_v_var, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Label(row_n, text="n_q:", font=F['body']).pack(side=tk.LEFT, padx=(8, 0))
        self._n_q_var = tk.StringVar(value="200")
        ttk.Entry(row_n, textvariable=self._n_q_var, width=5).pack(side=tk.LEFT, padx=2)

        # Nonlinear
        row_nl = ttk.Frame(sec4)
        row_nl.pack(fill=tk.X, pady=2)
        self._use_nonlinear_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row_nl, text="비선형 보정 (f,g)", variable=self._use_nonlinear_var).pack(side=tk.LEFT)

        # Flash temperature
        row_flash = ttk.Frame(sec4)
        row_flash.pack(fill=tk.X, pady=2)
        self._use_flash_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row_flash, text="Flash Temperature", variable=self._use_flash_var).pack(side=tk.LEFT)

        # gamma
        row_gamma = ttk.Frame(sec4)
        row_gamma.pack(fill=tk.X, pady=2)
        ttk.Label(row_gamma, text="γ:", font=F['body']).pack(side=tk.LEFT)
        self._gamma_var = tk.StringVar(value="0.6")
        ttk.Entry(row_gamma, textvariable=self._gamma_var, width=8).pack(side=tk.RIGHT, padx=2)

        # n_phi
        row_phi = ttk.Frame(sec4)
        row_phi.pack(fill=tk.X, pady=2)
        ttk.Label(row_phi, text="n_φ:", font=F['body']).pack(side=tk.LEFT)
        self._n_phi_var = tk.StringVar(value="72")
        ttk.Entry(row_phi, textvariable=self._n_phi_var, width=8).pack(side=tk.RIGHT, padx=2)

        # ── Calculate All Button ──
        sec5 = self._make_section(content, "4) 일괄 계산")
        self._calc_button = ttk.Button(sec5, text="★ 전체 계산 실행 ★",
                                       command=self._on_calculate_all)
        self._calc_button.pack(fill=tk.X, pady=4, ipady=6)

        self._progress_var = tk.DoubleVar(value=0)
        self._progress_bar = ttk.Progressbar(sec5, variable=self._progress_var, maximum=100)
        self._progress_bar.pack(fill=tk.X, pady=2)

        self._calc_status_var = tk.StringVar(value="대기 중")
        ttk.Label(sec5, textvariable=self._calc_status_var, font=F['small'],
                  foreground=C['text_secondary']).pack(fill=tk.X, pady=2)

        # ── Right panel (preview plot) ──
        right = ttk.Frame(main)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._fig = Figure(figsize=(8, 6), dpi=100, facecolor='white')
        self._fig.subplots_adjust(hspace=0.35, wspace=0.3)
        self._canvas = FigureCanvasTkAgg(self._fig, right)
        toolbar = NavigationToolbar2Tk(self._canvas, right)
        toolbar.update()
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Initialize with 1 compound
        self._rebuild_compound_columns(1)

    # ================================================================
    #  UI helpers
    # ================================================================
    def _make_section(self, parent, title):
        frame = ttk.LabelFrame(parent, text=title, padding=8)
        frame.pack(fill=tk.X, pady=4, padx=2)
        return frame

    def _on_compound_count_confirm(self):
        n = self._compound_count_var.get()
        n = max(1, min(n, self.MAX_COMPOUNDS))
        self._rebuild_compound_columns(n)

    def _rebuild_compound_columns(self, n):
        """Rebuild compound input columns."""
        # Clear existing
        for w in self._compounds_container.winfo_children():
            w.destroy()
        self._compound_frames.clear()
        self._compound_widgets.clear()
        self._status_labels.clear()

        # Adjust compound list
        while len(self.compounds) < n:
            idx = len(self.compounds) + 1
            self.compounds.append(CompoundData(f'Compound {idx}'))
        while len(self.compounds) > n:
            self.compounds.pop()

        F = self.app.FONTS
        C = self.app.COLORS

        for i in range(n):
            cpd = self.compounds[i]
            color = self.COMPOUND_COLORS[i % len(self.COMPOUND_COLORS)]

            lf = ttk.LabelFrame(self._compounds_container,
                                text=f"  ● {cpd.name}  ", padding=6)
            lf.pack(fill=tk.X, pady=3, padx=2)
            self._compound_frames.append(lf)

            widgets = {}

            # Name
            row_name = ttk.Frame(lf)
            row_name.pack(fill=tk.X, pady=1)
            ttk.Label(row_name, text="이름:", font=F['small']).pack(side=tk.LEFT)
            name_var = tk.StringVar(value=cpd.name)
            name_entry = ttk.Entry(row_name, textvariable=name_var, width=18)
            name_entry.pack(side=tk.LEFT, padx=2)
            name_var.trace_add('write', lambda *a, idx=i, v=name_var: self._on_name_changed(idx, v))
            widgets['name_var'] = name_var

            # Master curve (필수)
            row_mc = ttk.Frame(lf)
            row_mc.pack(fill=tk.X, pady=1)
            ttk.Button(row_mc, text="마스터커브 (DMA)", width=18,
                       command=lambda idx=i: self._load_dma(idx)).pack(side=tk.LEFT)
            mc_label = ttk.Label(row_mc, text="✗", font=F['small'], foreground=C['danger'])
            mc_label.pack(side=tk.LEFT, padx=4)
            widgets['mc_label'] = mc_label

            # aT shift factor (선택)
            row_at = ttk.Frame(lf)
            row_at.pack(fill=tk.X, pady=1)
            ttk.Button(row_at, text="aT 시프트 팩터", width=18,
                       command=lambda idx=i: self._load_aT(idx)).pack(side=tk.LEFT)
            at_label = ttk.Label(row_at, text="-", font=F['small'], foreground=C['text_secondary'])
            at_label.pack(side=tk.LEFT, padx=4)
            widgets['at_label'] = at_label

            # Strain / f,g curves (선택) — 2가지 모드
            row_str_header = ttk.Frame(lf)
            row_str_header.pack(fill=tk.X, pady=(4, 0))
            ttk.Label(row_str_header, text="Strain 보정:", font=F['small_bold']).pack(side=tk.LEFT)

            row_str_btns = ttk.Frame(lf)
            row_str_btns.pack(fill=tk.X, pady=1)
            ttk.Button(row_str_btns, text="온도별 sweep", width=14,
                       command=lambda idx=i: self._load_strain_raw(idx)).pack(side=tk.LEFT, padx=(0, 2))
            ttk.Button(row_str_btns, text="f,g 파일", width=10,
                       command=lambda idx=i: self._load_fg_file(idx)).pack(side=tk.LEFT)
            str_label = ttk.Label(lf, text="-", font=F['small'], foreground=C['text_secondary'])
            str_label.pack(fill=tk.X, pady=1)
            widgets['str_label'] = str_label

            # mu_dry (선택)
            row_mud = ttk.Frame(lf)
            row_mud.pack(fill=tk.X, pady=1)
            ttk.Button(row_mud, text="μ_dry (실측)", width=18,
                       command=lambda idx=i: self._load_mu_dry(idx)).pack(side=tk.LEFT)
            mud_label = ttk.Label(row_mud, text="-", font=F['small'], foreground=C['text_secondary'])
            mud_label.pack(side=tk.LEFT, padx=4)
            widgets['mud_label'] = mud_label

            self._compound_widgets[i] = widgets

    def _on_name_changed(self, idx, var):
        try:
            self.compounds[idx].name = var.get()
        except Exception:
            pass

    # ================================================================
    #  Data loading
    # ================================================================
    def _load_psd_file(self):
        fp = filedialog.askopenfilename(
            title="PSD 파일 선택",
            filetypes=[("CSV/TXT", "*.csv *.txt"), ("All", "*.*")])
        if not fp:
            return
        try:
            q, C_q = load_psd_from_file(fp)
            self.psd_model = create_psd_from_data(q, C_q)
            self.psd_file = fp
            fname = os.path.basename(fp)
            self.psd_info = f"{fname} ({len(q)} pts, q: {q.min():.1e}~{q.max():.1e})"
            self._psd_status_var.set(f"✓ {self.psd_info}")
        except Exception as e:
            messagebox.showerror("PSD 로드 실패", str(e))

    def _load_profile_file(self):
        fp = filedialog.askopenfilename(
            title="노면 프로파일 선택",
            filetypes=[("All", "*.*"), ("TXT", "*.txt"), ("CSV", "*.csv")])
        if not fp:
            return
        try:
            from monte_carlo_tab import PSDComputer
            pc = PSDComputer()
            info = pc.load_profile(fp)
            q, C_q = pc.compute_psd()
            self.psd_model = create_psd_from_data(q, C_q)
            self.psd_file = fp
            fname = os.path.basename(fp)
            self.psd_info = f"{fname} ({info['n_points']} pts → PSD {len(q)} bins)"
            self._psd_status_var.set(f"✓ {self.psd_info}")
        except Exception as e:
            messagebox.showerror("프로파일 로드 실패", str(e))

    def _use_existing_psd(self):
        if self.app.psd_model is not None:
            self.psd_model = self.app.psd_model
            self.psd_info = "기존 PSD (Tab 0)"
            self._psd_status_var.set(f"✓ {self.psd_info}")
        else:
            messagebox.showinfo("알림", "기존 PSD가 없습니다.")

    def _load_dma(self, idx):
        fp = filedialog.askopenfilename(
            title=f"DMA 파일 선택 ({self.compounds[idx].name})",
            filetypes=[("CSV/TXT", "*.csv *.txt"), ("All", "*.*")])
        if not fp:
            return
        try:
            omega, E_stor, E_loss = load_dma_from_file(fp)
            smoothed = smooth_dma_data(omega, E_stor, E_loss)
            material = create_material_from_dma(
                smoothed['omega'],
                smoothed['E_storage_smooth'],
                smoothed['E_loss_smooth'],
                material_name=self.compounds[idx].name
            )
            self.compounds[idx].material = material
            self.compounds[idx].dma_file = fp
            fname = os.path.basename(fp)
            w = self._compound_widgets.get(idx, {})
            if 'mc_label' in w:
                w['mc_label'].config(text=f"✓ {fname}", foreground=self.app.COLORS['success'])
        except Exception as e:
            messagebox.showerror("DMA 로드 실패", str(e))

    def _load_aT(self, idx):
        """aT shift factor 파일 로드. 형식: T(°C), aT [, bT]"""
        fp = filedialog.askopenfilename(
            title=f"aT 시프트 팩터 ({self.compounds[idx].name})",
            filetypes=[("CSV/TXT", "*.csv *.txt"), ("All", "*.*")])
        if not fp:
            return
        try:
            import pandas as pd
            from scipy.interpolate import interp1d

            df = None
            for sep in [r'\s+', '\t', ',', ';']:
                try:
                    df = pd.read_csv(fp, sep=sep, skipinitialspace=True,
                                     comment='#', header=None, engine='python')
                    if len(df.columns) >= 2:
                        break
                except Exception:
                    continue

            if df is None or len(df.columns) < 2:
                raise ValueError("2열 이상 (T, aT [, bT]) 형식이 필요합니다.")

            df = df.apply(pd.to_numeric, errors='coerce').dropna()
            if len(df) < 2:
                raise ValueError("유효한 데이터 행이 부족합니다.")

            T = df.iloc[:, 0].values
            aT = df.iloc[:, 1].values
            has_bT = len(df.columns) >= 3
            bT = df.iloc[:, 2].values if has_bT else np.ones_like(T)

            log_aT = np.log10(np.maximum(aT, 1e-20))
            sort_idx = np.argsort(T)
            T, aT, log_aT, bT = T[sort_idx], aT[sort_idx], log_aT[sort_idx], bT[sort_idx]

            ref_idx = np.argmin(np.abs(aT - 1.0))
            T_ref = T[ref_idx]

            cpd = self.compounds[idx]
            cpd.aT_data = {
                'T': T, 'aT': aT, 'log_aT': log_aT,
                'bT': bT, 'T_ref': T_ref, 'has_bT': has_bT,
            }
            cpd.aT_interp = interp1d(T, log_aT, kind='linear',
                                     bounds_error=False, fill_value='extrapolate')
            cpd.bT_interp = interp1d(T, bT, kind='linear',
                                     bounds_error=False, fill_value='extrapolate')
            cpd.aT_file = fp

            fname = os.path.basename(fp)
            bT_str = ", +bT" if has_bT else ""
            w = self._compound_widgets.get(idx, {})
            if 'at_label' in w:
                w['at_label'].config(
                    text=f"✓ {fname} ({len(T)}pts, Tref={T_ref:.0f}°C{bT_str})",
                    foreground=self.app.COLORS['success'])
        except Exception as e:
            messagebox.showerror("aT 로드 실패", str(e))

    def _load_strain_raw(self, idx):
        """온도별 strain sweep 원시 데이터 로드 → f,g 계산."""
        fp = filedialog.askopenfilename(
            title=f"Strain sweep (온도별) ({self.compounds[idx].name})",
            filetypes=[("CSV/TXT", "*.csv *.txt"), ("All", "*.*")])
        if not fp:
            return
        try:
            data_by_T = load_strain_sweep_file(fp)
            fg_by_T = compute_fg_from_strain_sweep(data_by_T)
            fg_avg = average_fg_curves(fg_by_T)

            f_interp, g_interp = create_fg_interpolator(
                fg_avg['strain'], fg_avg['f'], fg_avg.get('g'))

            cpd = self.compounds[idx]
            cpd.strain_mode = 'raw'
            cpd.strain_data = data_by_T
            cpd.fg_by_T = fg_by_T
            cpd.fg_raw = None
            cpd.f_interpolator = f_interp
            cpd.g_interpolator = g_interp
            cpd.strain_file = fp

            fname = os.path.basename(fp)
            n_T = len(fg_by_T)
            w = self._compound_widgets.get(idx, {})
            if 'str_label' in w:
                w['str_label'].config(text=f"✓ [sweep] {fname} ({n_T}T)",
                                      foreground=self.app.COLORS['success'])
        except Exception as e:
            messagebox.showerror("Strain sweep 로드 실패", str(e))

    def _load_fg_file(self, idx):
        """완성된 f,g 커브 파일 로드 (strain, f [, g])."""
        fp = filedialog.askopenfilename(
            title=f"f,g 커브 파일 ({self.compounds[idx].name})",
            filetypes=[("CSV/TXT", "*.csv *.txt"), ("All", "*.*")])
        if not fp:
            return
        try:
            fg_data = load_fg_curve_file(fp)
            if fg_data is None:
                raise ValueError("f,g 파일 파싱에 실패했습니다.")

            f_interp, g_interp = create_fg_interpolator(
                fg_data['strain'], fg_data['f'], fg_data.get('g'))

            cpd = self.compounds[idx]
            cpd.strain_mode = 'fg'
            cpd.fg_raw = fg_data
            cpd.strain_data = None
            cpd.fg_by_T = None
            cpd.f_interpolator = f_interp
            cpd.g_interpolator = g_interp
            cpd.strain_file = fp

            fname = os.path.basename(fp)
            n_pts = len(fg_data['strain'])
            has_g = fg_data.get('g') is not None
            g_str = "f+g" if has_g else "f only"
            w = self._compound_widgets.get(idx, {})
            if 'str_label' in w:
                w['str_label'].config(text=f"✓ [f,g] {fname} ({n_pts}pts, {g_str})",
                                      foreground=self.app.COLORS['success'])
        except Exception as e:
            messagebox.showerror("f,g 파일 로드 실패", str(e))

    def _load_mu_dry(self, idx):
        fp = filedialog.askopenfilename(
            title=f"μ_dry 파일 ({self.compounds[idx].name})",
            filetypes=[("CSV/TXT", "*.csv *.txt"), ("All", "*.*")])
        if not fp:
            return
        try:
            data = np.loadtxt(fp, delimiter=None, comments='#')
            if data.ndim == 1:
                raise ValueError("2열 (log10_v, mu_dry) 형식이 필요합니다")
            log_v = data[:, 0]
            mu_dry = data[:, 1]
            self.compounds[idx].mu_dry_data = (log_v, mu_dry)
            self.compounds[idx].mu_dry_file = fp

            fname = os.path.basename(fp)
            w = self._compound_widgets.get(idx, {})
            if 'mud_label' in w:
                w['mud_label'].config(text=f"✓ {fname} ({len(mu_dry)} pts)",
                                      foreground=self.app.COLORS['success'])
        except Exception as e:
            messagebox.showerror("μ_dry 로드 실패", str(e))

    # ================================================================
    #  Validation
    # ================================================================
    def _validate(self):
        """Validate inputs before calculation. Returns (ok, message)."""
        if self.psd_model is None:
            return False, "PSD가 로드되지 않았습니다."
        for i, cpd in enumerate(self.compounds):
            if cpd.material is None:
                return False, f"컴파운드 {i+1} ({cpd.name})에 마스터커브가 없습니다."
        return True, ""

    # ================================================================
    #  Calculate All
    # ================================================================
    def _on_calculate_all(self):
        if self._calculating:
            return
        ok, msg = self._validate()
        if not ok:
            messagebox.showwarning("입력 확인", msg)
            return
        self._calculating = True
        self._calc_button.config(state='disabled')
        self._progress_var.set(0)
        self._calc_status_var.set("계산 시작...")

        thread = threading.Thread(target=self._calculate_all_worker, daemon=True)
        thread.start()

    def _calculate_all_worker(self):
        """Background worker: calculate mu_visc for all compounds."""
        try:
            sigma_0 = float(self._sigma0_var.get()) * 1e6  # MPa → Pa
            temperature = float(self._temp_var.get())
            v_min = float(self._v_min_var.get())
            v_max = float(self._v_max_var.get())
            n_v = int(self._n_v_var.get())
            n_q = int(self._n_q_var.get())
            gamma = float(self._gamma_var.get())
            n_phi = int(self._n_phi_var.get())
            use_nonlinear = self._use_nonlinear_var.get()

            v_array = np.logspace(np.log10(v_min), np.log10(v_max), n_v)

            # Get q range from PSD model
            if hasattr(self.psd_model, 'q_data'):
                q_min_psd = self.psd_model.q_data.min()
                q_max_psd = self.psd_model.q_data.max()
            else:
                q_min_psd = 1e2
                q_max_psd = 1e8
            q_array = np.logspace(np.log10(q_min_psd), np.log10(q_max_psd), n_q)

            n_compounds = len(self.compounds)
            total_steps = n_compounds * 2  # G calc + mu calc per compound

            for ci, cpd in enumerate(self.compounds):
                self._update_status(f"[{ci+1}/{n_compounds}] {cpd.name}: G(q,v) 계산 중...")

                # ── aT temperature shift ──
                # If aT data is available, shift frequency: ω_eff = ω * aT(T)
                # E*(ω, T) = E*(ω·aT, T_ref), so modulus_func uses shifted ω
                has_aT = cpd.aT_interp is not None
                if has_aT:
                    log_aT_val = float(cpd.aT_interp(temperature))
                    aT_val = 10 ** log_aT_val
                    T_ref = cpd.aT_data['T_ref']
                else:
                    aT_val = 1.0
                    T_ref = temperature

                # Create modulus function (with aT shift applied)
                def modulus_func(omega, mat=cpd.material, _aT=aT_val, _Tref=T_ref):
                    return mat.get_modulus(omega * _aT, temperature=_Tref)

                # Create G calculator
                g_calc = GCalculator(
                    psd_func=self.psd_model,
                    modulus_func=modulus_func,
                    sigma_0=sigma_0,
                    velocity=v_array[0],
                    poisson_ratio=0.5,
                    n_angle_points=36,
                )

                # Nonlinear correction
                if use_nonlinear and cpd.f_interpolator is not None and cpd.g_interpolator is not None:
                    fixed_strain = 0.01
                    strain_for_G = np.full(len(q_array), fixed_strain)
                    g_calc.storage_modulus_func = lambda w, m=cpd.material, _aT=aT_val, _Tref=T_ref: \
                        m.get_storage_modulus(w * _aT, temperature=_Tref)
                    g_calc.loss_modulus_func = lambda w, m=cpd.material, _aT=aT_val, _Tref=T_ref: \
                        m.get_loss_modulus(w * _aT, temperature=_Tref)
                    g_calc.set_nonlinear_correction(
                        f_interpolator=cpd.f_interpolator,
                        g_interpolator=cpd.g_interpolator,
                        strain_array=strain_for_G,
                        strain_q_array=q_array,
                    )

                # Calculate G(q,v) matrix
                def g_progress(pct, ci=ci):
                    overall = (ci * 2 / total_steps + pct / 100 / total_steps) * 100
                    self._update_progress(overall)

                results_2d = g_calc.calculate_G_multi_velocity(
                    q_array, v_array, q_min=q_array[0], progress_callback=g_progress)

                G_matrix = results_2d['G_matrix']

                # Clear nonlinear correction
                if use_nonlinear and cpd.f_interpolator is not None:
                    g_calc.clear_nonlinear_correction()

                # ── mu_visc calculation ──
                self._update_status(f"[{ci+1}/{n_compounds}] {cpd.name}: μ_visc 계산 중...")

                # Loss modulus with aT shift
                def loss_mod_func(omega, T, mat=cpd.material, _aT=aT_val, _Tref=T_ref):
                    return mat.get_loss_modulus(omega * _aT, temperature=_Tref)

                g_interp = cpd.g_interpolator if (use_nonlinear and cpd.g_interpolator) else None

                friction_calc = FrictionCalculator(
                    psd_func=self.psd_model,
                    loss_modulus_func=loss_mod_func,
                    sigma_0=sigma_0,
                    velocity=v_array[0],
                    temperature=temperature,
                    poisson_ratio=0.5,
                    gamma=gamma,
                    n_angle_points=n_phi,
                    g_interpolator=g_interp,
                    strain_estimate=0.01,
                )

                C_q = self.psd_model(q_array)

                def mu_progress(pct, ci=ci):
                    overall = ((ci * 2 + 1) / total_steps + pct / 100 / total_steps) * 100
                    self._update_progress(overall)

                mu_array, details = friction_calc.calculate_mu_visc_multi_velocity(
                    q_array, G_matrix, v_array, C_q, mu_progress)

                # ── mu_adh from mu_dry (if available) ──
                mu_adh_array = None
                if cpd.mu_dry_data is not None:
                    from scipy.interpolate import interp1d
                    log_v_dry, mu_dry = cpd.mu_dry_data
                    mu_dry_interp = interp1d(log_v_dry, mu_dry, kind='linear',
                                             bounds_error=False, fill_value='extrapolate')
                    # Simple adhesion model: mu_adh ≈ mu_dry * (A/A0)
                    # A/A0 from the last P(q) value
                    A_A0_arr = np.zeros(n_v)
                    for j in range(n_v):
                        d_j = details['details'][j]
                        A_A0_arr[j] = d_j['P'][-1] if 'P' in d_j else 0.5
                    mu_dry_at_v = mu_dry_interp(np.log10(v_array))
                    mu_adh_array = mu_dry_at_v * A_A0_arr

                # Store results
                mu_total = mu_array.copy()
                if mu_adh_array is not None:
                    mu_total = mu_array + mu_adh_array

                cpd.results = {
                    'v': v_array,
                    'q': q_array,
                    'mu_visc': mu_array,
                    'mu_adh': mu_adh_array,
                    'mu_total': mu_total,
                    'G_matrix': G_matrix,
                    'details': details,
                    'C_q': C_q,
                    'sigma_0': sigma_0,
                    'temperature': temperature,
                }

            # All done
            self._update_progress(100)
            self._update_status(f"계산 완료 ({n_compounds}개 컴파운드)")

            # Push to app state
            self.app.all_compound_results = [cpd.results for cpd in self.compounds]
            self.app.compound_data = self.compounds
            self.app.data_input_finalized = True

            # Also update app's shared PSD
            if self.psd_model is not None:
                self.app.shared_psd_model = self.psd_model

            # Update results tab
            self.app.root.after(100, self._after_calculation)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self._update_status(f"오류: {e}")
        finally:
            self._calculating = False
            self.app.root.after(0, lambda: self._calc_button.config(state='normal'))

    def _after_calculation(self):
        """Post-calculation: update plots and results tab."""
        self._plot_preview()
        # Notify results overview tab
        if hasattr(self.app, '_results_overview_tab'):
            self.app._results_overview_tab.refresh()

    def _update_status(self, msg):
        self.app.root.after(0, lambda: self._calc_status_var.set(msg))

    def _update_progress(self, pct):
        self.app.root.after(0, lambda: self._progress_var.set(min(pct, 100)))

    # ================================================================
    #  Preview Plot
    # ================================================================
    def _plot_preview(self):
        """Draw a quick preview of calculation results."""
        self._fig.clear()

        compounds_with_results = [c for c in self.compounds if c.results is not None]
        if not compounds_with_results:
            self._canvas.draw_idle()
            return

        ax = self._fig.add_subplot(111)
        ax.set_xlabel('Velocity (m/s)')
        ax.set_ylabel('μ')
        ax.set_title('μ_total vs Velocity (All Compounds)')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)

        for i, cpd in enumerate(compounds_with_results):
            r = cpd.results
            color = self.COMPOUND_COLORS[i % len(self.COMPOUND_COLORS)]
            ax.plot(r['v'], r['mu_total'], '-', color=color,
                    linewidth=2, label=cpd.name)
            if r.get('mu_adh') is not None:
                ax.plot(r['v'], r['mu_visc'], '--', color=color,
                        linewidth=1, alpha=0.6, label=f'{cpd.name} (visc)')
                ax.plot(r['v'], r['mu_adh'], ':', color=color,
                        linewidth=1, alpha=0.6, label=f'{cpd.name} (adh)')

        ax.legend(fontsize=9, loc='best')
        self._fig.tight_layout()
        self._canvas.draw_idle()
