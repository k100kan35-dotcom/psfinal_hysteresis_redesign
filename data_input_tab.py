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
import tkinter.simpledialog
from tkinter import ttk, filedialog, messagebox
import numpy as np
import os
import json

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from persson_model.core.psd_models import MeasuredPSD
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
    # 1=검정, 2=빨강, 3=파랑, 이후 구분 가능한 색상
    COMPOUND_COLORS = [
        '#000000', '#DC2626', '#2563EB', '#059669',
        '#D97706', '#7C3AED', '#DB2777', '#0891B2',
    ]
    # 그래프 기본 스타일
    PLOT_FONT_SIZE = 12
    PLOT_LINE_WIDTH = 2.0
    PLOT_LINE_WIDTH_SUB = 1.3  # 보조 선 (visc, adh)
    PLOT_LEGEND_SIZE = 12

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

        # ── Section 0: Dataset Preset ──
        sec0 = self._make_section(content, "데이터셋 프리셋")
        row_preset = ttk.Frame(sec0)
        row_preset.pack(fill=tk.X, pady=2)
        self._dataset_combo_var = tk.StringVar(value="(선택)")
        self._dataset_combo = ttk.Combobox(row_preset, textvariable=self._dataset_combo_var,
                                           state='readonly', width=22)
        self._dataset_combo.pack(side=tk.LEFT, padx=2)
        ttk.Button(row_preset, text="불러오기", command=self._load_dataset_preset).pack(side=tk.LEFT, padx=2)

        row_preset2 = ttk.Frame(sec0)
        row_preset2.pack(fill=tk.X, pady=2)
        ttk.Button(row_preset2, text="현재 데이터 저장", command=self._save_dataset_preset).pack(side=tk.LEFT, padx=2)
        ttk.Button(row_preset2, text="삭제", command=self._delete_dataset_preset).pack(side=tk.LEFT, padx=2)
        self._refresh_dataset_list()

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

        # q range (q0, q1) with surface preset
        row_q_preset = ttk.Frame(sec4)
        row_q_preset.pack(fill=tk.X, pady=2)
        ttk.Label(row_q_preset, text="노면 프리셋:", font=F['body']).pack(side=tk.LEFT)
        self._surface_q1_combo_var = tk.StringVar(value="(직접 입력)")
        self._surface_q1_combo = ttk.Combobox(
            row_q_preset, textvariable=self._surface_q1_combo_var,
            state='readonly', width=14)
        self._surface_q1_combo.pack(side=tk.LEFT, padx=2)
        self._surface_q1_combo.bind('<<ComboboxSelected>>', self._on_surface_q1_selected)
        self._refresh_surface_q1_list()

        row_q = ttk.Frame(sec4)
        row_q.pack(fill=tk.X, pady=2)
        ttk.Label(row_q, text="q₀:", font=F['body']).pack(side=tk.LEFT)
        self._q0_var = tk.StringVar(value="500")
        ttk.Entry(row_q, textvariable=self._q0_var, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(row_q, text="q₁:", font=F['body']).pack(side=tk.LEFT, padx=(8, 0))
        self._q1_var = tk.StringVar(value="1.05e5")
        ttk.Entry(row_q, textvariable=self._q1_var, width=8).pack(side=tk.LEFT, padx=2)

        # h'rms (ξ) display + calculate button
        row_hrms = ttk.Frame(sec4)
        row_hrms.pack(fill=tk.X, pady=2)
        ttk.Button(row_hrms, text="h'rms 계산", width=10,
                   command=self._calc_hrms_from_q1).pack(side=tk.LEFT)
        self._hrms_var = tk.StringVar(value="ξ = -")
        ttk.Label(row_hrms, textvariable=self._hrms_var, font=F['body_bold'],
                  foreground=C['primary']).pack(side=tk.LEFT, padx=8)

        # n_v, n_q
        row_n = ttk.Frame(sec4)
        row_n.pack(fill=tk.X, pady=2)
        ttk.Label(row_n, text="n_v:", font=F['body']).pack(side=tk.LEFT)
        self._n_v_var = tk.StringVar(value="30")
        ttk.Entry(row_n, textvariable=self._n_v_var, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Label(row_n, text="n_q:", font=F['body']).pack(side=tk.LEFT, padx=(8, 0))
        self._n_q_var = tk.StringVar(value="36")
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
        self._gamma_var = tk.StringVar(value="0.57")
        ttk.Entry(row_gamma, textvariable=self._gamma_var, width=8).pack(side=tk.RIGHT, padx=2)

        # n_phi
        row_phi = ttk.Frame(sec4)
        row_phi.pack(fill=tk.X, pady=2)
        ttk.Label(row_phi, text="n_φ:", font=F['body']).pack(side=tk.LEFT)
        self._n_phi_var = tk.StringVar(value="14")
        ttk.Entry(row_phi, textvariable=self._n_phi_var, width=8).pack(side=tk.RIGHT, padx=2)

        # ── Section 5: mu_adh parameters ──
        sec5 = self._make_section(content, "4) μ_adh 설정")

        # mu_adh model: τ_f = τ_f0 × exp[-c × (log10(v/v0*))²]
        row_tau = ttk.Frame(sec5)
        row_tau.pack(fill=tk.X, pady=2)
        ttk.Label(row_tau, text="τ_f0 (MPa):", font=F['body']).pack(side=tk.LEFT)
        self._tau_f0_var = tk.StringVar(value="6.5")
        ttk.Entry(row_tau, textvariable=self._tau_f0_var, width=8).pack(side=tk.RIGHT, padx=2)

        row_v0 = ttk.Frame(sec5)
        row_v0.pack(fill=tk.X, pady=2)
        ttk.Label(row_v0, text="v₀* (m/s):", font=F['body']).pack(side=tk.LEFT)
        self._v0_star_var = tk.StringVar(value="10")
        ttk.Entry(row_v0, textvariable=self._v0_star_var, width=8).pack(side=tk.RIGHT, padx=2)

        row_c = ttk.Frame(sec5)
        row_c.pack(fill=tk.X, pady=2)
        ttk.Label(row_c, text="c (Gauss폭):", font=F['body']).pack(side=tk.LEFT)
        self._c_gauss_var = tk.StringVar(value="0.1")
        ttk.Entry(row_c, textvariable=self._c_gauss_var, width=8).pack(side=tk.RIGHT, padx=2)

        row_eps = ttk.Frame(sec5)
        row_eps.pack(fill=tk.X, pady=2)
        ttk.Label(row_eps, text="ε (eV):", font=F['body']).pack(side=tk.LEFT)
        self._epsilon_var = tk.StringVar(value="0.97")
        ttk.Entry(row_eps, textvariable=self._epsilon_var, width=8).pack(side=tk.RIGHT, padx=2)

        self._auto_fit_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(sec5, text="μ_dry 데이터로 자동 피팅",
                         variable=self._auto_fit_var).pack(fill=tk.X, pady=2)

        # ── Calculate All Button ──
        sec6 = self._make_section(content, "5) 일괄 계산")
        self._calc_button = ttk.Button(sec6, text="★ 전체 계산 실행 ★",
                                       command=self._on_calculate_all)
        self._calc_button.pack(fill=tk.X, pady=4, ipady=6)

        self._progress_var = tk.DoubleVar(value=0)
        self._progress_bar = ttk.Progressbar(sec6, variable=self._progress_var, maximum=100)
        self._progress_bar.pack(fill=tk.X, pady=2)

        self._calc_status_var = tk.StringVar(value="대기 중")
        ttk.Label(sec6, textvariable=self._calc_status_var, font=F['small'],
                  foreground=C['text_secondary']).pack(fill=tk.X, pady=2)

        # ── Right panel (sub-notebook with multiple plot tabs) ──
        right = ttk.Frame(main)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._plot_notebook = ttk.Notebook(right)
        self._plot_notebook.pack(fill=tk.BOTH, expand=True)

        # Create sub-tabs with individual figures
        self._plot_tabs = {}  # key -> (fig, canvas)
        _plot_defs = [
            ('input_data',   '입력 데이터'),
            ('results',      '마찰 결과'),
            ('cold_hot',     'Cold & Hot Branch'),
            ('flash_temp',   'Flash Temperature'),
        ]
        for key, label in _plot_defs:
            tab_frame = ttk.Frame(self._plot_notebook)
            self._plot_notebook.add(tab_frame, text=f'  {label}  ')
            fig = Figure(dpi=100, facecolor='white')
            canvas = FigureCanvasTkAgg(fig, tab_frame)
            toolbar = NavigationToolbar2Tk(canvas, tab_frame)
            toolbar.update()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self._plot_tabs[key] = (fig, canvas)

            # Auto-resize figure to widget on <Configure>
            _after_id = [None]
            widget = canvas.get_tk_widget()
            def _on_resize(event, _fig=fig, _cv=canvas, _w=widget, _aid=_after_id):
                w, h = event.width, event.height
                if w > 1 and h > 1:
                    _fig.set_size_inches(w / _fig.dpi, h / _fig.dpi, forward=False)
                    try:
                        _fig.tight_layout(pad=0.5, h_pad=0.6, w_pad=0.4)
                    except Exception:
                        pass
                    if _aid[0] is not None:
                        try:
                            _w.after_cancel(_aid[0])
                        except Exception:
                            pass
                    _aid[0] = _w.after(50, lambda: _cv.draw_idle())
            widget.bind('<Configure>', _on_resize, add='+')

        # Legacy single fig/canvas references (for backward compatibility)
        self._fig, self._canvas = self._plot_tabs['results']

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

    def _refresh_surface_q1_list(self):
        """surface_q1 프리셋 목록 갱신."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        preset_dir = os.path.join(base_dir, 'preset_data', 'surface_q1')
        items = ['(직접 입력)']
        if os.path.isdir(preset_dir):
            items += sorted([f.replace('.txt', '') for f in os.listdir(preset_dir) if f.endswith('.txt')])
        self._surface_q1_combo['values'] = items

    def _on_surface_q1_selected(self, event=None):
        """노면 프리셋 선택 시 q_max/q1 값 적용."""
        selected = self._surface_q1_combo_var.get()
        if selected == '(직접 입력)':
            return
        base_dir = os.path.dirname(os.path.abspath(__file__))
        fp = os.path.join(base_dir, 'preset_data', 'surface_q1', selected + '.txt')
        if not os.path.exists(fp):
            return
        try:
            q_max_val = None
            q1_val = None
            with open(fp, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('#') or not line:
                        continue
                    if line.startswith('q_max='):
                        q_max_val = line.split('=', 1)[1].strip()
                    elif line.startswith('q1='):
                        q1_val = line.split('=', 1)[1].strip()
            if q1_val:
                self._q1_var.set(q1_val)
            if q_max_val:
                self._q1_var.set(q_max_val)  # q_max = q1 for upper cutoff
            # 자동 h'rms 계산
            self._calc_hrms_from_q1()
        except Exception as e:
            print(f"[surface_q1 preset] error: {e}")

    # ================================================================
    #  h'rms calculation from q1
    # ================================================================
    def _calc_hrms_from_q1(self):
        """PSD와 q1으로부터 h'rms (ξ) 계산.
        ξ²(q1) = 2π ∫[q0→q1] k³ C(k) dk
        """
        if self.psd_model is None:
            self._hrms_var.set("ξ = - (PSD 먼저 로드)")
            return
        try:
            q0 = float(self._q0_var.get())
            q1 = float(self._q1_var.get())
            n_pts = 500

            q_arr = np.logspace(np.log10(q0), np.log10(q1), n_pts)
            C_arr = self.psd_model(q_arr)

            # ξ² = 2π ∫ q³ C(q) dq  (log-spaced trapezoid)
            integrand = q_arr ** 3 * C_arr
            xi_sq = 2 * np.pi * np.trapz(integrand, q_arr)
            xi = np.sqrt(max(xi_sq, 0))

            self._hrms_var.set(f"ξ = {xi:.4f}")
            print(f"[DataInput] h'rms: q0={q0:.1e}, q1={q1:.1e} → ξ={xi:.4f}")
        except Exception as e:
            self._hrms_var.set(f"ξ = 오류")
            print(f"[DataInput] h'rms calc error: {e}")

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
            omega, E_stor, E_loss = self._read_dma_auto_unit(fp)
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
            self._plot_input_data()
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
            self._plot_input_data()
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
            self._plot_input_data()
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
            self._plot_input_data()
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

        # Must run on main thread because existing GUI methods update UI widgets
        self.app.root.after(50, self._calculate_all_sequential, 0)

    def _calculate_all_sequential(self, ci):
        """Process compound ci using existing pipeline (main thread).
        Called sequentially via root.after() for each compound."""
        app = self.app
        n_compounds = len(self.compounds)

        if ci >= n_compounds:
            # All done
            self._progress_var.set(100)
            self._calc_status_var.set(f"계산 완료 ({n_compounds}개 컴파운드)")
            self._calculating = False
            self._calc_button.config(state='normal')

            # Collect results
            app.all_compound_results = [cpd.results for cpd in self.compounds]
            app.compound_data = self.compounds
            app.data_input_finalized = True
            if self.psd_model is not None:
                app.shared_psd_model = self.psd_model

            self._after_calculation()
            return

        cpd = self.compounds[ci]
        self._calc_status_var.set(f"[{ci+1}/{n_compounds}] {cpd.name}...")
        self._progress_var.set(ci / n_compounds * 100)
        app.root.update()

        try:
            self._run_pipeline_for_compound(ci, cpd)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._calc_status_var.set(f"오류 ({cpd.name}): {e}")
            self._calculating = False
            self._calc_button.config(state='normal')
            return

        # Next compound
        app.root.after(50, self._calculate_all_sequential, ci + 1)

    def _run_pipeline_for_compound(self, ci, cpd):
        """Run the full pipeline for one compound, reusing existing app methods.

        Mirrors the '시험 Run' pipeline in Friction Map tab:
        1) Set PSD → tab0_finalized
        2) Set material + aT → tab1_finalized
        3) Set calculation params → _run_calculation() → G(q,v)
        4) Set strain/f,g data
        5) _calculate_mu_visc()
        6) Load mu_dry → _run_auto_fit_mu_dry() → _complete_adh_fitting()
        """
        app = self.app
        n_compounds = len(self.compounds)

        # ── Step 1: PSD 설정 (시험 Run과 동일한 방식) ──
        # 시험 Run: _load_preset_psd() → _apply_profile_psd_to_tab3()
        #   → MeasuredPSD(q, C) [linear 보간, 기본값]
        # 데이터 입력 탭: create_psd_from_data() → MeasuredPSD(q, C, 'log-log')
        # → 동일하게 맞추기: raw 데이터로 MeasuredPSD 재생성 (linear 보간)
        from persson_model.core.psd_models import MeasuredPSD as _MeasuredPSD
        if hasattr(self.psd_model, 'q_data') and self.psd_model.q_data is not None:
            app.psd_model = _MeasuredPSD(self.psd_model.q_data, self.psd_model.C_data)
            # q_min을 PSD 실제 범위로 설정 (시험 Run의 _apply_profile_psd_to_tab3 line 2645 동일)
            psd_q_min = float(self.psd_model.q_data[0])
            user_q_min = float(self._q0_var.get())
            if user_q_min < psd_q_min:
                print(f"[DataInput] q₀ 보정: {user_q_min:.2e} → {psd_q_min:.2e} (PSD 범위에 맞춤)")
                app.q_min_var.set(f"{psd_q_min:.2e}")
        else:
            app.psd_model = self.psd_model
        app.tab0_finalized = True
        app.psd_source = "데이터 입력 탭"

        # ── Step 2: Material 설정 ──
        app.material = cpd.material
        app.tab1_finalized = True
        app.material_source = f"데이터 입력: {cpd.name}"

        # ── DEBUG: material modulus 값 확인 ──
        try:
            E_test = cpd.material.get_modulus(2 * np.pi * 1.0, temperature=20.0)
            E_stor_test = cpd.material.get_storage_modulus(2 * np.pi * 1.0, temperature=20.0)
            print(f"[DataInput] {cpd.name} material check: "
                  f"|E*|(1Hz)={abs(E_test):.3e} Pa, E'(1Hz)={E_stor_test:.3e} Pa")
            if abs(E_test) > 1e10:
                print(f"[DataInput] WARNING: |E*| > 1e10 Pa — 단위가 과대할 수 있음!")
        except Exception as e_dbg:
            print(f"[DataInput] material debug error: {e_dbg}")

        # aT 시프트 팩터 설정
        if cpd.aT_interp is not None:
            app.persson_aT_interp = cpd.aT_interp
            app.persson_aT_data = cpd.aT_data
            if cpd.bT_interp is not None:
                app.persson_bT_interp = cpd.bT_interp
            # persson_master_curve 설정 (Flash Temperature aT 시프트에 필요)
            if cpd.material is not None:
                mat = cpd.material
                freq = getattr(mat, '_frequencies', None)
                E_s = getattr(mat, '_storage_modulus', None)
                E_l = getattr(mat, '_loss_modulus', None)
                if freq is not None and E_s is not None:
                    if E_l is None:
                        E_l = np.zeros_like(E_s)
                    app.persson_master_curve = {
                        'omega': freq.copy(),
                        'E_storage': E_s.copy(),
                        'E_loss': E_l.copy(),
                    }
        else:
            app.persson_aT_interp = None
            app.persson_aT_data = None

        # ── Step 3: 계산 파라미터 설정 + G(q,v) 계산 ──
        # 시험 Run과 동일하게 surface_q1 프리셋 적용
        q1_str = self._q1_var.get()
        app.sigma_0_var.set(self._sigma0_var.get())
        app.temperature_var.set(self._temp_var.get())
        app.v_min_var.set(self._v_min_var.get())
        app.v_max_var.set(self._v_max_var.get())
        app.n_velocity_var.set(self._n_v_var.get())
        app.q_min_var.set(self._q0_var.get())
        app.q_max_var.set(q1_str)
        app.n_q_var.set(self._n_q_var.get())
        app.gamma_var.set(self._gamma_var.get())
        app.n_phi_var.set(self._n_phi_var.get())

        # input_q1_var 동기화 (h'rms 계산에서 사용)
        if hasattr(app, 'input_q1_var'):
            app.input_q1_var.set(q1_str)

        # q1→h'rms 모드 설정 + h'rms 계산 (시험 Run Step 3과 동일)
        if hasattr(app, 'hrms_q1_mode_var'):
            app.hrms_q1_mode_var.set("q1_to_hrms")
            if hasattr(app, '_on_hrms_q1_mode_changed'):
                app._on_hrms_q1_mode_changed()
            app.root.update_idletasks()
            if hasattr(app, '_calculate_hrms_q1'):
                try:
                    app._calculate_hrms_q1()
                except Exception as e_hrms:
                    print(f"[DataInput] h'rms 자동 계산 건너뜀: {e_hrms}")

        # _calculate_hrms_q1()가 q_max_var를 갱신했을 수 있으므로 최종 확인
        final_q_max = app.q_max_var.get()
        if hasattr(app, 'calculated_q1') and app.calculated_q1 is not None:
            final_q_max = f"{app.calculated_q1:.6e}"
            app.q_max_var.set(final_q_max)
        print(f"[DataInput] 최종 q_max={final_q_max} (calculated_q1={getattr(app, 'calculated_q1', 'N/A')})")

        self._calc_status_var.set(f"[{ci+1}/{n_compounds}] {cpd.name}: G(q,v)...")
        app.root.update()
        app._run_calculation()
        app.root.update()

        # ── Step 4: Strain / f,g 설정 ──
        if cpd.f_interpolator is not None:
            app.strain_data = cpd.strain_data
            app.fg_by_T = cpd.fg_by_T
            app.f_interpolator = cpd.f_interpolator
            app.g_interpolator = cpd.g_interpolator

            # fg_averaged 설정 (mu_visc에서 사용될 수 있음)
            if cpd.fg_raw is not None:
                app.fg_averaged = cpd.fg_raw
            elif cpd.fg_by_T is not None:
                try:
                    app.fg_averaged = average_fg_curves(cpd.fg_by_T)
                except Exception:
                    pass
        else:
            app.strain_data = None
            app.fg_by_T = None
            app.f_interpolator = None
            app.g_interpolator = None
            app.fg_averaged = None

        # 비선형 보정 설정
        app.use_fg_correction_var.set(self._use_nonlinear_var.get()
                                      and cpd.f_interpolator is not None)
        # Flash temperature
        app.use_flash_temp_var.set(self._use_flash_var.get())

        # ── Step 5: mu_visc 계산 ──
        self._calc_status_var.set(f"[{ci+1}/{n_compounds}] {cpd.name}: μ_visc...")
        app.root.update()
        app._calculate_mu_visc()
        app.root.update()

        # ── Step 6: mu_adh (mu_dry 있을 때) ──
        if cpd.mu_dry_data is not None:
            self._calc_status_var.set(f"[{ci+1}/{n_compounds}] {cpd.name}: μ_adh 피팅...")
            app.root.update()

            # mu_dry 데이터를 Treeview에 로드 (기존 피팅이 읽는 곳)
            log_v_dry, mu_dry_vals = cpd.mu_dry_data
            if hasattr(app, 'meas_mu_tree'):
                # 기존 데이터 삭제
                for item in app.meas_mu_tree.get_children():
                    app.meas_mu_tree.delete(item)
                # 새 데이터 삽입
                for lv, mv in zip(log_v_dry, mu_dry_vals):
                    app.meas_mu_tree.insert('', tk.END,
                                            values=(f"{lv:.4f}", f"{mv:.6f}"))

            # mu_adh 초기 파라미터 설정
            if hasattr(app, 'adh_tau_f0_var'):
                app.adh_tau_f0_var.set(self._tau_f0_var.get())
            if hasattr(app, 'adh_v0_star_var'):
                app.adh_v0_star_var.set(self._v0_star_var.get())
            if hasattr(app, 'adh_c_var'):
                app.adh_c_var.set(self._c_gauss_var.get())
            if hasattr(app, 'adh_epsilon_var'):
                app.adh_epsilon_var.set(self._epsilon_var.get())

            # 피팅 상태만 리셋 (mu_visc_results는 보존!)
            # _reset_adh_fitting()은 mu_visc_results를 None으로 지우므로 호출하면 안 됨
            app._adh_fitting_completed = False
            app._fit_results = None
            app.mu_adh_results = None

            if self._auto_fit_var.get() and hasattr(app, '_run_auto_fit_mu_dry'):
                app._run_auto_fit_mu_dry()
                app.root.update()
            if hasattr(app, '_complete_adh_fitting'):
                app._complete_adh_fitting()
                app.root.update()

        # ── Snapshot results ──
        v_array = None
        mu_visc = None
        mu_adh = None
        mu_total = None

        if app.mu_visc_results is not None:
            v_array = app.mu_visc_results.get('v', None)
            use_flash = app.mu_visc_results.get('use_flash', False)

            # Cold branch (항상 존재)
            mu_visc_cold = app.mu_visc_results.get('mu', None)
            if mu_visc_cold is None:
                mu_visc_cold = app.mu_visc_results.get('mu_visc', None)

            # Hot branch (flash temp 활성 시)
            mu_visc_hot = app.mu_visc_results.get('mu_hot', None) if use_flash else None

            # 대표값: flash 있으면 hot, 없으면 cold
            mu_visc = mu_visc_hot if mu_visc_hot is not None else mu_visc_cold

        if app.mu_adh_results is not None:
            mu_adh = app.mu_adh_results.get('mu_adh', None)

        if v_array is not None and mu_visc is not None:
            if mu_adh is not None and len(mu_adh) == len(mu_visc):
                mu_total = mu_visc + mu_adh
            else:
                mu_adh = np.zeros_like(mu_visc)
                mu_total = mu_visc.copy()

            # A/A0
            A_A0_cold = app.mu_visc_results.get('P_qmax', np.zeros_like(mu_visc))
            A_A0_hot = app.mu_visc_results.get('A_A0_hot', None) if use_flash else None
            A_A0 = A_A0_hot if A_A0_hot is not None else A_A0_cold

            cpd.results = {
                'v': v_array,
                'mu_visc': mu_visc,
                'mu_adh': mu_adh,
                'mu_total': mu_total,
                'A_A0': A_A0,
                'sigma_0': float(self._sigma0_var.get()) * 1e6,
                'temperature': float(self._temp_var.get()),
                'use_flash': use_flash,
            }

            # Cold & Hot Branch 데이터 (10번 탭과 동일 구조)
            if mu_visc_cold is not None:
                mu_adh_cold = mu_adh  # cold adh = 기본 adh 피팅 결과
                cpd.results['mu_cold_hys'] = mu_visc_cold
                cpd.results['mu_cold_adh'] = mu_adh_cold
                cpd.results['mu_cold_total'] = mu_visc_cold + mu_adh_cold
                cpd.results['A_A0_cold'] = A_A0_cold

                # tau_f_cold (전단응력): mu_adh 탭의 원시 tau_f
                tau_f_cold = app.mu_adh_results.get('tau_f', None) if app.mu_adh_results else None
                if tau_f_cold is not None and len(tau_f_cold) == len(v_array):
                    cpd.results['tau_f_cold'] = tau_f_cold.copy()
                elif tau_f_cold is not None:
                    # 보간
                    v_adh = app.mu_adh_results.get('v', v_array)
                    cpd.results['tau_f_cold'] = np.interp(
                        np.log10(np.maximum(v_array, 1e-30)),
                        np.log10(np.maximum(v_adh, 1e-30)),
                        tau_f_cold)

            if mu_visc_hot is not None:
                # Hot adhesion: A_A0_hot 기반 재계산
                # flash_results에서 T_hot 배열을 가져와 Arrhenius 시프트 적용
                flash_results = app.mu_visc_results.get('flash_results', None)
                if A_A0_hot is not None and flash_results is not None:
                    T_hot_arr = flash_results.get('T_hot', None)
                    delta_T_arr = flash_results.get('delta_T', None)
                    cpd.results['T_hot'] = T_hot_arr
                    cpd.results['delta_T'] = delta_T_arr
                    cpd.results['flash_results'] = flash_results

                    # Hot adhesion: aT' Arrhenius 시프트 적용
                    mu_adh_hot, tau_f_hot = self._calc_hot_adhesion(
                        app, cpd, v_array, A_A0_hot, T_hot_arr)
                    cpd.results['mu_hot_hys'] = mu_visc_hot
                    cpd.results['mu_hot_adh'] = mu_adh_hot
                    cpd.results['mu_hot_total'] = mu_visc_hot + mu_adh_hot
                    cpd.results['A_A0_hot'] = A_A0_hot
                    cpd.results['tau_f_hot'] = tau_f_hot
                    # 대표값도 hot total로 갱신
                    cpd.results['mu_total'] = mu_visc_hot + mu_adh_hot
                else:
                    # flash_results 없으면 cold adh 그대로
                    cpd.results['mu_hot_hys'] = mu_visc_hot
                    cpd.results['mu_hot_adh'] = mu_adh
                    cpd.results['mu_hot_total'] = mu_visc_hot + mu_adh
                    cpd.results['A_A0_hot'] = A_A0_hot if A_A0_hot is not None else A_A0_cold

            # Copy adh fit params if available
            if app.mu_adh_results is not None:
                cpd.results['adh_params'] = app.mu_adh_results.get('params', None)
        else:
            cpd.results = None
            print(f"[WARNING] {cpd.name}: mu_visc_results가 비어있습니다.")

    def _calc_hot_adhesion(self, app, cpd, v_array, A_A0_hot, T_hot_arr):
        """Hot adhesion: Arrhenius aT 시프트로 τ_f(T_hot) 계산 → μ_adh_hot.

        10번 탭(Cold & Hot Branch)과 **완전히 동일한 로직**:
        - 피팅된 adh_params (τ_f0, v0*, c, ε, T_ref) 사용
        - per-velocity Arrhenius 시프트: aT'(T_hot(v))
        - vectorized 계산 (for 루프 제거)

        Returns (mu_adh_hot, tau_f_hot) tuple.
        """
        zeros = np.zeros_like(v_array)
        if T_hot_arr is None or A_A0_hot is None:
            return zeros, zeros

        # ── 피팅된 파라미터 사용 (Tab 10과 동일) ──
        adh_params = app.mu_adh_results.get('params') if app.mu_adh_results else None
        if adh_params is None:
            return zeros, zeros

        tau_f0 = adh_params['tau_f0']       # Pa (이미 Pa 단위)
        v0_star = adh_params['v0_star']     # m/s
        c_val = adh_params['c']             # Gaussian width
        epsilon = adh_params['epsilon']     # eV
        T_ref_K = adh_params['T_ref']       # K (피팅 시 사용된 기준 온도)
        p0 = adh_params['p0']              # Pa
        k_B = 8.6173e-5                     # eV/K

        if p0 <= 0:
            return zeros, zeros

        # Per-velocity Arrhenius shift: aT'(T_hot(v))
        T_hot_arr_K = np.where(T_hot_arr > -200, T_hot_arr + 273.15, T_ref_K)
        aT_prime_hot = np.exp((epsilon / k_B) * (1.0 / T_hot_arr_K - 1.0 / T_ref_K))

        # τ_f_hot: per-velocity Arrhenius shifted Gaussian
        v_eff_hot = v_array * aT_prime_hot
        log_ratio_hot = np.log10(np.maximum(v_eff_hot, 1e-30) / v0_star)
        tau_f_hot = tau_f0 * np.exp(-c_val * log_ratio_hot**2)

        mu_adh_hot = (A_A0_hot * tau_f_hot) / p0

        return mu_adh_hot, tau_f_hot

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
        """Draw all preview plots across sub-tabs."""
        self._plot_input_data()
        self._plot_results()
        self._plot_cold_hot()
        self._plot_flash_temp()

    @staticmethod
    def _style_ax(ax, title='', xlabel='', ylabel='', xscale=None, yscale=None):
        """Apply consistent compact styling to an axis."""
        FS = DataInputTab.PLOT_FONT_SIZE
        if title:
            ax.set_title(title, fontsize=FS, fontweight='bold', pad=4)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=FS, labelpad=3)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=FS, labelpad=3)
        if xscale:
            ax.set_xscale(xscale)
        if yscale:
            ax.set_yscale(yscale)
        ax.tick_params(axis='both', labelsize=FS, pad=2,
                       length=3, width=0.6)
        ax.grid(True, alpha=0.3, linewidth=0.5)

    def _finalize_data_plot(self, fig, canvas):
        """Figure 크기를 위젯에 맞추고 tight_layout 적용 후 draw."""
        try:
            w = canvas.get_tk_widget().winfo_width()
            h = canvas.get_tk_widget().winfo_height()
            if w > 1 and h > 1:
                fig.set_size_inches(w / fig.dpi, h / fig.dpi, forward=False)
            fig.tight_layout(pad=0.5, h_pad=0.6, w_pad=0.4)
        except Exception:
            pass
        canvas.draw_idle()

    # ── Tab 1: 입력 데이터 (E'/E'' + aT + f,g) ──
    def _plot_input_data(self):
        """3×2 grid: E', E'', log(aT), bT, f(strain), g(strain)."""
        FS = self.PLOT_FONT_SIZE
        LW = self.PLOT_LINE_WIDTH
        fig, canvas = self._plot_tabs['input_data']
        fig.clear()

        ax_Es  = fig.add_subplot(3, 2, 1)
        ax_El  = fig.add_subplot(3, 2, 2)
        ax_aT  = fig.add_subplot(3, 2, 3)
        ax_bT  = fig.add_subplot(3, 2, 4)
        ax_f   = fig.add_subplot(3, 2, 5)
        ax_g   = fig.add_subplot(3, 2, 6)

        self._style_ax(ax_Es, "E'(ω)", 'ω (rad/s)', "E' (Pa)", 'log', 'log')
        self._style_ax(ax_El, "E''(ω)", 'ω (rad/s)', "E'' (Pa)", 'log', 'log')
        self._style_ax(ax_aT, 'log₁₀(aT)', 'T (°C)', 'log₁₀(aT)')
        self._style_ax(ax_bT, 'bT', 'T (°C)', 'bT')
        self._style_ax(ax_f,  'f(strain)', 'Strain', 'f')
        self._style_ax(ax_g,  'g(strain)', 'Strain', 'g')

        has_bT = False
        for i, cpd in enumerate(self.compounds):
            color = self.COMPOUND_COLORS[i % len(self.COMPOUND_COLORS)]
            lbl = cpd.name

            # E', E''
            if cpd.material is not None:
                mat = cpd.material
                if getattr(mat, '_frequencies', None) is not None:
                    omega = mat._frequencies
                    ax_Es.plot(omega, mat._storage_modulus, '-', color=color,
                               linewidth=LW, label=lbl)
                    E_loss = mat._loss_modulus if mat._loss_modulus is not None else np.zeros_like(omega)
                    ax_El.plot(omega, E_loss, '-', color=color, linewidth=LW, label=lbl)

            # aT, bT
            if cpd.aT_data is not None:
                d = cpd.aT_data
                ax_aT.plot(d['T'], d['log_aT'], 'o-', color=color, markersize=4,
                           linewidth=LW, label=f"{lbl} (Tref={d['T_ref']:.0f}°C)")
                if d.get('has_bT', False):
                    ax_bT.plot(d['T'], d['bT'], 's-', color=color, markersize=4,
                               linewidth=LW, label=lbl)
                    has_bT = True

            # f, g
            if cpd.fg_raw is not None:
                strain = cpd.fg_raw.get('strain')
                if strain is not None:
                    f_v = cpd.fg_raw.get('f')
                    g_v = cpd.fg_raw.get('g')
                    if f_v is not None:
                        ax_f.plot(strain, f_v, 'o-', color=color, markersize=3,
                                  linewidth=LW, label=lbl)
                    if g_v is not None:
                        ax_g.plot(strain, g_v, 's-', color=color, markersize=3,
                                  linewidth=LW, label=lbl)
            elif cpd.f_interpolator is not None:
                strain = np.linspace(0.001, 1.0, 200)
                try:
                    ax_f.plot(strain, cpd.f_interpolator(strain), '-', color=color,
                              linewidth=LW, label=lbl)
                except Exception:
                    pass
                if cpd.g_interpolator is not None:
                    try:
                        ax_g.plot(strain, cpd.g_interpolator(strain), '-', color=color,
                                  linewidth=LW, label=lbl)
                    except Exception:
                        pass

        if not has_bT:
            ax_bT.text(0.5, 0.5, 'bT 없음', ha='center', va='center',
                       fontsize=FS - 1, color='gray', transform=ax_bT.transAxes)

        for ax in [ax_Es, ax_El, ax_aT, ax_bT, ax_f, ax_g]:
            if ax.get_legend_handles_labels()[1]:
                ax.legend(fontsize=self.PLOT_LEGEND_SIZE, loc='best',
                          handlelength=1.2, handletextpad=0.4,
                          borderpad=0.3, labelspacing=0.2)
        self._finalize_data_plot(fig, canvas)

    # ── Tab 2: 마찰 결과 (A/A0 + hys/adh + total) ──
    def _plot_results(self):
        """3×2: A/A0, μ_hys, τ_f, μ_adh, μ_total, peak 비교."""
        FS = self.PLOT_FONT_SIZE
        LW = self.PLOT_LINE_WIDTH
        LWs = self.PLOT_LINE_WIDTH_SUB
        fig, canvas = self._plot_tabs['results']
        fig.clear()

        cpds = [c for c in self.compounds if c.results is not None]
        if not cpds:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, '계산 실행 후 표시됩니다', ha='center', va='center',
                    fontsize=12, color='gray', transform=ax.transAxes)
            ax.set_axis_off()
            canvas.draw_idle()
            return

        ax1 = fig.add_subplot(3, 2, 1)
        ax2 = fig.add_subplot(3, 2, 2)
        ax3 = fig.add_subplot(3, 2, 3)
        ax4 = fig.add_subplot(3, 2, 4)
        ax5 = fig.add_subplot(3, 2, 5)
        ax6 = fig.add_subplot(3, 2, 6)

        self._style_ax(ax1, 'A/A₀', 'v (m/s)', 'A/A₀', 'log')
        self._style_ax(ax2, 'μ_hys', 'v (m/s)', 'μ_hys', 'log')
        self._style_ax(ax3, 'τ_f (전단응력)', 'v (m/s)', 'τ_f (MPa)', 'log')
        self._style_ax(ax4, 'μ_adh', 'v (m/s)', 'μ_adh', 'log')
        self._style_ax(ax5, 'μ_total', 'v (m/s)', 'μ', 'log')
        self._style_ax(ax6, 'μ_total 피크 비교', '', 'μ_peak')

        peak_names = []
        peak_vals = []

        for i, cpd in enumerate(cpds):
            color = self.COMPOUND_COLORS[i % len(self.COMPOUND_COLORS)]
            r = cpd.results

            if r.get('A_A0') is not None:
                ax1.plot(r['v'], r['A_A0'], '-', color=color, linewidth=LW, label=cpd.name)
            ax2.plot(r['v'], r['mu_visc'], '-', color=color, linewidth=LW, label=cpd.name)

            # τ_f 플롯
            tau_f = r.get('tau_f_cold')
            if tau_f is not None:
                ax3.plot(r['v'], tau_f / 1e6, '-', color=color, linewidth=LW, label=cpd.name)

            if r.get('mu_adh') is not None:
                ax4.plot(r['v'], r['mu_adh'], '-', color=color, linewidth=LW, label=cpd.name)

            ax5.plot(r['v'], r['mu_total'], '-', color=color, linewidth=LW, label=cpd.name)
            ax5.plot(r['v'], r['mu_visc'], '--', color=color, linewidth=LWs, alpha=0.5)
            if r.get('mu_adh') is not None:
                ax5.plot(r['v'], r['mu_adh'], ':', color=color, linewidth=LWs, alpha=0.5)

            # 피크 비교
            peak_names.append(cpd.name)
            peak_vals.append(np.max(r['mu_total']))

        # 바 차트: μ_total 피크 비교
        if peak_names:
            colors = [self.COMPOUND_COLORS[i % len(self.COMPOUND_COLORS)]
                      for i in range(len(peak_names))]
            bars = ax6.bar(range(len(peak_names)), peak_vals, color=colors, alpha=0.8)
            ax6.set_xticks(range(len(peak_names)))
            ax6.set_xticklabels(peak_names, fontsize=FS, rotation=30, ha='right')
            for bar, val in zip(bars, peak_vals):
                ax6.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                         f'{val:.3f}', ha='center', va='bottom', fontsize=FS)

        for ax in [ax1, ax2, ax3, ax4, ax5]:
            if ax.get_legend_handles_labels()[1]:
                ax.legend(fontsize=self.PLOT_LEGEND_SIZE, loc='best',
                          handlelength=1.2, handletextpad=0.4,
                          borderpad=0.3, labelspacing=0.2)
        self._finalize_data_plot(fig, canvas)

    # ── Tab 3: Cold & Hot Branch ──
    def _plot_cold_hot(self):
        """3×2: μ_hys C/H, A/A0 C/H, τ_f C/H, μ_adh C/H, μ_total C/H, 피크 비교."""
        FS = self.PLOT_FONT_SIZE
        LW = self.PLOT_LINE_WIDTH
        LWs = self.PLOT_LINE_WIDTH_SUB
        fig, canvas = self._plot_tabs['cold_hot']
        fig.clear()

        has_hot = any(c.results is not None and c.results.get('mu_hot_hys') is not None
                      for c in self.compounds)
        if not has_hot:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5,
                    'Flash Temperature 활성화 후 계산하세요',
                    ha='center', va='center', fontsize=FS, color='gray',
                    transform=ax.transAxes)
            ax.set_axis_off()
            canvas.draw_idle()
            return

        ax_hys   = fig.add_subplot(3, 2, 1)
        ax_aa0   = fig.add_subplot(3, 2, 2)
        ax_tau   = fig.add_subplot(3, 2, 3)
        ax_adh   = fig.add_subplot(3, 2, 4)
        ax_total = fig.add_subplot(3, 2, 5)
        ax_peak  = fig.add_subplot(3, 2, 6)

        self._style_ax(ax_hys,   'μ_hys: Cold vs Hot',   'v (m/s)', 'μ_hys', 'log')
        self._style_ax(ax_aa0,   'A/A₀: Cold vs Hot',    'v (m/s)', 'A/A₀', 'log')
        self._style_ax(ax_tau,   'τ_f: Cold vs Hot',     'v (m/s)', 'τ_f (MPa)', 'log')
        self._style_ax(ax_adh,   'μ_adh: Cold vs Hot',   'v (m/s)', 'μ_adh', 'log')
        self._style_ax(ax_total, 'μ_total: Cold vs Hot', 'v (m/s)', 'μ_total', 'log')
        self._style_ax(ax_peak,  'μ_peak: Cold vs Hot',  '', 'μ_peak')

        peak_cold_names = []
        peak_cold_vals = []
        peak_hot_vals = []

        for i, cpd in enumerate(self.compounds):
            if cpd.results is None:
                continue
            r = cpd.results
            v = r['v']
            color = self.COMPOUND_COLORS[i % len(self.COMPOUND_COLORS)]
            lbl = cpd.name

            # μ_hys
            for key, style in [('mu_cold_hys', '-'), ('mu_hot_hys', '--')]:
                d = r.get(key)
                if d is not None:
                    branch = 'Cold' if 'cold' in key else 'Hot'
                    ax_hys.plot(v, d, style, color=color, linewidth=LW if style == '-' else LWs,
                                label=f'{lbl} {branch}')

            # A/A0
            for key, style in [('A_A0_cold', '-'), ('A_A0_hot', '--')]:
                d = r.get(key)
                if d is not None:
                    branch = 'Cold' if 'cold' in key.lower() else 'Hot'
                    ax_aa0.plot(v, d, style, color=color, linewidth=LW if style == '-' else LWs,
                                label=f'{lbl} {branch}')

            # τ_f (전단응력) Cold vs Hot
            tau_cold = r.get('tau_f_cold')
            tau_hot = r.get('tau_f_hot')
            if tau_cold is not None:
                ax_tau.plot(v, tau_cold / 1e6, '-', color=color, linewidth=LW,
                            label=f'{lbl} Cold')
            if tau_hot is not None:
                ax_tau.plot(v, tau_hot / 1e6, '--', color=color, linewidth=LWs,
                            label=f'{lbl} Hot')

            # μ_adh (= A/A0 × τ_f / p0)
            for key, style in [('mu_cold_adh', '-'), ('mu_hot_adh', '--')]:
                d = r.get(key)
                if d is not None:
                    branch = 'Cold' if 'cold' in key else 'Hot'
                    ax_adh.plot(v, d, style, color=color, linewidth=LW if style == '-' else LWs,
                                label=f'{lbl} {branch}')

            # μ_total
            for key, style in [('mu_cold_total', '-'), ('mu_hot_total', '--')]:
                d = r.get(key)
                if d is not None:
                    branch = 'Cold' if 'cold' in key else 'Hot'
                    ax_total.plot(v, d, style, color=color, linewidth=LW if style == '-' else LWs,
                                  label=f'{lbl} {branch}')

            # 피크 수집
            mu_c = r.get('mu_cold_total')
            mu_h = r.get('mu_hot_total')
            if mu_c is not None and mu_h is not None:
                peak_cold_names.append(lbl)
                peak_cold_vals.append(np.max(mu_c))
                peak_hot_vals.append(np.max(mu_h))

        # 피크 비교 바 차트 (Cold vs Hot)
        if peak_cold_names:
            x = np.arange(len(peak_cold_names))
            w = 0.35
            colors = [self.COMPOUND_COLORS[i % len(self.COMPOUND_COLORS)]
                      for i in range(len(peak_cold_names))]
            bars_c = ax_peak.bar(x - w/2, peak_cold_vals, w, color=colors, alpha=0.7, label='Cold')
            bars_h = ax_peak.bar(x + w/2, peak_hot_vals, w, color=colors, alpha=0.4,
                                 edgecolor=colors, linewidth=1.5, label='Hot', hatch='//')
            ax_peak.set_xticks(x)
            ax_peak.set_xticklabels(peak_cold_names, fontsize=FS, rotation=30, ha='right')
            for bar, val in zip(bars_c, peak_cold_vals):
                ax_peak.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                             f'{val:.3f}', ha='center', va='bottom', fontsize=FS)
            for bar, val in zip(bars_h, peak_hot_vals):
                ax_peak.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                             f'{val:.3f}', ha='center', va='bottom', fontsize=FS)
            ax_peak.legend(fontsize=self.PLOT_LEGEND_SIZE)

        for ax in [ax_hys, ax_aa0, ax_tau, ax_adh, ax_total]:
            if ax.get_legend_handles_labels()[1]:
                ax.legend(fontsize=self.PLOT_LEGEND_SIZE, loc='best',
                          handlelength=1.2, handletextpad=0.4,
                          borderpad=0.3, labelspacing=0.2)
        self._finalize_data_plot(fig, canvas)

    # ── Tab 4: Flash Temperature (T_hot, 파수별 누적, 히트맵) ──
    def _plot_flash_temp(self):
        """3×2: ΔT(v), T_hot(v), ΔT(q) 누적, ΔT(q,v) 히트맵, ζ vs ΔT, A/A0 C/H."""
        import matplotlib.pyplot as _plt
        FS = self.PLOT_FONT_SIZE
        LW = self.PLOT_LINE_WIDTH
        fig, canvas = self._plot_tabs['flash_temp']
        fig.clear()

        cpds = [c for c in self.compounds
                if c.results is not None and c.results.get('flash_results') is not None]
        if not cpds:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'Flash Temperature 계산 후 표시됩니다',
                    ha='center', va='center', fontsize=FS, color='gray',
                    transform=ax.transAxes)
            ax.set_axis_off()
            canvas.draw_idle()
            return

        ax_dT    = fig.add_subplot(3, 2, 1)
        ax_Thot  = fig.add_subplot(3, 2, 2)
        ax_dTq   = fig.add_subplot(3, 2, 3)
        ax_hmap  = fig.add_subplot(3, 2, 4)
        ax_aa0   = fig.add_subplot(3, 2, 5)
        ax_mu    = fig.add_subplot(3, 2, 6)

        self._style_ax(ax_dT,   'ΔT(v)', 'v (m/s)', 'ΔT (°C)', 'log')
        self._style_ax(ax_Thot, 'T_hot(v)', 'v (m/s)', 'T (°C)', 'log')
        self._style_ax(ax_dTq,  'ΔT(q) 파수별 누적', 'q (1/m)', 'ΔT (°C)', 'log')
        self._style_ax(ax_aa0,  'A/A₀: Cold vs Hot', 'v (m/s)', 'A/A₀', 'log')
        self._style_ax(ax_mu,   'μ: Cold vs Hot', 'v (m/s)', 'μ', 'log')

        hmap_drawn = False
        for idx, cpd in enumerate(cpds):
            color = self.COMPOUND_COLORS[idx % len(self.COMPOUND_COLORS)]
            r = cpd.results
            v = r['v']
            fr = r['flash_results']

            # ΔT(v)
            dT = fr.get('delta_T')
            if dT is not None:
                ax_dT.plot(v, dT, '-', color=color, linewidth=LW, label=cpd.name)

            # T_hot(v)
            T_hot = fr.get('T_hot')
            if T_hot is not None:
                ax_Thot.plot(v, T_hot, '-', color=color, linewidth=LW, label=cpd.name)
                T0 = r.get('temperature', 20.0)
                ax_Thot.axhline(y=T0, color=color, linestyle=':', linewidth=0.8, alpha=0.5)

            # ΔT(q) 누적 (대표 속도 몇 개만)
            dT_profile = fr.get('delta_T_profile')
            q_arr = fr.get('q_array')
            if dT_profile is not None and q_arr is not None:
                n_v_total = dT_profile.shape[1]
                n_curves = min(6, n_v_total)
                v_indices = np.linspace(0, n_v_total - 1, n_curves, dtype=int) if n_v_total > n_curves else np.arange(n_v_total)
                cmap = _plt.colormaps['coolwarm'].resampled(max(n_curves, 2))
                for k, j_idx in enumerate(v_indices):
                    c_k = cmap(k / max(n_curves - 1, 1))
                    ax_dTq.plot(q_arr, dT_profile[:, j_idx], '-', color=c_k, linewidth=1.2,
                                label=f'v={v[j_idx]:.2g}' if idx == 0 else None)

                # 히트맵 (첫 번째 컴파운드만)
                if not hmap_drawn and len(v) > 1:
                    log_v = np.log10(np.maximum(v, 1e-20))
                    log_q = np.log10(np.maximum(q_arr, 1e-20))
                    dT_max = np.max(dT_profile)
                    if dT_max > 0:
                        pcm = ax_hmap.pcolormesh(log_v, log_q, dT_profile,
                                                  cmap='hot', shading='auto',
                                                  vmin=0, vmax=dT_max)
                        cb = fig.colorbar(pcm, ax=ax_hmap, pad=0.02, fraction=0.046)
                        cb.set_label('ΔT (°C)', fontsize=FS)
                        cb.ax.tick_params(labelsize=FS)
                    ax_hmap.set_xlabel('log₁₀(v)', fontsize=FS, labelpad=3)
                    ax_hmap.set_ylabel('log₁₀(q)', fontsize=FS, labelpad=3)
                    ax_hmap.set_title(f'ΔT(q,v) 히트맵 ({cpd.name})',
                                      fontsize=FS, fontweight='bold', pad=4)
                    ax_hmap.tick_params(labelsize=FS)
                    hmap_drawn = True

            # A/A0 Cold vs Hot
            aa0_c = r.get('A_A0_cold')
            aa0_h = r.get('A_A0_hot')
            if aa0_c is not None:
                ax_aa0.plot(v, aa0_c, '-', color=color, linewidth=LW, label=f'{cpd.name} Cold')
            if aa0_h is not None:
                ax_aa0.plot(v, aa0_h, '--', color=color, linewidth=LW - 0.5, label=f'{cpd.name} Hot')

            # μ Cold vs Hot
            mu_c = r.get('mu_cold_total')
            mu_h = r.get('mu_hot_total')
            if mu_c is not None:
                ax_mu.plot(v, mu_c, '-', color=color, linewidth=LW, label=f'{cpd.name} Cold')
            if mu_h is not None:
                ax_mu.plot(v, mu_h, '--', color=color, linewidth=LW - 0.5, label=f'{cpd.name} Hot')

        if not hmap_drawn:
            ax_hmap.text(0.5, 0.5, 'ΔT profile 없음', ha='center', va='center',
                         fontsize=FS - 1, color='gray', transform=ax_hmap.transAxes)
            ax_hmap.set_axis_off()

        for ax in [ax_dT, ax_Thot, ax_dTq, ax_aa0, ax_mu]:
            if ax.get_legend_handles_labels()[1]:
                ax.legend(fontsize=self.PLOT_LEGEND_SIZE, loc='best',
                          handlelength=1.2, handletextpad=0.4,
                          borderpad=0.3, labelspacing=0.2)
        self._finalize_data_plot(fig, canvas)

    # ================================================================
    #  Dataset Preset (저장/불러오기)
    # ================================================================
    def _get_dataset_dir(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        d = os.path.join(base_dir, 'preset_data', 'dataset')
        os.makedirs(d, exist_ok=True)
        return d

    def _refresh_dataset_list(self):
        d = self._get_dataset_dir()
        files = sorted([f.replace('.json', '') for f in os.listdir(d) if f.endswith('.json')])
        self._dataset_combo['values'] = files if files else ['(없음)']

    def _save_dataset_preset(self):
        """현재 로드된 데이터셋(PSD + 컴파운드별 파일 경로)을 JSON으로 저장."""
        name = tk.simpledialog.askstring("데이터셋 저장", "프리셋 이름:")
        if not name:
            return

        dataset = {
            'name': name,
            'psd_file': self.psd_file,
            'compound_count': len(self.compounds),
            'params': {
                'sigma_0': self._sigma0_var.get(),
                'temperature': self._temp_var.get(),
                'v_min': self._v_min_var.get(),
                'v_max': self._v_max_var.get(),
                'n_v': self._n_v_var.get(),
                'q0': self._q0_var.get(),
                'q1': self._q1_var.get(),
                'n_q': self._n_q_var.get(),
                'gamma': self._gamma_var.get(),
                'n_phi': self._n_phi_var.get(),
                'nonlinear': self._use_nonlinear_var.get(),
                'flash_temp': self._use_flash_var.get(),
                'tau_f0': self._tau_f0_var.get(),
                'v0_star': self._v0_star_var.get(),
                'c_gauss': self._c_gauss_var.get(),
                'epsilon': self._epsilon_var.get(),
                'auto_fit': self._auto_fit_var.get(),
            },
            'compounds': [],
        }

        for cpd in self.compounds:
            c = {
                'name': cpd.name,
                'dma_file': cpd.dma_file,
                'aT_file': cpd.aT_file,
                'strain_mode': cpd.strain_mode,
                'strain_file': cpd.strain_file,
                'mu_dry_file': cpd.mu_dry_file,
            }
            dataset['compounds'].append(c)

        fp = os.path.join(self._get_dataset_dir(), name + '.json')
        with open(fp, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

        self._refresh_dataset_list()
        self._dataset_combo_var.set(name)
        messagebox.showinfo("저장 완료", f"데이터셋 '{name}' 저장 완료")

    def _load_dataset_preset(self):
        """저장된 데이터셋 프리셋을 불러와서 모든 파일을 자동 로드."""
        selected = self._dataset_combo_var.get()
        if not selected or selected == '(없음)' or selected == '(선택)':
            messagebox.showinfo("알림", "프리셋을 선택하세요.")
            return

        fp = os.path.join(self._get_dataset_dir(), selected + '.json')
        if not os.path.exists(fp):
            messagebox.showerror("오류", f"파일 없음: {fp}")
            return

        with open(fp, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        # ── Restore parameters ──
        params = dataset.get('params', {})
        for key, var in [
            ('sigma_0', self._sigma0_var), ('temperature', self._temp_var),
            ('v_min', self._v_min_var), ('v_max', self._v_max_var),
            ('n_v', self._n_v_var), ('q0', self._q0_var), ('q1', self._q1_var),
            ('n_q', self._n_q_var), ('gamma', self._gamma_var),
            ('n_phi', self._n_phi_var), ('tau_f0', self._tau_f0_var),
            ('v0_star', self._v0_star_var), ('c_gauss', self._c_gauss_var),
            ('epsilon', self._epsilon_var),
        ]:
            if key in params:
                var.set(str(params[key]))
        if 'nonlinear' in params:
            self._use_nonlinear_var.set(bool(params['nonlinear']))
        if 'flash_temp' in params:
            self._use_flash_var.set(bool(params['flash_temp']))
        if 'auto_fit' in params:
            self._auto_fit_var.set(bool(params['auto_fit']))

        # ── Load PSD ──
        psd_file = dataset.get('psd_file')
        if psd_file and os.path.exists(psd_file):
            try:
                q, C_q = load_psd_from_file(psd_file)
                self.psd_model = create_psd_from_data(q, C_q)
                self.psd_file = psd_file
                self.psd_info = f"{os.path.basename(psd_file)} ({len(q)} pts)"
                self._psd_status_var.set(f"✓ {self.psd_info}")
            except Exception as e:
                print(f"[Dataset] PSD 로드 실패: {e}")

        # ── Load compounds ──
        compound_defs = dataset.get('compounds', [])
        n = len(compound_defs)
        if n < 1:
            n = 1
        self._compound_count_var.set(n)
        self._rebuild_compound_columns(n)

        for i, cdef in enumerate(compound_defs):
            if i >= len(self.compounds):
                break
            cpd = self.compounds[i]

            # Name
            cpd.name = cdef.get('name', f'Compound {i+1}')
            w = self._compound_widgets.get(i, {})
            if 'name_var' in w:
                w['name_var'].set(cpd.name)

            # DMA
            dma_file = cdef.get('dma_file')
            if dma_file and os.path.exists(dma_file):
                try:
                    self._load_dma_from_path(i, dma_file)
                except Exception as e:
                    print(f"[Dataset] DMA 로드 실패 ({cpd.name}): {e}")

            # aT
            aT_file = cdef.get('aT_file')
            if aT_file and os.path.exists(aT_file):
                try:
                    self._load_aT_from_path(i, aT_file)
                except Exception as e:
                    print(f"[Dataset] aT 로드 실패 ({cpd.name}): {e}")

            # Strain / f,g
            strain_mode = cdef.get('strain_mode')
            strain_file = cdef.get('strain_file')
            if strain_file and os.path.exists(strain_file):
                try:
                    if strain_mode == 'fg':
                        self._load_fg_from_path(i, strain_file)
                    else:
                        self._load_strain_raw_from_path(i, strain_file)
                except Exception as e:
                    print(f"[Dataset] Strain 로드 실패 ({cpd.name}): {e}")

            # mu_dry
            mu_dry_file = cdef.get('mu_dry_file')
            if mu_dry_file and os.path.exists(mu_dry_file):
                try:
                    self._load_mu_dry_from_path(i, mu_dry_file)
                except Exception as e:
                    print(f"[Dataset] mu_dry 로드 실패 ({cpd.name}): {e}")

        # Refresh input data plots
        self._plot_input_data()

        messagebox.showinfo("불러오기 완료",
                            f"데이터셋 '{selected}' 로드 완료 ({n}개 컴파운드)")

    def _delete_dataset_preset(self):
        selected = self._dataset_combo_var.get()
        if not selected or selected in ('(없음)', '(선택)'):
            return
        if not messagebox.askyesno("삭제 확인", f"'{selected}' 프리셋을 삭제하시겠습니까?"):
            return
        fp = os.path.join(self._get_dataset_dir(), selected + '.json')
        if os.path.exists(fp):
            os.remove(fp)
        self._refresh_dataset_list()
        self._dataset_combo_var.set('(선택)')

    # ── DMA unit auto-detection ──
    @staticmethod
    def _read_dma_auto_unit(fp):
        """Load DMA file with automatic unit detection from header.

        마스터커브 탭 프리셋과 동일한 방식:
        - 헤더에 '(Pa)' → Pa 단위 (변환 없음)
        - 헤더에 '(MPa)' → MPa 단위 (×1e6)
        - 헤더에 '(GPa)' → GPa 단위 (×1e9)
        - 또는 값 크기로 자동 감지: E' > 10000 이면 Pa로 간주
        """
        # Read header to detect unit
        modulus_unit = None
        first_E_value = None

        for enc in ['utf-8-sig', 'utf-8', 'cp949', 'latin-1']:
            try:
                with open(fp, 'r', encoding=enc) as f:
                    for line in f:
                        line_s = line.strip()
                        if not line_s:
                            continue
                        if line_s.startswith('#'):
                            ll = line_s.lower()
                            if '(gpa)' in ll:
                                modulus_unit = 'GPa'
                            elif '(mpa)' in ll:
                                modulus_unit = 'MPa'
                            elif '(pa)' in ll:
                                modulus_unit = 'Pa'
                        else:
                            # First data line: check E' magnitude
                            parts = line_s.split()
                            if len(parts) >= 2:
                                try:
                                    first_E_value = float(parts[1])
                                except ValueError:
                                    pass
                            break
                break
            except UnicodeDecodeError:
                continue

        # If header didn't specify unit, detect from value magnitude
        if modulus_unit is None and first_E_value is not None:
            if first_E_value > 1e4:
                # E' > 10000 → likely Pa (typical rubber E' ~ 1e6-1e9 Pa)
                modulus_unit = 'Pa'
                print(f"[DMA unit] Auto-detected Pa (first E'={first_E_value:.2e})")
            else:
                # E' < 10000 → likely MPa (typical rubber E' ~ 1-1000 MPa)
                modulus_unit = 'MPa'
                print(f"[DMA unit] Auto-detected MPa (first E'={first_E_value:.2e})")
        elif modulus_unit is None:
            modulus_unit = 'MPa'  # fallback

        print(f"[DMA unit] File: {os.path.basename(fp)}, unit={modulus_unit}")
        omega, E_stor, E_loss = load_dma_from_file(fp, modulus_unit=modulus_unit)
        return omega, E_stor, E_loss

    # ── Path-based loaders (for dataset preset, no file dialog) ──
    def _load_dma_from_path(self, idx, fp):
        omega, E_stor, E_loss = self._read_dma_auto_unit(fp)
        smoothed = smooth_dma_data(omega, E_stor, E_loss)
        material = create_material_from_dma(
            smoothed['omega'], smoothed['E_storage_smooth'], smoothed['E_loss_smooth'],
            material_name=self.compounds[idx].name)
        self.compounds[idx].material = material
        self.compounds[idx].dma_file = fp
        w = self._compound_widgets.get(idx, {})
        if 'mc_label' in w:
            w['mc_label'].config(text=f"✓ {os.path.basename(fp)}",
                                 foreground=self.app.COLORS['success'])

    def _load_aT_from_path(self, idx, fp):
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
            raise ValueError("aT 파일 형식 오류")
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
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
        cpd.aT_data = {'T': T, 'aT': aT, 'log_aT': log_aT, 'bT': bT, 'T_ref': T_ref, 'has_bT': has_bT}
        cpd.aT_interp = interp1d(T, log_aT, kind='linear', bounds_error=False, fill_value='extrapolate')
        cpd.bT_interp = interp1d(T, bT, kind='linear', bounds_error=False, fill_value='extrapolate')
        cpd.aT_file = fp
        w = self._compound_widgets.get(idx, {})
        if 'at_label' in w:
            w['at_label'].config(text=f"✓ {os.path.basename(fp)} ({len(T)}pts)",
                                 foreground=self.app.COLORS['success'])

    def _load_strain_raw_from_path(self, idx, fp):
        data_by_T = load_strain_sweep_file(fp)
        fg_by_T = compute_fg_from_strain_sweep(data_by_T)
        fg_avg = average_fg_curves(fg_by_T)
        f_interp, g_interp = create_fg_interpolator(fg_avg['strain'], fg_avg['f'], fg_avg.get('g'))
        cpd = self.compounds[idx]
        cpd.strain_mode = 'raw'
        cpd.strain_data = data_by_T
        cpd.fg_by_T = fg_by_T
        cpd.f_interpolator = f_interp
        cpd.g_interpolator = g_interp
        cpd.strain_file = fp
        w = self._compound_widgets.get(idx, {})
        if 'str_label' in w:
            w['str_label'].config(text=f"✓ [sweep] {os.path.basename(fp)}",
                                  foreground=self.app.COLORS['success'])

    def _load_fg_from_path(self, idx, fp):
        fg_data = load_fg_curve_file(fp)
        if fg_data is None:
            raise ValueError("f,g 파일 파싱 실패")
        f_interp, g_interp = create_fg_interpolator(fg_data['strain'], fg_data['f'], fg_data.get('g'))
        cpd = self.compounds[idx]
        cpd.strain_mode = 'fg'
        cpd.fg_raw = fg_data
        cpd.f_interpolator = f_interp
        cpd.g_interpolator = g_interp
        cpd.strain_file = fp
        w = self._compound_widgets.get(idx, {})
        if 'str_label' in w:
            w['str_label'].config(text=f"✓ [f,g] {os.path.basename(fp)}",
                                  foreground=self.app.COLORS['success'])

    def _load_mu_dry_from_path(self, idx, fp):
        data = np.loadtxt(fp, delimiter=None, comments='#')
        if data.ndim == 1:
            raise ValueError("2열 형식 필요")
        log_v = data[:, 0]
        mu_dry = data[:, 1]
        self.compounds[idx].mu_dry_data = (log_v, mu_dry)
        self.compounds[idx].mu_dry_file = fp
        w = self._compound_widgets.get(idx, {})
        if 'mud_label' in w:
            w['mud_label'].config(text=f"✓ {os.path.basename(fp)} ({len(mu_dry)}pts)",
                                  foreground=self.app.COLORS['success'])
