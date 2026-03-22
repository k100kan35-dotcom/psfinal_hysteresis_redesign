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
    PLOT_LINE_WIDTH = 2.5
    PLOT_LINE_WIDTH_SUB = 1.8  # 보조 선 (visc, adh)

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
            ('viscoelastic', '점탄성 (E\', E\'\')'),
            ('aT',           'aT 비교'),
            ('strain',       'Strain / f,g 비교'),
            ('A_A0',         'A/A0 비교'),
            ('adh_hys',      'μ_adh / μ_hys 비교'),
            ('total',        'μ_total 비교'),
        ]
        for key, label in _plot_defs:
            tab_frame = ttk.Frame(self._plot_notebook)
            self._plot_notebook.add(tab_frame, text=f'  {label}  ')
            fig = Figure(figsize=(8, 6), dpi=100, facecolor='white')
            fig.subplots_adjust(hspace=0.35, wspace=0.3)
            canvas = FigureCanvasTkAgg(fig, tab_frame)
            toolbar = NavigationToolbar2Tk(canvas, tab_frame)
            toolbar.update()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self._plot_tabs[key] = (fig, canvas)

        # Legacy single fig/canvas references (for backward compatibility)
        self._fig, self._canvas = self._plot_tabs['total']

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
            self._plot_viscoelastic()
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
            self._plot_aT()
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
            self._plot_strain()
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
            self._plot_strain()
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
            # persson_master_curve 설정 (aT 시프트에 필요)
            if cpd.material is not None:
                freq = cpd.material.frequencies if hasattr(cpd.material, 'frequencies') else None
                E_s = cpd.material.storage_modulus if hasattr(cpd.material, 'storage_modulus') else None
                E_l = cpd.material.loss_modulus if hasattr(cpd.material, 'loss_modulus') else None
                if freq is not None and E_s is not None and E_l is not None:
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

            # Flash temperature: mu_hot 사용, 없으면 mu (cold) 사용
            if use_flash and app.mu_visc_results.get('mu_hot') is not None:
                mu_visc = app.mu_visc_results['mu_hot']
            else:
                mu_visc = app.mu_visc_results.get('mu', None)
                if mu_visc is None:
                    mu_visc = app.mu_visc_results.get('mu_visc', None)

        if app.mu_adh_results is not None:
            mu_adh = app.mu_adh_results.get('mu_adh', None)

        if v_array is not None and mu_visc is not None:
            if mu_adh is not None and len(mu_adh) == len(mu_visc):
                mu_total = mu_visc + mu_adh
            else:
                mu_adh = np.zeros_like(mu_visc)
                mu_total = mu_visc.copy()

            # A/A0: flash temp 시 A_A0_hot 사용
            use_flash = app.mu_visc_results.get('use_flash', False) if app.mu_visc_results else False
            if use_flash and app.mu_visc_results.get('A_A0_hot') is not None:
                A_A0 = app.mu_visc_results['A_A0_hot']
            else:
                A_A0 = app.mu_visc_results.get('P_qmax', np.zeros_like(mu_visc))

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

            # Copy adh fit params if available
            if app.mu_adh_results is not None:
                cpd.results['adh_params'] = app.mu_adh_results.get('params', None)
        else:
            cpd.results = None
            print(f"[WARNING] {cpd.name}: mu_visc_results가 비어있습니다.")

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
        self._plot_viscoelastic()
        self._plot_aT()
        self._plot_strain()
        self._plot_A_A0()
        self._plot_adh_hys()
        self._plot_total()

    @staticmethod
    def _style_ax(ax, title='', xlabel='', ylabel='', xscale=None, yscale=None):
        """Apply consistent styling to an axis: 12pt fonts, grid."""
        FS = DataInputTab.PLOT_FONT_SIZE
        if title:
            ax.set_title(title, fontsize=FS + 1, fontweight='bold')
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=FS)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=FS)
        if xscale:
            ax.set_xscale(xscale)
        if yscale:
            ax.set_yscale(yscale)
        ax.tick_params(axis='both', labelsize=FS - 1)
        ax.grid(True, alpha=0.3)

    def _plot_viscoelastic(self):
        """Plot E'(ω), E''(ω) master curves for all compounds."""
        FS = self.PLOT_FONT_SIZE
        LW = self.PLOT_LINE_WIDTH
        fig, canvas = self._plot_tabs['viscoelastic']
        fig.clear()
        compounds_with_material = [c for c in self.compounds if c.material is not None]
        if not compounds_with_material:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'DMA 데이터를 로드하세요', ha='center', va='center',
                    fontsize=14, color='gray', transform=ax.transAxes)
            ax.set_axis_off()
            canvas.draw_idle()
            return

        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        self._style_ax(ax1, "Storage Modulus E'(ω)", 'ω (rad/s)', "E' (Pa)", 'log', 'log')
        self._style_ax(ax2, "Loss Modulus E''(ω)", 'ω (rad/s)', "E'' (Pa)", 'log', 'log')

        for i, cpd in enumerate(compounds_with_material):
            color = self.COMPOUND_COLORS[i % len(self.COMPOUND_COLORS)]
            mat = cpd.material
            if hasattr(mat, '_frequencies') and mat._frequencies is not None:
                omega = mat._frequencies
                E_stor = mat._storage_modulus
                E_loss = mat._loss_modulus if mat._loss_modulus is not None else np.zeros_like(E_stor)
                ax1.plot(omega, E_stor, '-', color=color, linewidth=LW, label=cpd.name)
                ax2.plot(omega, E_loss, '-', color=color, linewidth=LW, label=cpd.name)

        ax1.legend(fontsize=FS - 1, loc='best')
        ax2.legend(fontsize=FS - 1, loc='best')
        fig.tight_layout()
        canvas.draw_idle()

    def _plot_aT(self):
        """Plot aT shift factor comparison."""
        FS = self.PLOT_FONT_SIZE
        LW = self.PLOT_LINE_WIDTH
        fig, canvas = self._plot_tabs['aT']
        fig.clear()
        compounds_with_aT = [c for c in self.compounds if c.aT_data is not None]
        if not compounds_with_aT:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'aT 데이터를 로드하세요', ha='center', va='center',
                    fontsize=14, color='gray', transform=ax.transAxes)
            ax.set_axis_off()
            canvas.draw_idle()
            return

        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        self._style_ax(ax1, 'log₁₀(aT) vs Temperature', 'Temperature (°C)', 'log₁₀(aT)')
        self._style_ax(ax2, 'bT vs Temperature', 'Temperature (°C)', 'bT')

        has_bT = False
        for i, cpd in enumerate(compounds_with_aT):
            color = self.COMPOUND_COLORS[i % len(self.COMPOUND_COLORS)]
            d = cpd.aT_data
            ax1.plot(d['T'], d['log_aT'], 'o-', color=color, markersize=5,
                     linewidth=LW, label=f"{cpd.name} (Tref={d['T_ref']:.0f}°C)")
            if d.get('has_bT', False):
                ax2.plot(d['T'], d['bT'], 's-', color=color, markersize=5,
                         linewidth=LW, label=cpd.name)
                has_bT = True

        ax1.legend(fontsize=FS - 1, loc='best')
        if has_bT:
            ax2.legend(fontsize=FS - 1, loc='best')
        else:
            ax2.text(0.5, 0.5, 'bT 데이터 없음', ha='center', va='center',
                     fontsize=FS, color='gray', transform=ax2.transAxes)
            ax2.set_axis_off()
        fig.tight_layout()
        canvas.draw_idle()

    def _plot_strain(self):
        """Plot strain sweep / f,g curves comparison."""
        FS = self.PLOT_FONT_SIZE
        LW = self.PLOT_LINE_WIDTH
        fig, canvas = self._plot_tabs['strain']
        fig.clear()
        compounds_with_strain = [c for c in self.compounds
                                 if c.fg_raw is not None or c.fg_by_T is not None
                                 or c.f_interpolator is not None]
        if not compounds_with_strain:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'Strain/f,g 데이터를 로드하세요', ha='center', va='center',
                    fontsize=14, color='gray', transform=ax.transAxes)
            ax.set_axis_off()
            canvas.draw_idle()
            return

        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        self._style_ax(ax1, 'f(strain) 비교', 'Strain', 'f(strain)')
        self._style_ax(ax2, 'g(strain) 비교', 'Strain', 'g(strain)')

        for i, cpd in enumerate(compounds_with_strain):
            color = self.COMPOUND_COLORS[i % len(self.COMPOUND_COLORS)]
            if cpd.fg_raw is not None:
                strain = cpd.fg_raw.get('strain', None)
                f_vals = cpd.fg_raw.get('f', None)
                g_vals = cpd.fg_raw.get('g', None)
                if strain is not None and f_vals is not None:
                    ax1.plot(strain, f_vals, 'o-', color=color, markersize=4,
                             linewidth=LW, label=cpd.name)
                if strain is not None and g_vals is not None:
                    ax2.plot(strain, g_vals, 's-', color=color, markersize=4,
                             linewidth=LW, label=cpd.name)
            elif cpd.f_interpolator is not None:
                strain = np.linspace(0.001, 1.0, 200)
                try:
                    f_vals = cpd.f_interpolator(strain)
                    ax1.plot(strain, f_vals, '-', color=color, linewidth=LW, label=cpd.name)
                except Exception:
                    pass
                if cpd.g_interpolator is not None:
                    try:
                        g_vals = cpd.g_interpolator(strain)
                        ax2.plot(strain, g_vals, '-', color=color, linewidth=LW, label=cpd.name)
                    except Exception:
                        pass

        ax1.legend(fontsize=FS - 1, loc='best')
        ax2.legend(fontsize=FS - 1, loc='best')
        fig.tight_layout()
        canvas.draw_idle()

    def _plot_A_A0(self):
        """Plot A/A0 (contact area ratio) comparison."""
        FS = self.PLOT_FONT_SIZE
        LW = self.PLOT_LINE_WIDTH
        fig, canvas = self._plot_tabs['A_A0']
        fig.clear()
        compounds_with_results = [c for c in self.compounds
                                  if c.results is not None and c.results.get('A_A0') is not None]
        if not compounds_with_results:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, '계산 결과가 없습니다\n(계산 실행 후 표시)', ha='center', va='center',
                    fontsize=14, color='gray', transform=ax.transAxes)
            ax.set_axis_off()
            canvas.draw_idle()
            return

        ax = fig.add_subplot(111)
        self._style_ax(ax, 'A/A₀ vs Velocity (All Compounds)', 'Velocity (m/s)', 'A/A₀', 'log')

        for i, cpd in enumerate(compounds_with_results):
            color = self.COMPOUND_COLORS[i % len(self.COMPOUND_COLORS)]
            r = cpd.results
            ax.plot(r['v'], r['A_A0'], '-', color=color, linewidth=LW, label=cpd.name)

        ax.legend(fontsize=FS, loc='best')
        fig.tight_layout()
        canvas.draw_idle()

    def _plot_adh_hys(self):
        """Plot μ_adh and μ_hys (visc) comparison."""
        FS = self.PLOT_FONT_SIZE
        LW = self.PLOT_LINE_WIDTH
        fig, canvas = self._plot_tabs['adh_hys']
        fig.clear()
        compounds_with_results = [c for c in self.compounds if c.results is not None]
        if not compounds_with_results:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, '계산 결과가 없습니다\n(계산 실행 후 표시)', ha='center', va='center',
                    fontsize=14, color='gray', transform=ax.transAxes)
            ax.set_axis_off()
            canvas.draw_idle()
            return

        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        self._style_ax(ax1, 'μ_hys (Hysteresis) 비교', 'Velocity (m/s)', 'μ_hys', 'log')
        self._style_ax(ax2, 'μ_adh (Adhesion) 비교', 'Velocity (m/s)', 'μ_adh', 'log')

        has_adh = False
        for i, cpd in enumerate(compounds_with_results):
            color = self.COMPOUND_COLORS[i % len(self.COMPOUND_COLORS)]
            r = cpd.results
            ax1.plot(r['v'], r['mu_visc'], '-', color=color, linewidth=LW, label=cpd.name)
            if r.get('mu_adh') is not None:
                ax2.plot(r['v'], r['mu_adh'], '-', color=color, linewidth=LW, label=cpd.name)
                has_adh = True

        ax1.legend(fontsize=FS - 1, loc='best')
        if has_adh:
            ax2.legend(fontsize=FS - 1, loc='best')
        else:
            ax2.text(0.5, 0.5, 'μ_adh 없음', ha='center', va='center',
                     fontsize=FS, color='gray', transform=ax2.transAxes)
            ax2.set_axis_off()
        fig.tight_layout()
        canvas.draw_idle()

    def _plot_total(self):
        """Plot μ_total comparison (original _plot_preview)."""
        FS = self.PLOT_FONT_SIZE
        LW = self.PLOT_LINE_WIDTH
        LWs = self.PLOT_LINE_WIDTH_SUB
        fig, canvas = self._plot_tabs['total']
        fig.clear()
        compounds_with_results = [c for c in self.compounds if c.results is not None]
        if not compounds_with_results:
            canvas.draw_idle()
            return

        ax = fig.add_subplot(111)
        self._style_ax(ax, 'μ_total vs Velocity (All Compounds)', 'Velocity (m/s)', 'μ', 'log')

        for i, cpd in enumerate(compounds_with_results):
            r = cpd.results
            color = self.COMPOUND_COLORS[i % len(self.COMPOUND_COLORS)]
            ax.plot(r['v'], r['mu_total'], '-', color=color,
                    linewidth=LW, label=cpd.name)
            if r.get('mu_adh') is not None:
                ax.plot(r['v'], r['mu_visc'], '--', color=color,
                        linewidth=LWs, alpha=0.6, label=f'{cpd.name} (visc)')
                ax.plot(r['v'], r['mu_adh'], ':', color=color,
                        linewidth=LWs, alpha=0.6, label=f'{cpd.name} (adh)')

        ax.legend(fontsize=FS, loc='best')
        fig.tight_layout()
        canvas.draw_idle()

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
        self._plot_viscoelastic()
        self._plot_aT()
        self._plot_strain()

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
