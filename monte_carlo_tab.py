"""
Monte Carlo Tab for Persson Friction Model
==========================================
유효 표면 PSD와 차단 파수 q₁의 Monte Carlo 역추정
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import pickle
import threading
import time

from persson_model.core.psd_models import MeasuredPSD
from persson_model.core.viscoelastic import ViscoelasticMaterial
from persson_model.core.g_calculator import GCalculator
from persson_model.core.friction import FrictionCalculator

# PSD computation & ensemble from psd repo
from psd_ensemble import PSDEnsemble


def bind_monte_carlo_tab(app):
    """Monte Carlo 탭을 app에 바인딩한다."""
    MonteCarloTab(app)


class MonteCarloTab:
    def __init__(self, app):
        self.app = app
        app._create_monte_carlo_tab = self._create_tab

        # 공유 데이터
        self.C_pool = None       # (N_valid, N_q) ndarray — 생성된 PSD 샘플
        self.q_grid = None       # (N_q,) ndarray — 공통 q 그리드
        self.q1_pool = None      # (M,) ndarray — q₁ 풀
        self.mc_results = None   # list of dicts
        self._stop_flag = False  # Monte Carlo 중단 플래그

        # 원본 스캔 데이터
        self._raw_scans = []     # list of (q, C_q) tuples
        self._idada_files = []   # IDADA 프로파일 파일 경로 목록
        self._ensemble = None    # PSDEnsemble 인스턴스

        # 실험 데이터 (q₁ 탭)
        self._granite_data = {}  # {compound: {v_array, mu_array}}
        self._asphalt_data = {}
        # 건식 마찰 데이터 (MC 탭)
        self._dry_data = {}      # {compound: {v_array, mu_array}}

        # q₁ 사전 탐색 결과
        self._q1_valid_min = None
        self._q1_valid_max = None

        # 컴파운드 기본 이름
        self._compound_names = ['R0', 'R20', 'R40', 'R60']

    # ================================================================
    # 메인 탭 생성
    # ================================================================
    def _create_tab(self, parent):
        """메인 탭 생성 — 내부에 서브탭 4개"""
        nb = ttk.Notebook(parent)
        nb.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self.frame_psd = ttk.Frame(nb)
        self.frame_q1 = ttk.Frame(nb)
        self.frame_mc = ttk.Frame(nb)
        self.frame_result = ttk.Frame(nb)

        nb.add(self.frame_psd, text='  PSD 앙상블  ')
        nb.add(self.frame_q1, text='  q₁ 생성  ')
        nb.add(self.frame_mc, text='  Monte Carlo  ')
        nb.add(self.frame_result, text='  결과  ')

        self._build_psd_tab(self.frame_psd)
        self._build_q1_tab(self.frame_q1)
        self._build_mc_tab(self.frame_mc)
        self._build_result_tab(self.frame_result)

    # ================================================================
    # 유틸리티
    # ================================================================
    def _show_status(self, msg, level='info'):
        if hasattr(self.app, '_show_status'):
            self.app._show_status(msg, level)

    def _get_fonts(self):
        return self.app.FONTS

    def _get_colors(self):
        return self.app.COLORS

    def _get_plot_fonts(self):
        return getattr(self.app, 'PLOT_FONTS', {
            'title': 10, 'label': 9, 'tick': 8, 'legend': 8,
            'suptitle': 11, 'annotation': 8,
        })

    # ================================================================
    # 서브탭 1: PSD 앙상블
    # ================================================================
    def _build_psd_tab(self, parent):
        FONTS = self._get_fonts()
        COLORS = self._get_colors()

        main = ttk.Frame(parent)
        main.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        # 좌측 제어판 (스크롤 가능)
        left = ttk.Frame(main, width=360)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 4))
        left.pack_propagate(False)

        psd_canvas = tk.Canvas(left, highlightthickness=0)
        psd_sb = ttk.Scrollbar(left, orient='vertical', command=psd_canvas.yview)
        psd_content = ttk.Frame(psd_canvas)
        psd_content.bind('<Configure>',
                         lambda e: psd_canvas.configure(scrollregion=psd_canvas.bbox('all')))
        psd_canvas.create_window((0, 0), window=psd_content, anchor='nw', width=340)
        psd_canvas.configure(yscrollcommand=psd_sb.set)
        psd_sb.pack(side=tk.RIGHT, fill=tk.Y)
        psd_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        def _mw(event):
            if event.delta:
                psd_canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')
            elif event.num == 4:
                psd_canvas.yview_scroll(-1, 'units')
            elif event.num == 5:
                psd_canvas.yview_scroll(1, 'units')
        psd_canvas.bind('<Enter>', lambda e: (
            psd_canvas.bind_all('<MouseWheel>', _mw),
            psd_canvas.bind_all('<Button-4>', _mw),
            psd_canvas.bind_all('<Button-5>', _mw)))
        psd_canvas.bind('<Leave>', lambda e: (
            psd_canvas.unbind_all('<MouseWheel>'),
            psd_canvas.unbind_all('<Button-4>'),
            psd_canvas.unbind_all('<Button-5>')))

        # --- 섹션 1: 입력 파일 ---
        sec1 = ttk.LabelFrame(psd_content, text="1) 입력 데이터", padding=6)
        sec1.pack(fill=tk.X, pady=3, padx=2)

        ttk.Label(sec1, text="── 기존 PSD 로드 ──",
                  font=FONTS['small'], foreground='#64748B').pack(anchor=tk.W, pady=(0, 1))
        ttk.Button(sec1, text="Tab 0 PSD 사용",
                   command=self._load_psd_from_tab0).pack(fill=tk.X, pady=1)
        ttk.Button(sec1, text="PSD CSV 폴더 선택",
                   command=self._load_psd_folder).pack(fill=tk.X, pady=1)

        ttk.Label(sec1, text="── IDADA 프로파일 → PSD 계산 ──",
                  font=FONTS['small'], foreground='#64748B').pack(anchor=tk.W, pady=(6, 1))
        ttk.Button(sec1, text="IDADA 프로파일 폴더 선택",
                   command=self._load_idada_folder).pack(fill=tk.X, pady=1)
        ttk.Button(sec1, text="IDADA 파일 개별 선택",
                   command=self._load_idada_files).pack(fill=tk.X, pady=1)

        self._psd_scan_label = ttk.Label(sec1, text="스캔 수: 0", font=FONTS['body'])
        self._psd_scan_label.pack(anchor=tk.W, pady=2)

        # --- 섹션 1b: PSD 계산 파라미터 (IDADA 전용) ---
        sec1b = ttk.LabelFrame(psd_content, text="1b) PSD 계산 파라미터 (IDADA)", padding=6)
        sec1b.pack(fill=tk.X, pady=3, padx=2)

        r = ttk.Frame(sec1b); r.pack(fill=tk.X, pady=1)
        ttk.Label(r, text="Detrend:", font=FONTS['body']).pack(side=tk.LEFT)
        self._psd_detrend_var = tk.StringVar(value='linear')
        ttk.Combobox(r, textvariable=self._psd_detrend_var, width=12,
                     values=['none', 'mean', 'linear', 'quadratic'],
                     state='readonly').pack(side=tk.RIGHT)

        r = ttk.Frame(sec1b); r.pack(fill=tk.X, pady=1)
        ttk.Label(r, text="Window:", font=FONTS['body']).pack(side=tk.LEFT)
        self._psd_window_var = tk.StringVar(value='none')
        ttk.Combobox(r, textvariable=self._psd_window_var, width=12,
                     values=['none', 'hanning', 'hamming', 'blackman'],
                     state='readonly').pack(side=tk.RIGHT)

        r = ttk.Frame(sec1b); r.pack(fill=tk.X, pady=1)
        ttk.Label(r, text="Top PSD:", font=FONTS['body']).pack(side=tk.LEFT)
        self._psd_top_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(r, variable=self._psd_top_var).pack(side=tk.RIGHT)

        r = ttk.Frame(sec1b); r.pack(fill=tk.X, pady=1)
        ttk.Label(r, text="1D→2D 변환:", font=FONTS['body']).pack(side=tk.LEFT)
        self._psd_conv_var = tk.StringVar(value='standard')
        ttk.Combobox(r, textvariable=self._psd_conv_var, width=12,
                     values=['standard', 'gamma', 'sqrt'],
                     state='readonly').pack(side=tk.RIGHT)

        r = ttk.Frame(sec1b); r.pack(fill=tk.X, pady=1)
        ttk.Label(r, text="보정계수:", font=FONTS['body']).pack(side=tk.LEFT)
        self._psd_corr_var = tk.StringVar(value="1.1615")
        ttk.Entry(r, textvariable=self._psd_corr_var, width=8).pack(side=tk.RIGHT)

        r = ttk.Frame(sec1b); r.pack(fill=tk.X, pady=1)
        ttk.Label(r, text="Hurst H:", font=FONTS['body']).pack(side=tk.LEFT)
        self._psd_hurst_var = tk.StringVar(value="0.80")
        ttk.Entry(r, textvariable=self._psd_hurst_var, width=6).pack(side=tk.RIGHT)

        r = ttk.Frame(sec1b); r.pack(fill=tk.X, pady=1)
        ttk.Label(r, text="Log bins:", font=FONTS['body']).pack(side=tk.LEFT)
        self._psd_nbins_var = tk.StringVar(value="88")
        ttk.Entry(r, textvariable=self._psd_nbins_var, width=6).pack(side=tk.RIGHT)

        ttk.Button(sec1b, text="IDADA → PSD 계산",
                   command=self._compute_psd_from_idada).pack(fill=tk.X, pady=2)

        # --- 섹션 2: PCA 앙상블 설정 ---
        sec2 = ttk.LabelFrame(psd_content, text="2) PCA 앙상블 설정", padding=6)
        sec2.pack(fill=tk.X, pady=3, padx=2)

        r1 = ttk.Frame(sec2); r1.pack(fill=tk.X, pady=1)
        ttk.Label(r1, text="누적분산 임계값:", font=FONTS['body']).pack(side=tk.LEFT)
        self._pca_threshold_var = tk.StringVar(value="0.90")
        ttk.Entry(r1, textvariable=self._pca_threshold_var, width=6).pack(side=tk.LEFT, padx=2)

        r2 = ttk.Frame(sec2); r2.pack(fill=tk.X, pady=1)
        ttk.Label(r2, text="생성 샘플 수:", font=FONTS['body']).pack(side=tk.LEFT)
        self._n_samples_var = tk.StringVar(value="1000")
        ttk.Entry(r2, textvariable=self._n_samples_var, width=8).pack(side=tk.LEFT, padx=2)

        r3 = ttk.Frame(sec2); r3.pack(fill=tk.X, pady=1)
        ttk.Label(r3, text="Random seed:", font=FONTS['body']).pack(side=tk.LEFT)
        self._pca_seed_var = tk.StringVar(value="42")
        ttk.Entry(r3, textvariable=self._pca_seed_var, width=6).pack(side=tk.LEFT, padx=2)

        # 물리적 타당성 필터
        ttk.Label(sec2, text="── 물리적 타당성 필터 ──",
                  font=FONTS['small'], foreground='#64748B').pack(anchor=tk.W, pady=(4, 1))

        r4 = ttk.Frame(sec2); r4.pack(fill=tk.X, pady=1)
        ttk.Label(r4, text="rms roughness ±N σ:", font=FONTS['body']).pack(side=tk.LEFT)
        self._rms_sigma_var = tk.StringVar(value="3")
        ttk.Entry(r4, textvariable=self._rms_sigma_var, width=4).pack(side=tk.LEFT, padx=2)

        r5 = ttk.Frame(sec2); r5.pack(fill=tk.X, pady=1)
        ttk.Label(r5, text="단조감소 위반 허용:", font=FONTS['body']).pack(side=tk.LEFT)
        self._mono_tol_var = tk.StringVar(value="0.1")
        ttk.Entry(r5, textvariable=self._mono_tol_var, width=5).pack(side=tk.LEFT, padx=2)

        # --- 섹션 3: 실행 ---
        sec3 = ttk.LabelFrame(psd_content, text="3) 앙상블 생성", padding=6)
        sec3.pack(fill=tk.X, pady=3, padx=2)

        ttk.Button(sec3, text="PSD 앙상블 생성 (PSDEnsemble)",
                   command=self._run_psd_ensemble_via_class,
                   style='Accent.TButton').pack(fill=tk.X, pady=2)
        ttk.Button(sec3, text="PSD 앙상블 생성 (인라인 PCA)",
                   command=self._run_psd_ensemble).pack(fill=tk.X, pady=1)
        self._psd_progress_label = ttk.Label(sec3, text="", font=FONTS['small'])
        self._psd_progress_label.pack(anchor=tk.W)

        # 우측 플롯
        right = ttk.Frame(main)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        PF = self._get_plot_fonts()
        self._fig_psd = Figure(figsize=(8, 7), dpi=90)
        self._ax_psd_orig = self._fig_psd.add_subplot(211)
        self._ax_psd_gen = self._fig_psd.add_subplot(212)
        self._fig_psd.tight_layout(pad=2.5)

        self._canvas_psd = FigureCanvasTkAgg(self._fig_psd, master=right)
        self._canvas_psd.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ── PSD 로드 ──
    def _load_psd_from_tab0(self):
        """app.raw_psd_data에서 PSD 읽어오기"""
        raw = getattr(self.app, 'raw_psd_data', None)
        if raw is None or 'q' not in raw:
            self._show_status("Tab 0에 PSD 데이터가 없습니다.", 'warning')
            return
        q = np.asarray(raw['q'])
        C = np.asarray(raw['C_q'])
        self._raw_scans = [(q.copy(), C.copy())]
        self._psd_scan_label.config(text=f"스캔 수: 1 (Tab 0)")
        self._show_status("Tab 0 PSD 로드 완료", 'success')
        self._plot_raw_scans()

    def _load_psd_folder(self):
        """폴더에서 CSV 파일 전부 로드"""
        folder = filedialog.askdirectory(title="PSD CSV 폴더 선택")
        if not folder:
            return
        csv_files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.csv')])
        if not csv_files:
            self._show_status("CSV 파일이 없습니다.", 'warning')
            return
        scans = []
        for fn in csv_files:
            try:
                data = np.loadtxt(os.path.join(folder, fn), delimiter=',', skiprows=1)
                if data.ndim == 2 and data.shape[1] >= 2:
                    q = data[:, 0]
                    C = data[:, 1]
                    # 양수 값만
                    mask = (q > 0) & (C > 0)
                    if np.sum(mask) >= 10:
                        scans.append((q[mask], C[mask]))
            except Exception:
                continue
        if not scans:
            self._show_status("유효한 CSV 데이터가 없습니다.", 'warning')
            return
        self._raw_scans = scans
        self._psd_scan_label.config(text=f"스캔 수: {len(scans)} ({os.path.basename(folder)})")
        self._show_status(f"PSD {len(scans)}개 로드 완료", 'success')
        self._plot_raw_scans()

    def _plot_raw_scans(self):
        """원본 스캔 플롯"""
        ax = self._ax_psd_orig
        ax.clear()
        PF = self._get_plot_fonts()
        for q, C in self._raw_scans:
            ax.loglog(q, C, color='gray', alpha=0.4, linewidth=0.8)
        if self._raw_scans:
            # 평균선 (공통 그리드에서)
            q_common = self._make_common_q_grid()
            if q_common is not None:
                C_matrix = self._interpolate_scans_to_grid(q_common)
                C_mean = np.exp(np.mean(np.log(C_matrix), axis=0))
                ax.loglog(q_common, C_mean, 'r-', linewidth=2, label='평균')
                ax.legend(fontsize=PF.get('legend', 8))
        ax.set_xlabel('q (1/m)', fontsize=PF.get('label', 9))
        ax.set_ylabel('C(q) (m⁴)', fontsize=PF.get('label', 9))
        ax.set_title('원본 PSD 스캔', fontsize=PF.get('title', 10))
        ax.tick_params(labelsize=PF.get('tick', 8))
        ax.grid(True, alpha=0.3)
        self._canvas_psd.draw_idle()

    # ── IDADA 프로파일 로드 ──
    def _load_idada_folder(self):
        """IDADA 프로파일 폴더 선택"""
        folder = filedialog.askdirectory(title="IDADA 프로파일 폴더 선택")
        if not folder:
            return
        exts = ('.csv', '.dat', '.txt')
        files = sorted([
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(exts)
        ])
        if not files:
            self._show_status("프로파일 파일이 없습니다.", 'warning')
            return
        self._idada_files = files
        self._psd_scan_label.config(
            text=f"IDADA 프로파일: {len(files)}개 ({os.path.basename(folder)})")
        self._show_status(f"IDADA 프로파일 {len(files)}개 선택됨", 'success')

    def _load_idada_files(self):
        """IDADA 프로파일 개별 파일 선택"""
        files = filedialog.askopenfilenames(
            title="IDADA 프로파일 파일 선택",
            filetypes=[("All", "*.*"), ("CSV", "*.csv"), ("DAT", "*.dat")])
        if not files:
            return
        self._idada_files = list(files)
        self._psd_scan_label.config(text=f"IDADA 프로파일: {len(files)}개")
        self._show_status(f"IDADA 프로파일 {len(files)}개 선택됨", 'success')

    def _get_psd_params(self):
        """PSD 계산 파라미터 딕셔너리 반환"""
        return {
            'detrend': self._psd_detrend_var.get(),
            'window': self._psd_window_var.get(),
            'use_top_psd': self._psd_top_var.get(),
            'conversion_method': self._psd_conv_var.get(),
            'hurst': float(self._psd_hurst_var.get()),
            'correction_factor': float(self._psd_corr_var.get()),
            'n_bins': int(self._psd_nbins_var.get()),
        }

    def _compute_psd_from_idada(self):
        """IDADA 프로파일 파일 → PSDComputer로 PSD 계산 → _raw_scans에 저장"""
        if not self._idada_files:
            self._show_status("IDADA 프로파일을 먼저 선택하세요.", 'warning')
            return

        from psd_generator import PSDComputer

        psd_params = self._get_psd_params()
        scans = []
        errors = []

        self._psd_progress_label.config(text="IDADA PSD 계산 중...")
        self.app.root.update_idletasks()

        for fpath in self._idada_files:
            try:
                comp = PSDComputer()
                comp.load_profile(fpath)
                q_bin, C2D_bin, *_ = comp.compute_psd(**psd_params)
                if len(q_bin) >= 10:
                    scans.append((q_bin, C2D_bin))
            except Exception as e:
                errors.append(f"{os.path.basename(fpath)}: {e}")

        if not scans:
            self._show_status(f"유효한 PSD 없음. 오류: {len(errors)}개", 'error')
            return

        self._raw_scans = scans
        msg = f"IDADA PSD 계산 완료: {len(scans)}개 성공"
        if errors:
            msg += f", {len(errors)}개 실패"
        self._psd_scan_label.config(text=f"스캔 수: {len(scans)} (IDADA)")
        self._psd_progress_label.config(text=msg)
        self._show_status(msg, 'success')
        self._plot_raw_scans()

    def _run_psd_ensemble_via_class(self):
        """PSDEnsemble 클래스를 사용한 PCA 앙상블 생성"""
        # IDADA 파일이 있으면 PSDEnsemble.load_profiles() 사용
        # 아니면 _raw_scans 기반 커스텀 경로 사용
        try:
            threshold = float(self._pca_threshold_var.get())
            n_samples = int(self._n_samples_var.get())
            seed = int(self._pca_seed_var.get())
        except ValueError:
            self._show_status("PCA 설정값을 확인하세요.", 'error')
            return

        if self._idada_files and len(self._idada_files) >= 2:
            # IDADA 파일 → PSDEnsemble 풀 파이프라인
            self._psd_progress_label.config(text="PSDEnsemble: 프로파일 로드 중...")
            self.app.root.update_idletasks()

            psd_params = self._get_psd_params()
            ens = PSDEnsemble(psd_params=psd_params)

            try:
                ens.load_profiles(self._idada_files)
            except Exception as e:
                self._show_status(f"프로파일 로드 실패: {e}", 'error')
                return

            self._psd_progress_label.config(text="PSDEnsemble: PCA 분해 중...")
            self.app.root.update_idletasks()

            ens.fit_pca(var_threshold=threshold)

            self._psd_progress_label.config(text=f"PSDEnsemble: {n_samples}개 샘플 생성 중...")
            self.app.root.update_idletasks()

            samples = ens.generate_samples(n_samples=n_samples, random_seed=seed)

            # 결과를 내부 변수에 저장
            self._ensemble = ens
            self.q_grid = ens.q_grid
            self.C_pool = samples

            # _raw_scans도 업데이트 (원본 프로파일의 PSD)
            self._raw_scans = [
                (ens.q_grid, ens.C_matrix[i]) for i in range(ens.n_profiles)
            ]

            msg = (f"PSDEnsemble 완료: {len(samples)}개 생성 "
                   f"(PC={ens.K}, 프로파일={ens.n_profiles})")
            self._psd_progress_label.config(text=msg)
            self._show_status(msg, 'success')
            self._plot_raw_scans()
            self._plot_generated_samples()

        elif self._raw_scans:
            # _raw_scans 기반: 수동 PCA (기존 로직과 유사하지만 PSDEnsemble 스타일)
            if len(self._raw_scans) < 2:
                self._show_status("PSDEnsemble에는 2개 이상 프로파일 필요. "
                                  "인라인 PCA를 사용하세요.", 'warning')
                return

            self._psd_progress_label.config(text="PCA 앙상블 생성 중 (raw scans)...")
            self.app.root.update_idletasks()

            q_common = self._make_common_q_grid(n_points=100)
            if q_common is None:
                self._show_status("q 범위 교집합이 없습니다.", 'error')
                return

            C_matrix = self._interpolate_scans_to_grid(q_common)

            # PSDEnsemble 스타일 PCA
            ens = PSDEnsemble()
            ens.q_grid = q_common
            ens.C_matrix = C_matrix
            ens.n_profiles = C_matrix.shape[0]
            ens.fit_pca(var_threshold=threshold)
            samples = ens.generate_samples(n_samples=n_samples, random_seed=seed)

            self._ensemble = ens
            self.q_grid = q_common
            self.C_pool = samples

            msg = (f"PCA 앙상블 완료: {len(samples)}개 생성 "
                   f"(PC={ens.K}, 스캔={ens.n_profiles})")
            self._psd_progress_label.config(text=msg)
            self._show_status(msg, 'success')
            self._plot_generated_samples()
        else:
            self._show_status("데이터를 먼저 로드하세요.", 'warning')

    def _make_common_q_grid(self, n_points=100):
        """모든 스캔의 q 범위 교집합으로 공통 그리드 생성"""
        if not self._raw_scans:
            return None
        q_min = max(s[0].min() for s in self._raw_scans)
        q_max = min(s[0].max() for s in self._raw_scans)
        if q_min >= q_max:
            return None
        return np.logspace(np.log10(q_min), np.log10(q_max), n_points)

    def _interpolate_scans_to_grid(self, q_common):
        """모든 스캔을 공통 q 그리드에 log-log 보간"""
        from scipy.interpolate import interp1d
        C_matrix = np.zeros((len(self._raw_scans), len(q_common)))
        for i, (q, C) in enumerate(self._raw_scans):
            log_interp = interp1d(np.log10(q), np.log10(C),
                                  kind='linear', fill_value='extrapolate')
            C_matrix[i] = 10 ** log_interp(np.log10(q_common))
        return C_matrix

    # ── PSD 앙상블 생성 ──
    def _run_psd_ensemble(self):
        """PCA 기반 PSD 앙상블 생성"""
        if not self._raw_scans:
            self._show_status("PSD 데이터를 먼저 로드하세요.", 'warning')
            return

        try:
            threshold = float(self._pca_threshold_var.get())
            n_samples = int(self._n_samples_var.get())
            seed = int(self._pca_seed_var.get())
            rms_n_sigma = float(self._rms_sigma_var.get())
            mono_tol = float(self._mono_tol_var.get())
        except ValueError:
            self._show_status("PCA 설정값을 확인하세요.", 'error')
            return

        self._psd_progress_label.config(text="공통 q 그리드 생성 중...")
        self.app.root.update_idletasks()

        # 1) 공통 q 그리드
        q_common = self._make_common_q_grid(n_points=100)
        if q_common is None:
            self._show_status("q 범위 교집합이 없습니다.", 'error')
            return

        # 2) 공통 그리드로 보간
        C_matrix = self._interpolate_scans_to_grid(q_common)
        N = C_matrix.shape[0]

        # 3) log 변환
        Y = np.log(C_matrix)
        Y_mean = np.mean(Y, axis=0)

        # 4) 편차 행렬 + SVD
        dY = Y - Y_mean
        if N < 2:
            # 스캔 1개면 PCA 불가 → 노이즈 추가로 샘플 생성
            self._psd_progress_label.config(text="스캔 1개: 노이즈 기반 생성...")
            self.app.root.update_idletasks()
            rng = np.random.default_rng(seed)
            noise_std = 0.1  # log 공간에서 10% 변동
            valid_samples = []
            for _ in range(n_samples * 3):
                Y_new = Y_mean + rng.normal(0, noise_std, len(q_common))
                C_new = np.exp(Y_new)
                valid_samples.append(C_new)
                if len(valid_samples) >= n_samples:
                    break
            self.C_pool = np.array(valid_samples[:n_samples])
            self.q_grid = q_common
            self._psd_progress_label.config(text=f"완료: {len(self.C_pool)}개 생성")
            self._show_status(f"PSD 앙상블 {len(self.C_pool)}개 생성 (노이즈 기반)", 'success')
            self._plot_generated_samples()
            return

        self._psd_progress_label.config(text="SVD 분해 중...")
        self.app.root.update_idletasks()

        U, S, Vt = np.linalg.svd(dY, full_matrices=False)
        eigenvalues = S ** 2 / (N - 1)

        # 5) K 선택 (누적분산 >= threshold)
        cumvar = np.cumsum(eigenvalues) / np.sum(eigenvalues)
        K = int(np.argmax(cumvar >= threshold)) + 1
        K = max(K, 1)

        self._psd_progress_label.config(text=f"PC {K}개 사용, 샘플 생성 중...")
        self.app.root.update_idletasks()

        # 6) 원본 rms roughness 통계 (필터용)
        rms_orig = np.array([
            np.sqrt(np.trapz(C_matrix[i] * q_common ** 2, q_common) / (2 * np.pi))
            for i in range(N)
        ])
        rms_mean = np.mean(rms_orig)
        rms_std = np.std(rms_orig) if N > 1 else rms_mean * 0.1

        # 7) 샘플 생성
        rng = np.random.default_rng(seed)
        valid_samples = []
        attempts = 0
        max_attempts = n_samples * 10

        while len(valid_samples) < n_samples and attempts < max_attempts:
            attempts += 1
            z_k = rng.normal(0, np.sqrt(eigenvalues[:K]))
            Y_new = Y_mean + np.sum(z_k[:, None] * Vt[:K], axis=0)
            C_new = np.exp(Y_new)

            # 물리적 필터 1: rms roughness
            rms_new = np.sqrt(np.trapz(C_new * q_common ** 2, q_common) / (2 * np.pi))
            if abs(rms_new - rms_mean) > rms_n_sigma * rms_std:
                continue

            # 물리적 필터 2: 단조감소 위반 비율
            diffs = np.diff(C_new)
            n_increase = np.sum(diffs > 0)
            violation_ratio = n_increase / len(diffs) if len(diffs) > 0 else 0
            if violation_ratio > mono_tol:
                continue

            valid_samples.append(C_new)

            if len(valid_samples) % 100 == 0:
                self._psd_progress_label.config(
                    text=f"생성: {len(valid_samples)}/{n_samples} (시도: {attempts})")
                self.app.root.update_idletasks()

        self.C_pool = np.array(valid_samples)
        self.q_grid = q_common

        msg = f"PSD 앙상블 {len(self.C_pool)}개 생성 (PC={K}, 시도={attempts})"
        self._psd_progress_label.config(text=msg)
        self._show_status(msg, 'success')
        self._plot_generated_samples()

    def _plot_generated_samples(self):
        """생성된 PSD 샘플 플롯"""
        ax = self._ax_psd_gen
        ax.clear()
        PF = self._get_plot_fonts()

        if self.C_pool is None or self.q_grid is None:
            self._canvas_psd.draw_idle()
            return

        # 원본 (회색)
        for q, C in self._raw_scans:
            ax.loglog(q, C, color='gray', alpha=0.3, linewidth=0.5)

        # 생성 샘플 (최대 100개, 파란 반투명)
        n_show = min(100, len(self.C_pool))
        for i in range(n_show):
            ax.loglog(self.q_grid, self.C_pool[i], color='#3B82F6', alpha=0.08, linewidth=0.5)

        # 평균 (빨간선)
        C_mean = np.exp(np.mean(np.log(self.C_pool), axis=0))
        ax.loglog(self.q_grid, C_mean, 'r-', linewidth=2, label='생성 평균')
        ax.legend(fontsize=PF.get('legend', 8))
        ax.set_xlabel('q (1/m)', fontsize=PF.get('label', 9))
        ax.set_ylabel('C(q) (m⁴)', fontsize=PF.get('label', 9))
        ax.set_title(f'생성된 PSD 앙상블 ({len(self.C_pool)}개)', fontsize=PF.get('title', 10))
        ax.tick_params(labelsize=PF.get('tick', 8))
        ax.grid(True, alpha=0.3)
        self._fig_psd.tight_layout(pad=2.5)
        self._canvas_psd.draw_idle()

    # ================================================================
    # 서브탭 2: q₁ 생성
    # ================================================================
    def _build_q1_tab(self, parent):
        FONTS = self._get_fonts()

        main = ttk.Frame(parent)
        main.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        # 좌측 제어판
        left = ttk.Frame(main, width=380)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 4))
        left.pack_propagate(False)

        # 스크롤 가능한 내부 영역
        q1_canvas = tk.Canvas(left, highlightthickness=0)
        q1_scrollbar = ttk.Scrollbar(left, orient='vertical', command=q1_canvas.yview)
        q1_content = ttk.Frame(q1_canvas)
        q1_content.bind('<Configure>',
                        lambda e: q1_canvas.configure(scrollregion=q1_canvas.bbox('all')))
        q1_canvas.create_window((0, 0), window=q1_content, anchor='nw', width=360)
        q1_canvas.configure(yscrollcommand=q1_scrollbar.set)
        q1_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        q1_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 마우스휠 스크롤
        def _mw(event):
            if event.delta:
                q1_canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')
            elif event.num == 4:
                q1_canvas.yview_scroll(-1, 'units')
            elif event.num == 5:
                q1_canvas.yview_scroll(1, 'units')
        q1_canvas.bind('<Enter>', lambda e: (
            q1_canvas.bind_all('<MouseWheel>', _mw),
            q1_canvas.bind_all('<Button-4>', _mw),
            q1_canvas.bind_all('<Button-5>', _mw)))
        q1_canvas.bind('<Leave>', lambda e: (
            q1_canvas.unbind_all('<MouseWheel>'),
            q1_canvas.unbind_all('<Button-4>'),
            q1_canvas.unbind_all('<Button-5>')))

        # --- 섹션 1: 실험 데이터 (화강암/아스팔트) ---
        sec1 = ttk.LabelFrame(q1_content, text="1) 실험 데이터 — 화강암 비눗물 μ (하한)", padding=6)
        sec1.pack(fill=tk.X, pady=3, padx=2)

        r = ttk.Frame(sec1); r.pack(fill=tk.X, pady=1)
        ttk.Label(r, text="속도 (m/s):", font=FONTS['body']).pack(side=tk.LEFT)
        self._granite_v_var = tk.StringVar(value="0.002, 0.01, 0.046")
        ttk.Entry(r, textvariable=self._granite_v_var, width=24).pack(side=tk.LEFT, padx=2)

        self._granite_mu_vars = {}
        for cname in self._compound_names:
            row = ttk.Frame(sec1); row.pack(fill=tk.X, pady=1)
            ttk.Label(row, text=f"  {cname}:", font=FONTS['body']).pack(side=tk.LEFT)
            var = tk.StringVar(value="")
            ttk.Entry(row, textvariable=var, width=24).pack(side=tk.LEFT, padx=2)
            self._granite_mu_vars[cname] = var

        sec1b = ttk.LabelFrame(q1_content, text="   아스팔트 비눗물 μ (상한)", padding=6)
        sec1b.pack(fill=tk.X, pady=3, padx=2)

        r = ttk.Frame(sec1b); r.pack(fill=tk.X, pady=1)
        ttk.Label(r, text="속도 (m/s):", font=FONTS['body']).pack(side=tk.LEFT)
        self._asphalt_v_var = tk.StringVar(value="0.01, 0.046")
        ttk.Entry(r, textvariable=self._asphalt_v_var, width=24).pack(side=tk.LEFT, padx=2)

        self._asphalt_mu_vars = {}
        for cname in self._compound_names:
            row = ttk.Frame(sec1b); row.pack(fill=tk.X, pady=1)
            ttk.Label(row, text=f"  {cname}:", font=FONTS['body']).pack(side=tk.LEFT)
            var = tk.StringVar(value="")
            ttk.Entry(row, textvariable=var, width=24).pack(side=tk.LEFT, padx=2)
            self._asphalt_mu_vars[cname] = var

        ttk.Button(sec1b, text="실측값 불러오기 (CSV)",
                   command=self._load_bounds_csv).pack(fill=tk.X, pady=2)

        # --- 섹션 2: q₁ 탐색 범위 ---
        sec2 = ttk.LabelFrame(q1_content, text="2) q₁ 탐색 범위", padding=6)
        sec2.pack(fill=tk.X, pady=3, padx=2)

        r = ttk.Frame(sec2); r.pack(fill=tk.X, pady=1)
        ttk.Label(r, text="q₁_min:", font=FONTS['body']).pack(side=tk.LEFT)
        self._q1_min_var = tk.StringVar(value="1e4")
        ttk.Entry(r, textvariable=self._q1_min_var, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(r, text="q₁_max:", font=FONTS['body']).pack(side=tk.LEFT)
        self._q1_max_var = tk.StringVar(value="1e7")
        ttk.Entry(r, textvariable=self._q1_max_var, width=8).pack(side=tk.LEFT, padx=2)

        r2 = ttk.Frame(sec2); r2.pack(fill=tk.X, pady=1)
        ttk.Label(r2, text="탐색 포인트:", font=FONTS['body']).pack(side=tk.LEFT)
        self._q1_scan_n_var = tk.StringVar(value="50")
        ttk.Entry(r2, textvariable=self._q1_scan_n_var, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(r2, text="풀 크기:", font=FONTS['body']).pack(side=tk.LEFT)
        self._q1_pool_size_var = tk.StringVar(value="1000")
        ttk.Entry(r2, textvariable=self._q1_pool_size_var, width=8).pack(side=tk.LEFT, padx=2)

        # --- 섹션 3: 사전 탐색 ---
        sec3 = ttk.LabelFrame(q1_content, text="3) 사전 탐색", padding=6)
        sec3.pack(fill=tk.X, pady=3, padx=2)

        ttk.Button(sec3, text="q₁ 유효 범위 사전 탐색",
                   command=self._prescan_q1,
                   style='Accent.TButton').pack(fill=tk.X, pady=2)
        self._q1_scan_label = ttk.Label(sec3, text="유효 범위: 미탐색", font=FONTS['body'])
        self._q1_scan_label.pack(anchor=tk.W, pady=2)

        # --- 섹션 4: 풀 생성 ---
        sec4 = ttk.LabelFrame(q1_content, text="4) q₁ 풀 생성", padding=6)
        sec4.pack(fill=tk.X, pady=3, padx=2)

        ttk.Button(sec4, text="q₁ 풀 생성",
                   command=self._generate_q1_pool,
                   style='Accent.TButton').pack(fill=tk.X, pady=2)
        self._q1_pool_label = ttk.Label(sec4, text="풀 크기: 0", font=FONTS['body'])
        self._q1_pool_label.pack(anchor=tk.W, pady=2)

        # 우측 플롯
        right = ttk.Frame(main)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        PF = self._get_plot_fonts()
        self._fig_q1 = Figure(figsize=(8, 7), dpi=90)
        self._ax_q1_mu = self._fig_q1.add_subplot(211)
        self._ax_q1_pool = self._fig_q1.add_subplot(212)
        self._fig_q1.tight_layout(pad=2.5)

        self._canvas_q1 = FigureCanvasTkAgg(self._fig_q1, master=right)
        self._canvas_q1.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ── q₁ 탭 콜백 ──
    def _parse_float_list(self, s):
        """콤마 구분 문자열 → float 배열"""
        return np.array([float(x.strip()) for x in s.split(',') if x.strip()])

    def _load_bounds_csv(self):
        """실측 μ 범위 CSV 로드"""
        path = filedialog.askopenfilename(
            title="실측 μ 범위 CSV", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        try:
            data = np.loadtxt(path, delimiter=',', skiprows=1, dtype=str)
            # 형식: surface, compound, v1, v2, ... (첫 행: header)
            self._show_status(f"실측 데이터 로드: {os.path.basename(path)}", 'success')
        except Exception as e:
            self._show_status(f"CSV 로드 실패: {e}", 'error')

    def _get_material_for_compound(self, compound_idx):
        """컴파운드별 ViscoelasticMaterial 반환 (현재 app.material 공유)"""
        mat = getattr(self.app, 'material', None)
        if mat is None:
            return None
        return mat

    def _compute_mu_hys_single(self, psd_func, q_use, C_q, material, v, sigma0, temperature, nu):
        """단일 (PSD, q₁, v) 조합의 μ_hys 계산"""
        modulus_func = lambda omega: material.get_modulus(omega, temperature)
        loss_func = lambda omega, T: material.get_loss_modulus(omega, T)

        g_calc = GCalculator(
            psd_func=psd_func,
            modulus_func=modulus_func,
            sigma_0=sigma0,
            velocity=v,
            poisson_ratio=nu,
            storage_modulus_func=lambda omega: material.get_storage_modulus(omega, temperature),
            loss_modulus_func=lambda omega: material.get_loss_modulus(omega, temperature),
        )
        G_array = g_calc.calculate_G(q_use)

        fc = FrictionCalculator(
            psd_func=psd_func,
            loss_modulus_func=loss_func,
            sigma_0=sigma0,
            velocity=v,
            temperature=temperature,
            poisson_ratio=nu,
        )
        mu_hys, _ = fc.calculate_mu_visc(q_use, G_array, C_q)
        return mu_hys

    def _prescan_q1(self):
        """평균 C(q)로 q₁ 유효 범위 사전 탐색"""
        if self.C_pool is None or self.q_grid is None:
            self._show_status("PSD 앙상블을 먼저 생성하세요.", 'warning')
            return

        material = getattr(self.app, 'material', None)
        if material is None:
            self._show_status("마스터 커브 데이터가 없습니다.", 'warning')
            return

        try:
            q1_min = float(self._q1_min_var.get())
            q1_max = float(self._q1_max_var.get())
            n_scan = int(self._q1_scan_n_var.get())
        except ValueError:
            self._show_status("q₁ 범위 설정을 확인하세요.", 'error')
            return

        # 실험 범위 파싱
        granite_bounds, asphalt_bounds = self._parse_experimental_bounds()
        if granite_bounds is None:
            self._show_status("화강암/아스팔트 μ 데이터를 입력하세요.", 'warning')
            return

        self._q1_scan_label.config(text="탐색 중...")
        self.app.root.update_idletasks()

        # 평균 C(q)
        C_mean = np.exp(np.mean(np.log(self.C_pool), axis=0))
        psd_mean = MeasuredPSD(self.q_grid, C_mean, interpolation_kind='linear')

        q1_candidates = np.logspace(np.log10(q1_min), np.log10(q1_max), n_scan)
        valid_q1s = []
        mu_results = {}  # q1 → {compound: [mu at each v]}

        sigma0 = 0.3e6  # 기본 0.3 MPa
        temperature = 20.0
        nu = 0.5

        for q1 in q1_candidates:
            q_use = self.q_grid[self.q_grid <= q1]
            if len(q_use) < 10:
                continue
            C_q = psd_mean(q_use)

            all_pass = True
            mu_results[q1] = {}

            for ci, cname in enumerate(self._compound_names):
                mat = self._get_material_for_compound(ci)
                if mat is None:
                    continue

                # 화강암 속도점에서 하한 체크
                if cname in granite_bounds:
                    gb = granite_bounds[cname]
                    for vi, v in enumerate(gb['v']):
                        mu = self._compute_mu_hys_single(
                            psd_mean, q_use, C_q, mat, v, sigma0, temperature, nu)
                        if vi < len(gb['mu']) and mu < gb['mu'][vi]:
                            all_pass = False

                # 아스팔트 속도점에서 상한 체크
                if cname in asphalt_bounds:
                    ab = asphalt_bounds[cname]
                    for vi, v in enumerate(ab['v']):
                        mu = self._compute_mu_hys_single(
                            psd_mean, q_use, C_q, mat, v, sigma0, temperature, nu)
                        if vi < len(ab['mu']) and mu > ab['mu'][vi]:
                            all_pass = False

            if all_pass:
                valid_q1s.append(q1)

        if valid_q1s:
            self._q1_valid_min = min(valid_q1s)
            self._q1_valid_max = max(valid_q1s)
            self._q1_scan_label.config(
                text=f"유효 범위: [{self._q1_valid_min:.2e}, {self._q1_valid_max:.2e}] (1/m)")
            self._show_status(f"q₁ 유효 범위: {len(valid_q1s)}/{n_scan}개 통과", 'success')
        else:
            self._q1_scan_label.config(text="유효 범위: 없음 (범위/데이터 확인)")
            self._show_status("유효한 q₁이 없습니다. 범위를 넓히세요.", 'warning')

        self._plot_q1_scan(q1_candidates, valid_q1s)

    def _parse_experimental_bounds(self):
        """실험 데이터 Entry에서 범위 파싱"""
        granite_bounds = {}
        asphalt_bounds = {}
        try:
            g_v = self._parse_float_list(self._granite_v_var.get())
            a_v = self._parse_float_list(self._asphalt_v_var.get())
        except (ValueError, AttributeError):
            return None, None

        if len(g_v) == 0 and len(a_v) == 0:
            return None, None

        for cname in self._compound_names:
            # 화강암
            g_str = self._granite_mu_vars[cname].get().strip()
            if g_str:
                try:
                    g_mu = self._parse_float_list(g_str)
                    granite_bounds[cname] = {'v': g_v[:len(g_mu)], 'mu': g_mu}
                except ValueError:
                    pass
            # 아스팔트
            a_str = self._asphalt_mu_vars[cname].get().strip()
            if a_str:
                try:
                    a_mu = self._parse_float_list(a_str)
                    asphalt_bounds[cname] = {'v': a_v[:len(a_mu)], 'mu': a_mu}
                except ValueError:
                    pass

        if not granite_bounds and not asphalt_bounds:
            return None, None
        return granite_bounds, asphalt_bounds

    def _plot_q1_scan(self, q1_candidates, valid_q1s):
        """q₁ 사전 탐색 결과 플롯"""
        ax = self._ax_q1_mu
        ax.clear()
        PF = self._get_plot_fonts()

        log_q1 = np.log10(q1_candidates)
        ax.axhline(y=0, color='gray', alpha=0.3)

        if valid_q1s:
            log_valid = np.log10(valid_q1s)
            ax.axvspan(log_valid.min(), log_valid.max(),
                       alpha=0.2, color='#3B82F6', label='유효 범위')
            for lq in log_valid:
                ax.axvline(x=lq, color='#3B82F6', alpha=0.1, linewidth=0.5)

        ax.set_xlabel('log₁₀(q₁) (1/m)', fontsize=PF.get('label', 9))
        ax.set_ylabel('μ_hys', fontsize=PF.get('label', 9))
        ax.set_title('q₁ 사전 탐색', fontsize=PF.get('title', 10))
        ax.legend(fontsize=PF.get('legend', 8))
        ax.tick_params(labelsize=PF.get('tick', 8))
        ax.grid(True, alpha=0.3)

        # 하단: q₁ 풀 분포
        ax2 = self._ax_q1_pool
        ax2.clear()
        if self.q1_pool is not None:
            ax2.hist(np.log10(self.q1_pool), bins=50, color='#3B82F6', alpha=0.7)
            ax2.set_xlabel('log₁₀(q₁)', fontsize=PF.get('label', 9))
            ax2.set_ylabel('빈도', fontsize=PF.get('label', 9))
            ax2.set_title(f'q₁ 풀 분포 ({len(self.q1_pool)}개)', fontsize=PF.get('title', 10))
        ax2.tick_params(labelsize=PF.get('tick', 8))
        ax2.grid(True, alpha=0.3)

        self._fig_q1.tight_layout(pad=2.5)
        self._canvas_q1.draw_idle()

    def _generate_q1_pool(self):
        """유효 범위에서 q₁ 풀 log-uniform 랜덤 샘플링"""
        if self._q1_valid_min is None or self._q1_valid_max is None:
            # 유효 범위 없으면 전체 범위 사용
            try:
                q1_min = float(self._q1_min_var.get())
                q1_max = float(self._q1_max_var.get())
            except ValueError:
                self._show_status("q₁ 범위를 확인하세요.", 'error')
                return
        else:
            q1_min = self._q1_valid_min
            q1_max = self._q1_valid_max

        try:
            pool_size = int(self._q1_pool_size_var.get())
        except ValueError:
            pool_size = 1000

        rng = np.random.default_rng(42)
        log_q1 = rng.uniform(np.log10(q1_min), np.log10(q1_max), pool_size)
        self.q1_pool = 10 ** log_q1

        self._q1_pool_label.config(text=f"풀 크기: {len(self.q1_pool)}")
        self._show_status(f"q₁ 풀 {len(self.q1_pool)}개 생성 완료", 'success')
        self._plot_q1_scan(
            np.logspace(np.log10(q1_min), np.log10(q1_max), 50),
            self.q1_pool)

    # ================================================================
    # 서브탭 3: Monte Carlo
    # ================================================================
    def _build_mc_tab(self, parent):
        FONTS = self._get_fonts()

        main = ttk.Frame(parent)
        main.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        # 좌측 제어판
        left = ttk.Frame(main, width=380)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 4))
        left.pack_propagate(False)

        # 스크롤 영역
        mc_canvas = tk.Canvas(left, highlightthickness=0)
        mc_sb = ttk.Scrollbar(left, orient='vertical', command=mc_canvas.yview)
        mc_content = ttk.Frame(mc_canvas)
        mc_content.bind('<Configure>',
                        lambda e: mc_canvas.configure(scrollregion=mc_canvas.bbox('all')))
        mc_canvas.create_window((0, 0), window=mc_content, anchor='nw', width=360)
        mc_canvas.configure(yscrollcommand=mc_sb.set)
        mc_sb.pack(side=tk.RIGHT, fill=tk.Y)
        mc_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        def _mw(event):
            if event.delta:
                mc_canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')
            elif event.num == 4:
                mc_canvas.yview_scroll(-1, 'units')
            elif event.num == 5:
                mc_canvas.yview_scroll(1, 'units')
        mc_canvas.bind('<Enter>', lambda e: (
            mc_canvas.bind_all('<MouseWheel>', _mw),
            mc_canvas.bind_all('<Button-4>', _mw),
            mc_canvas.bind_all('<Button-5>', _mw)))
        mc_canvas.bind('<Leave>', lambda e: (
            mc_canvas.unbind_all('<MouseWheel>'),
            mc_canvas.unbind_all('<Button-4>'),
            mc_canvas.unbind_all('<Button-5>')))

        # --- 섹션 1: DMA 데이터 확인 ---
        sec1 = ttk.LabelFrame(mc_content, text="1) DMA / 컴파운드 설정", padding=6)
        sec1.pack(fill=tk.X, pady=3, padx=2)

        ttk.Button(sec1, text="Tab 1 마스터 커브 사용",
                   command=self._load_material_from_tab1).pack(fill=tk.X, pady=1)

        self._compound_vars = {}
        for cname in self._compound_names:
            cf = ttk.LabelFrame(sec1, text=cname, padding=4)
            cf.pack(fill=tk.X, pady=2)
            vars_dict = {}

            r1 = ttk.Frame(cf); r1.pack(fill=tk.X, pady=1)
            ttk.Label(r1, text="σ₀ (MPa):", font=FONTS['body']).pack(side=tk.LEFT)
            var_s = tk.StringVar(value="0.3")
            ttk.Entry(r1, textvariable=var_s, width=6).pack(side=tk.LEFT, padx=2)
            vars_dict['sigma0'] = var_s

            ttk.Label(r1, text="T(°C):", font=FONTS['body']).pack(side=tk.LEFT)
            var_t = tk.StringVar(value="20")
            ttk.Entry(r1, textvariable=var_t, width=5).pack(side=tk.LEFT, padx=2)
            vars_dict['temperature'] = var_t

            r2 = ttk.Frame(cf); r2.pack(fill=tk.X, pady=1)
            ttk.Label(r2, text="ν:", font=FONTS['body']).pack(side=tk.LEFT)
            var_nu = tk.StringVar(value="0.5")
            ttk.Entry(r2, textvariable=var_nu, width=5).pack(side=tk.LEFT, padx=2)
            vars_dict['nu'] = var_nu

            self._compound_vars[cname] = vars_dict

        # --- 섹션 2: 건식 마찰 데이터 ---
        sec2 = ttk.LabelFrame(mc_content, text="2) 건식 마찰 데이터 (Likelihood)", padding=6)
        sec2.pack(fill=tk.X, pady=3, padx=2)

        r = ttk.Frame(sec2); r.pack(fill=tk.X, pady=1)
        ttk.Label(r, text="log₁₀(V) 속도점:", font=FONTS['body']).pack(side=tk.LEFT)
        self._dry_logv_var = tk.StringVar(value="-4,-3.337,-2.699,-2,-1.337,-0.678")
        ttk.Entry(r, textvariable=self._dry_logv_var, width=28).pack(side=tk.LEFT, padx=2)

        self._dry_mu_vars = {}
        for cname in self._compound_names:
            row = ttk.Frame(sec2); row.pack(fill=tk.X, pady=1)
            ttk.Label(row, text=f"  {cname} μ_dry:", font=FONTS['body']).pack(side=tk.LEFT)
            var = tk.StringVar(value="")
            ttk.Entry(row, textvariable=var, width=28).pack(side=tk.LEFT, padx=2)
            self._dry_mu_vars[cname] = var

        ttk.Button(sec2, text="실측값 불러오기 (CSV)",
                   command=self._load_dry_csv).pack(fill=tk.X, pady=2)

        # --- 섹션 3: Monte Carlo 설정 ---
        sec3 = ttk.LabelFrame(mc_content, text="3) Monte Carlo 설정", padding=6)
        sec3.pack(fill=tk.X, pady=3, padx=2)

        r1 = ttk.Frame(sec3); r1.pack(fill=tk.X, pady=1)
        ttk.Label(r1, text="반복 횟수 N:", font=FONTS['body']).pack(side=tk.LEFT)
        self._mc_n_var = tk.StringVar(value="3000")
        ttk.Entry(r1, textvariable=self._mc_n_var, width=8).pack(side=tk.LEFT, padx=2)

        r2 = ttk.Frame(sec3); r2.pack(fill=tk.X, pady=1)
        ttk.Label(r2, text="Random seed:", font=FONTS['body']).pack(side=tk.LEFT)
        self._mc_seed_var = tk.StringVar(value="42")
        ttk.Entry(r2, textvariable=self._mc_seed_var, width=6).pack(side=tk.LEFT, padx=2)

        r3 = ttk.Frame(sec3); r3.pack(fill=tk.X, pady=1)
        ttk.Label(r3, text="상위 X %:", font=FONTS['body']).pack(side=tk.LEFT)
        self._mc_top_pct_var = tk.StringVar(value="10")
        ttk.Entry(r3, textvariable=self._mc_top_pct_var, width=5).pack(side=tk.LEFT, padx=2)

        r4 = ttk.Frame(sec3); r4.pack(fill=tk.X, pady=1)
        ttk.Label(r4, text="1단계 필터 속도 (m/s):", font=FONTS['body']).pack(side=tk.LEFT)
        self._mc_filter_v_var = tk.StringVar(value="0.01, 0.046")
        ttk.Entry(r4, textvariable=self._mc_filter_v_var, width=16).pack(side=tk.LEFT, padx=2)

        # --- 섹션 4: 실행 ---
        sec4 = ttk.LabelFrame(mc_content, text="4) 실행", padding=6)
        sec4.pack(fill=tk.X, pady=3, padx=2)

        btn_row = ttk.Frame(sec4); btn_row.pack(fill=tk.X, pady=2)
        ttk.Button(btn_row, text="Monte Carlo 시작",
                   command=self._start_monte_carlo,
                   style='Accent.TButton').pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(btn_row, text="중단",
                   command=self._stop_monte_carlo).pack(side=tk.LEFT)

        self._mc_progress_var = tk.IntVar(value=0)
        self._mc_progressbar = ttk.Progressbar(
            sec4, variable=self._mc_progress_var, maximum=100)
        self._mc_progressbar.pack(fill=tk.X, pady=2)

        self._mc_status_label = ttk.Label(sec4, text="대기 중", font=FONTS['body'])
        self._mc_status_label.pack(anchor=tk.W)
        self._mc_eta_label = ttk.Label(sec4, text="", font=FONTS['small'],
                                       foreground='#64748B')
        self._mc_eta_label.pack(anchor=tk.W)

        # 우측 패널: 로그 + 진행 차트
        right = ttk.Frame(main)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 로그 텍스트
        log_frame = ttk.LabelFrame(right, text="계산 로그", padding=4)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 4))

        self._mc_log_text = tk.Text(log_frame, wrap=tk.WORD, height=15,
                                    font=self._get_fonts()['mono_small'],
                                    bg='#1E293B', fg='#E2E8F0',
                                    insertbackground='#E2E8F0')
        mc_log_sb = ttk.Scrollbar(log_frame, orient='vertical',
                                  command=self._mc_log_text.yview)
        self._mc_log_text.configure(yscrollcommand=mc_log_sb.set)
        mc_log_sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._mc_log_text.pack(fill=tk.BOTH, expand=True)

        # 진행 차트
        chart_frame = ttk.LabelFrame(right, text="누적 생존 조합", padding=4)
        chart_frame.pack(fill=tk.BOTH, expand=True)

        PF = self._get_plot_fonts()
        self._fig_mc_progress = Figure(figsize=(6, 3), dpi=90)
        self._ax_mc_progress = self._fig_mc_progress.add_subplot(111)
        self._fig_mc_progress.tight_layout(pad=2.0)
        self._canvas_mc_progress = FigureCanvasTkAgg(self._fig_mc_progress, master=chart_frame)
        self._canvas_mc_progress.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ── MC 탭 콜백 ──
    def _load_material_from_tab1(self):
        """app.material 또는 app.persson_master_curve에서 재료 정보 확인"""
        mat = getattr(self.app, 'material', None)
        if mat is not None:
            self._show_status("Tab 1 마스터 커브 확인 완료", 'success')
        else:
            self._show_status("마스터 커브가 설정되지 않았습니다.", 'warning')

    def _load_dry_csv(self):
        """건식 마찰 CSV 로드"""
        path = filedialog.askopenfilename(
            title="건식 마찰 데이터 CSV", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        try:
            # 형식: 첫 행 header, 첫 열 compound, 나머지 열 μ at each velocity
            with open(path, 'r') as f:
                lines = f.readlines()
            if len(lines) < 2:
                return
            header = lines[0].strip().split(',')
            # 속도 값 (header[1:])
            velocities = [float(x) for x in header[1:]]
            self._dry_logv_var.set(','.join(f'{np.log10(v):.3f}' for v in velocities))
            for line in lines[1:]:
                parts = line.strip().split(',')
                cname = parts[0].strip()
                if cname in self._dry_mu_vars:
                    self._dry_mu_vars[cname].set(','.join(parts[1:]))
            self._show_status(f"건식 마찰 데이터 로드: {os.path.basename(path)}", 'success')
        except Exception as e:
            self._show_status(f"CSV 로드 실패: {e}", 'error')

    def _mc_log(self, msg):
        """로그 텍스트에 메시지 추가 (thread-safe)"""
        def _append():
            self._mc_log_text.insert(tk.END, msg + '\n')
            self._mc_log_text.see(tk.END)
        self.app.root.after(0, _append)

    def _start_monte_carlo(self):
        """Monte Carlo 시작 (별도 스레드)"""
        if self.C_pool is None or self.q_grid is None:
            self._show_status("PSD 앙상블을 먼저 생성하세요.", 'warning')
            return
        if self.q1_pool is None:
            self._show_status("q₁ 풀을 먼저 생성하세요.", 'warning')
            return

        material = getattr(self.app, 'material', None)
        if material is None:
            self._show_status("마스터 커브가 없습니다.", 'warning')
            return

        self._stop_flag = False
        self._mc_log_text.delete('1.0', tk.END)

        thread = threading.Thread(target=self._run_monte_carlo_thread, daemon=True)
        thread.start()

    def _stop_monte_carlo(self):
        """Monte Carlo 중단"""
        self._stop_flag = True

    def _run_monte_carlo_thread(self):
        """Monte Carlo 계산 (별도 스레드)"""
        try:
            N_iter = int(self._mc_n_var.get())
            seed = int(self._mc_seed_var.get())
            top_pct = float(self._mc_top_pct_var.get()) / 100.0
            filter_v = self._parse_float_list(self._mc_filter_v_var.get())
        except ValueError:
            self._mc_log("설정값 오류")
            return

        # 건식 마찰 데이터 파싱
        try:
            dry_logv = self._parse_float_list(self._dry_logv_var.get())
            dry_v = 10 ** dry_logv
        except ValueError:
            self._mc_log("건식 마찰 속도 파싱 오류")
            return

        dry_mu = {}
        for cname in self._compound_names:
            s = self._dry_mu_vars[cname].get().strip()
            if s:
                try:
                    dry_mu[cname] = self._parse_float_list(s)
                except ValueError:
                    pass

        # 화강암/아스팔트 범위
        granite_bounds, asphalt_bounds = self._parse_experimental_bounds()

        # 컴파운드 파라미터
        compound_params = {}
        for cname in self._compound_names:
            cv = self._compound_vars[cname]
            try:
                compound_params[cname] = {
                    'sigma0': float(cv['sigma0'].get()) * 1e6,  # MPa → Pa
                    'temperature': float(cv['temperature'].get()),
                    'nu': float(cv['nu'].get()),
                }
            except ValueError:
                compound_params[cname] = {'sigma0': 0.3e6, 'temperature': 20.0, 'nu': 0.5}

        material = self.app.material
        rng = np.random.default_rng(seed)

        results = []
        n_pass1 = 0
        n_total = 0
        t_start = time.time()

        # 진행 차트 데이터
        progress_x = []
        progress_y = []

        self._mc_log(f"Monte Carlo 시작: N={N_iter}, seed={seed}")
        self._mc_log(f"PSD 풀: {len(self.C_pool)}, q₁ 풀: {len(self.q1_pool)}")
        self._mc_log(f"1단계 필터 속도: {filter_v}")
        self._mc_log(f"2단계 속도점: {len(dry_v)}개")
        self._mc_log("=" * 50)

        for iteration in range(N_iter):
            if self._stop_flag:
                self._mc_log(f"\n중단됨 (iter={iteration})")
                break

            n_total += 1

            # 1. 랜덤 샘플링
            idx_C = rng.integers(0, len(self.C_pool))
            idx_q1 = rng.integers(0, len(self.q1_pool))
            C_sample = self.C_pool[idx_C]
            q1_sample = self.q1_pool[idx_q1]

            # 2. q 배열
            q_use = self.q_grid[self.q_grid <= q1_sample]
            if len(q_use) < 10:
                continue

            # 3. PSD 생성
            C_q_use = C_sample[:len(q_use)]
            psd = MeasuredPSD(q_use, C_q_use, interpolation_kind='linear')

            # 4. 1단계 필터
            pass_filter = True
            if granite_bounds or asphalt_bounds:
                for ci, cname in enumerate(self._compound_names):
                    if not pass_filter:
                        break
                    cp = compound_params.get(cname, compound_params[self._compound_names[0]])
                    mat = self._get_material_for_compound(ci)
                    if mat is None:
                        continue

                    for v in filter_v:
                        C_q = psd(q_use)
                        mu = self._compute_mu_hys_single(
                            psd, q_use, C_q, mat, v, cp['sigma0'], cp['temperature'], cp['nu'])

                        # 하한 체크
                        if cname in (granite_bounds or {}):
                            gb = granite_bounds[cname]
                            for vi, gv in enumerate(gb['v']):
                                if abs(v - gv) / max(gv, 1e-10) < 0.1:
                                    if vi < len(gb['mu']) and mu < gb['mu'][vi]:
                                        pass_filter = False
                                        break
                        # 상한 체크
                        if cname in (asphalt_bounds or {}):
                            ab = asphalt_bounds[cname]
                            for vi, av in enumerate(ab['v']):
                                if abs(v - av) / max(av, 1e-10) < 0.1:
                                    if vi < len(ab['mu']) and mu > ab['mu'][vi]:
                                        pass_filter = False
                                        break

            if not pass_filter:
                continue

            n_pass1 += 1

            # 5. 2단계 점수 계산 (전체 속도점)
            score = 0.0
            mu_pred_all = {}

            for ci, cname in enumerate(self._compound_names):
                if cname not in dry_mu:
                    continue
                cp = compound_params.get(cname, compound_params[self._compound_names[0]])
                mat = self._get_material_for_compound(ci)
                if mat is None:
                    continue

                mu_exp = dry_mu[cname]
                mu_pred = []

                for vi, v in enumerate(dry_v):
                    if vi >= len(mu_exp):
                        break
                    C_q = psd(q_use)
                    mu = self._compute_mu_hys_single(
                        psd, q_use, C_q, mat, v, cp['sigma0'], cp['temperature'], cp['nu'])
                    mu_pred.append(mu)
                    score += (mu - mu_exp[vi]) ** 2

                mu_pred_all[cname] = mu_pred

            score = np.exp(-score)
            results.append({
                'idx_C': idx_C, 'idx_q1': idx_q1,
                'q1': q1_sample, 'score': score,
                'mu_pred': mu_pred_all,
            })

            # 진행 업데이트
            if iteration % 50 == 0 or iteration == N_iter - 1:
                elapsed = time.time() - t_start
                rate = (iteration + 1) / max(elapsed, 0.01)
                eta = (N_iter - iteration - 1) / max(rate, 0.01)
                pct = int((iteration + 1) / N_iter * 100)

                progress_x.append(iteration)
                progress_y.append(n_pass1)

                def _update(it=iteration, p=pct, np1=n_pass1, nt=n_total,
                            eta_s=eta, px=list(progress_x), py=list(progress_y)):
                    self._mc_progress_var.set(p)
                    self._mc_status_label.config(
                        text=f"iter {it+1}/{N_iter} | 통과: {np1}/{nt}")
                    m, s = divmod(int(eta_s), 60)
                    self._mc_eta_label.config(text=f"예상 잔여: {m}m {s}s")
                    # 진행 차트 업데이트
                    self._ax_mc_progress.clear()
                    self._ax_mc_progress.plot(px, py, '-', color='#3B82F6', linewidth=1.5)
                    self._ax_mc_progress.set_xlabel('반복')
                    self._ax_mc_progress.set_ylabel('누적 생존')
                    self._ax_mc_progress.grid(True, alpha=0.3)
                    self._fig_mc_progress.tight_layout(pad=2.0)
                    self._canvas_mc_progress.draw_idle()
                self.app.root.after(0, _update)

            if iteration % 200 == 0:
                self._mc_log(f"[{iteration+1}/{N_iter}] 통과: {n_pass1}/{n_total}, "
                             f"score_last={score:.4e}")

        # 결과 정리
        if results:
            scores = np.array([r['score'] for r in results])
            threshold = np.percentile(scores, 100 * (1 - top_pct))
            survivors = [r for r in results if r['score'] >= threshold]
        else:
            survivors = []

        self.mc_results = {
            'all_results': results,
            'survivors': survivors,
            'n_total': n_total,
            'n_pass1': n_pass1,
            'N_iter': N_iter,
            'dry_v': dry_v,
            'dry_mu': dry_mu,
            'top_pct': top_pct,
        }

        # 자동 저장
        try:
            pkl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    'results_mc.pkl')
            with open(pkl_path, 'wb') as f:
                pickle.dump(self.mc_results, f)
            self._mc_log(f"\n결과 저장: {pkl_path}")
        except Exception as e:
            self._mc_log(f"저장 실패: {e}")

        msg = (f"\nMonte Carlo 완료: {n_total}회 중 1단계 통과 {n_pass1}개, "
               f"최종 생존 {len(survivors)}개")
        self._mc_log(msg)
        self._mc_log("=" * 50)

        def _final():
            self._mc_progress_var.set(100)
            self._mc_status_label.config(text=f"완료: 생존 {len(survivors)}개")
            self._mc_eta_label.config(text="")
            self._show_status(msg.strip(), 'success')
            # 결과 탭 자동 업데이트
            self._update_result_plots()
        self.app.root.after(0, _final)

    # ================================================================
    # 서브탭 4: 결과
    # ================================================================
    def _build_result_tab(self, parent):
        FONTS = self._get_fonts()

        main = ttk.Frame(parent)
        main.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        # 좌측 요약 패널
        left = ttk.Frame(main, width=300)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 4))
        left.pack_propagate(False)

        sec_summary = ttk.LabelFrame(left, text="결과 요약", padding=6)
        sec_summary.pack(fill=tk.X, pady=3, padx=2)

        self._result_summary_text = tk.Text(sec_summary, wrap=tk.WORD, height=18,
                                            font=FONTS['mono_small'],
                                            bg='#F8FAFC', fg='#1E293B')
        self._result_summary_text.pack(fill=tk.BOTH, expand=True)

        # 버튼들
        btn_frame = ttk.LabelFrame(left, text="파일 관리", padding=6)
        btn_frame.pack(fill=tk.X, pady=3, padx=2)

        ttk.Button(btn_frame, text="결과 불러오기 (pkl)",
                   command=self._load_results_pkl).pack(fill=tk.X, pady=1)
        ttk.Button(btn_frame, text="결과 저장 (pkl)",
                   command=self._save_results_pkl).pack(fill=tk.X, pady=1)
        ttk.Button(btn_frame, text="플롯 저장 (PNG)",
                   command=self._save_result_plot).pack(fill=tk.X, pady=1)

        # 우측 플롯 (2×2)
        right = ttk.Frame(main)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        PF = self._get_plot_fonts()
        self._fig_result = Figure(figsize=(10, 8), dpi=90)
        self._ax_r_q1_hist = self._fig_result.add_subplot(221)
        self._ax_r_cq_band = self._fig_result.add_subplot(222)
        self._ax_r_score = self._fig_result.add_subplot(223)
        self._ax_r_scatter = self._fig_result.add_subplot(224)
        self._fig_result.tight_layout(pad=2.5)

        self._canvas_result = FigureCanvasTkAgg(self._fig_result, master=right)
        self._canvas_result.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ── 결과 탭 콜백 ──
    def _update_result_plots(self):
        """MC 결과로 4개 서브플롯 업데이트"""
        if self.mc_results is None:
            return

        PF = self._get_plot_fonts()
        results = self.mc_results
        all_res = results.get('all_results', [])
        survivors = results.get('survivors', [])
        n_total = results.get('n_total', 0)
        n_pass1 = results.get('n_pass1', 0)

        if not all_res:
            return

        # 요약 텍스트
        all_q1 = np.array([r['q1'] for r in all_res])
        all_scores = np.array([r['score'] for r in all_res])
        surv_q1 = np.array([r['q1'] for r in survivors]) if survivors else np.array([])

        self._result_summary_text.delete('1.0', tk.END)
        summary = f"총 반복 수: {results.get('N_iter', 0)}\n"
        summary += f"1단계 통과율: {n_pass1}/{n_total}"
        if n_total > 0:
            summary += f" ({n_pass1/n_total*100:.1f}%)"
        summary += f"\n2단계 최종 생존: {len(survivors)}\n"
        if len(surv_q1) > 0:
            q1_med = np.median(surv_q1)
            q1_lo = np.percentile(surv_q1, 5)
            q1_hi = np.percentile(surv_q1, 95)
            summary += f"\nq₁ 중앙값: {q1_med:.3e} m⁻¹\n"
            summary += f"90% CI: [{q1_lo:.3e}, {q1_hi:.3e}]\n"
        self._result_summary_text.insert('1.0', summary)

        # 플롯 1 (좌상): q₁ 분포
        ax1 = self._ax_r_q1_hist
        ax1.clear()
        log_all_q1 = np.log10(all_q1)
        ax1.hist(log_all_q1, bins=40, color='#CBD5E1', alpha=0.7, label='전체')
        if len(surv_q1) > 0:
            log_surv = np.log10(surv_q1)
            ax1.hist(log_surv, bins=40, color='#3B82F6', alpha=0.7, label='생존')
            ax1.axvline(np.median(log_surv), color='red', linestyle='--',
                        linewidth=1.5, label=f'중앙값')
            ax1.axvline(np.percentile(log_surv, 5), color='orange', linestyle=':',
                        linewidth=1, label='90% CI')
            ax1.axvline(np.percentile(log_surv, 95), color='orange', linestyle=':',
                        linewidth=1)
        ax1.set_xlabel('log₁₀(q₁)', fontsize=PF.get('label', 9))
        ax1.set_ylabel('빈도', fontsize=PF.get('label', 9))
        ax1.set_title('q₁ 분포', fontsize=PF.get('title', 10))
        ax1.legend(fontsize=PF.get('legend', 8))
        ax1.tick_params(labelsize=PF.get('tick', 8))
        ax1.grid(True, alpha=0.3)

        # 플롯 2 (우상): C(q) Posterior 밴드
        ax2 = self._ax_r_cq_band
        ax2.clear()
        if self.q_grid is not None and self.C_pool is not None:
            # 원본 스캔 (회색)
            for q, C in self._raw_scans:
                ax2.loglog(q, C, color='gray', alpha=0.3, linewidth=0.5)
            # 생존 C(q)
            if survivors:
                surv_C_indices = [r['idx_C'] for r in survivors]
                surv_C = self.C_pool[surv_C_indices]
                # 중앙값, 5%, 95%
                C_median = np.median(surv_C, axis=0)
                C_lo = np.percentile(surv_C, 5, axis=0)
                C_hi = np.percentile(surv_C, 95, axis=0)
                ax2.fill_between(self.q_grid, C_lo, C_hi,
                                 alpha=0.3, color='#3B82F6', label='90% CI')
                ax2.loglog(self.q_grid, C_median, 'r-', linewidth=2, label='중앙값')
                # 생존 샘플 일부
                for i in range(min(30, len(surv_C))):
                    ax2.loglog(self.q_grid, surv_C[i], color='#3B82F6',
                               alpha=0.05, linewidth=0.5)
                ax2.legend(fontsize=PF.get('legend', 8))
        ax2.set_xlabel('q (1/m)', fontsize=PF.get('label', 9))
        ax2.set_ylabel('C(q) (m⁴)', fontsize=PF.get('label', 9))
        ax2.set_title('C(q) Posterior', fontsize=PF.get('title', 10))
        ax2.tick_params(labelsize=PF.get('tick', 8))
        ax2.grid(True, alpha=0.3)

        # 플롯 3 (좌하): 점수 분포
        ax3 = self._ax_r_score
        ax3.clear()
        if len(all_scores) > 0:
            # log 변환 (0 제거)
            valid_scores = all_scores[all_scores > 0]
            if len(valid_scores) > 0:
                ax3.hist(np.log10(valid_scores), bins=50, color='#CBD5E1', alpha=0.7)
                top_pct = results.get('top_pct', 0.1)
                threshold = np.percentile(valid_scores, 100 * (1 - top_pct))
                if threshold > 0:
                    ax3.axvline(np.log10(threshold), color='red', linestyle='--',
                                linewidth=1.5, label=f'상위 {top_pct*100:.0f}%')
                    ax3.legend(fontsize=PF.get('legend', 8))
        ax3.set_xlabel('log₁₀(score)', fontsize=PF.get('label', 9))
        ax3.set_ylabel('빈도', fontsize=PF.get('label', 9))
        ax3.set_title('점수 분포', fontsize=PF.get('title', 10))
        ax3.tick_params(labelsize=PF.get('tick', 8))
        ax3.grid(True, alpha=0.3)

        # 플롯 4 (우하): (q₁, C_rms) 산점도
        ax4 = self._ax_r_scatter
        ax4.clear()
        if self.q_grid is not None and self.C_pool is not None:
            # 전체
            all_c_rms = []
            all_log_q1_s = []
            for r in all_res:
                C_s = self.C_pool[r['idx_C']]
                rms = np.sqrt(np.mean(C_s ** 2))
                all_c_rms.append(rms)
                all_log_q1_s.append(np.log10(r['q1']))
            ax4.scatter(all_log_q1_s, all_c_rms, s=3, color='gray', alpha=0.3, label='전체')

            # 생존
            if survivors:
                surv_rms = []
                surv_lq1 = []
                for r in survivors:
                    C_s = self.C_pool[r['idx_C']]
                    rms = np.sqrt(np.mean(C_s ** 2))
                    surv_rms.append(rms)
                    surv_lq1.append(np.log10(r['q1']))
                ax4.scatter(surv_lq1, surv_rms, s=40, color='red', marker='*',
                            alpha=0.7, label='생존')
            ax4.legend(fontsize=PF.get('legend', 8))

        ax4.set_xlabel('log₁₀(q₁)', fontsize=PF.get('label', 9))
        ax4.set_ylabel('C(q) rms', fontsize=PF.get('label', 9))
        ax4.set_title('(q₁, C_rms) 산점도', fontsize=PF.get('title', 10))
        ax4.tick_params(labelsize=PF.get('tick', 8))
        ax4.grid(True, alpha=0.3)

        self._fig_result.tight_layout(pad=2.5)
        self._canvas_result.draw_idle()

    def _load_results_pkl(self):
        """pkl 파일에서 결과 불러오기"""
        path = filedialog.askopenfilename(
            title="결과 불러오기", filetypes=[("Pickle", "*.pkl")])
        if not path:
            return
        try:
            with open(path, 'rb') as f:
                self.mc_results = pickle.load(f)
            self._show_status(f"결과 로드: {os.path.basename(path)}", 'success')
            self._update_result_plots()
        except Exception as e:
            self._show_status(f"로드 실패: {e}", 'error')

    def _save_results_pkl(self):
        """결과를 pkl로 저장"""
        if self.mc_results is None:
            self._show_status("저장할 결과가 없습니다.", 'warning')
            return
        path = filedialog.asksaveasfilename(
            title="결과 저장", defaultextension=".pkl",
            filetypes=[("Pickle", "*.pkl")])
        if not path:
            return
        try:
            with open(path, 'wb') as f:
                pickle.dump(self.mc_results, f)
            self._show_status(f"결과 저장: {os.path.basename(path)}", 'success')
        except Exception as e:
            self._show_status(f"저장 실패: {e}", 'error')

    def _save_result_plot(self):
        """결과 플롯 PNG 저장"""
        path = filedialog.asksaveasfilename(
            title="플롯 저장", defaultextension=".png",
            filetypes=[("PNG", "*.png")])
        if not path:
            return
        try:
            self._fig_result.savefig(path, dpi=150, bbox_inches='tight')
            self._show_status(f"플롯 저장: {os.path.basename(path)}", 'success')
        except Exception as e:
            self._show_status(f"저장 실패: {e}", 'error')
