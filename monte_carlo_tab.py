"""
Monte Carlo Tab for Persson Friction Model
==========================================
유효 표면 PSD와 차단 파수 q₁의 Monte Carlo 역추정

Usage in main.py:
    from monte_carlo_tab import bind_monte_carlo_tab
    # __init__ 내에서, _create_main_layout() 호출 전에:
    bind_monte_carlo_tab(self)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import os
import pickle
import threading
import time

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from scipy.interpolate import interp1d
from scipy.fft import fft, fftfreq
from scipy.signal import detrend as scipy_detrend
from scipy.special import gamma as gamma_func

from persson_model.core.psd_models import MeasuredPSD
from persson_model.core.g_calculator import GCalculator
from persson_model.core.friction import FrictionCalculator

# numpy >= 2.0 renamed trapz → trapezoid
_trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))


# ================================================================
# ====  PSDComputer — IDADA 프로파일 → PSD 계산 (psd repo 동일)  ====
# ================================================================

class PSDComputer:
    """1D 노면 프로파일에서 2D PSD C(q)를 계산한다.

    psd_generator.py (k100kan35-dotcom/psd)와 동일한 파이프라인:
      detrend → (top profile) → window → FFT → 1D→2D 변환 → log-bin
    """

    def __init__(self):
        self.profile_name = ""
        self.h_raw = None
        self.dx = None
        self.N = None
        self.L = None
        self.unit_factor = 1e-6

    def load_profile(self, filepath):
        """노면 프로파일 로드 (포맷 자동 감지).

        지원 포맷:
          1) IDADA 포맷: 5줄 헤더 (이름, 플래그, N, dx, unit) + 'x  h' 데이터
          2) 선 거칠기 CSV: 헤더 "X(μm)","Z(μm)" + 콤마 구분 데이터 (μm 단위)
          3) 일반 2열 데이터: x, h (m 단위) — CSV 또는 공백/탭 구분
        """
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()

        if not lines:
            raise ValueError(f"빈 파일: {filepath}")

        first_line = lines[0].strip()

        # ── 포맷 감지 ──
        # (A) 선 거칠기 CSV: 헤더에 "X(" 또는 "Z(" 포함
        if 'X(' in first_line.upper() or 'Z(' in first_line.upper():
            return self._load_line_roughness_csv(filepath, lines)

        # (B) IDADA 포맷: 3번째 줄이 큰 정수 (데이터 포인트 수)
        try:
            if len(lines) >= 5:
                n_check = int(lines[2].strip())
                dx_check = float(lines[3].strip())
                if n_check > 10 and dx_check > 0:
                    return self._load_idada_format(lines)
        except (ValueError, IndexError):
            pass

        # (C) 일반 2열 데이터 (x, h)
        return self._load_generic_xy(filepath)

    def _load_idada_format(self, lines):
        """IDADA 포맷 파일 파싱."""
        self.profile_name = lines[0].strip()
        self.dx = float(lines[3].strip())
        self.unit_factor = float(lines[4].strip())

        h_list = []
        for line in lines[5:]:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    h_list.append(float(parts[1]))
                except ValueError:
                    continue

        self.h_raw = np.array(h_list) * self.unit_factor
        self.N = len(self.h_raw)
        self.L = self.N * self.dx

        return {
            'name': self.profile_name,
            'n_points': self.N,
            'dx_um': self.dx * 1e6,
            'L_mm': self.L * 1e3,
            'h_rms_um': np.std(self.h_raw) * 1e6,
            'q_min': 2 * np.pi / self.L,
            'q_max': np.pi / self.dx,
        }

    def _load_line_roughness_csv(self, filepath, lines):
        """선 거칠기 CSV 파싱: "X(μm)","Z(μm)" 형식.

        μm 단위 → m 변환, 빈 값은 NaN으로 처리 후 제거.
        """
        import csv as csv_mod
        x_list, h_list = [], []

        reader = csv_mod.reader(lines[1:])  # 헤더 스킵
        for row in reader:
            if len(row) < 2:
                continue
            x_str = row[0].strip().strip('"')
            h_str = row[1].strip().strip('"')
            if not x_str or not h_str:
                continue
            try:
                x_val = float(x_str)
                h_val = float(h_str)
                x_list.append(x_val)
                h_list.append(h_val)
            except ValueError:
                continue

        if len(x_list) < 10:
            raise ValueError(f"유효 데이터 부족: {len(x_list)}개")

        x_arr = np.array(x_list) * 1e-6   # μm → m
        h_arr = np.array(h_list) * 1e-6   # μm → m

        self.dx = np.median(np.diff(x_arr))
        self.unit_factor = 1.0  # 이미 m 단위
        self.h_raw = h_arr
        self.N = len(self.h_raw)
        self.L = self.N * self.dx
        self.profile_name = os.path.splitext(os.path.basename(filepath))[0]

        return {
            'name': self.profile_name,
            'n_points': self.N,
            'dx_um': self.dx * 1e6,
            'L_mm': self.L * 1e3,
            'h_rms_um': np.std(self.h_raw) * 1e6,
            'q_min': 2 * np.pi / self.L,
            'q_max': np.pi / self.dx,
        }

    def _load_generic_xy(self, filepath):
        """일반 2열 (x, h) 데이터 파일 로드 (m 단위 가정)."""
        # 구분자 자동 감지
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            first_data = ''
            for line in f:
                s = line.strip()
                if s and not s.startswith('#') and not s[0].isalpha():
                    first_data = s
                    break
        delim = ',' if ',' in first_data else None
        data = np.loadtxt(filepath, delimiter=delim, comments='#',
                          encoding='utf-8-sig')

        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError(f"2열 이상 데이터 필요 (현재 shape: {data.shape})")
        if len(data) < 10:
            raise ValueError(f"유효 데이터 부족: {len(data)}개")

        x_arr = data[:, 0]
        h_arr = data[:, 1]

        self.dx = np.median(np.diff(x_arr))
        self.unit_factor = 1.0
        self.h_raw = h_arr
        self.N = len(self.h_raw)
        self.L = self.N * self.dx
        self.profile_name = os.path.splitext(os.path.basename(filepath))[0]

        return {
            'name': self.profile_name,
            'n_points': self.N,
            'dx_um': self.dx * 1e6,
            'L_mm': self.L * 1e3,
            'h_rms_um': np.std(self.h_raw) * 1e6,
            'q_min': 2 * np.pi / self.L,
            'q_max': np.pi / self.dx,
        }

    def compute_psd(self, detrend='linear', window='none', use_top_psd=False,
                    conversion_method='standard', hurst=0.8,
                    correction_factor=1.1615, n_bins=88):
        """전체 PSD 파이프라인: detrend → window → FFT → 1D→2D → log-bin."""
        if self.h_raw is None:
            raise ValueError("No profile loaded")
        h = self.h_raw.copy()

        # Detrend
        if detrend == 'mean':
            h = h - np.mean(h)
        elif detrend == 'linear':
            h = scipy_detrend(h, type='linear')
        elif detrend == 'quadratic':
            x_idx = np.arange(len(h), dtype=np.float64)
            coeffs = np.polyfit(x_idx, h, 2)
            h = h - np.polyval(coeffs, x_idx)

        # Top profile
        if use_top_psd:
            h_c = h - np.mean(h)
            h_c[h_c < 0] = 0.0
            h = h_c

        # Window
        N = self.N
        if window in ('hanning', 'hamming', 'blackman'):
            w = getattr(np, window)(N)
            correction = np.sqrt(N / np.sum(w ** 2))
            h = h * w * correction

        # FFT → 1D PSD
        H_fft = fft(h)
        freqs = fftfreq(N, d=self.dx)
        q = 2.0 * np.pi * freqs
        C1D = (self.dx / (2.0 * np.pi * N)) * np.abs(H_fft) ** 2
        mask = q > 0
        q_pos, C1D_pos = q[mask], C1D[mask]

        # 1D → 2D 변환
        if conversion_method == 'standard':
            C2D = C1D_pos / (np.pi * q_pos) * correction_factor
        elif conversion_method == 'gamma':
            f_g = gamma_func(1.0 + hurst) / (np.sqrt(np.pi) * gamma_func(hurst + 0.5))
            C2D = (C1D_pos / q_pos) * f_g
        elif conversion_method == 'sqrt':
            C2D = (C1D_pos / q_pos) * np.sqrt(1.0 + 3.0 * hurst)
        else:
            C2D = C1D_pos / (np.pi * q_pos) * correction_factor

        # Log-bin
        log_q = np.log10(q_pos)
        edges = np.linspace(log_q.min(), log_q.max(), n_bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2.0
        C_binned = np.full(n_bins, np.nan)
        for i in range(n_bins):
            m = (log_q >= edges[i]) & (log_q < edges[i + 1])
            if np.any(m):
                C_binned[i] = np.mean(C2D[m])
        valid = ~np.isnan(C_binned) & (C_binned > 0)
        return 10.0 ** centers[valid], C_binned[valid]


# ================================================================
# ====  Monte Carlo 역추정 탭  ====
# ================================================================

def bind_monte_carlo_tab(app):
    """Monte Carlo 탭을 app에 바인딩한다."""
    mc = MonteCarloTab(app)
    app._mc_tab = mc


class MonteCarloTab:
    """Monte Carlo inverse estimation tab — 별도 클래스로 관리."""

    # 컴파운드 이름 기본값
    COMPOUND_NAMES = ['R0', 'R20', 'R40', 'R60']
    COMPOUND_COLORS = ['#2563EB', '#059669', '#D97706', '#DC2626']

    def __init__(self, app):
        self.app = app
        # 탭 빌더를 app에 등록
        app._create_monte_carlo_tab = self._create_tab

        # ── 공유 데이터 ──
        self.psd_scans = []       # list of (q, C) tuples — 원본 스캔
        self.C_pool = None        # (N_valid, N_q) ndarray — 앙상블
        self.q_grid = None        # (N_q,) ndarray — 공통 q 그리드
        self.q1_pool = None       # (M,) ndarray — q₁ 후보 풀
        self.q1_valid_range = None  # (q1_min, q1_max)
        self.mc_results = None    # list of dicts
        self._mc_stop = False     # 중단 플래그
        self._mc_thread = None

        # 컴파운드별 마스터커브 (리스트, 각 원소는 dict or None)
        self._compound_materials = [None] * 4

    # ================================================================
    #  메인 탭 생성
    # ================================================================
    def _create_tab(self, parent):
        """메인 탭 — 내부 서브탭 4개."""
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
    #  유틸리티: 좌우 레이아웃 생성
    # ================================================================
    def _make_lr_layout(self, parent, panel_width=340):
        """좌측 제어판(스크롤) + 우측 플롯 프레임 생성. (left_content, right) 반환."""
        C = self.app.COLORS

        main = ttk.Frame(parent)
        main.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        # 좌측 고정폭
        left = ttk.Frame(main, width=panel_width)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 4))
        left.pack_propagate(False)

        # 스크롤 캔버스
        scroll_canvas = tk.Canvas(left, highlightthickness=0, bg=C['bg'])
        scrollbar = ttk.Scrollbar(left, orient='vertical',
                                  command=scroll_canvas.yview)
        content = ttk.Frame(scroll_canvas)
        content.bind('<Configure>',
                     lambda e: scroll_canvas.configure(
                         scrollregion=scroll_canvas.bbox('all')))
        cw_id = scroll_canvas.create_window((0, 0), window=content, anchor='nw',
                                            width=panel_width - 18)
        scroll_canvas.configure(yscrollcommand=scrollbar.set)
        scroll_canvas.bind('<Configure>',
                           lambda e, _c=scroll_canvas, _id=cw_id:
                               _c.itemconfigure(_id, width=e.width))
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 마우스휠 스크롤
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

        # 우측
        right = ttk.Frame(main)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        return content, right

    def _make_section(self, parent, title):
        """LabelFrame 섹션 생성."""
        frame = ttk.LabelFrame(parent, text=title, padding=8)
        frame.pack(fill=tk.X, pady=4, padx=2)
        return frame

    def _make_entry_row(self, parent, label, default, width=12):
        """라벨 + Entry 한 줄. StringVar 반환."""
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=label, font=self.app.FONTS['body']).pack(side=tk.LEFT)
        var = tk.StringVar(value=str(default))
        ttk.Entry(row, textvariable=var, width=width).pack(side=tk.RIGHT, padx=2)
        return var

    # ================================================================
    #  서브탭 1: PSD 앙상블
    # ================================================================
    def _build_psd_tab(self, parent):
        left, right = self._make_lr_layout(parent)

        # ── 섹션 1: 입력 ──
        sec1 = self._make_section(left, "1) 입력 파일")

        ttk.Button(sec1, text="Tab 0 PSD 사용",
                   command=self._load_psd_from_app).pack(fill=tk.X, pady=2)
        ttk.Button(sec1, text="PSD 파일 추가 (CSV/TXT)",
                   command=self._load_psd_files).pack(fill=tk.X, pady=2)
        ttk.Button(sec1, text="노면 프로파일 추가",
                   command=self._load_profile_files).pack(fill=tk.X, pady=2)
        ttk.Button(sec1, text="스캔 초기화",
                   command=self._clear_psd_scans).pack(fill=tk.X, pady=2)

        self._psd_scan_label = ttk.Label(sec1, text="로드된 스캔: 0개",
                                         font=self.app.FONTS['small'],
                                         foreground='#64748B')
        self._psd_scan_label.pack(anchor='w', pady=2)

        # 노면 프로파일 PSD 계산 파라미터
        psd_plf = ttk.LabelFrame(sec1, text="프로파일 → PSD 파라미터", padding=4)
        psd_plf.pack(fill=tk.X, pady=2)
        psd_p_row = ttk.Frame(psd_plf)
        psd_p_row.pack(fill=tk.X, pady=1)
        ttk.Label(psd_p_row, text="Detrend:").pack(side=tk.LEFT)
        self._psd_detrend_var = tk.StringVar(value='linear')
        ttk.Combobox(psd_p_row, textvariable=self._psd_detrend_var, width=10,
                     values=['none', 'mean', 'linear', 'quadratic'],
                     state='readonly').pack(side=tk.RIGHT, padx=2)
        self._psd_nbins_var = self._make_entry_row(psd_plf, "Log-bin 수:", "88", width=6)
        self._psd_corr_var = self._make_entry_row(psd_plf, "보정계수 (1D→2D):", "1.1615")

        # ── 섹션 2: PCA 설정 ──
        sec2 = self._make_section(left, "2) PCA 설정")

        self._pca_var_thresh = self._make_entry_row(sec2, "누적분산 임계값:", "0.90")
        self._pca_n_samples = self._make_entry_row(sec2, "생성 샘플 수:", "1000")
        self._pca_seed = self._make_entry_row(sec2, "Random seed:", "42")

        ttk.Separator(sec2, orient='horizontal').pack(fill=tk.X, pady=4)
        ttk.Label(sec2, text="물리적 타당성 필터:",
                  font=self.app.FONTS['body_bold']).pack(anchor='w')
        self._pca_rms_sigma = self._make_entry_row(sec2, "rms roughness ±N σ:", "3")
        self._pca_mono_tol = self._make_entry_row(sec2, "단조감소 위반 허용:", "0.1")

        # ── 섹션 3: 실행 ──
        sec3 = self._make_section(left, "3) 실행")
        ttk.Button(sec3, text="PSD 앙상블 생성",
                   command=self._run_psd_ensemble,
                   style='Accent.TButton').pack(fill=tk.X, pady=4)
        self._psd_status_label = ttk.Label(sec3, text="대기 중",
                                           font=self.app.FONTS['small'],
                                           foreground='#64748B')
        self._psd_status_label.pack(anchor='w')

        # ── 우측 플롯 ──
        self._fig_psd = Figure(figsize=(8, 6), dpi=100, facecolor='white')
        self._ax_psd_orig = self._fig_psd.add_subplot(2, 1, 1)
        self._ax_psd_gen = self._fig_psd.add_subplot(2, 1, 2)
        self._fig_psd.tight_layout(pad=2.0)

        self._canvas_psd = FigureCanvasTkAgg(self._fig_psd, master=right)
        self._canvas_psd.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._canvas_psd.draw_idle()

    # ── PSD 로드 ──
    def _load_psd_from_app(self):
        """Tab 0에서 확정된 PSD를 단일 스캔으로 로드."""
        try:
            raw = self.app.raw_psd_data
            if raw is None:
                messagebox.showwarning("PSD 로드",
                    "Tab 0에서 PSD를 먼저 확정하세요.")
                return
            q = np.asarray(raw['q']).copy()
            C = np.asarray(raw['C_q']).copy()
            self.psd_scans = [(q, C)]
            self._psd_scan_label.config(text=f"로드된 스캔: 1개 (Tab 0)")
            self._plot_psd_originals()
            self.app._show_status("Tab 0 PSD를 MC 탭에 로드했습니다.", 'success')
        except Exception as e:
            messagebox.showerror("오류", f"PSD 로드 실패:\n{e}")

    def _clear_psd_scans(self):
        """로드된 PSD 스캔 초기화."""
        self.psd_scans = []
        self._psd_scan_label.config(text="로드된 스캔: 0개")
        self._ax_psd_orig.clear()
        self._ax_psd_orig.set_title("원본 PSD 스캔")
        self._canvas_psd.draw_idle()
        self.app._show_status("PSD 스캔 초기화됨", 'info')

    def _load_psd_files(self):
        """CSV/TXT 파일 선택으로 PSD 스캔 추가 (중복 선택 가능)."""
        fpaths = filedialog.askopenfilenames(
            title="PSD 파일 선택 (복수 선택 가능)",
            filetypes=[("CSV/TXT 파일", "*.csv *.txt"),
                       ("모든 파일", "*.*")])
        if not fpaths:
            return
        try:
            new_scans = []
            for fpath in fpaths:
                try:
                    with open(fpath, 'r', encoding='utf-8-sig') as f:
                        first_data = ''
                        for line in f:
                            s = line.strip()
                            if s and not s.startswith('#'):
                                first_data = s
                                break
                    delim = ',' if ',' in first_data else None
                    data = np.loadtxt(fpath, delimiter=delim, comments='#',
                                      encoding='utf-8-sig')
                except Exception:
                    continue
                if data.ndim == 2 and data.shape[1] >= 2:
                    q = data[:, 0]
                    C = data[:, 1]
                    mask = (q > 0) & (C > 0)
                    if np.sum(mask) >= 5:
                        new_scans.append((q[mask], C[mask]))

            if not new_scans:
                messagebox.showwarning("PSD 로드",
                    "유효한 PSD 데이터를 찾을 수 없습니다.")
                return

            self.psd_scans.extend(new_scans)
            self._psd_scan_label.config(
                text=f"로드된 스캔: {len(self.psd_scans)}개 (추가 {len(new_scans)}개)")
            self._plot_psd_originals()
            self.app._show_status(
                f"PSD 스캔 {len(new_scans)}개 추가 (총 {len(self.psd_scans)}개)",
                'success')

        except Exception as e:
            messagebox.showerror("오류", f"PSD 파일 로드 실패:\n{e}")

    def _load_profile_files(self):
        """노면 프로파일 파일 선택으로 PSD 계산하여 추가 (중복 선택 가능)."""
        fpaths = filedialog.askopenfilenames(
            title="노면 프로파일 파일 선택 (복수 선택 가능)",
            filetypes=[("프로파일 파일", "*.dat *.csv *.txt"),
                       ("모든 파일", "*.*")])
        if not fpaths:
            return
        try:
            detrend = self._psd_detrend_var.get()
            n_bins = int(self._psd_nbins_var.get())
            corr = float(self._psd_corr_var.get())

            self._psd_status_label.config(
                text=f"프로파일 {len(fpaths)}개 PSD 계산 중...")
            self.app.root.update_idletasks()

            new_scans = []
            loaded_names = []
            skip_msgs = []
            for fpath in fpaths:
                try:
                    comp = PSDComputer()
                    info = comp.load_profile(fpath)
                    q_bin, C_bin = comp.compute_psd(
                        detrend=detrend, window='none',
                        use_top_psd=False,
                        conversion_method='standard',
                        correction_factor=corr, n_bins=n_bins,
                    )
                    if len(q_bin) >= 5:
                        new_scans.append((q_bin, C_bin))
                        loaded_names.append(
                            info.get('name', os.path.basename(fpath)))
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    skip_msgs.append(f"{os.path.basename(fpath)}: {e}")
                    continue

            if not new_scans:
                detail = "\n".join(skip_msgs[:10]) if skip_msgs else "알 수 없는 오류"
                messagebox.showwarning("프로파일 로드",
                    f"유효한 프로파일을 찾을 수 없습니다.\n\n{detail}")
                return

            self.psd_scans.extend(new_scans)
            self._psd_scan_label.config(
                text=f"로드된 스캔: {len(self.psd_scans)}개 (추가 {len(new_scans)}개)")
            self._plot_psd_originals()
            self._psd_status_label.config(
                text=f"프로파일 {len(new_scans)}개 PSD 계산 완료")
            self.app._show_status(
                f"노면 프로파일 {len(new_scans)}개 → PSD 계산 완료 "
                f"(총 {len(self.psd_scans)}개)\n"
                f"  파일: {', '.join(loaded_names[:5])}"
                + (f" 외 {len(loaded_names)-5}개" if len(loaded_names) > 5 else ""),
                'success')

        except Exception as e:
            messagebox.showerror("오류", f"프로파일 로드 실패:\n{e}")
            import traceback
            traceback.print_exc()

    # ── PSD 앙상블 생성 ──
    def _run_psd_ensemble(self):
        """PCA 기반 PSD 앙상블 생성.

        PSDEnsemble (k100kan35-dotcom/psd)과 동일한 파이프라인:
          1. C(q) → ln() 변환 (자연로그 기반 — psd_ensemble.py와 동일)
          2. 공통 q 그리드에 log-log 보간
          3. 평균 + 편차 행렬
          4. SVD → 고유값/고유벡터
          5. K 선택 (누적분산 ≥ threshold)
          6. z_k ~ N(0, sqrt(eigenvalue_k)) 샘플링
          7. Y_new = Y_mean + z @ Vt[:K] → C_new = exp(Y_new)
          8. 물리 필터: rms ±3σ + 단조감소 위반
        """
        if not self.psd_scans:
            messagebox.showwarning("PSD 앙상블", "먼저 PSD 스캔을 로드하세요.")
            return

        try:
            self._psd_status_label.config(text="앙상블 생성 중...")
            self.app.root.update_idletasks()

            threshold = float(self._pca_var_thresh.get())
            n_samples = int(self._pca_n_samples.get())
            seed = int(self._pca_seed.get())
            rms_n_sigma = float(self._pca_rms_sigma.get())
            mono_tol = float(self._pca_mono_tol.get())

            # 1) 공통 q 그리드 (교집합 범위, 100 포인트)
            q_mins = [np.min(s[0]) for s in self.psd_scans]
            q_maxs = [np.max(s[0]) for s in self.psd_scans]
            q_lo = max(q_mins)
            q_hi = min(q_maxs)
            if q_lo >= q_hi:
                messagebox.showerror("오류", "스캔 q 범위가 겹치지 않습니다.")
                return
            n_q = 100
            q_common = np.logspace(np.log10(q_lo), np.log10(q_hi), n_q)

            # 2) log-log 보간 → C_matrix (선형 스케일)
            N = len(self.psd_scans)
            C_matrix = np.zeros((N, n_q))
            for i, (q_i, C_i) in enumerate(self.psd_scans):
                log_C_interp = np.interp(np.log10(q_common),
                                         np.log10(q_i), np.log10(C_i))
                C_matrix[i] = 10.0 ** log_C_interp

            # 3) ln 변환 (PSDEnsemble과 동일 — 자연로그)
            Y = np.log(C_matrix)        # shape (N, n_q)
            Y_mean = np.mean(Y, axis=0)  # shape (n_q,)
            dY = Y - Y_mean              # shape (N, n_q)

            if N < 2:
                # 스캔 1개 → PCA 불가, 노이즈로 앙상블 생성
                rng = np.random.default_rng(seed)
                C_pool_list = []
                for _ in range(n_samples):
                    noise = rng.normal(0, 0.05, n_q)
                    C_new = np.exp(Y_mean + noise)
                    C_pool_list.append(C_new)
                self.C_pool = np.array(C_pool_list)
                self.q_grid = q_common
                K = 0
            else:
                # 4) SVD
                U, S, Vt = np.linalg.svd(dY, full_matrices=False)
                eigenvalues = S ** 2 / (N - 1)
                eigenvectors = Vt  # shape (n_components, n_q)

                # 5) K 선택
                total_var = np.sum(eigenvalues)
                cumvar = np.cumsum(eigenvalues) / total_var
                K = int(np.searchsorted(cumvar, threshold) + 1)
                K = min(K, len(eigenvalues))

                # PCA 결과 저장 (플롯용)
                self._pca_eigenvalues = eigenvalues
                self._pca_eigenvectors = eigenvectors
                self._pca_Y_mean = Y_mean
                self._pca_C_matrix = C_matrix

                # 6) 원본 rms roughness 통계 (필터용)
                rms_orig = np.array([
                    np.sqrt(_trapz(C_matrix[i], q_common))
                    for i in range(N)
                ])
                rms_mean = np.mean(rms_orig)
                rms_std = np.std(rms_orig, ddof=0)
                rms_lo = rms_mean - rms_n_sigma * rms_std
                rms_hi = rms_mean + rms_n_sigma * rms_std

                # 7) 샘플 생성 + 물리 필터
                rng = np.random.default_rng(seed)
                std_k = np.sqrt(eigenvalues[:K])
                ev_k = eigenvectors[:K]  # shape (K, n_q)

                C_pool_list = []
                n_rejected = 0
                max_attempts = n_samples * 50

                for attempt in range(max_attempts):
                    z = rng.normal(0.0, std_k)      # shape (K,)
                    Y_new = Y_mean + z @ ev_k        # shape (n_q,)
                    C_new = np.exp(Y_new)

                    # 필터 1: rms roughness ±Nσ
                    rms_new = np.sqrt(_trapz(C_new, q_common))
                    if rms_new < rms_lo or rms_new > rms_hi:
                        n_rejected += 1
                        continue

                    # 필터 2: 단조감소 위반 (평균 기울기 양수 → 거부)
                    dC = np.diff(C_new)
                    violation_ratio = np.sum(dC > 0) / max(len(dC), 1)
                    if violation_ratio > mono_tol:
                        n_rejected += 1
                        continue

                    C_pool_list.append(C_new)
                    if len(C_pool_list) >= n_samples:
                        break

                self.C_pool = np.array(C_pool_list)
                self.q_grid = q_common

                print(f"[PSD Ensemble] {len(C_pool_list)} accepted, "
                      f"{n_rejected} rejected ({attempt+1} attempts)")
                print(f"  PCA: K={K}, eigenvalues={eigenvalues[:K]}")
                print(f"  RMS filter: [{rms_lo:.4e}, {rms_hi:.4e}]")

            n_valid = len(self.C_pool)
            self._psd_status_label.config(
                text=f"완료: {n_valid}개 생성 (K={K})")
            self._plot_psd_ensemble()
            self.app._show_status(
                f"PSD 앙상블 {n_valid}개 생성 완료 (PCA K={K})", 'success')

        except Exception as e:
            self._psd_status_label.config(text=f"오류: {e}")
            messagebox.showerror("오류", f"PSD 앙상블 생성 실패:\n{e}")
            import traceback
            traceback.print_exc()

    # ── PSD 플롯 ──
    def _plot_psd_originals(self):
        """원본 스캔 플롯."""
        ax = self._ax_psd_orig
        ax.clear()
        ax.set_title('원본 PSD 스캔', fontweight='bold')
        ax.set_xlabel('q (1/m)')
        ax.set_ylabel('C(q) (m⁴)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        for i, (q, C) in enumerate(self.psd_scans):
            ax.plot(q, C, '-', color='gray', alpha=0.5, linewidth=0.8)

        if self.psd_scans:
            # 평균 (log-log 보간)
            q0 = self.psd_scans[0][0]
            C_mean = np.mean([s[1] for s in self.psd_scans], axis=0) \
                if len(self.psd_scans) > 1 and all(len(s[0]) == len(q0) for s in self.psd_scans) \
                else self.psd_scans[0][1]
            ax.plot(q0, C_mean, 'r-', linewidth=2, label='평균')
            ax.legend(loc='best')

        self._ax_psd_gen.clear()
        self._ax_psd_gen.set_title('생성된 앙상블', fontweight='bold')
        self._ax_psd_gen.set_xlabel('q (1/m)')
        self._ax_psd_gen.set_ylabel('C(q) (m⁴)')
        self._ax_psd_gen.grid(True, alpha=0.3)
        self._ax_psd_gen.text(0.5, 0.5, '앙상블 생성 전',
                              transform=self._ax_psd_gen.transAxes,
                              ha='center', va='center', color='gray')

        self._fig_psd.tight_layout(pad=2.0)
        self._canvas_psd.draw_idle()

    def _plot_psd_ensemble(self):
        """원본 + 생성 앙상블 플롯."""
        # 상단: 원본
        self._plot_psd_originals()

        # 하단: 생성된 앙상블
        ax = self._ax_psd_gen
        ax.clear()
        ax.set_title('생성된 PSD 앙상블', fontweight='bold')
        ax.set_xlabel('q (1/m)')
        ax.set_ylabel('C(q) (m⁴)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        if self.C_pool is not None and self.q_grid is not None:
            # 원본 (회색)
            for q, C in self.psd_scans:
                ax.plot(q, C, '-', color='gray', alpha=0.3, linewidth=0.5)

            # 생성 샘플 (최대 100개)
            n_show = min(100, len(self.C_pool))
            for i in range(n_show):
                ax.plot(self.q_grid, self.C_pool[i], '-',
                        color='#3B82F6', alpha=0.08, linewidth=0.5)

            # 평균
            C_mean = np.mean(self.C_pool, axis=0)
            ax.plot(self.q_grid, C_mean, 'r-', linewidth=2, label='앙상블 평균')
            ax.legend(loc='best')

        self._fig_psd.tight_layout(pad=2.0)
        self._canvas_psd.draw_idle()

    # ================================================================
    #  서브탭 2: q₁ 생성
    # ================================================================
    def _build_q1_tab(self, parent):
        left, right = self._make_lr_layout(parent)

        # ── 섹션 1: 실험 데이터 — 화강암(하한) ──
        sec1 = self._make_section(left, "1) 화강암 비눗물 μ (하한)")
        self._q1_granite_v = self._make_entry_row(sec1, "속도점 (m/s):", "0.002, 0.01, 0.046", width=20)
        self._q1_granite_mu = {}
        for name in self.COMPOUND_NAMES:
            self._q1_granite_mu[name] = self._make_entry_row(sec1, f"  {name}:", "", width=20)

        # ── 아스팔트(상한) ──
        sec1b = self._make_section(left, "   아스팔트 비눗물 μ (상한)")
        self._q1_asphalt_v = self._make_entry_row(sec1b, "속도점 (m/s):", "0.01, 0.046", width=20)
        self._q1_asphalt_mu = {}
        for name in self.COMPOUND_NAMES:
            self._q1_asphalt_mu[name] = self._make_entry_row(sec1b, f"  {name}:", "", width=20)

        btn_csv = ttk.Frame(left)
        btn_csv.pack(fill=tk.X, pady=2, padx=2)
        ttk.Button(btn_csv, text="실측값 불러오기 (CSV)",
                   command=self._load_q1_exp_csv).pack(side=tk.LEFT)

        # ── 섹션 2: 탐색 범위 ──
        sec2 = self._make_section(left, "2) q₁ 탐색 범위")
        self._q1_min_var = self._make_entry_row(sec2, "q₁_min (1/m):", "1e4")
        self._q1_max_var = self._make_entry_row(sec2, "q₁_max (1/m):", "1e7")
        self._q1_n_scan = self._make_entry_row(sec2, "탐색 포인트 수:", "50")
        self._q1_pool_size = self._make_entry_row(sec2, "풀 크기:", "1000")

        # ── 섹션 3: 사전 탐색 ──
        sec3 = self._make_section(left, "3) 사전 탐색")
        ttk.Button(sec3, text="q₁ 유효 범위 사전 탐색",
                   command=self._run_q1_prescan,
                   style='Accent.TButton').pack(fill=tk.X, pady=4)
        self._q1_range_label = ttk.Label(sec3, text="유효 범위: 미탐색",
                                         font=self.app.FONTS['small'],
                                         foreground='#64748B')
        self._q1_range_label.pack(anchor='w')

        # ── 섹션 4: 풀 생성 ──
        sec4 = self._make_section(left, "4) 풀 생성")
        ttk.Button(sec4, text="q₁ 풀 생성",
                   command=self._generate_q1_pool).pack(fill=tk.X, pady=4)
        self._q1_pool_label = ttk.Label(sec4, text="풀: 미생성",
                                        font=self.app.FONTS['small'],
                                        foreground='#64748B')
        self._q1_pool_label.pack(anchor='w')

        # ── 우측 플롯 ──
        self._fig_q1 = Figure(figsize=(8, 6), dpi=100, facecolor='white')
        self._ax_q1 = self._fig_q1.add_subplot(1, 1, 1)
        self._fig_q1.tight_layout(pad=2.0)
        self._canvas_q1 = FigureCanvasTkAgg(self._fig_q1, master=right)
        self._canvas_q1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._canvas_q1.draw_idle()

    def _load_q1_exp_csv(self):
        """q₁ 탭용 실측 μ CSV 로드."""
        fpath = filedialog.askopenfilename(
            title="실측 μ CSV 선택",
            filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if not fpath:
            return
        try:
            messagebox.showinfo("CSV 로드",
                "CSV 형식:\n행: 속도점, 열: 컴파운드별 μ 값\n"
                "첫 행 헤더, 첫 열 속도(m/s)")
            self.app._show_status(f"CSV 로드: {os.path.basename(fpath)}", 'info')
        except Exception as e:
            messagebox.showerror("오류", str(e))

    def _parse_float_list(self, s):
        """콤마 구분 문자열을 float 배열로 변환."""
        return np.array([float(x.strip()) for x in s.split(',') if x.strip()])

    def _compute_mu_hys_for_q1(self, q1_val, C_arr, q_arr, compound_idx):
        """주어진 q₁, C(q), 컴파운드에 대해 μ_hys 계산 (단일 속도 v=0.01 m/s).

        Returns mu_hys (float) or None on failure.
        """
        try:
            # q 범위 제한
            mask = q_arr <= q1_val
            if np.sum(mask) < 5:
                return None
            q_use = q_arr[mask]
            C_use = C_arr[mask]

            psd = MeasuredPSD(q_use, C_use, interpolation_kind='log-log')

            # 마스터커브 — app에서 가져옴
            mat = self._get_compound_material(compound_idx)
            if mat is None:
                return None

            sigma_0 = 0.3e6  # 0.3 MPa
            v = 0.01  # m/s
            poisson = 0.5

            modulus_func = lambda omega: mat.get_modulus(omega, mat.reference_temp)

            g_calc = GCalculator(
                psd_func=psd,
                modulus_func=modulus_func,
                sigma_0=sigma_0,
                velocity=v,
                poisson_ratio=poisson,
            )
            G_array = g_calc.calculate_G(q_use)
            C_q = psd(q_use)

            loss_func = lambda omega, T: mat.get_loss_modulus(omega, mat.reference_temp)
            fc = FrictionCalculator(
                psd_func=psd,
                loss_modulus_func=loss_func,
                sigma_0=sigma_0,
                velocity=v,
                temperature=mat.reference_temp,
                poisson_ratio=poisson,
            )
            mu, _ = fc.calculate_mu_visc(q_use, G_array, C_q)
            return mu
        except Exception:
            return None

    def _get_compound_material(self, idx):
        """idx번째 컴파운드의 재료 객체 반환 (없으면 app.material)."""
        if self._compound_materials[idx] is not None:
            return self._compound_materials[idx]
        if self.app.material is not None:
            return self.app.material
        return None

    def _run_q1_prescan(self):
        """평균 C(q)로 q₁ 유효 범위 사전 탐색."""
        if self.C_pool is None or self.q_grid is None:
            messagebox.showwarning("q₁ 탐색", "먼저 PSD 앙상블을 생성하세요.")
            return

        try:
            self._q1_range_label.config(text="탐색 중...")
            self.app.root.update_idletasks()

            q1_min = float(self._q1_min_var.get())
            q1_max = float(self._q1_max_var.get())
            n_scan = int(self._q1_n_scan.get())

            q1_candidates = np.logspace(np.log10(q1_min), np.log10(q1_max), n_scan)
            C_mean = np.mean(self.C_pool, axis=0)

            # 각 q₁에서 μ_hys 계산 (첫 번째 컴파운드만)
            mu_vs_q1 = []
            for q1 in q1_candidates:
                mu = self._compute_mu_hys_for_q1(q1, C_mean, self.q_grid, 0)
                mu_vs_q1.append(mu if mu is not None else 0.0)

            mu_vs_q1 = np.array(mu_vs_q1)

            # 유효 범위: μ > 0인 q₁ 구간
            valid = mu_vs_q1 > 0
            if not np.any(valid):
                self._q1_range_label.config(text="유효 범위 없음 — 파라미터 확인 필요")
                return

            valid_q1 = q1_candidates[valid]
            self.q1_valid_range = (valid_q1.min(), valid_q1.max())
            self._q1_range_label.config(
                text=f"유효 범위: [{valid_q1.min():.2e}, {valid_q1.max():.2e}] 1/m")

            # 플롯
            ax = self._ax_q1
            ax.clear()
            ax.set_title('μ_hys vs q₁ (평균 PSD)', fontweight='bold')
            ax.set_xlabel('q₁ (1/m)')
            ax.set_ylabel('μ_hys')
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
            ax.plot(q1_candidates, mu_vs_q1, 'o-', color='#2563EB', linewidth=1.5,
                    markersize=3, label='μ_hys (compound 0)')
            if self.q1_valid_range:
                ax.axvspan(self.q1_valid_range[0], self.q1_valid_range[1],
                          alpha=0.1, color='green', label='유효 범위')
            ax.legend(loc='best')
            self._fig_q1.tight_layout(pad=2.0)
            self._canvas_q1.draw_idle()

            self.app._show_status(
                f"q₁ 사전 탐색 완료: [{valid_q1.min():.2e}, {valid_q1.max():.2e}]",
                'success')

        except Exception as e:
            self._q1_range_label.config(text=f"오류: {e}")
            messagebox.showerror("오류", f"q₁ 탐색 실패:\n{e}")
            import traceback
            traceback.print_exc()

    def _generate_q1_pool(self):
        """유효 범위에서 log-uniform q₁ 풀 생성."""
        if self.q1_valid_range is None:
            messagebox.showwarning("q₁ 풀", "먼저 사전 탐색을 수행하세요.")
            return
        try:
            pool_size = int(self._q1_pool_size.get())
            rng = np.random.RandomState(42)
            lo, hi = self.q1_valid_range
            self.q1_pool = 10.0 ** rng.uniform(np.log10(lo), np.log10(hi), pool_size)
            self._q1_pool_label.config(text=f"풀: {pool_size}개 생성 완료")
            self.app._show_status(f"q₁ 풀 {pool_size}개 생성", 'success')
        except Exception as e:
            messagebox.showerror("오류", f"q₁ 풀 생성 실패:\n{e}")

    # ================================================================
    #  서브탭 3: Monte Carlo
    # ================================================================
    def _build_mc_tab(self, parent):
        left, right = self._make_lr_layout(parent)

        # ── 섹션 1: DMA 데이터 / 컴파운드 설정 ──
        sec1 = self._make_section(left, "1) 컴파운드 설정")
        ttk.Button(sec1, text="Tab 1 마스터 커브 사용",
                   command=self._load_material_from_app).pack(fill=tk.X, pady=2)

        self._mc_compound_vars = []
        for i, name in enumerate(self.COMPOUND_NAMES):
            row = ttk.Frame(sec1)
            row.pack(fill=tk.X, pady=1)
            ttk.Label(row, text=f"{name}:", font=self.app.FONTS['body'],
                      width=5).pack(side=tk.LEFT)
            sigma_var = tk.StringVar(value="0.3")
            temp_var = tk.StringVar(value="20")
            poisson_var = tk.StringVar(value="0.5")
            ttk.Label(row, text="σ₀(MPa)").pack(side=tk.LEFT)
            ttk.Entry(row, textvariable=sigma_var, width=5).pack(side=tk.LEFT, padx=1)
            ttk.Label(row, text="T(°C)").pack(side=tk.LEFT)
            ttk.Entry(row, textvariable=temp_var, width=4).pack(side=tk.LEFT, padx=1)
            ttk.Label(row, text="ν").pack(side=tk.LEFT)
            ttk.Entry(row, textvariable=poisson_var, width=4).pack(side=tk.LEFT, padx=1)
            self._mc_compound_vars.append({
                'sigma': sigma_var, 'temp': temp_var, 'poisson': poisson_var
            })

        # ── 섹션 2: 건식 마찰 데이터 ──
        sec2 = self._make_section(left, "2) 건식 마찰 데이터 (Likelihood)")
        self._mc_dry_v = self._make_entry_row(
            sec2, "속도점 (log V):", "-4,-3.337,-2.699,-2,-1.337,-0.678", width=25)
        self._mc_dry_mu = {}
        for name in self.COMPOUND_NAMES:
            self._mc_dry_mu[name] = self._make_entry_row(sec2, f"  {name} μ_dry:", "", width=25)
        ttk.Button(sec2, text="실측값 불러오기 (CSV)",
                   command=self._load_mc_dry_csv).pack(anchor='w', pady=2)

        # ── 섹션 3: MC 설정 ──
        sec3 = self._make_section(left, "3) Monte Carlo 설정")
        self._mc_n_iter = self._make_entry_row(sec3, "반복 횟수 N:", "3000")
        self._mc_seed = self._make_entry_row(sec3, "Random seed:", "42")
        self._mc_top_pct = self._make_entry_row(sec3, "상위 X %:", "10")
        self._mc_filter_v = self._make_entry_row(sec3, "1단계 필터 속도 (m/s):", "0.01, 0.046", width=20)

        # ── 섹션 4: 실행 ──
        sec4 = self._make_section(left, "4) 실행")
        btn_row = ttk.Frame(sec4)
        btn_row.pack(fill=tk.X, pady=4)
        ttk.Button(btn_row, text="Monte Carlo 시작",
                   command=self._start_mc,
                   style='Accent.TButton').pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="중단",
                   command=self._stop_mc).pack(side=tk.LEFT, padx=2)

        self._mc_progress = ttk.Progressbar(sec4, mode='determinate')
        self._mc_progress.pack(fill=tk.X, pady=2)
        self._mc_status_label = ttk.Label(sec4, text="대기 중",
                                          font=self.app.FONTS['small'],
                                          foreground='#64748B')
        self._mc_status_label.pack(anchor='w')
        self._mc_eta_label = ttk.Label(sec4, text="",
                                       font=self.app.FONTS['small'],
                                       foreground='#64748B')
        self._mc_eta_label.pack(anchor='w')

        # ── 우측: 로그 + 진행 차트 ──
        # 로그 텍스트
        log_frame = ttk.LabelFrame(right, text="계산 로그", padding=4)
        log_frame.pack(fill=tk.BOTH, expand=True)

        self._mc_log_text = tk.Text(log_frame, height=15, wrap=tk.WORD,
                                    font=self.app.FONTS.get('mono', ('Courier', 10)),
                                    bg='#1E293B', fg='#E2E8F0',
                                    insertbackground='white')
        log_scroll = ttk.Scrollbar(log_frame, orient='vertical',
                                   command=self._mc_log_text.yview)
        self._mc_log_text.configure(yscrollcommand=log_scroll.set)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._mc_log_text.pack(fill=tk.BOTH, expand=True)

        # 간이 진행 차트
        chart_frame = ttk.LabelFrame(right, text="진행 차트", padding=4)
        chart_frame.pack(fill=tk.BOTH, expand=True)
        self._fig_mc_progress = Figure(figsize=(6, 2.5), dpi=100, facecolor='white')
        self._ax_mc_progress = self._fig_mc_progress.add_subplot(1, 1, 1)
        self._canvas_mc_progress = FigureCanvasTkAgg(self._fig_mc_progress,
                                                     master=chart_frame)
        self._canvas_mc_progress.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _load_material_from_app(self):
        """Tab 1 마스터 커브를 4컴파운드에 공통 적용."""
        if self.app.material is None:
            messagebox.showwarning("재료 로드",
                "Tab 1에서 마스터 커브를 먼저 확정하세요.")
            return
        for i in range(4):
            self._compound_materials[i] = self.app.material
        self.app._show_status("Tab 1 마스터 커브 → 4컴파운드 공통 적용", 'success')

    def _load_mc_dry_csv(self):
        """건식 마찰 실측 CSV 로드."""
        fpath = filedialog.askopenfilename(
            title="건식 마찰 CSV 선택",
            filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if not fpath:
            return
        try:
            data = np.loadtxt(fpath, delimiter=',', comments='#',
                              encoding='utf-8-sig', skiprows=1)
            if data.ndim != 2 or data.shape[1] < 2:
                messagebox.showerror("오류", "CSV 형식: 첫열=log(v), 이후열=컴파운드별 μ")
                return
            v_str = ', '.join(f"{x:.3f}" for x in data[:, 0])
            self._mc_dry_v.set(v_str)
            for i, name in enumerate(self.COMPOUND_NAMES):
                if i + 1 < data.shape[1]:
                    mu_str = ', '.join(f"{x:.4f}" for x in data[:, i + 1])
                    self._mc_dry_mu[name].set(mu_str)
            self.app._show_status(f"건식 마찰 CSV 로드 완료: {data.shape[0]}점", 'success')
        except Exception as e:
            messagebox.showerror("오류", f"CSV 로드 실패:\n{e}")

    def _mc_log(self, msg):
        """MC 로그 텍스트에 메시지 추가 (thread-safe)."""
        def _append():
            self._mc_log_text.insert(tk.END, msg + '\n')
            self._mc_log_text.see(tk.END)
        self.app.root.after(0, _append)

    def _stop_mc(self):
        """MC 중단 요청."""
        self._mc_stop = True
        self._mc_log("⚠ 사용자 중단 요청...")

    def _start_mc(self):
        """MC 시작 (별도 스레드)."""
        if self.C_pool is None:
            messagebox.showwarning("MC", "먼저 PSD 앙상블을 생성하세요.")
            return
        if self.q1_pool is None:
            messagebox.showwarning("MC", "먼저 q₁ 풀을 생성하세요.")
            return
        if self._mc_thread is not None and self._mc_thread.is_alive():
            messagebox.showwarning("MC", "이미 실행 중입니다.")
            return

        self._mc_stop = False
        self._mc_log_text.delete('1.0', tk.END)
        self._mc_thread = threading.Thread(target=self._run_monte_carlo, daemon=True)
        self._mc_thread.start()

    def _run_monte_carlo(self):
        """Monte Carlo 메인 루프 (별도 스레드)."""
        try:
            N = int(self._mc_n_iter.get())
            seed = int(self._mc_seed.get())
            top_pct = float(self._mc_top_pct.get()) / 100.0
            filter_v_str = self._mc_filter_v.get()
            filter_v = self._parse_float_list(filter_v_str)

            # 건식 마찰 실측값
            log_v_exp = self._parse_float_list(self._mc_dry_v.get())
            v_exp = 10.0 ** log_v_exp
            n_v = len(v_exp)
            mu_exp_all = {}
            for name in self.COMPOUND_NAMES:
                s = self._mc_dry_mu[name].get().strip()
                if s:
                    mu_exp_all[name] = self._parse_float_list(s)
                else:
                    mu_exp_all[name] = None

            # 컴파운드별 파라미터
            compound_params = []
            for i, name in enumerate(self.COMPOUND_NAMES):
                sigma = float(self._mc_compound_vars[i]['sigma'].get()) * 1e6  # MPa → Pa
                temp = float(self._mc_compound_vars[i]['temp'].get())
                poisson = float(self._mc_compound_vars[i]['poisson'].get())
                mat = self._get_compound_material(i)
                compound_params.append({
                    'name': name, 'sigma': sigma, 'temp': temp,
                    'poisson': poisson, 'material': mat
                })

            rng = np.random.RandomState(seed)
            results = []
            n_pass1 = 0
            survival_curve = []
            t_start = time.time()

            self._mc_log(f"=== Monte Carlo 시작: N={N} ===")
            self._mc_log(f"C_pool: {self.C_pool.shape}, q1_pool: {len(self.q1_pool)}")
            self._mc_log(f"필터 속도: {filter_v} m/s")

            def _update_progress(i_iter, n_total, n_survived):
                elapsed = time.time() - t_start
                rate = (i_iter + 1) / max(elapsed, 0.01)
                eta = (n_total - i_iter - 1) / max(rate, 0.01)
                pct = int((i_iter + 1) / n_total * 100)
                self._mc_progress['value'] = pct
                self._mc_status_label.config(
                    text=f"진행: {i_iter+1}/{n_total} ({pct}%) — 생존: {n_survived}")
                self._mc_eta_label.config(
                    text=f"경과: {elapsed:.0f}s, 예상 잔여: {eta:.0f}s")

            for i_iter in range(N):
                if self._mc_stop:
                    self._mc_log("중단됨.")
                    break

                # 1. 랜덤 샘플
                idx_C = rng.randint(0, len(self.C_pool))
                idx_q1 = rng.randint(0, len(self.q1_pool))
                C_sample = self.C_pool[idx_C]
                q1_sample = self.q1_pool[idx_q1]

                # 2. q 배열 구성
                mask = self.q_grid <= q1_sample
                if np.sum(mask) < 10:
                    continue
                q_use = self.q_grid[mask]
                C_use = C_sample[mask]

                # 3. MeasuredPSD 생성
                try:
                    psd = MeasuredPSD(q_use, C_use, interpolation_kind='log-log')
                except Exception:
                    continue

                # 4. 2단계 전체 속도 계산 + 점수
                all_mu_pred = {}
                score = 0.0
                valid = True

                for ci, cp in enumerate(compound_params):
                    if cp['material'] is None:
                        continue
                    mu_exp_c = mu_exp_all.get(cp['name'])
                    if mu_exp_c is None or len(mu_exp_c) == 0:
                        continue

                    mu_pred_list = []
                    for vi, v in enumerate(v_exp):
                        if vi >= len(mu_exp_c):
                            break
                        try:
                            mat = cp['material']
                            modulus_func = lambda omega, _m=mat, _T=cp['temp']: \
                                _m.get_modulus(omega, _T)
                            g_calc = GCalculator(
                                psd_func=psd,
                                modulus_func=modulus_func,
                                sigma_0=cp['sigma'],
                                velocity=v,
                                poisson_ratio=cp['poisson'],
                            )
                            G_arr = g_calc.calculate_G(q_use)
                            C_q = psd(q_use)
                            loss_func = lambda omega, T, _m=mat, _T=cp['temp']: \
                                _m.get_loss_modulus(omega, _T)
                            fc = FrictionCalculator(
                                psd_func=psd,
                                loss_modulus_func=loss_func,
                                sigma_0=cp['sigma'],
                                velocity=v,
                                temperature=cp['temp'],
                                poisson_ratio=cp['poisson'],
                            )
                            mu, _ = fc.calculate_mu_visc(q_use, G_arr, C_q)
                            mu_pred_list.append(mu)
                        except Exception:
                            valid = False
                            break

                    if not valid:
                        break

                    if mu_pred_list:
                        mu_pred_arr = np.array(mu_pred_list)
                        mu_exp_arr = np.array(mu_exp_c[:len(mu_pred_list)])
                        score += np.sum((mu_pred_arr - mu_exp_arr) ** 2)
                        all_mu_pred[cp['name']] = mu_pred_arr

                if not valid or not all_mu_pred:
                    continue

                likelihood = np.exp(-score)
                results.append({
                    'idx_C': idx_C,
                    'idx_q1': idx_q1,
                    'q1': q1_sample,
                    'score': likelihood,
                    'mu_pred': all_mu_pred,
                })
                n_pass1 += 1
                survival_curve.append(n_pass1)

                # 100회마다 UI 업데이트
                if (i_iter + 1) % 100 == 0 or i_iter == N - 1:
                    self.app.root.after(0, _update_progress, i_iter, N, n_pass1)

                if (i_iter + 1) % 500 == 0:
                    self._mc_log(f"  [{i_iter+1}/{N}] 생존: {n_pass1}")

            # 결과 정리
            if results:
                scores = np.array([r['score'] for r in results])
                threshold = np.percentile(scores, 100 * (1.0 - top_pct))
                for r in results:
                    r['survived'] = r['score'] >= threshold
                n_final = sum(1 for r in results if r['survived'])
            else:
                n_final = 0

            self.mc_results = results
            self._mc_survival_curve = survival_curve

            elapsed_total = time.time() - t_start
            self._mc_log(f"\n=== 완료 ===")
            self._mc_log(f"총 반복: {N}, 유효 결과: {len(results)}, 상위 선별: {n_final}")
            self._mc_log(f"소요 시간: {elapsed_total:.1f}s")

            # 자동 저장
            try:
                pkl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        'results_mc.pkl')
                with open(pkl_path, 'wb') as f:
                    pickle.dump({'results': results, 'C_pool': self.C_pool,
                                 'q_grid': self.q_grid, 'q1_pool': self.q1_pool,
                                 'survival_curve': survival_curve}, f)
                self._mc_log(f"결과 저장: {pkl_path}")
            except Exception as e:
                self._mc_log(f"자동 저장 실패: {e}")

            def _finalize():
                self._mc_progress['value'] = 100
                self._mc_status_label.config(
                    text=f"완료 — 유효: {len(results)}, 선별: {n_final}")
                self._mc_eta_label.config(text=f"소요: {elapsed_total:.1f}s")
                self.app._show_status(
                    f"MC 완료: {len(results)}개 유효, {n_final}개 선별", 'success')
                self._update_mc_progress_chart()
                self._update_result_plots()
            self.app.root.after(0, _finalize)

        except Exception as e:
            self._mc_log(f"오류: {e}")
            import traceback
            self._mc_log(traceback.format_exc())
            self.app.root.after(0, lambda: messagebox.showerror("MC 오류", str(e)))

    def _update_mc_progress_chart(self):
        """MC 진행 차트 업데이트."""
        ax = self._ax_mc_progress
        ax.clear()
        ax.set_title('누적 생존 조합', fontweight='bold')
        ax.set_xlabel('반복 횟수')
        ax.set_ylabel('생존 수')
        ax.grid(True, alpha=0.3)
        if hasattr(self, '_mc_survival_curve') and self._mc_survival_curve:
            ax.plot(range(1, len(self._mc_survival_curve) + 1),
                    self._mc_survival_curve, '-', color='#2563EB', linewidth=1.5)
        self._fig_mc_progress.tight_layout(pad=2.0)
        self._canvas_mc_progress.draw_idle()

    # ================================================================
    #  서브탭 4: 결과
    # ================================================================
    def _build_result_tab(self, parent):
        left, right = self._make_lr_layout(parent, panel_width=300)

        # ── 좌측: 요약 + 버튼 ──
        sec1 = self._make_section(left, "결과 요약")
        self._result_summary = tk.Text(sec1, height=12, wrap=tk.WORD,
                                       font=self.app.FONTS.get('mono', ('Courier', 10)),
                                       bg='#F8FAFC', fg='#1E293B',
                                       state='disabled')
        self._result_summary.pack(fill=tk.BOTH, expand=True, pady=2)

        sec2 = self._make_section(left, "파일 관리")
        ttk.Button(sec2, text="결과 불러오기 (pkl)",
                   command=self._load_results_pkl).pack(fill=tk.X, pady=2)
        ttk.Button(sec2, text="결과 저장 (pkl)",
                   command=self._save_results_pkl).pack(fill=tk.X, pady=2)
        ttk.Button(sec2, text="플롯 저장 (PNG)",
                   command=self._save_result_png).pack(fill=tk.X, pady=2)

        # ── 우측: 2×2 플롯 ──
        self._fig_result = Figure(figsize=(10, 8), dpi=100, facecolor='white')
        self._ax_r1 = self._fig_result.add_subplot(2, 2, 1)
        self._ax_r2 = self._fig_result.add_subplot(2, 2, 2)
        self._ax_r3 = self._fig_result.add_subplot(2, 2, 3)
        self._ax_r4 = self._fig_result.add_subplot(2, 2, 4)
        self._fig_result.tight_layout(pad=2.5)

        self._canvas_result = FigureCanvasTkAgg(self._fig_result, master=right)
        self._canvas_result.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._canvas_result.draw_idle()

    def _update_result_plots(self):
        """결과 4-패널 플롯 업데이트."""
        if not self.mc_results:
            return

        results = self.mc_results
        survived = [r for r in results if r.get('survived', False)]
        all_q1 = np.array([r['q1'] for r in results])
        all_scores = np.array([r['score'] for r in results])

        # ── 요약 텍스트 ──
        q1_surv = np.array([r['q1'] for r in survived]) if survived else np.array([])
        self._result_summary.config(state='normal')
        self._result_summary.delete('1.0', tk.END)
        summary_lines = [
            f"총 유효 결과: {len(results)}",
            f"상위 선별: {len(survived)}",
        ]
        if len(q1_surv) > 0:
            med = np.median(q1_surv)
            ci_lo = np.percentile(q1_surv, 5)
            ci_hi = np.percentile(q1_surv, 95)
            summary_lines += [
                f"",
                f"q₁ 중앙값: {med:.3e} m⁻¹",
                f"90% CI: [{ci_lo:.3e}, {ci_hi:.3e}]",
            ]
        self._result_summary.insert('1.0', '\n'.join(summary_lines))
        self._result_summary.config(state='disabled')

        # ── 플롯 1: q₁ 분포 ──
        ax1 = self._ax_r1
        ax1.clear()
        ax1.set_title('q₁ 분포', fontweight='bold')
        ax1.set_xlabel('log₁₀(q₁)')
        ax1.set_ylabel('빈도')

        log_q1_all = np.log10(all_q1)
        ax1.hist(log_q1_all, bins=30, color='gray', alpha=0.5, label='전체')
        if len(q1_surv) > 0:
            log_q1_surv = np.log10(q1_surv)
            ax1.hist(log_q1_surv, bins=30, color='#2563EB', alpha=0.7, label='생존')
            med = np.median(log_q1_surv)
            ci_lo = np.percentile(log_q1_surv, 5)
            ci_hi = np.percentile(log_q1_surv, 95)
            ax1.axvline(med, color='red', linewidth=2, label=f'중앙값 {10**med:.2e}')
            ax1.axvline(ci_lo, color='red', linewidth=1, linestyle='--')
            ax1.axvline(ci_hi, color='red', linewidth=1, linestyle='--')
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # ── 플롯 2: C(q) Posterior 밴드 ──
        ax2 = self._ax_r2
        ax2.clear()
        ax2.set_title('C(q) Posterior 밴드', fontweight='bold')
        ax2.set_xlabel('q (1/m)')
        ax2.set_ylabel('C(q) (m⁴)')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)

        if self.C_pool is not None and self.q_grid is not None:
            # 원본 (회색)
            for q, C in self.psd_scans:
                ax2.plot(q, C, '-', color='gray', alpha=0.3, linewidth=0.5)

            # 생존 C(q) (파란색)
            surv_indices = [r['idx_C'] for r in survived]
            n_show = min(50, len(surv_indices))
            for i in range(n_show):
                ax2.plot(self.q_grid, self.C_pool[surv_indices[i]], '-',
                        color='#3B82F6', alpha=0.1, linewidth=0.5)

            if surv_indices:
                C_surv = self.C_pool[surv_indices]
                C_median = np.median(C_surv, axis=0)
                C_lo = np.percentile(C_surv, 5, axis=0)
                C_hi = np.percentile(C_surv, 95, axis=0)
                ax2.plot(self.q_grid, C_median, 'r-', linewidth=2, label='중앙값')
                ax2.fill_between(self.q_grid, C_lo, C_hi,
                                color='#3B82F6', alpha=0.2, label='90% CI')
                ax2.legend(loc='best', fontsize=8)

        # ── 플롯 3: 점수 분포 ──
        ax3 = self._ax_r3
        ax3.clear()
        ax3.set_title('Likelihood 점수 분포', fontweight='bold')
        ax3.set_xlabel('점수')
        ax3.set_ylabel('빈도')
        ax3.grid(True, alpha=0.3)

        if len(all_scores) > 0:
            ax3.hist(all_scores, bins=50, color='#64748B', alpha=0.7)
            if survived:
                threshold = min(r['score'] for r in survived)
                ax3.axvline(threshold, color='red', linewidth=2,
                           linestyle='--', label=f'임계값 {threshold:.4e}')
                ax3.legend(loc='best', fontsize=8)

        # ── 플롯 4: (q₁, rms) 산점도 ──
        ax4 = self._ax_r4
        ax4.clear()
        ax4.set_title('(q₁, C_rms) 산점도', fontweight='bold')
        ax4.set_xlabel('log₁₀(q₁)')
        ax4.set_ylabel('C(q) rms')
        ax4.grid(True, alpha=0.3)

        if self.C_pool is not None and self.q_grid is not None:
            rms_all = np.array([
                np.sqrt(2 * np.pi * np.trapz(
                    self.C_pool[r['idx_C']] * self.q_grid, self.q_grid))
                for r in results
            ])
            log_q1 = np.log10(all_q1)
            ax4.scatter(log_q1, rms_all, s=5, c='gray', alpha=0.3, label='전체')

            if survived:
                rms_surv = np.array([
                    np.sqrt(2 * np.pi * np.trapz(
                        self.C_pool[r['idx_C']] * self.q_grid, self.q_grid))
                    for r in survived
                ])
                log_q1_s = np.log10(q1_surv)
                ax4.scatter(log_q1_s, rms_surv, s=40, c='red', marker='*',
                           label='생존', zorder=5)
            ax4.legend(loc='best', fontsize=8)

        self._fig_result.tight_layout(pad=2.5)
        self._canvas_result.draw_idle()

    # ── 파일 I/O ──
    def _load_results_pkl(self):
        """pkl 파일에서 결과 불러오기."""
        fpath = filedialog.askopenfilename(
            title="MC 결과 파일 선택",
            filetypes=[("Pickle", "*.pkl"), ("All", "*.*")])
        if not fpath:
            return
        try:
            with open(fpath, 'rb') as f:
                data = pickle.load(f)
            self.mc_results = data.get('results', [])
            if 'C_pool' in data:
                self.C_pool = data['C_pool']
            if 'q_grid' in data:
                self.q_grid = data['q_grid']
            if 'q1_pool' in data:
                self.q1_pool = data['q1_pool']
            if 'survival_curve' in data:
                self._mc_survival_curve = data['survival_curve']
            self._update_result_plots()
            self.app._show_status(
                f"MC 결과 로드 완료: {len(self.mc_results)}개", 'success')
        except Exception as e:
            messagebox.showerror("오류", f"결과 로드 실패:\n{e}")

    def _save_results_pkl(self):
        """결과를 pkl로 저장."""
        if not self.mc_results:
            messagebox.showwarning("저장", "저장할 결과가 없습니다.")
            return
        fpath = filedialog.asksaveasfilename(
            title="MC 결과 저장",
            defaultextension='.pkl',
            filetypes=[("Pickle", "*.pkl")])
        if not fpath:
            return
        try:
            with open(fpath, 'wb') as f:
                pickle.dump({
                    'results': self.mc_results,
                    'C_pool': self.C_pool,
                    'q_grid': self.q_grid,
                    'q1_pool': self.q1_pool,
                    'survival_curve': getattr(self, '_mc_survival_curve', []),
                }, f)
            self.app._show_status(f"결과 저장 완료: {fpath}", 'success')
        except Exception as e:
            messagebox.showerror("오류", f"저장 실패:\n{e}")

    def _save_result_png(self):
        """결과 플롯을 PNG로 저장."""
        if not self.mc_results:
            messagebox.showwarning("저장", "저장할 결과가 없습니다.")
            return
        fpath = filedialog.asksaveasfilename(
            title="플롯 PNG 저장",
            defaultextension='.png',
            filetypes=[("PNG", "*.png")])
        if not fpath:
            return
        try:
            self._fig_result.savefig(fpath, dpi=150, bbox_inches='tight')
            self.app._show_status(f"플롯 저장 완료: {fpath}", 'success')
        except Exception as e:
            messagebox.showerror("오류", f"저장 실패:\n{e}")
