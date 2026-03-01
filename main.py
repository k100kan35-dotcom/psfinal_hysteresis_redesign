"""
Enhanced Main GUI for Persson Friction Model (v3.0)
===================================================

Work Instruction v3.0 Implementation:
- Log-log cubic spline interpolation for viscoelastic data
- Inner integral visualization with savgol_filter smoothing
- G(q,v) heatmap using pcolormesh
- Korean labels for all graphs and UI elements
- Velocity range: 0.0001~10 m/s (log scale)
- G(q,v) 2D matrix calculation
- Input data verification tab
- Multi-velocity G(q) plotting
- Default measured data loading
"""

import sys
import os

# PyInstaller frozen exe: matplotlib 폰트 캐시 디렉토리를 쓰기 가능한 임시 경로로 설정
# (반드시 import matplotlib 전에 실행해야 함)
if getattr(sys, 'frozen', False):
    import tempfile
    _mpl_cfg = os.path.join(tempfile.gettempdir(), 'mpl_persson')
    os.makedirs(_mpl_cfg, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = _mpl_cfg
    # 빌드 머신 경로가 담긴 stale 캐시 삭제 → 실행 환경에서 재생성
    for _f in os.listdir(_mpl_cfg):
        if _f.startswith('fontlist') and _f.endswith('.json'):
            try:
                os.remove(os.path.join(_mpl_cfg, _f))
            except OSError:
                pass

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# === 기본 폰트/수식 설정 (모든 환경에서 안전) ===
matplotlib.rcParams.update({
    'axes.unicode_minus': False,       # ASCII 마이너스 (유니코드 − 깨짐 방지)
    'text.usetex': False,              # LaTeX 비활성화
    'mathtext.fontset': 'cm',  # 수식 폰트: Computer Modern (LaTeX 스타일, Cambria Math 유사)
    'font.size': 14,
    'axes.titlesize': 15,
    'axes.labelsize': 13,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm
from scipy.signal import savgol_filter
from typing import Optional
import io
import json

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from persson_model.core.g_calculator import GCalculator
from persson_model.core.psd_models import FractalPSD, MeasuredPSD
from persson_model.core.viscoelastic import ViscoelasticMaterial
from persson_model.core.contact import ContactMechanics
from persson_model.utils.output import (
    save_calculation_details_csv,
    save_summary_txt,
    export_for_plotting,
    format_parameters_dict
)
from persson_model.utils.data_loader import (
    load_psd_from_file,
    load_dma_from_file,
    create_material_from_dma,
    create_psd_from_data,
    smooth_dma_data,
    load_strain_sweep_file,
    load_fg_curve_file,
    compute_fg_from_strain_sweep,
    create_fg_interpolator,
    average_fg_curves,
    create_strain_grid,
    DEFAULT_STRAIN_SPLIT
)
from persson_model.core.friction import (
    FrictionCalculator,
    calculate_mu_visc_simple,
    apply_nonlinear_strain_correction,
    calculate_rms_slope_profile,
    calculate_strain_profile,
    calculate_hrms_profile,
    RMSSlopeCalculator
)
from persson_model.core.master_curve import MasterCurveGenerator, load_multi_temp_dma
from persson_model.core.psd_from_profile import ProfilePSDAnalyzer, self_affine_psd_model

# === 한글 폰트 설정 (PyInstaller frozen exe 호환) ===
try:
    import matplotlib.font_manager as fm
    import platform

    # Frozen exe: 시스템 폰트 디렉토리에서 직접 등록 (캐시 우회)
    if getattr(sys, 'frozen', False):
        _font_dirs = []
        if platform.system() == 'Windows':
            _windir = os.environ.get('WINDIR', r'C:\Windows')
            _font_dirs.append(os.path.join(_windir, 'Fonts'))
            # 사용자별 폰트 디렉토리
            _localappdata = os.environ.get('LOCALAPPDATA', '')
            if _localappdata:
                _font_dirs.append(os.path.join(_localappdata, r'Microsoft\Windows\Fonts'))
        elif platform.system() == 'Darwin':
            _font_dirs.extend(['/Library/Fonts', os.path.expanduser('~/Library/Fonts')])
        else:
            _font_dirs.extend(['/usr/share/fonts', os.path.expanduser('~/.local/share/fonts')])

        _korean_ttfs = ['malgun.ttf', 'malgunbd.ttf', 'malgunsl.ttf',
                        'NanumGothic.ttf', 'NanumGothicBold.ttf',
                        'gulim.ttc', 'batang.ttc']
        for _fdir in _font_dirs:
            if os.path.isdir(_fdir):
                for _fname in _korean_ttfs:
                    _fpath = os.path.join(_fdir, _fname)
                    if os.path.exists(_fpath):
                        try:
                            if hasattr(fm.fontManager, 'addfont'):
                                fm.fontManager.addfont(_fpath)
                            else:
                                fm.fontManager.ttflist.append(
                                    fm.FontEntry(fname=_fpath, name=_fname.split('.')[0]))
                        except Exception:
                            pass

    # 시스템에서 한글 폰트 검색
    korean_fonts = []
    for font in fm.fontManager.ttflist:
        if any(name in font.name for name in ['Malgun', 'NanumGothic', 'NanumBarun',
                                                'AppleGothic', 'Gulim', 'Dotum']):
            if font.name not in korean_fonts:
                korean_fonts.append(font.name)

    if korean_fonts:
        matplotlib.rcParams['font.family'] = 'sans-serif'
        matplotlib.rcParams['font.sans-serif'] = korean_fonts + ['DejaVu Sans', 'Arial']
    else:
        matplotlib.rcParams['font.family'] = 'sans-serif'
        matplotlib.rcParams['font.sans-serif'] = ['Malgun Gothic', 'DejaVu Sans', 'Arial']
except Exception:
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']

matplotlib.rcParams['axes.labelweight'] = 'bold'
matplotlib.rcParams['axes.labelsize'] = 13
matplotlib.rcParams['figure.titleweight'] = 'bold'
matplotlib.rcParams['figure.titlesize'] = 16


class PerssonModelGUI_V2:
    """Enhanced GUI for Persson friction model (Work Instruction v2.1)."""

    # ── Modern Color Palette ──
    COLORS = {
        'bg':           '#F0F2F5',   # 전체 배경 (연한 회색-블루)
        'surface':      '#FFFFFF',   # 카드/패널 배경
        'sidebar':      '#1B2A4A',   # 사이드바 / 헤더 (딥 네이비)
        'sidebar_text': '#E2E8F0',   # 사이드바 텍스트
        'primary':      '#2563EB',   # 메인 액센트 (블루)
        'primary_hover':'#1D4ED8',
        'primary_fg':   '#FFFFFF',
        'success':      '#059669',   # 녹색 (확정/완료)
        'success_fg':   '#FFFFFF',
        'danger':       '#DC2626',   # 적색 (중요/경고)
        'danger_fg':    '#FFFFFF',
        'warning':      '#D97706',   # 오렌지 (주의)
        'text':         '#1E293B',   # 메인 텍스트
        'text_secondary':'#64748B',  # 보조 텍스트
        'border':       '#CBD5E1',   # 테두리
        'tab_active':   '#FFFFFF',   # 선택된 탭
        'tab_inactive': '#E2E8F0',   # 비선택 탭
        'input_bg':     '#FFFFFF',   # 입력 필드 배경
        'input_border': '#94A3B8',   # 입력 필드 테두리
        'statusbar_bg': '#1E293B',   # 상태바 배경
        'statusbar_fg': '#94A3B8',   # 상태바 텍스트
        'highlight':    '#DBEAFE',   # 강조 배경 (연한 블루)
    }

    # Font sizes designed for 96 DPI (100% scaling, 1600×1000 window).
    # On high-DPI displays, the Tk scaling factor is reset in main() so that
    # these point sizes render at the same physical size on every machine.
    FONTS = {
        'heading':   ('Segoe UI', 22, 'bold'),
        'subheading':('Segoe UI', 20, 'bold'),
        'body':      ('Segoe UI', 17),
        'body_bold': ('Segoe UI', 17, 'bold'),
        'small':     ('Segoe UI', 16),
        'small_bold':('Segoe UI', 16, 'bold'),
        'tiny':      ('Segoe UI', 15),
        'mono':      ('Consolas', 17),
        'mono_small':('Consolas', 16),
    }

    # ── NEXEN TIRE Logo ──
    # Logo loaded from assets/nexen_logo.png at runtime

    # ── Plot Font Size Constants (base sizes for 1600px window) ──
    PLOT_FONTS = {
        'title': 15,          # subplot titles
        'label': 13,          # axis labels (x, y)
        'tick': 12,           # tick labels
        'legend': 12,         # legend text
        'suptitle': 16,       # figure suptitle
        'annotation': 12,     # annotation text
        'title_sm': 13,       # titles in dense grids (strain map, VE advisor)
        'label_sm': 12,       # labels in dense grids
        'legend_sm': 10,      # legends in dense grids
    }
    _REFERENCE_WIDTH = 1600   # reference window width for font scaling

    def __init__(self, root):
        """Initialize enhanced GUI."""
        self.root = root
        self.root.title("NEXEN Rubber Friction Modelling Program  v3.0")
        self.root.geometry("1600x1000")
        self.root.configure(bg=self.COLORS['bg'])
        self.root.minsize(1200, 700)

        # Store DPI scale for any component that needs it
        self._dpi_scale = _get_system_dpi_scale()

        # ── Apply modern theme ──
        self._setup_modern_theme()

        # Initialize variables
        self.material = None
        self.psd_model = None
        self.g_calculator = None
        self.results = {}
        self.raw_dma_data = None  # Store raw DMA data for plotting
        self.raw_psd_data = None  # Store raw PSD data for comparison plotting
        self.target_xi = None  # Target h'rms from Tab 2 PSD settings

        # Data source tracking
        self.material_source = None  # 마스터 커브 출처: "기본 파일", "예제 SBR", "Tab 1 확정", etc.
        self.psd_source = None  # PSD 출처: "Tab 0 확정", etc.

        # Strain/mu_visc related variables
        self.strain_data = None  # Strain sweep raw data by temperature
        self.fg_by_T = None  # f,g curves by temperature
        self.fg_averaged = None  # Averaged f,g curves
        self.f_interpolator = None  # f(strain) function
        self.g_interpolator = None  # g(strain) function
        self.mu_visc_results = None  # mu_visc calculation results

        # h'rms / Local Strain related variables
        self.rms_slope_calculator = None  # RMSSlopeCalculator instance
        self.rms_slope_profiles = None  # Calculated profiles (q, xi, strain, hrms)
        self.local_strain_array = None  # Local strain for mu_visc calculation

        # Profile PSD analyzer (Tab 0)
        self.profile_psd_analyzer = None  # ProfilePSDAnalyzer instance
        self.profile_psd_data = None  # Loaded profile data (x, h)

        # Graph data registry for automatic data listing
        self.graph_data_registry = {}  # {name: {x, y, header, description, timestamp}}

        # Reference μ_visc data for comparison (Persson program output)
        self._init_reference_mu_data()
        # Multiple reference datasets for overlay plotting
        self.plotted_ref_datasets = []  # list of {'name':str, 'mu_log_v':[], 'mu_vals':[], 'area_log_v':[], 'area_vals':[]}

        # Initialize tkinter variables that were previously in verification tab
        # These are needed by other functions that reference them
        self.dma_import_status_var = tk.StringVar(value="데이터 미로드")
        self.psd_status_var = tk.StringVar(value="PSD 미설정")
        self.psd_q_range_var = tk.StringVar(value="- ~ - (1/m)")
        self.psd_H_var = tk.StringVar(value="0.8")
        self.psd_xi_var = tk.StringVar(value="1.3")
        self.verify_smooth_var = tk.BooleanVar(value=True)
        self.verify_smooth_window_var = tk.IntVar(value=11)
        self.verify_extrap_var = tk.BooleanVar(value=True)
        self.dma_extrap_fmin_var = tk.StringVar(value="1e-2")
        self.dma_extrap_fmax_var = tk.StringVar(value="1e12")
        self.psd_q0_var = tk.StringVar(value="500")
        self.psd_q1_var = tk.StringVar(value="1e5")
        self.psd_Cq0_var = tk.StringVar(value="3.5e-13")

        # Create UI
        self._create_menu()
        self._create_main_layout()
        self._create_status_bar()

        # Load default measured data
        self._load_default_data()

    # ================================================================
    #  THEME / STYLE  CONFIGURATION
    # ================================================================
    def _setup_modern_theme(self):
        """Configure ttk styles for a modern, professional look."""
        C = self.COLORS
        F = self.FONTS
        style = ttk.Style(self.root)

        # Base theme
        try:
            style.theme_use('clam')
        except tk.TclError:
            pass

        # ── Override Tk default fonts (affects all Entry/Combobox field text) ──
        import tkinter.font as tkfont
        _body_size = F['body'][1]
        _mono_size = F['mono'][1]
        for fname in ('TkDefaultFont', 'TkTextFont', 'TkFixedFont'):
            try:
                f = tkfont.nametofont(fname)
                if fname == 'TkFixedFont':
                    f.configure(family='Consolas', size=_mono_size)
                else:
                    f.configure(family='Segoe UI', size=_body_size)
            except Exception:
                pass

        # ── Global defaults ──
        style.configure('.', background=C['bg'], foreground=C['text'],
                        font=F['body'], borderwidth=0)

        # ── TFrame ──
        style.configure('TFrame', background=C['bg'])
        style.configure('Card.TFrame', background=C['surface'], relief='flat')

        # ── TLabel ──
        style.configure('TLabel', background=C['bg'], foreground=C['text'],
                        font=F['body'])
        style.configure('Heading.TLabel', font=F['heading'], foreground=C['sidebar'])
        style.configure('Sub.TLabel', font=F['subheading'], foreground=C['text'])
        style.configure('Small.TLabel', font=F['small'], foreground=C['text_secondary'])
        style.configure('Tiny.TLabel', font=F['tiny'], foreground=C['text_secondary'])
        style.configure('Status.TLabel', background=C['statusbar_bg'],
                        foreground=C['statusbar_fg'], font=F['small'],
                        padding=(12, 6))

        # ── TLabelframe ──
        style.configure('TLabelframe', background=C['surface'],
                        relief='flat', borderwidth=1, bordercolor=C['border'])
        style.configure('TLabelframe.Label', background=C['surface'],
                        foreground=C['primary'], font=F['body_bold'])

        # ── TNotebook (Tabs) ──
        style.configure('TNotebook', background=C['bg'], borderwidth=0,
                        tabmargins=[4, 4, 4, 0])
        style.configure('TNotebook.Tab', background=C['tab_inactive'],
                        foreground=C['text_secondary'], font=F['body_bold'],
                        padding=[14, 6], borderwidth=0)
        style.map('TNotebook.Tab',
                  background=[('selected', C['tab_active']),
                              ('active', C['highlight'])],
                  foreground=[('selected', C['primary']),
                              ('active', C['text'])],
                  expand=[('selected', [0, 0, 0, 2])])

        # ── TButton (Default) ──
        style.configure('TButton', font=F['body'], padding=[12, 5],
                        background=C['surface'], foreground=C['text'],
                        borderwidth=1, relief='raised', anchor='center')
        style.map('TButton',
                  background=[('pressed', C['border']),
                              ('active', C['highlight'])],
                  relief=[('pressed', 'sunken'), ('!pressed', 'raised')])

        # ── Accent.TButton (Primary action - blue) ──
        style.configure('Accent.TButton', font=F['body_bold'],
                        background=C['primary'], foreground=C['primary_fg'],
                        padding=[14, 6], borderwidth=2, relief='raised')
        style.map('Accent.TButton',
                  background=[('pressed', '#0F172A'),
                              ('active', C['primary_hover']),
                              ('disabled', C['border'])],
                  foreground=[('disabled', C['text_secondary'])],
                  relief=[('pressed', 'sunken'), ('!pressed', 'raised')],
                  padding=[('pressed', [16, 8])])

        # ── Outline.TButton (Blue outline / border) ──
        style.configure('Outline.TButton', font=F['body_bold'],
                        background=C['surface'], foreground=C['primary'],
                        padding=[14, 6], borderwidth=2, relief='solid',
                        bordercolor=C['primary'])
        style.map('Outline.TButton',
                  background=[('active', '#EBF5FF'),
                              ('pressed', '#DBEAFE')],
                  foreground=[('disabled', C['text_secondary'])])

        # ── Success.TButton (Confirm - green) ──
        style.configure('Success.TButton', font=F['body_bold'],
                        background=C['success'], foreground=C['success_fg'],
                        padding=[14, 6], borderwidth=2, relief='raised')
        style.map('Success.TButton',
                  background=[('pressed', '#064E3B'),
                              ('active', '#047857'),
                              ('disabled', C['border'])],
                  foreground=[('disabled', C['text_secondary'])],
                  relief=[('pressed', 'sunken'), ('!pressed', 'raised')],
                  padding=[('pressed', [16, 8])])

        # ── Danger.TButton (Warning action - red) ──
        style.configure('Danger.TButton', font=F['body_bold'],
                        background=C['danger'], foreground=C['danger_fg'],
                        padding=[14, 6], borderwidth=2, relief='raised')
        style.map('Danger.TButton',
                  background=[('pressed', '#7F1D1D'),
                              ('active', '#B91C1C'),
                              ('disabled', C['border'])],
                  foreground=[('disabled', C['text_secondary'])],
                  relief=[('pressed', 'sunken'), ('!pressed', 'raised')],
                  padding=[('pressed', [16, 8])])

        # ── TEntry ──
        style.configure('TEntry', fieldbackground=C['input_bg'],
                        foreground=C['text'], font=F['body'],
                        borderwidth=1, relief='solid', padding=[6, 4])
        style.map('TEntry',
                  fieldbackground=[('focus', '#F0F7FF'), ('readonly', C['bg'])],
                  bordercolor=[('focus', C['primary'])])

        # ── TCombobox ──
        style.configure('TCombobox', fieldbackground=C['input_bg'],
                        foreground=C['text'], font=F['body'], padding=[6, 4])
        style.map('TCombobox',
                  fieldbackground=[('readonly', C['input_bg']),
                                   ('focus', '#F0F7FF')])

        # ── TCheckbutton / TRadiobutton ──
        style.configure('TCheckbutton', background=C['bg'],
                        foreground=C['text'], font=F['body'],
                        indicatorsize=20, padding=[4, 2])
        style.configure('TRadiobutton', background=C['bg'],
                        foreground=C['text'], font=F['body'],
                        indicatorsize=20, padding=[4, 2])
        # Inside LabelFrames (surface bg)
        style.configure('Surface.TCheckbutton', background=C['surface'],
                        foreground=C['text'], font=F['body'],
                        indicatorsize=20, padding=[4, 2])
        style.configure('Surface.TRadiobutton', background=C['surface'],
                        foreground=C['text'], font=F['body'],
                        indicatorsize=20, padding=[4, 2])

        # ── Horizontal.TProgressbar ──
        style.configure('Horizontal.TProgressbar',
                        troughcolor=C['border'], background=C['primary'],
                        thickness=6, borderwidth=0)

        # ── TSeparator ──
        style.configure('TSeparator', background=C['border'])

        # ── TScrollbar ──
        style.configure('Vertical.TScrollbar', background=C['bg'],
                        troughcolor=C['bg'], borderwidth=0, arrowsize=12)
        style.map('Vertical.TScrollbar',
                  background=[('active', C['border']),
                              ('pressed', C['input_border'])])

        # ── Combobox dropdown list font ──
        self.root.option_add('*TCombobox*Listbox.font', F['body'])
        self.root.option_add('*TCombobox*Listbox.foreground', C['text'])

    def _load_default_data(self):
        """Load default measured data on startup."""
        try:
            # Get data directory
            base_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(base_dir, 'examples', 'data')

            psd_file = os.path.join(data_dir, 'measured_psd.txt')
            dma_file = os.path.join(data_dir, 'measured_dma.txt')

            if os.path.exists(psd_file) and os.path.exists(dma_file):
                # Load DMA data
                omega_raw, E_storage_raw, E_loss_raw = load_dma_from_file(
                    dma_file,
                    skip_header=1,
                    freq_unit='Hz',
                    modulus_unit='MPa'
                )

                # Apply smoothing/fitting
                smoothed = smooth_dma_data(omega_raw, E_storage_raw, E_loss_raw)

                # Store raw data for visualization
                self.raw_dma_data = {
                    'omega': omega_raw,
                    'E_storage': E_storage_raw,
                    'E_loss': E_loss_raw
                }

                # Create material from smoothed data
                self.material = create_material_from_dma(
                    omega=smoothed['omega'],
                    E_storage=smoothed['E_storage_smooth'],
                    E_loss=smoothed['E_loss_smooth'],
                    material_name="Measured Rubber (smoothed)",
                    reference_temp=20.0
                )
                self.material_source = f"기본 파일 ({os.path.basename(dma_file)})"

                # PSD is NOT loaded here - must come from Tab 0
                # User must use Tab 0 (PSD 생성) to set PSD data

                self._update_material_display()

                self.status_var.set(f"초기 DMA 데이터 로드 완료 ({len(omega_raw)}개). PSD는 Tab 0에서 설정하세요.")
            else:
                # Use example material
                self.material = ViscoelasticMaterial.create_example_sbr()
                self.material_source = "내장 예제 (SBR)"
                self._update_material_display()
                self.status_var.set("예제 재료 (SBR) 로드됨")

        except Exception as e:
            print(f"Default data loading error: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to example material
            self.material = ViscoelasticMaterial.create_example_sbr()
            self.material_source = "내장 예제 (SBR)"
            self.status_var.set("예제 재료 (SBR) 로드됨")

    def _init_reference_mu_data(self):
        """Initialize reference μ_visc and A/A0 data as empty (no default data loaded).

        Users can load reference data via the '참조 편집' dialog and saved datasets.
        """
        self.reference_mu_data = None
        self.reference_area_data = None
        print("[참조 데이터] 초기 참조 데이터 없음 (저장된 데이터셋에서 불러오기 가능)")

    def _create_menu(self):
        """Create menu bar."""
        C = self.COLORS
        menu_cfg = dict(
            bg=C['surface'], fg=C['text'], activebackground=C['primary'],
            activeforeground=C['primary_fg'], relief='flat', borderwidth=0,
            font=self.FONTS['body'],
        )

        menubar = tk.Menu(self.root, **menu_cfg)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0, **menu_cfg)
        menubar.add_cascade(label="  File  ", menu=file_menu)
        file_menu.add_command(label="  Load DMA Data", command=self._load_material)
        file_menu.add_separator()
        file_menu.add_command(label="  Save Results (CSV)", command=self._save_detailed_csv)
        file_menu.add_command(label="  Export All", command=self._export_all_results)
        file_menu.add_command(label="  Graph Data Export...", command=self._show_graph_data_export_popup)
        file_menu.add_separator()
        file_menu.add_command(label="  Exit", command=self.root.quit)

        # Settings menu
        settings_menu = tk.Menu(menubar, tearoff=0, **menu_cfg)
        menubar.add_cascade(label="  Settings  ", menu=settings_menu)
        settings_menu.add_command(label="  레이아웃 설정...", command=self._open_layout_settings)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0, **menu_cfg)
        menubar.add_cascade(label="  Help  ", menu=help_menu)
        help_menu.add_command(label="  User Guide", command=self._show_help)
        help_menu.add_command(label="  About", command=self._show_about)

    def _create_main_layout(self):
        """Create main application layout with tabs."""

        # ── Header bar ──
        header = tk.Frame(self.root, bg='#FFFFFF', height=48)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)
        tk.Label(header, text="NEXEN Rubber Friction Modelling Program",
                 bg='#FFFFFF', fg=self.COLORS['sidebar'],
                 font=('Segoe UI', 20, 'bold')).pack(side=tk.LEFT, padx=16)
        tk.Label(header, text="v3.0",
                 bg='#FFFFFF', fg=self.COLORS['text_secondary'],
                 font=('Segoe UI', 17)).pack(side=tk.LEFT, padx=(0, 8))

        # ── Load company logo (for left panel footers) ──
        self._logo_image = None
        try:
            if getattr(sys, 'frozen', False):
                _base_dir = sys._MEIPASS
            else:
                _base_dir = os.path.dirname(os.path.abspath(__file__))
            _logo_path = os.path.join(_base_dir, 'assets', 'nexen_logo.png')

            if os.path.exists(_logo_path):
                _logo_full = tk.PhotoImage(file=_logo_path)
                # Scale to ~70px height for panel footer
                _orig_h = _logo_full.height()
                _scale = max(1, _orig_h // 70)
                self._logo_image = _logo_full.subsample(_scale, _scale)
        except Exception:
            pass

        # Header bottom border
        tk.Frame(self.root, bg=self.COLORS['border'], height=1).pack(fill=tk.X, side=tk.TOP)

        # ── Activity Log Panel (작업 로그) ──
        self._create_log_panel()

        # Create notebook (tabbed interface)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=(4, 0))

        # ── Tab definitions ──
        tabs = [
            ('tab_psd_profile',     'PSD 생성',           self._create_psd_profile_tab),
            ('tab_master_curve',    '마스터 커브',         self._create_master_curve_tab),
            ('tab_parameters',      '계산 설정',           self._create_parameters_tab),
            ('tab_results',         'G(q,v) 결과',        self._create_results_tab),
            ('tab_rms_slope',       "h'rms / Strain",     self._create_rms_slope_tab),
            ('tab_mu_visc',         'μ_visc 계산',        self._create_mu_visc_tab),
            ('tab_ve_advisor',      '점탄성 설계',        self._create_ve_advisor_tab),
            ('tab_strain_map',      'Strain Map',         self._create_strain_map_tab),
            ('tab_integrand',       '피적분함수',          self._create_integrand_tab),
            ('tab_equations',       '수식 정리',           self._create_equations_tab),
            ('tab_variables',       '변수 관계',           self._create_variables_tab),
            ('tab_debug',           '디버그',              self._create_debug_tab),
            ('tab_friction_factors','영향 인자',           self._create_friction_factors_tab),
        ]

        for attr, label, builder in tabs:
            frame = ttk.Frame(self.notebook)
            setattr(self, attr, frame)
            self.notebook.add(frame, text=f'  {label}  ')
            builder(frame)

        # ── Handle tab switch: prevent graph resize flicker ──
        self._tab_switch_after_id = None

        def _on_tab_changed(event):
            # Cancel any pending redraw
            if self._tab_switch_after_id is not None:
                try:
                    self.root.after_cancel(self._tab_switch_after_id)
                except Exception:
                    pass

            # Freeze notebook size during transition to prevent flicker
            nb_w = self.notebook.winfo_width()
            nb_h = self.notebook.winfo_height()
            if nb_w > 1 and nb_h > 1:
                self.notebook.configure(width=nb_w, height=nb_h)

            def _stabilize():
                self._tab_switch_after_id = None
                # Release fixed size so it can adapt normally
                self.notebook.configure(width=0, height=0)
                # Redraw visible canvas
                for canvas_attr in ('canvas_psd_profile', 'canvas_mc',
                                    'canvas_calc_progress', 'canvas_results',
                                    'canvas_rms', 'canvas_mu_visc',
                                    'canvas_integrand', 'canvas_strain_map',
                                    'canvas_ve_advisor'):
                    canvas = getattr(self, canvas_attr, None)
                    if canvas is not None:
                        try:
                            widget = canvas.get_tk_widget()
                            if widget.winfo_ismapped():
                                canvas.draw_idle()
                        except Exception:
                            pass

            self._tab_switch_after_id = self.root.after(80, _stabilize)

        self.notebook.bind('<<NotebookTabChanged>>', _on_tab_changed)

        # Ctrl+Enter shortcut: trigger mu_visc 계산 when on the mu_visc tab
        def _on_ctrl_enter(event):
            current_tab = self.notebook.index(self.notebook.select())
            if current_tab == 5:  # tab_mu_visc (μ_visc 계산)
                self._calculate_mu_visc()
        self.root.bind('<Control-Return>', _on_ctrl_enter)

        # Initialize debug log storage
        self.debug_log_messages = []

        # ── Responsive font scaling ──
        self._setup_responsive_fonts()

    # ──────────────────────────────────────────────────────────
    # Responsive font scaling for different monitor sizes
    # ──────────────────────────────────────────────────────────
    def _setup_responsive_fonts(self):
        """Setup responsive font scaling based on window size."""
        self._font_scale = 1.0
        self._resize_after_id = None
        self._last_resize_width = 0

        # Base font sizes (match PLOT_FONTS)
        self._base_font_cfg = {
            'font.size':        self.PLOT_FONTS['label'],
            'axes.titlesize':   self.PLOT_FONTS['title'],
            'axes.labelsize':   self.PLOT_FONTS['label'],
            'xtick.labelsize':  self.PLOT_FONTS['tick'],
            'ytick.labelsize':  self.PLOT_FONTS['tick'],
            'legend.fontsize':  self.PLOT_FONTS['legend'],
            'figure.titlesize': self.PLOT_FONTS['suptitle'],
        }

        self.root.bind('<Configure>', self._on_root_configure)

    def _on_root_configure(self, event):
        """Handle window resize with debouncing."""
        if event.widget is not self.root:
            return
        new_width = event.width
        # Ignore trivial changes
        if abs(new_width - self._last_resize_width) < 80:
            return
        self._last_resize_width = new_width

        if self._resize_after_id is not None:
            self.root.after_cancel(self._resize_after_id)
        self._resize_after_id = self.root.after(250, self._rescale_all_fonts)

    def _rescale_all_fonts(self):
        """Rescale all plot fonts based on current window size."""
        self._resize_after_id = None
        scale = max(0.65, min(1.4, self._last_resize_width / self._REFERENCE_WIDTH))

        if abs(scale - self._font_scale) < 0.05:
            return
        self._font_scale = scale

        # Update rcParams for any future plots
        for key, base_size in self._base_font_cfg.items():
            matplotlib.rcParams[key] = max(8, round(base_size * scale))

        # Update all existing figures
        for fig, canvas in self._get_all_figures_and_canvases():
            for ax in fig.axes:
                # Title
                if ax.get_title():
                    ax.title.set_fontsize(max(9, round(self.PLOT_FONTS['title'] * scale)))
                # Axis labels
                ax.xaxis.label.set_fontsize(max(8, round(self.PLOT_FONTS['label'] * scale)))
                ax.yaxis.label.set_fontsize(max(8, round(self.PLOT_FONTS['label'] * scale)))
                # Tick labels
                ax.tick_params(labelsize=max(8, round(self.PLOT_FONTS['tick'] * scale)))
                # Legend
                legend = ax.get_legend()
                if legend:
                    for text in legend.get_texts():
                        text.set_fontsize(max(7, round(self.PLOT_FONTS['legend'] * scale)))
            try:
                fig.tight_layout()
            except Exception:
                pass
            canvas.draw_idle()

    def _get_all_figures_and_canvases(self):
        """Return list of (figure, canvas) tuples for all plot figures."""
        pairs = []
        for attr_fig, attr_canvas in [
            ('fig_psd_profile', 'canvas_psd_profile'),
            ('fig_mc', 'canvas_mc'),
            ('fig_calc_progress', 'canvas_calc_progress'),
            ('fig_results', 'canvas_results'),
            ('fig_rms', 'canvas_rms'),
            ('fig_mu_visc', 'canvas_mu_visc'),
            ('fig_strain_map', 'canvas_strain_map'),
            ('fig_integrand', 'canvas_integrand'),
            ('fig_ve_advisor', 'canvas_ve_advisor'),
        ]:
            fig = getattr(self, attr_fig, None)
            canvas = getattr(self, attr_canvas, None)
            if fig is not None and canvas is not None:
                pairs.append((fig, canvas))
        return pairs

    def _add_logo_to_panel(self, parent_frame):
        """Add company logo fixed at the bottom of a left panel frame.
        Must be called BEFORE packing scrollable content so it stays at bottom."""
        if self._logo_image is None:
            return
        logo_container = tk.Frame(parent_frame, bg='#F0F2F5', height=80)
        logo_container.pack(side=tk.BOTTOM, fill=tk.X)
        logo_container.pack_propagate(False)
        tk.Frame(logo_container, bg=self.COLORS['border'], height=1).pack(fill=tk.X, side=tk.TOP)
        tk.Label(logo_container, image=self._logo_image,
                 bg='#F0F2F5').pack(expand=True)

    def _create_panel_toolbar(self, parent, buttons=None):
        """Create a fixed toolbar at the top of a panel.

        Args:
            parent: Parent frame (usually left_frame)
            buttons: List of (text, command, style) tuples. style can be
                     'Accent.TButton', 'Success.TButton', 'TButton', etc.

        Returns:
            toolbar frame (tk.Frame)
        """
        C = self.COLORS
        toolbar = tk.Frame(parent, bg='#E2E8F0', padx=4, pady=3)
        toolbar.pack(side=tk.TOP, fill=tk.X)
        # Bottom border
        tk.Frame(parent, bg=C['border'], height=1).pack(side=tk.TOP, fill=tk.X)

        # Toolbar indicator
        tk.Label(toolbar, text="\u25A0", bg='#E2E8F0',
                 fg=C['primary'], font=('Segoe UI', 12)).pack(side=tk.LEFT, padx=(0, 4))

        if buttons is None:
            # Reference-only toolbar (no action buttons)
            tk.Label(toolbar, text="참조", bg='#E2E8F0',
                     fg=C['text_secondary'],
                     font=('Segoe UI', 14)).pack(side=tk.LEFT, padx=2)
        elif buttons:
            for item in buttons:
                text, command = item[0], item[1]
                style = item[2] if len(item) > 2 else 'Accent.TButton'
                btn = ttk.Button(toolbar, text=text, command=command, style=style)
                btn.pack(side=tk.LEFT, padx=3, pady=1)

        return toolbar

    def _create_psd_profile_tab(self, parent):
        """Create PSD from Profile tab for calculating PSD from surface height data."""
        # Main container
        main_container = ttk.Frame(parent)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel for controls (fixed width)
        left_frame = ttk.Frame(main_container, width=getattr(self, '_left_panel_width', 600))
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        left_frame.pack_propagate(False)

        # Logo at bottom (pack before canvas so it stays at bottom)
        self._add_logo_to_panel(left_frame)

        # Toolbar (fixed at top, always accessible)
        self._create_panel_toolbar(left_frame, buttons=[
            ("PSD 계산", self._calculate_profile_psd, 'Accent.TButton'),
            ("PSD 확정 \u2192 계산에 사용", self._apply_profile_psd_to_tab3, 'Success.TButton'),
        ])

        # Add scrollbar to left panel
        left_canvas = tk.Canvas(left_frame, highlightthickness=0)
        left_scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=left_canvas.yview)
        left_scrollable = ttk.Frame(left_canvas)

        left_scrollable.bind(
            "<Configure>",
            lambda e: left_canvas.configure(scrollregion=left_canvas.bbox("all"))
        )

        left_canvas.create_window((0, 0), window=left_scrollable, anchor="nw")
        left_canvas.configure(yscrollcommand=left_scrollbar.set)

        left_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Enable mousewheel scrolling (local only - not bind_all)
        def _on_mousewheel_tab0(event):
            left_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        def _bind_mousewheel_tab0(event):
            left_canvas.bind_all("<MouseWheel>", _on_mousewheel_tab0)
        def _unbind_mousewheel_tab0(event):
            left_canvas.unbind_all("<MouseWheel>")
        left_canvas.bind("<Enter>", _bind_mousewheel_tab0)
        left_canvas.bind("<Leave>", _unbind_mousewheel_tab0)

        # ============== Left Panel: Controls ==============

        # 1. Description
        desc_frame = ttk.LabelFrame(left_scrollable, text="탭 설명", padding=5)
        desc_frame.pack(fill=tk.X, pady=3)

        desc_text = (
            "표면 높이 프로파일 데이터로부터\n"
            "Power Spectral Density (PSD)를 계산합니다.\n\n"
            "• Full PSD: 전체 표면 거칠기\n"
            "• Top PSD: 상부 표면만 (h>0)\n"
            "  (Ref: J. Chem. Phys. 162, 074704)"
        )
        ttk.Label(desc_frame, text=desc_text, font=('Segoe UI', 17), justify=tk.LEFT).pack(anchor=tk.W)

        # 2. Data Loading
        load_frame = ttk.LabelFrame(left_scrollable, text="1. 데이터 로드", padding=5)
        load_frame.pack(fill=tk.X, pady=3)

        # File path display
        self.psd_profile_file_var = tk.StringVar(value="(파일 선택 안됨)")
        ttk.Label(load_frame, textvariable=self.psd_profile_file_var,
                  font=('Segoe UI', 17), foreground='#64748B').pack(fill=tk.X)

        # Load button
        ttk.Button(load_frame, text="프로파일 데이터 로드 (.txt, .csv)",
                   command=self._load_profile_data).pack(fill=tk.X, pady=2)

        # ===== 내장 PSD 선택 섹션 =====
        ttk.Separator(load_frame, orient='horizontal').pack(fill=tk.X, pady=5)
        ttk.Label(load_frame, text="─ 내장 PSD 데이터 선택 ─",
                  font=('Segoe UI', 17, 'bold'), foreground='#059669').pack(anchor=tk.CENTER)

        preset_psd_frame = ttk.Frame(load_frame)
        preset_psd_frame.pack(fill=tk.X, pady=2)
        ttk.Label(preset_psd_frame, text="내장 PSD:", font=('Segoe UI', 17)).pack(side=tk.LEFT)
        self.preset_psd_var = tk.StringVar(value="(선택...)")
        self.preset_psd_combo = ttk.Combobox(preset_psd_frame, textvariable=self.preset_psd_var,
                                              state='readonly', width=18, font=self.FONTS['body'])
        self.preset_psd_combo.pack(side=tk.LEFT, padx=3)
        ttk.Button(preset_psd_frame, text="로드", command=self._load_preset_psd, width=4,
                   style='Outline.TButton').pack(side=tk.LEFT)
        ttk.Button(preset_psd_frame, text="삭제", command=self._delete_preset_psd, width=4).pack(side=tk.LEFT, padx=2)

        # 프로그램 시작 시 내장 PSD 목록 로드
        self._refresh_preset_psd_list()

        # Direct PSD loading section
        ttk.Separator(load_frame, orient='horizontal').pack(fill=tk.X, pady=5)
        ttk.Label(load_frame, text="─ 또는: PSD 파일 직접 로드 ─",
                  font=('Segoe UI', 17), foreground='#2563EB').pack(anchor=tk.CENTER)

        # PSD 직접 로드 버튼 + 리스트에 추가 버튼
        psd_direct_btn_frame = ttk.Frame(load_frame)
        psd_direct_btn_frame.pack(fill=tk.X, pady=2)

        ttk.Button(psd_direct_btn_frame, text="PSD 직접 로드 (q, C(q))",
                   command=self._load_psd_direct,
                   style='Accent.TButton').pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Button(psd_direct_btn_frame, text="→ 리스트 추가",
                   command=self._add_preset_psd, width=10).pack(side=tk.LEFT, padx=(3, 0))

        self.psd_direct_info_var = tk.StringVar(value="PSD 직접 로드: -")
        ttk.Label(load_frame, textvariable=self.psd_direct_info_var,
                  font=('Segoe UI', 17), foreground='#64748B').pack(fill=tk.X)

        # PSD 확정 버튼 (직접 로드 바로 아래)
        ttk.Separator(load_frame, orient='horizontal').pack(fill=tk.X, pady=5)
        apply_frame_top = ttk.Frame(load_frame)
        apply_frame_top.pack(fill=tk.X, pady=2)
        self.apply_psd_type_var = tk.StringVar(value="full")
        ttk.Label(apply_frame_top, text="적용할 PSD:", font=('Segoe UI', 17)).pack(side=tk.LEFT)
        ttk.Radiobutton(apply_frame_top, text="Full", variable=self.apply_psd_type_var,
                        value="full").pack(side=tk.LEFT)
        ttk.Radiobutton(apply_frame_top, text="Top", variable=self.apply_psd_type_var,
                        value="top").pack(side=tk.LEFT)
        ttk.Radiobutton(apply_frame_top, text="Param", variable=self.apply_psd_type_var,
                        value="param").pack(side=tk.LEFT)
        ttk.Radiobutton(apply_frame_top, text="직접로드", variable=self.apply_psd_type_var,
                        value="direct").pack(side=tk.LEFT)

        # Column settings
        col_frame = ttk.Frame(load_frame)
        col_frame.pack(fill=tk.X, pady=2)

        ttk.Label(col_frame, text="X열:", font=('Segoe UI', 17)).pack(side=tk.LEFT)
        self.profile_x_col_var = tk.StringVar(value="0")
        ttk.Entry(col_frame, textvariable=self.profile_x_col_var, width=3).pack(side=tk.LEFT, padx=2)

        ttk.Label(col_frame, text="H열:", font=('Segoe UI', 17)).pack(side=tk.LEFT, padx=(10, 0))
        self.profile_h_col_var = tk.StringVar(value="1")
        ttk.Entry(col_frame, textvariable=self.profile_h_col_var, width=3).pack(side=tk.LEFT, padx=2)

        ttk.Label(col_frame, text="Skip:", font=('Segoe UI', 17)).pack(side=tk.LEFT, padx=(10, 0))
        self.profile_skip_var = tk.StringVar(value="0")
        ttk.Entry(col_frame, textvariable=self.profile_skip_var, width=3).pack(side=tk.LEFT, padx=2)

        # Unit conversion
        unit_frame = ttk.LabelFrame(load_frame, text="단위 변환", padding=3)
        unit_frame.pack(fill=tk.X, pady=2)

        unit_row1 = ttk.Frame(unit_frame)
        unit_row1.pack(fill=tk.X)
        ttk.Label(unit_row1, text="X 단위:", font=('Segoe UI', 17)).pack(side=tk.LEFT)
        self.profile_x_unit_var = tk.StringVar(value="um")
        unit_combo_x = ttk.Combobox(unit_row1, textvariable=self.profile_x_unit_var,
                                     values=['m', 'mm', 'um', 'nm'], width=6, state='readonly', font=self.FONTS['body'])
        unit_combo_x.pack(side=tk.LEFT, padx=2)

        ttk.Label(unit_row1, text="H 단위:", font=('Segoe UI', 17)).pack(side=tk.LEFT, padx=(10, 0))
        self.profile_h_unit_var = tk.StringVar(value="um")
        unit_combo_h = ttk.Combobox(unit_row1, textvariable=self.profile_h_unit_var,
                                     values=['m', 'mm', 'um', 'nm'], width=6, state='readonly', font=self.FONTS['body'])
        unit_combo_h.pack(side=tk.LEFT, padx=2)

        # Data info
        self.profile_data_info_var = tk.StringVar(value="데이터: -")
        ttk.Label(load_frame, textvariable=self.profile_data_info_var,
                  font=('Segoe UI', 17)).pack(fill=tk.X, pady=2)

        # 3. PSD Calculation Settings
        calc_frame = ttk.LabelFrame(left_scrollable, text="2. PSD 계산 설정", padding=5)
        calc_frame.pack(fill=tk.X, pady=3)

        # Detrend method
        detrend_row = ttk.Frame(calc_frame)
        detrend_row.pack(fill=tk.X, pady=2)
        ttk.Label(detrend_row, text="Detrend:", font=('Segoe UI', 17)).pack(side=tk.LEFT)
        self.profile_detrend_var = tk.StringVar(value="mean")
        ttk.Combobox(detrend_row, textvariable=self.profile_detrend_var,
                     values=['mean', 'linear', 'quadratic'], width=10, state='readonly', font=self.FONTS['body']).pack(side=tk.LEFT, padx=5)

        # Window function
        window_row = ttk.Frame(calc_frame)
        window_row.pack(fill=tk.X, pady=2)
        ttk.Label(window_row, text="Window:", font=('Segoe UI', 17)).pack(side=tk.LEFT)
        self.profile_window_var = tk.StringVar(value="hann")
        ttk.Combobox(window_row, textvariable=self.profile_window_var,
                     values=['hann', 'hamming', 'blackman', 'none'], width=10, state='readonly', font=self.FONTS['body']).pack(side=tk.LEFT, padx=5)

        # PSD type selection
        psd_type_frame = ttk.Frame(calc_frame)
        psd_type_frame.pack(fill=tk.X, pady=2)

        self.calc_full_psd_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(psd_type_frame, text="Full PSD",
                        variable=self.calc_full_psd_var).pack(side=tk.LEFT)

        self.calc_top_psd_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(psd_type_frame, text="Top PSD",
                        variable=self.calc_top_psd_var).pack(side=tk.LEFT, padx=10)

        # Logarithmic binning options
        bin_frame = ttk.Frame(calc_frame)
        bin_frame.pack(fill=tk.X, pady=2)

        self.apply_binning_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(bin_frame, text="로그 구간 평균화",
                        variable=self.apply_binning_var).pack(side=tk.LEFT)

        ttk.Label(bin_frame, text="점/decade:", font=('Segoe UI', 17)).pack(side=tk.LEFT, padx=(10, 0))
        self.points_per_decade_var = tk.StringVar(value="20")
        ttk.Entry(bin_frame, textvariable=self.points_per_decade_var, width=4).pack(side=tk.LEFT, padx=2)

        # Calculate button
        ttk.Button(calc_frame, text="PSD 계산",
                   command=self._calculate_profile_psd).pack(fill=tk.X, pady=5)

        # 4. Fitting Settings
        fit_frame = ttk.LabelFrame(left_scrollable, text="3. Self-Affine 피팅", padding=5)
        fit_frame.pack(fill=tk.X, pady=3)

        # Fitting range
        range_row = ttk.Frame(fit_frame)
        range_row.pack(fill=tk.X, pady=2)

        ttk.Label(range_row, text="q_min:", font=('Segoe UI', 17)).pack(side=tk.LEFT)
        self.fit_q_min_var = tk.StringVar(value="auto")
        ttk.Entry(range_row, textvariable=self.fit_q_min_var, width=8).pack(side=tk.LEFT, padx=2)

        ttk.Label(range_row, text="q_max:", font=('Segoe UI', 17)).pack(side=tk.LEFT, padx=(10, 0))
        self.fit_q_max_var = tk.StringVar(value="auto")
        ttk.Entry(range_row, textvariable=self.fit_q_max_var, width=8).pack(side=tk.LEFT, padx=2)

        # Auto range option
        self.fit_auto_range_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(fit_frame, text="자동 직선 구간 탐지",
                        variable=self.fit_auto_range_var).pack(anchor=tk.W)

        # PSD to fit
        fit_target_row = ttk.Frame(fit_frame)
        fit_target_row.pack(fill=tk.X, pady=2)
        ttk.Label(fit_target_row, text="피팅 대상:", font=('Segoe UI', 17)).pack(side=tk.LEFT)
        self.fit_target_var = tk.StringVar(value="full")
        ttk.Radiobutton(fit_target_row, text="Full", variable=self.fit_target_var,
                        value="full").pack(side=tk.LEFT)
        ttk.Radiobutton(fit_target_row, text="Top", variable=self.fit_target_var,
                        value="top").pack(side=tk.LEFT)

        # Fit button
        ttk.Button(fit_frame, text="Self-Affine 피팅 실행",
                   command=self._fit_profile_psd).pack(fill=tk.X, pady=5)

        # 4. q1 Extrapolation Settings
        extrap_frame = ttk.LabelFrame(left_scrollable, text="4. q1 외삽 (Extrapolation)", padding=5)
        extrap_frame.pack(fill=tk.X, pady=3)

        ttk.Label(extrap_frame, text="※ 피팅 결과를 사용하여 q1까지 PSD 외삽",
                  font=('Segoe UI', 17), foreground='#64748B').pack(anchor=tk.W)

        # Enable extrapolation
        self.enable_q1_extrap_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(extrap_frame, text="q1 외삽 활성화",
                        variable=self.enable_q1_extrap_var).pack(anchor=tk.W)

        # Target q1 input
        q1_row = ttk.Frame(extrap_frame)
        q1_row.pack(fill=tk.X, pady=2)
        ttk.Label(q1_row, text="Target q1:", font=('Segoe UI', 17)).pack(side=tk.LEFT)
        self.target_q1_extrap_var = tk.StringVar(value="1e6")
        ttk.Entry(q1_row, textvariable=self.target_q1_extrap_var, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(q1_row, text="1/m", font=('Segoe UI', 17)).pack(side=tk.LEFT)

        # Extrapolation points per decade
        extrap_pts_row = ttk.Frame(extrap_frame)
        extrap_pts_row.pack(fill=tk.X, pady=2)
        ttk.Label(extrap_pts_row, text="외삽 pts/decade:", font=('Segoe UI', 17)).pack(side=tk.LEFT)
        self.extrap_pts_per_decade_var = tk.StringVar(value="20")
        ttk.Entry(extrap_pts_row, textvariable=self.extrap_pts_per_decade_var, width=5).pack(side=tk.LEFT, padx=5)

        # Apply extrapolation button
        ttk.Button(extrap_frame, text="외삽 적용 및 그래프 업데이트",
                   command=self._apply_q1_extrapolation).pack(fill=tk.X, pady=5)

        # 5. Parameter-based PSD Generation
        param_psd_frame = ttk.LabelFrame(left_scrollable, text="5. 파라미터 PSD 생성", padding=5)
        param_psd_frame.pack(fill=tk.X, pady=3)

        ttk.Label(param_psd_frame, text="※ H, q0, C(q0), q1으로 Self-Affine PSD 생성",
                  font=('Segoe UI', 17), foreground='#64748B').pack(anchor=tk.W)

        # Enable parameter PSD overlay
        self.enable_param_psd_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(param_psd_frame, text="파라미터 PSD 표시",
                        variable=self.enable_param_psd_var).pack(anchor=tk.W)

        # H input
        h_row = ttk.Frame(param_psd_frame)
        h_row.pack(fill=tk.X, pady=2)
        ttk.Label(h_row, text="H (Hurst):", font=('Segoe UI', 17), width=10).pack(side=tk.LEFT)
        self.param_H_var = tk.StringVar(value="0.8")
        ttk.Entry(h_row, textvariable=self.param_H_var, width=10).pack(side=tk.LEFT, padx=5)

        # q0 input
        q0_row = ttk.Frame(param_psd_frame)
        q0_row.pack(fill=tk.X, pady=2)
        ttk.Label(q0_row, text="q0:", font=('Segoe UI', 17), width=10).pack(side=tk.LEFT)
        self.param_q0_var = tk.StringVar(value="1e4")
        ttk.Entry(q0_row, textvariable=self.param_q0_var, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(q0_row, text="1/m", font=('Segoe UI', 17)).pack(side=tk.LEFT)

        # q1 input
        q1_row = ttk.Frame(param_psd_frame)
        q1_row.pack(fill=tk.X, pady=2)
        ttk.Label(q1_row, text="q1:", font=('Segoe UI', 17), width=10).pack(side=tk.LEFT)
        self.param_q1_var = tk.StringVar(value="1e9")
        ttk.Entry(q1_row, textvariable=self.param_q1_var, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(q1_row, text="1/m", font=('Segoe UI', 17)).pack(side=tk.LEFT)

        # h0 (h_rms) input - C(q0) will be auto-calculated
        h0_row = ttk.Frame(param_psd_frame)
        h0_row.pack(fill=tk.X, pady=2)
        ttk.Label(h0_row, text="h0 (h_rms):", font=('Segoe UI', 17), width=10).pack(side=tk.LEFT)
        self.param_h0_var = tk.StringVar(value="1e-6")
        ttk.Entry(h0_row, textvariable=self.param_h0_var, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(h0_row, text="m", font=('Segoe UI', 17)).pack(side=tk.LEFT)

        # Points per decade
        ppd_row = ttk.Frame(param_psd_frame)
        ppd_row.pack(fill=tk.X, pady=2)
        ttk.Label(ppd_row, text="pts/decade:", font=('Segoe UI', 17), width=10).pack(side=tk.LEFT)
        self.param_ppd_var = tk.StringVar(value="20")
        ttk.Entry(ppd_row, textvariable=self.param_ppd_var, width=10).pack(side=tk.LEFT, padx=5)

        # Buttons
        param_btn_row = ttk.Frame(param_psd_frame)
        param_btn_row.pack(fill=tk.X, pady=5)
        ttk.Button(param_btn_row, text="PSD 생성 및 플롯",
                   command=self._generate_param_psd).pack(side=tk.LEFT, padx=2)
        ttk.Button(param_btn_row, text="피팅값 가져오기",
                   command=self._copy_fit_to_param).pack(side=tk.LEFT, padx=2)

        # 6. Results Display
        result_frame = ttk.LabelFrame(left_scrollable, text="6. 결과", padding=5)
        result_frame.pack(fill=tk.X, pady=3)

        self.psd_profile_result_text = tk.Text(result_frame, height=12, width=45, font=('Consolas', 17))
        self.psd_profile_result_text.pack(fill=tk.BOTH, expand=True)

        # 7. Export Options
        export_frame = ttk.LabelFrame(left_scrollable, text="7. 내보내기", padding=5)
        export_frame.pack(fill=tk.X, pady=3)

        export_btn_row = ttk.Frame(export_frame)
        export_btn_row.pack(fill=tk.X)

        ttk.Button(export_btn_row, text="PSD CSV 저장",
                   command=self._export_profile_psd_csv).pack(side=tk.LEFT, padx=2)
        ttk.Button(export_btn_row, text="그래프 저장",
                   command=self._save_profile_psd_plot).pack(side=tk.LEFT, padx=2)

        # (PSD 확정 버튼은 '데이터 로드' 섹션으로 이동됨)

        # ============== Right Panel: Plots ==============
        right_frame = ttk.Frame(main_container)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Create figure with subplots
        self.fig_psd_profile = Figure(figsize=(12, 9), dpi=100)

        # 2x2 subplot layout
        # Top-left: Raw profile
        self.ax_profile_raw = self.fig_psd_profile.add_subplot(221)
        self.ax_profile_raw.set_title('표면 프로파일', fontweight='bold', fontsize=15)
        self.ax_profile_raw.set_xlabel('Position (m)', fontsize=13)
        self.ax_profile_raw.set_ylabel('Height (m)', fontsize=13)
        self.ax_profile_raw.grid(True, alpha=0.3)

        # Top-right: Profile histogram
        self.ax_profile_hist = self.fig_psd_profile.add_subplot(222)
        self.ax_profile_hist.set_title('높이 분포', fontweight='bold', fontsize=15)
        self.ax_profile_hist.set_xlabel('Height (m)', fontsize=13)
        self.ax_profile_hist.set_ylabel('Count', fontsize=13)
        self.ax_profile_hist.grid(True, alpha=0.3)

        # Bottom-left: h_rms (거칠기) & Parseval 검증
        self.ax_hrms_parseval = self.fig_psd_profile.add_subplot(223)
        self.ax_hrms_parseval.set_title('h_rms 거칠기 & Parseval 검증', fontweight='bold', fontsize=15)
        self.ax_hrms_parseval.set_xlabel('Wavenumber q (1/m)', fontsize=13)
        self.ax_hrms_parseval.set_ylabel('누적 h_rms (m)', fontsize=13)
        self.ax_hrms_parseval.set_xscale('log')
        self.ax_hrms_parseval.set_yscale('log')
        self.ax_hrms_parseval.grid(True, alpha=0.3, which='both')

        # Bottom-right: 2D isotropic PSD (main result)
        self.ax_psd_2d = self.fig_psd_profile.add_subplot(224)
        self.ax_psd_2d.set_title('2D Isotropic PSD C(q)', fontweight='bold', fontsize=15)
        self.ax_psd_2d.set_xlabel('Wavenumber q (1/m)', fontsize=13)
        self.ax_psd_2d.set_ylabel(r'C(q) (m$^4$)', fontsize=13)
        self.ax_psd_2d.set_xscale('log')
        self.ax_psd_2d.set_yscale('log')
        self.ax_psd_2d.grid(True, alpha=0.3, which='both')

        self.fig_psd_profile.subplots_adjust(left=0.12, right=0.95, top=0.94, bottom=0.10, hspace=0.42, wspace=0.35)

        # Canvas
        self.canvas_psd_profile = FigureCanvasTkAgg(self.fig_psd_profile, master=right_frame)
        self.canvas_psd_profile.draw()
        self.canvas_psd_profile.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Toolbar
        toolbar_frame = ttk.Frame(right_frame)
        toolbar_frame.pack(fill=tk.X)
        NavigationToolbar2Tk(self.canvas_psd_profile, toolbar_frame)

    def _load_profile_data(self):
        """Load surface profile data from file."""
        filepath = filedialog.askopenfilename(
            title="표면 프로파일 데이터 선택",
            filetypes=[
                ("Text files", "*.txt"),
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )

        if not filepath:
            return

        try:
            # Get settings
            x_col = int(self.profile_x_col_var.get())
            h_col = int(self.profile_h_col_var.get())
            skip_header = int(self.profile_skip_var.get())
            x_unit = self.profile_x_unit_var.get()
            h_unit = self.profile_h_unit_var.get()

            # Create analyzer
            self.profile_psd_analyzer = ProfilePSDAnalyzer()
            self.profile_psd_analyzer.load_data(
                filepath,
                x_col=x_col,
                h_col=h_col,
                skip_header=skip_header,
                x_unit=x_unit,
                h_unit=h_unit
            )

            # Store data
            self.profile_psd_data = {
                'x': self.profile_psd_analyzer.x,
                'h': self.profile_psd_analyzer.h,
                'filepath': filepath
            }

            # Update UI
            filename = os.path.basename(filepath)
            self.psd_profile_file_var.set(f"파일: {filename}")

            n = len(self.profile_psd_analyzer.x)
            dx = np.abs(self.profile_psd_analyzer.x[1] - self.profile_psd_analyzer.x[0])
            L = n * dx
            # Calculate h_rms from detrended profile: h_rms = sqrt(mean(h_detrended^2))
            h_detrended = self.profile_psd_analyzer.h - np.mean(self.profile_psd_analyzer.h)
            h_rms = np.sqrt(np.mean(h_detrended**2))

            self.profile_data_info_var.set(
                f"Data: {n} pts, L={L*1e3:.3f} mm, dx={dx*1e6:.3f} um, h_rms={h_rms*1e6:.4f} um"
            )

            # Plot raw profile
            self._plot_profile_raw()

            self.status_var.set(f"프로파일 데이터 로드 완료: {filename}")

        except Exception as e:
            messagebox.showerror("오류", f"데이터 로드 실패: {e}")

    def _plot_profile_raw(self):
        """Plot raw profile data."""
        if self.profile_psd_analyzer is None:
            return

        x = self.profile_psd_analyzer.x
        h = self.profile_psd_analyzer.h

        # Plot 1: Raw profile
        self.ax_profile_raw.clear()
        self.ax_profile_raw.plot(x * 1e3, h * 1e6, 'b-', linewidth=0.5)
        self.ax_profile_raw.set_title('표면 프로파일', fontweight='bold', fontsize=15)
        self.ax_profile_raw.set_xlabel('Position (mm)', fontsize=13)
        self.ax_profile_raw.set_ylabel('Height (μm)', fontsize=13)
        self.ax_profile_raw.grid(True, alpha=0.3)

        # Plot 2: Height histogram
        self.ax_profile_hist.clear()
        h_detrended = h - np.mean(h)
        self.ax_profile_hist.hist(h_detrended * 1e6, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
        self.ax_profile_hist.axvline(x=0, color='r', linestyle='--', linewidth=1, label='Mean')
        self.ax_profile_hist.set_title('높이 분포 (Detrended)', fontweight='bold', fontsize=15)
        self.ax_profile_hist.set_xlabel('Height (μm)', fontsize=13)
        self.ax_profile_hist.set_ylabel('Count', fontsize=13)
        self.ax_profile_hist.grid(True, alpha=0.3)

        # Mark top region
        n_top = np.sum(h_detrended > 0)
        phi = n_top / len(h_detrended)
        self.ax_profile_hist.axvspan(0, np.max(h_detrended) * 1e6, alpha=0.2, color='green',
                                      label=f'Top (φ={phi:.2f})')
        self.ax_profile_hist.legend(fontsize=12)

        self.fig_psd_profile.subplots_adjust(left=0.12, right=0.95, top=0.94, bottom=0.10, hspace=0.42, wspace=0.35)
        self.canvas_psd_profile.draw()

    def _calculate_profile_psd(self):
        """Calculate PSD from profile data."""
        if self.profile_psd_analyzer is None:
            self._show_status("먼저 프로파일 데이터를 로드하세요.", 'warning')
            return

        try:
            window = self.profile_window_var.get()
            detrend = self.profile_detrend_var.get()
            calc_top = self.calc_top_psd_var.get()
            apply_binning = self.apply_binning_var.get()

            try:
                points_per_decade = int(self.points_per_decade_var.get())
            except ValueError:
                points_per_decade = 20

            # Calculate PSD with logarithmic binning
            self.profile_psd_analyzer.calculate_psd(
                window=window,
                detrend_method=detrend,
                calculate_top=calc_top,
                apply_binning=apply_binning,
                points_per_decade=points_per_decade
            )

            # Plot results
            self._plot_profile_psd()

            # Update results text
            self._update_psd_profile_results()

            self.status_var.set("PSD 계산 완료")

        except Exception as e:
            messagebox.showerror("오류", f"PSD 계산 실패: {e}")
            import traceback
            traceback.print_exc()

    def _plot_profile_psd(self):
        """Plot calculated PSD."""
        if self.profile_psd_analyzer is None or self.profile_psd_analyzer.q is None:
            return

        q = self.profile_psd_analyzer.q
        C_full_1d = self.profile_psd_analyzer.C_full_1d
        C_full_2d = self.profile_psd_analyzer.C_full_2d
        C_top_1d = self.profile_psd_analyzer.C_top_1d
        C_top_2d = self.profile_psd_analyzer.C_top_2d

        # Plot h_rms 거칠기 & Parseval 검증
        self.ax_hrms_parseval.clear()

        # 누적 h_rms 계산: h_rms²(q) = 2π ∫[q0→q] k⋅C(k) dk
        if C_full_2d is not None and len(q) > 1:
            # 누적 적분 계산
            hrms_sq_cumulative = np.zeros(len(q))
            integrand = q * C_full_2d  # k × C(k)
            for i in range(1, len(q)):
                hrms_sq_cumulative[i] = 2 * np.pi * np.trapezoid(integrand[:i+1], q[:i+1])
            hrms_cumulative = np.sqrt(np.maximum(hrms_sq_cumulative, 0))

            # 유효한 값만 플롯
            valid = hrms_cumulative > 0
            if np.any(valid):
                self.ax_hrms_parseval.loglog(q[valid], hrms_cumulative[valid]*1e6, 'b-',
                                             linewidth=2, label='Full PSD', alpha=0.8)

            # 프로파일에서 직접 계산한 h_rms 표시 (수평선)
            if hasattr(self.profile_psd_analyzer, 'surface_params') and self.profile_psd_analyzer.surface_params is not None:
                h_rms_direct = self.profile_psd_analyzer.surface_params.get('h_rms', None)
                if h_rms_direct is not None and h_rms_direct > 0:
                    self.ax_hrms_parseval.axhline(y=h_rms_direct*1e6, color='b', linestyle='--',
                                                  alpha=0.5, label=f'프로파일 h_rms={h_rms_direct*1e6:.2f}μm')

        if C_top_2d is not None and len(q) > 1:
            hrms_sq_cumulative_top = np.zeros(len(q))
            integrand_top = q * C_top_2d
            for i in range(1, len(q)):
                hrms_sq_cumulative_top[i] = 2 * np.pi * np.trapezoid(integrand_top[:i+1], q[:i+1])
            hrms_cumulative_top = np.sqrt(np.maximum(hrms_sq_cumulative_top, 0))

            valid_top = hrms_cumulative_top > 0
            if np.any(valid_top):
                self.ax_hrms_parseval.loglog(q[valid_top], hrms_cumulative_top[valid_top]*1e6, 'r-',
                                             linewidth=2, label='Top PSD', alpha=0.8)

        self.ax_hrms_parseval.set_title('h_rms 거칠기 & Parseval 검증', fontweight='bold', fontsize=15)
        self.ax_hrms_parseval.set_xlabel('Wavenumber q (1/m)', fontsize=13)
        self.ax_hrms_parseval.set_ylabel('누적 h_rms (μm)', fontsize=13)
        self.ax_hrms_parseval.grid(True, alpha=0.3, which='both')
        self.ax_hrms_parseval.legend(fontsize=12)

        # Plot 2D isotropic PSD
        self.ax_psd_2d.clear()
        if C_full_2d is not None:
            self.ax_psd_2d.loglog(q, C_full_2d, 'b-', linewidth=2, label='Full PSD', alpha=0.8)
        if C_top_2d is not None:
            self.ax_psd_2d.loglog(q, C_top_2d, 'r-', linewidth=2, label='Top PSD', alpha=0.8)

        # Add fitted lines if available
        if self.profile_psd_analyzer.fit_result_full is not None:
            fit = self.profile_psd_analyzer.fit_result_full
            H = fit['H']
            slope = fit['slope']
            self.ax_psd_2d.loglog(fit['q_fit'], fit['C_fit'], 'b--', linewidth=1.5,
                                   label=f'Fit (Full): H={H:.3f}, slope={slope:.2f}')

        if self.profile_psd_analyzer.fit_result_top is not None:
            fit = self.profile_psd_analyzer.fit_result_top
            H = fit['H']
            slope = fit['slope']
            self.ax_psd_2d.loglog(fit['q_fit'], fit['C_fit'], 'r--', linewidth=1.5,
                                   label=f'Fit (Top): H={H:.3f}, slope={slope:.2f}')

        # Plot extrapolated data if available
        if hasattr(self.profile_psd_analyzer, 'q_extrap') and self.profile_psd_analyzer.q_extrap is not None:
            q_extrap = self.profile_psd_analyzer.q_extrap
            C_extrap = self.profile_psd_analyzer.C_extrap
            extrap_info = self.profile_psd_analyzer.extrap_info

            # Plot extrapolated region with different style
            self.ax_psd_2d.loglog(q_extrap, C_extrap, 'g--', linewidth=2, alpha=0.8,
                                   label=f'Extrapolated (H={extrap_info["H"]:.3f})')

            # Add vertical line at data boundary
            q_max_data = extrap_info['q_max_data']
            self.ax_psd_2d.axvline(x=q_max_data, color='gray', linestyle=':', alpha=0.5,
                                    label=f'Data limit: {q_max_data:.1e}')

            # Add vertical line at target q1
            target_q1 = extrap_info['target_q1']
            self.ax_psd_2d.axvline(x=target_q1, color='green', linestyle=':', alpha=0.5,
                                    label=f'Target q1: {target_q1:.1e}')

        # Plot parameter-based PSD if enabled
        if (hasattr(self, 'enable_param_psd_var') and self.enable_param_psd_var.get() and
            hasattr(self, 'param_psd_data') and self.param_psd_data is not None):
            pdata = self.param_psd_data
            self.ax_psd_2d.loglog(pdata['q'], pdata['C'], 'm-', linewidth=2, alpha=0.7,
                                   label=f'Param PSD (H={pdata["H"]:.3f})')

        self.ax_psd_2d.set_title('2D Isotropic PSD C(q)', fontweight='bold', fontsize=15)
        self.ax_psd_2d.set_xlabel('Wavenumber q (1/m)', fontsize=13)
        self.ax_psd_2d.set_ylabel('C(q) (m^4)', fontsize=13)
        self.ax_psd_2d.grid(True, alpha=0.3, which='both')
        self.ax_psd_2d.legend(fontsize=12, loc='lower left')

        self.fig_psd_profile.subplots_adjust(left=0.12, right=0.95, top=0.94, bottom=0.10, hspace=0.42, wspace=0.35)
        self.canvas_psd_profile.draw()

        # Auto-register graph data
        if C_full_2d is not None:
            self._register_graph_data(
                "PSD_Full_2D", q, C_full_2d,
                "q(1/m)\tC(q)(m^4)", "Full PSD (2D isotropic)")
        if C_top_2d is not None:
            self._register_graph_data(
                "PSD_Top_2D", q, C_top_2d,
                "q(1/m)\tC(q)(m^4)", "Top PSD (2D isotropic)")
        if (hasattr(self, 'param_psd_data') and self.param_psd_data is not None):
            pdata = self.param_psd_data
            self._register_graph_data(
                "PSD_Param", pdata['q'], pdata['C'],
                "q(1/m)\tC(q)(m^4)", f"Param PSD (H={pdata['H']:.3f})")

    def _fit_profile_psd(self):
        """Fit PSD to self-affine fractal model."""
        if self.profile_psd_analyzer is None or self.profile_psd_analyzer.q is None:
            self._show_status("먼저 PSD를 계산하세요.", 'warning')
            return

        try:
            # Get settings
            auto_range = self.fit_auto_range_var.get()
            use_top = self.fit_target_var.get() == "top"

            q_min = None
            q_max = None
            if not auto_range:
                q_min_str = self.fit_q_min_var.get()
                q_max_str = self.fit_q_max_var.get()
                if q_min_str.lower() != "auto":
                    q_min = float(q_min_str)
                if q_max_str.lower() != "auto":
                    q_max = float(q_max_str)

            # Perform fitting
            result = self.profile_psd_analyzer.fit_model(
                q_min=q_min,
                q_max=q_max,
                fit_mode='slope_only',
                auto_range=auto_range,
                use_top_psd=use_top
            )

            # Also calculate surface parameters
            self.profile_psd_analyzer.get_surface_parameters(use_top_psd=use_top)

            # Update plot and results
            self._plot_profile_psd()
            self._update_psd_profile_results()

            self.status_var.set(f"피팅 완료: H = {result['H']:.4f}")

        except Exception as e:
            messagebox.showerror("오류", f"피팅 실패: {e}")
            import traceback
            traceback.print_exc()

    def _apply_q1_extrapolation(self):
        """Apply q1 extrapolation to extend PSD data using self-affine model."""
        if self.profile_psd_analyzer is None or self.profile_psd_analyzer.q is None:
            self._show_status("먼저 PSD를 계산하세요.", 'warning')
            return

        # Check if fitting has been done
        fit_result = self.profile_psd_analyzer.fit_result_full
        if self.fit_target_var.get() == "top":
            fit_result = self.profile_psd_analyzer.fit_result_top

        if fit_result is None:
            self._show_status("먼저 Self-Affine 피팅을 실행하세요.", 'warning')
            return

        try:
            # Get target q1
            target_q1 = float(self.target_q1_extrap_var.get())
            pts_per_decade = int(self.extrap_pts_per_decade_var.get())

            # Get current PSD data
            q_current = self.profile_psd_analyzer.q
            q_max_data = q_current[-1]

            if target_q1 <= q_max_data:
                self._show_status(f"Target q1 ({target_q1:.2e}) <= 현재 q_max ({q_max_data:.2e})\n"
                    f"외삽이 필요하지 않습니다.", 'success')
                return

            # Get fitting parameters
            H = fit_result['H']
            slope = fit_result['slope']  # slope = -2(1+H)

            # Create extrapolated q array
            log_q_max = np.log10(q_max_data)
            log_q1 = np.log10(target_q1)
            n_decades = log_q1 - log_q_max
            n_extrap_points = int(n_decades * pts_per_decade)

            if n_extrap_points < 1:
                n_extrap_points = 1

            q_extrap = np.logspace(log_q_max, log_q1, n_extrap_points + 1)[1:]  # Exclude first (overlap)

            # Extrapolate using power law: C(q) = C(q_max) * (q/q_max)^slope
            use_top = self.fit_target_var.get() == "top"
            if use_top:
                C_at_qmax = self.profile_psd_analyzer.C_top_2d[-1]
            else:
                C_at_qmax = self.profile_psd_analyzer.C_full_2d[-1]

            C_extrap = C_at_qmax * (q_extrap / q_max_data) ** slope

            # Store extrapolated data
            self.profile_psd_analyzer.q_extrap = q_extrap
            self.profile_psd_analyzer.C_extrap = C_extrap
            self.profile_psd_analyzer.extrap_info = {
                'target_q1': target_q1,
                'H': H,
                'slope': slope,
                'q_max_data': q_max_data,
                'n_points': n_extrap_points
            }

            # Concatenate for combined array (optional use)
            self.profile_psd_analyzer.q_combined = np.concatenate([q_current, q_extrap])
            if use_top:
                self.profile_psd_analyzer.C_combined = np.concatenate([
                    self.profile_psd_analyzer.C_top_2d, C_extrap])
            else:
                self.profile_psd_analyzer.C_combined = np.concatenate([
                    self.profile_psd_analyzer.C_full_2d, C_extrap])

            # Recalculate surface parameters with extrapolated data
            q_full = self.profile_psd_analyzer.q_combined
            C_full = self.profile_psd_analyzer.C_combined

            # h_rms from extrapolated PSD
            valid = (q_full > 0) & (C_full > 0) & np.isfinite(C_full)
            q_v = q_full[valid]
            C_v = C_full[valid]
            h_rms_sq = 2 * np.pi * np.trapezoid(q_v * C_v, q_v)
            h_rms_extrap = np.sqrt(max(h_rms_sq, 0))

            # h'_rms (slope) from extrapolated PSD
            slope_sq = 2 * np.pi * np.trapezoid(q_v**3 * C_v, q_v)
            h_rms_slope_extrap = np.sqrt(max(slope_sq, 0))

            self.profile_psd_analyzer.extrap_params = {
                'h_rms': h_rms_extrap,
                'h_rms_slope': h_rms_slope_extrap
            }

            # Update plot
            self._plot_profile_psd()
            self._update_psd_profile_results()

            self._show_status(f"q1 외삽 완료\n\n"
                f"데이터 q_max: {q_max_data:.2e} 1/m\n"
                f"외삽 q1: {target_q1:.2e} 1/m\n"
                f"외삽 점 수: {n_extrap_points}\n"
                f"사용된 H: {H:.4f}\n\n"
                f"외삽 포함 h_rms: {h_rms_extrap*1e6:.4f} um\n"
                f"외삽 포함 h'_rms (xi): {h_rms_slope_extrap:.6f}", 'success')

            self.status_var.set(f"외삽 완료: q1={target_q1:.2e}, h'_rms={h_rms_slope_extrap:.4f}")

        except Exception as e:
            messagebox.showerror("오류", f"외삽 실패: {e}")
            import traceback
            traceback.print_exc()

    def _generate_param_psd(self):
        """Generate PSD from parameters (H, q0, q1, h0).

        C(q0) is calculated from h0 (h_rms) using:
        C(q0) = h0^2 * H / (pi * q0^2 * (1 - (q0/q1)^(2H)))
        """
        try:
            # Get parameters
            H = float(self.param_H_var.get())
            q0 = float(self.param_q0_var.get())
            q1 = float(self.param_q1_var.get())
            h0 = float(self.param_h0_var.get())  # h_rms in meters
            pts_per_decade = int(self.param_ppd_var.get())

            # Validate
            if H <= 0 or H >= 1:
                self._show_status("H는 0 < H < 1 범위여야 합니다.", 'warning')
                return
            if q0 >= q1:
                self._show_status("q0 < q1 이어야 합니다.", 'warning')
                return
            if h0 <= 0:
                self._show_status("h0 > 0 이어야 합니다.", 'warning')
                return

            # Calculate C(q0) from h0
            # h_rms^2 = 2*pi * C0 * q0^2 * (1 - (q0/q1)^(2H)) / H  (for power-law region)
            # Therefore: C0 = h_rms^2 * H / (pi * q0^2 * (1 - (q0/q1)^(2H)))
            q_ratio = q0 / q1
            denom = np.pi * q0**2 * (1 - q_ratio**(2*H))
            if denom <= 0:
                self._show_status("파라미터 조합이 잘못되었습니다.", 'warning')
                return
            C_q0 = h0**2 * H / denom

            # Generate q array
            log_q0 = np.log10(q0)
            log_q1 = np.log10(q1)
            n_decades = log_q1 - log_q0
            n_points = max(10, int(n_decades * pts_per_decade))

            q_param = np.logspace(log_q0, log_q1, n_points)

            # Generate self-affine PSD: C(q) = C(q0) * (q/q0)^(-2(1+H))
            slope = -2 * (1 + H)
            C_param = C_q0 * (q_param / q0) ** slope

            # Store parameter PSD data
            self.param_psd_data = {
                'q': q_param,
                'C': C_param,
                'H': H,
                'q0': q0,
                'C_q0': C_q0,
                'q1': q1,
                'h0': h0,
                'slope': slope
            }

            # Calculate h'_rms from parameter PSD
            # h'_rms^2 = 2*pi * integral(q^3*C(q)*dq)
            slope_sq = 2 * np.pi * np.trapezoid(q_param**3 * C_param, q_param)
            h_rms_slope = np.sqrt(max(slope_sq, 0))

            # h_rms should match h0 (input)
            self.param_psd_data['h_rms'] = h0
            self.param_psd_data['h_rms_slope'] = h_rms_slope

            # Enable display and update plot
            self.enable_param_psd_var.set(True)
            self._plot_profile_psd()
            self._update_psd_profile_results()

            self._show_status(f"파라미터 PSD 생성 완료\n\n"
                f"H = {H:.4f}\n"
                f"q0 = {q0:.2e} 1/m\n"
                f"q1 = {q1:.2e} 1/m\n"
                f"h0 (입력) = {h0*1e6:.4f} um\n\n"
                f"계산된 C(q0) = {C_q0:.2e} m^4\n"
                f"h'_rms (xi) = {h_rms_slope:.6f}", 'success')

            self.status_var.set(f"파라미터 PSD 생성: H={H:.3f}, h'_rms={h_rms_slope:.4f}")

        except ValueError as e:
            messagebox.showerror("오류", f"파라미터 입력 오류: {e}")
        except Exception as e:
            messagebox.showerror("오류", f"PSD 생성 실패: {e}")
            import traceback
            traceback.print_exc()

    def _copy_fit_to_param(self):
        """Copy fitting results to parameter PSD inputs."""
        if self.profile_psd_analyzer is None:
            self._show_status("먼저 PSD를 계산하고 피팅을 실행하세요.", 'warning')
            return

        # Get fit result
        fit_result = self.profile_psd_analyzer.fit_result_full
        if self.fit_target_var.get() == "top":
            fit_result = self.profile_psd_analyzer.fit_result_top

        if fit_result is None:
            self._show_status("피팅 결과가 없습니다. 먼저 피팅을 실행하세요.", 'warning')
            return

        try:
            # Copy values
            H = fit_result['H']
            q0 = fit_result.get('q0', self.profile_psd_analyzer.q[0])

            # Get q1 from data or extrapolation setting
            if hasattr(self.profile_psd_analyzer, 'extrap_info') and self.profile_psd_analyzer.extrap_info is not None:
                q1 = self.profile_psd_analyzer.extrap_info['target_q1']
            else:
                q1 = self.profile_psd_analyzer.q[-1]

            # Get h_rms from surface parameters or extrapolation
            if hasattr(self.profile_psd_analyzer, 'extrap_params') and self.profile_psd_analyzer.extrap_params is not None:
                h0 = self.profile_psd_analyzer.extrap_params['h_rms']
            elif self.profile_psd_analyzer.surface_params is not None:
                h0 = self.profile_psd_analyzer.surface_params['h_rms']
            else:
                # Calculate from raw profile
                h_detrended = self.profile_psd_analyzer.h - np.mean(self.profile_psd_analyzer.h)
                h0 = np.sqrt(np.mean(h_detrended**2))

            # Update UI
            self.param_H_var.set(f"{H:.4f}")
            self.param_q0_var.set(f"{q0:.2e}")
            self.param_q1_var.set(f"{q1:.2e}")
            self.param_h0_var.set(f"{h0:.2e}")

            self.status_var.set(f"피팅값 복사 완료: H={H:.4f}, q0={q0:.2e}")

        except Exception as e:
            messagebox.showerror("오류", f"피팅값 복사 실패: {e}")

    def _update_psd_profile_results(self):
        """Update results text display."""
        self.psd_profile_result_text.delete(1.0, tk.END)

        if self.profile_psd_analyzer is None:
            return

        lines = []
        lines.append("=" * 45)
        lines.append("PSD 계산 결과")
        lines.append("=" * 45)

        # Data info
        if self.profile_psd_data is not None:
            x = self.profile_psd_analyzer.x
            h = self.profile_psd_analyzer.h
            n = len(x)
            dx = np.abs(x[1] - x[0])
            L = n * dx

            lines.append(f"\n[Data Information]")
            lines.append(f"  Data points: {n}")
            lines.append(f"  Total length: {L*1e3:.4f} mm")
            lines.append(f"  Sampling interval: {dx*1e6:.4f} um")
            # Calculate h_rms from detrended profile
            h_detrended = h - np.mean(h)
            h_rms_raw = np.sqrt(np.mean(h_detrended**2))
            lines.append(f"  h_rms (raw profile): {h_rms_raw*1e6:.4f} um")

            # Binning info
            if self.profile_psd_analyzer.q is not None:
                n_raw = len(self.profile_psd_analyzer.q_raw) if self.profile_psd_analyzer.q_raw is not None else 0
                n_binned = len(self.profile_psd_analyzer.q)
                ppd = self.profile_psd_analyzer.points_per_decade
                if n_raw != n_binned:
                    lines.append(f"\n[로그 구간 평균화]")
                    lines.append(f"  Raw 점: {n_raw} → Binned 점: {n_binned}")
                    lines.append(f"  점/decade: {ppd}")

        # Top PSD info
        if self.profile_psd_analyzer.phi is not None:
            phi = self.profile_psd_analyzer.phi
            lines.append(f"\n[Top PSD 정보]")
            lines.append(f"  φ (상부 비율): {phi:.4f} ({phi*100:.1f}%)")
            lines.append(f"  1/φ (보정 계수): {1/phi:.4f}")

        # Fit results
        if self.profile_psd_analyzer.fit_result_full is not None:
            fit = self.profile_psd_analyzer.fit_result_full
            lines.append(f"\n[Full PSD 피팅 결과]")
            lines.append(f"  Hurst Exponent H: {fit['H']:.4f}")
            lines.append(f"  기울기 (log-log): {fit['slope']:.4f}")
            lines.append(f"  이론 기울기: -2(1+H) = {-2*(1+fit['H']):.4f}")
            lines.append(f"  R²: {fit['r_squared']:.6f}")
            if 'q0' in fit:
                lines.append(f"  q0 (추정): {fit['q0']:.2e} 1/m")
            if 'C0' in fit:
                lines.append(f"  C(q0) (추정): {fit['C0']:.2e} m⁴")

        if self.profile_psd_analyzer.fit_result_top is not None:
            fit = self.profile_psd_analyzer.fit_result_top
            lines.append(f"\n[Top PSD 피팅 결과]")
            lines.append(f"  Hurst Exponent H: {fit['H']:.4f}")
            lines.append(f"  기울기 (log-log): {fit['slope']:.4f}")
            lines.append(f"  R²: {fit['r_squared']:.6f}")

        # Surface parameters
        if self.profile_psd_analyzer.surface_params is not None:
            params = self.profile_psd_analyzer.surface_params
            lines.append(f"\n[Surface Parameters (PSD Integration)]")
            lines.append(f"  h_rms: {params['h_rms']*1e6:.4f} um")
            lines.append(f"  h'_rms (xi): {params['h_rms_slope']:.6f}")
            lines.append(f"  h''_rms: {params['h_rms_curvature']:.2e} 1/m")

        # Extrapolation results
        if hasattr(self.profile_psd_analyzer, 'extrap_info') and self.profile_psd_analyzer.extrap_info is not None:
            info = self.profile_psd_analyzer.extrap_info
            params_ext = self.profile_psd_analyzer.extrap_params
            lines.append(f"\n[q1 Extrapolation]")
            lines.append(f"  Data q_max: {info['q_max_data']:.2e} 1/m")
            lines.append(f"  Target q1: {info['target_q1']:.2e} 1/m")
            lines.append(f"  Extrap points: {info['n_points']}")
            lines.append(f"  Used H: {info['H']:.4f}")
            lines.append(f"\n[With Extrapolation]")
            lines.append(f"  h_rms (extrap): {params_ext['h_rms']*1e6:.4f} um")
            lines.append(f"  h'_rms (xi, extrap): {params_ext['h_rms_slope']:.6f}")

        # Parameter PSD results
        if hasattr(self, 'param_psd_data') and self.param_psd_data is not None:
            pdata = self.param_psd_data
            lines.append(f"\n[Parameter PSD]")
            lines.append(f"  H: {pdata['H']:.4f}")
            lines.append(f"  q0: {pdata['q0']:.2e} 1/m")
            lines.append(f"  q1: {pdata['q1']:.2e} 1/m")
            lines.append(f"  h0 (input): {pdata.get('h0', pdata['h_rms'])*1e6:.4f} um")
            lines.append(f"  C(q0) (calc): {pdata['C_q0']:.2e} m^4")
            lines.append(f"  slope: {pdata['slope']:.4f}")
            lines.append(f"  h'_rms (xi): {pdata['h_rms_slope']:.6f}")

        self.psd_profile_result_text.insert(tk.END, "\n".join(lines))

    def _export_profile_psd_csv(self):
        """Export PSD data to CSV."""
        if self.profile_psd_analyzer is None or self.profile_psd_analyzer.q is None:
            self._show_status("먼저 PSD를 계산하세요.", 'warning')
            return

        filepath = filedialog.asksaveasfilename(
            title="PSD 데이터 저장",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if not filepath:
            return

        try:
            q = self.profile_psd_analyzer.q
            C_full_2d = self.profile_psd_analyzer.C_full_2d
            C_top_2d = self.profile_psd_analyzer.C_top_2d

            with open(filepath, 'w') as f:
                # Header
                header = ["q (1/m)", "C_full (m^4)"]
                if C_top_2d is not None:
                    header.append("C_top (m^4)")
                f.write(",".join(header) + "\n")

                # Data
                for i in range(len(q)):
                    row = [f"{q[i]:.6e}", f"{C_full_2d[i]:.6e}"]
                    if C_top_2d is not None:
                        row.append(f"{C_top_2d[i]:.6e}")
                    f.write(",".join(row) + "\n")

            self._show_status(f"PSD 데이터가 저장되었습니다:\n{filepath}", 'success')

        except Exception as e:
            messagebox.showerror("오류", f"저장 실패: {e}")

    def _save_profile_psd_plot(self):
        """Save PSD plot to file."""
        self._save_plot(self.fig_psd_profile, "profile_psd_plot")

    def _apply_profile_psd_to_tab3(self):
        """Apply PSD data directly to Tab 3 (Calculation Settings)."""
        psd_type = self.apply_psd_type_var.get()

        try:
            # Handle different PSD types
            if psd_type == "direct":
                # Use directly loaded PSD
                if not hasattr(self, 'psd_direct_data') or self.psd_direct_data is None:
                    self._show_status("먼저 PSD 파일을 직접 로드하세요.", 'warning')
                    return
                q = self.psd_direct_data['q'].copy()
                C = self.psd_direct_data['C_q'].copy()
                # Calculate H, xi, h_rms from data
                from scipy.integrate import simpson
                h_rms_sq = 2 * np.pi * simpson(q * C, x=q)
                h_rms = np.sqrt(max(h_rms_sq, 0))
                # Estimate H from slope (log-log)
                log_q = np.log10(q)
                log_C = np.log10(C)
                # Linear fit to get slope
                coeffs = np.polyfit(log_q, log_C, 1)
                slope = coeffs[0]  # C(q) ~ q^slope => slope = -2(1+H) => H = -slope/2 - 1
                H = max(0.1, min(1.0, -slope/2 - 1))
                # h_rms_slope (xi)
                h_rms_slope_sq = np.pi * simpson(q**3 * C, x=q)
                xi = np.sqrt(max(h_rms_slope_sq, 0))
                psd_type_str = f"직접 로드: {self.psd_direct_data['filename']}"
            elif psd_type == "param":
                # Use parameter-generated PSD
                if not hasattr(self, 'param_psd_data') or self.param_psd_data is None:
                    self._show_status("먼저 파라미터 PSD를 생성하세요.", 'warning')
                    return
                q = self.param_psd_data['q'].copy()
                C = self.param_psd_data['C'].copy()
                H = self.param_psd_data['H']
                xi = self.param_psd_data['h_rms_slope']
                h_rms = self.param_psd_data['h_rms']
                psd_type_str = "Param PSD"
            else:
                # Use profile-based PSD (Full or Top)
                if self.profile_psd_analyzer is None or self.profile_psd_analyzer.q is None:
                    self._show_status("먼저 PSD를 계산하세요.", 'warning')
                    return

                use_top = (psd_type == "top")

                # Check if extrapolated data should be used
                use_extrap = (self.enable_q1_extrap_var.get() and
                             hasattr(self.profile_psd_analyzer, 'q_combined') and
                             self.profile_psd_analyzer.q_combined is not None)

                if use_extrap:
                    q = self.profile_psd_analyzer.q_combined.copy()
                    C = self.profile_psd_analyzer.C_combined.copy()
                    xi = self.profile_psd_analyzer.extrap_params['h_rms_slope']
                    h_rms = self.profile_psd_analyzer.extrap_params['h_rms']
                else:
                    q, C = self.profile_psd_analyzer.get_psd_for_persson(use_top_psd=use_top)
                    params = self.profile_psd_analyzer.get_surface_parameters(use_top_psd=use_top)
                    xi = params['h_rms_slope']
                    h_rms = params['h_rms']

                fit_result = self.profile_psd_analyzer.fit_result_top if use_top else self.profile_psd_analyzer.fit_result_full
                H = fit_result['H'] if fit_result else 0.8
                psd_type_str = "Top PSD" if use_top else "Full PSD"

            # Store finalized PSD data
            self.finalized_psd = {
                'q': q,
                'C': C,
                'H': H,
                'xi': xi,
                'h_rms': h_rms,
                'type': psd_type_str
            }

            # Create PSD model for calculations
            self.psd_model = MeasuredPSD(q, C)
            self.raw_psd_data = {'q': q, 'C_q': C}  # Key must be 'C_q' for Tab 3 compatibility

            # Register finalized PSD
            self._register_graph_data(
                "PSD_Finalized", q, C,
                "q(1/m)\tC(q)(m^4)", f"Finalized PSD ({psd_type_str})")

            # Update Tab 3 PSD display
            if hasattr(self, 'psd_status_var'):
                self.psd_status_var.set(f"✓ PSD 설정됨: {psd_type_str}")
            if hasattr(self, 'psd_status_label'):
                self.psd_status_label.config(foreground='#059669')
            if hasattr(self, 'psd_q_range_var'):
                self.psd_q_range_var.set(f"{q[0]:.2e} ~ {q[-1]:.2e} (1/m)")
            if hasattr(self, 'psd_H_var'):
                self.psd_H_var.set(f"{H:.4f}")
            if hasattr(self, 'psd_xi_var'):
                self.psd_xi_var.set(f"{xi:.6f}")

            # Update q_min for calculation (q_max는 기본값 1.0e+6 유지)
            if hasattr(self, 'q_min_var'):
                self.q_min_var.set(f"{q[0]:.2e}")

            # Mark Tab 0 as finalized
            self.tab0_finalized = True
            self.psd_source = f"Tab 0 PSD 생성 ({psd_type_str})"

            # Check if Tab 2 should be disabled
            self._update_tab2_state()

            # Verify PSD normalization: ∫ q×C(q) dq (using 2π factor) should ≈ h_rms²
            # For 2D isotropic: h_rms² = 2π ∫ q C(q) dq
            from scipy.integrate import simpson
            integrand_qC = q * C  # q × C(q) for h_rms² check
            h_rms_sq_from_psd = 2 * np.pi * simpson(integrand_qC, x=q)
            h_rms_from_psd = np.sqrt(max(h_rms_sq_from_psd, 0))

            # Also check C(q0) value for typical range
            C_q0 = C[0]
            q0 = q[0]

            # Get scan length info if available
            scan_length_str = ""
            if hasattr(self.profile_psd_analyzer, 'scan_length'):
                L = self.profile_psd_analyzer.scan_length
                n_pts = self.profile_psd_analyzer.n_points
                dx = self.profile_psd_analyzer.sample_spacing
                scan_length_str = (
                    f"\n[프로파일 정보]\n"
                    f"• 스캔 길이 L: {L*1e3:.4f} mm ({L:.6f} m)\n"
                    f"• 데이터 포인트: {n_pts}\n"
                    f"• 샘플 간격 dx: {dx*1e6:.4f} μm\n"
                )

            # Show verification info
            verification_msg = (
                f"[PSD 검증]\n"
                f"• 원본 h_rms: {h_rms*1e6:.4f} μm\n"
                f"• PSD 적분 h_rms: {h_rms_from_psd*1e6:.4f} μm\n"
                f"• C(q₀={q0:.2e}): {C_q0:.2e} m⁴\n"
            )
            if abs(h_rms - h_rms_from_psd) / max(h_rms, 1e-12) > 0.5:
                verification_msg += "⚠ 경고: PSD 적분값과 h_rms 불일치! 정규화 확인 필요\n"
            verification_msg += scan_length_str

            self._show_status(f"{psd_type_str} → 계산 설정 전송 완료\n\n"
                f"q range: {q[0]:.2e} ~ {q[-1]:.2e} 1/m\n"
                f"H = {H:.4f}\n"
                f"h_rms = {h_rms*1e6:.4f} um\n"
                f"h'_rms (xi) = {xi:.6f}\n\n"
                f"{verification_msg}", 'success')

            self.status_var.set(f"PSD 확정: {psd_type_str}, ξ = {xi:.6f}")

        except Exception as e:
            messagebox.showerror("오류", f"적용 실패: {e}")
            import traceback
            traceback.print_exc()

    def _update_tab2_state(self):
        """Enable/disable Tab 2 based on Tab 0 and Tab 1 finalization status."""
        tab0_ready = getattr(self, 'tab0_finalized', False)
        tab1_ready = getattr(self, 'tab1_finalized', False)

        # Tab 2 is always enabled - no need to disable it
        # Users can use Tab 2 (계산 설정) at any time
        self.notebook.tab(2, state='normal')

    def _create_master_curve_tab(self, parent):
        """Create Master Curve generation tab using Time-Temperature Superposition."""
        # Main container
        main_container = ttk.Frame(parent)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel container (fixed width) with scrollable canvas
        left_container = ttk.Frame(main_container, width=getattr(self, '_left_panel_width', 600))
        left_container.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        left_container.pack_propagate(False)

        # Logo at bottom (pack before canvas so it stays at bottom)
        self._add_logo_to_panel(left_container)

        # Toolbar (fixed at top, always accessible)
        self._create_panel_toolbar(left_container, buttons=[
            ("마스터 커브 확정 \u2192 계산에 사용", self._use_persson_master_curve_for_calc, 'Success.TButton'),
        ])

        # Create canvas and scrollbar for scrolling
        mc_canvas = tk.Canvas(left_container, highlightthickness=0, bg='#F0F2F5')
        mc_scrollbar = ttk.Scrollbar(left_container, orient="vertical", command=mc_canvas.yview)

        # Create inner frame for content
        left_frame = ttk.Frame(mc_canvas)

        # Configure scroll region when frame size changes
        def _configure_scroll(event):
            mc_canvas.configure(scrollregion=mc_canvas.bbox("all"))
        left_frame.bind("<Configure>", _configure_scroll)

        # Create window inside canvas
        canvas_window = mc_canvas.create_window((0, 0), window=left_frame, anchor="nw", width=getattr(self, '_left_panel_width', 600) - 20)
        mc_canvas.configure(yscrollcommand=mc_scrollbar.set)

        # Pack scrollbar and canvas
        mc_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        mc_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Enable mousewheel scrolling (cross-platform)
        def _on_mousewheel(event):
            # Windows
            if event.delta:
                mc_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            # Linux scroll up
            elif event.num == 4:
                mc_canvas.yview_scroll(-1, "units")
            # Linux scroll down
            elif event.num == 5:
                mc_canvas.yview_scroll(1, "units")

        # Bind mousewheel events
        def _bind_mousewheel(event):
            mc_canvas.bind_all("<MouseWheel>", _on_mousewheel)  # Windows
            mc_canvas.bind_all("<Button-4>", _on_mousewheel)    # Linux scroll up
            mc_canvas.bind_all("<Button-5>", _on_mousewheel)    # Linux scroll down

        def _unbind_mousewheel(event):
            mc_canvas.unbind_all("<MouseWheel>")
            mc_canvas.unbind_all("<Button-4>")
            mc_canvas.unbind_all("<Button-5>")

        mc_canvas.bind("<Enter>", _bind_mousewheel)
        mc_canvas.bind("<Leave>", _unbind_mousewheel)

        # ============== Left Panel: Controls ==============

        # 1. Description
        desc_frame = ttk.LabelFrame(left_frame, text="탭 설명", padding=5)
        desc_frame.pack(fill=tk.X, pady=2, padx=3)

        desc_text = (
            "다중 온도 DMA 데이터로부터 마스터 커브를 생성합니다.\n"
            "시간-온도 중첩 원리(TTS) 적용:\n"
            "  - 수평 이동 aT: 주파수 시프트\n"
            "  - 수직 이동 bT: 모듈러스 시프트 (밀도/엔트로피 보정)"
        )
        ttk.Label(desc_frame, text=desc_text, font=('Segoe UI', 17), justify=tk.LEFT).pack(anchor=tk.W)

        # 2. File Loading
        load_frame = ttk.LabelFrame(left_frame, text="데이터 로드", padding=5)
        load_frame.pack(fill=tk.X, pady=2, padx=3)

        ttk.Button(
            load_frame,
            text="다중 온도 DMA 데이터 로드 (CSV/Excel)",
            command=self._load_multi_temp_dma
        ).pack(fill=tk.X, pady=2)

        # ===== 내장 마스터 커브/aT 선택 섹션 =====
        ttk.Separator(load_frame, orient='horizontal').pack(fill=tk.X, pady=5)
        ttk.Label(load_frame, text="─ 내장 데이터 선택 ─",
                  font=('Segoe UI', 17, 'bold'), foreground='#059669').pack(anchor=tk.CENTER)

        # 내장 마스터 커브 선택
        preset_mc_frame = ttk.Frame(load_frame)
        preset_mc_frame.pack(fill=tk.X, pady=2)
        ttk.Label(preset_mc_frame, text="마스터커브:", font=('Segoe UI', 17)).pack(side=tk.LEFT)
        self.preset_mc_var = tk.StringVar(value="(선택...)")
        self.preset_mc_combo = ttk.Combobox(preset_mc_frame, textvariable=self.preset_mc_var,
                                             state='readonly', width=15, font=self.FONTS['body'])
        self.preset_mc_combo.pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_mc_frame, text="로드", command=self._load_preset_mastercurve, width=4,
                   style='Outline.TButton').pack(side=tk.LEFT)
        ttk.Button(preset_mc_frame, text="삭제", command=self._delete_preset_mastercurve, width=4).pack(side=tk.LEFT, padx=2)

        # 내장 aT 선택
        preset_aT_frame = ttk.Frame(load_frame)
        preset_aT_frame.pack(fill=tk.X, pady=2)
        ttk.Label(preset_aT_frame, text="aT 팩터:", font=('Segoe UI', 17)).pack(side=tk.LEFT)
        self.preset_aT_var = tk.StringVar(value="(선택...)")
        self.preset_aT_combo = ttk.Combobox(preset_aT_frame, textvariable=self.preset_aT_var,
                                             state='readonly', width=15, font=self.FONTS['body'])
        self.preset_aT_combo.pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_aT_frame, text="로드", command=self._load_preset_aT, width=4,
                   style='Outline.TButton').pack(side=tk.LEFT)
        ttk.Button(preset_aT_frame, text="삭제", command=self._delete_preset_aT, width=4).pack(side=tk.LEFT, padx=2)

        # 프로그램 시작 시 내장 데이터 목록 로드
        self._refresh_preset_mastercurve_list()
        self._refresh_preset_aT_list()

        # Persson 정품 마스터 커브 직접 로드 버튼
        ttk.Separator(load_frame, orient='horizontal').pack(fill=tk.X, pady=5)
        ttk.Label(load_frame, text="─ 또는: 완성된 마스터 커브 직접 로드 ─",
                  font=('Segoe UI', 17), foreground='#2563EB').pack(anchor=tk.CENTER)

        # 마스터 커브 직접 로드 버튼 + 리스트에 추가 버튼
        mc_direct_btn_frame = ttk.Frame(load_frame)
        mc_direct_btn_frame.pack(fill=tk.X, pady=2)

        ttk.Button(
            mc_direct_btn_frame,
            text="마스터 커브 로드 (f, E', E'')",
            command=self._load_persson_master_curve,
            style='Accent.TButton'
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Button(mc_direct_btn_frame, text="→ 리스트 추가",
                   command=self._add_preset_mastercurve, width=10).pack(side=tk.LEFT, padx=(3, 0))

        # aT 시프트 팩터 로드 버튼 + 리스트에 추가 버튼
        aT_direct_btn_frame = ttk.Frame(load_frame)
        aT_direct_btn_frame.pack(fill=tk.X, pady=2)

        ttk.Button(
            aT_direct_btn_frame,
            text="aT 시프트 팩터 로드 (T, aT)",
            command=self._load_persson_aT,
            style='Accent.TButton'
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Button(aT_direct_btn_frame, text="→ 리스트 추가",
                   command=self._add_preset_aT, width=10).pack(side=tk.LEFT, padx=(3, 0))

        self.mc_data_info_var = tk.StringVar(value="데이터 미로드")
        ttk.Label(load_frame, textvariable=self.mc_data_info_var,
                  font=('Segoe UI', 17), foreground='#64748B').pack(anchor=tk.W)

        # aT 정보 표시
        self.mc_aT_info_var = tk.StringVar(value="aT: 미로드")
        ttk.Label(load_frame, textvariable=self.mc_aT_info_var,
                  font=('Segoe UI', 17), foreground='#64748B').pack(anchor=tk.W)

        # 마스터 커브 비교 버튼
        self.mc_compare_var = tk.BooleanVar(value=False)
        compare_frame = ttk.Frame(load_frame)
        compare_frame.pack(fill=tk.X, pady=2)
        ttk.Checkbutton(
            compare_frame,
            text="정품 vs 생성 비교 모드",
            variable=self.mc_compare_var
        ).pack(side=tk.LEFT)
        ttk.Button(
            compare_frame,
            text="비교 플롯",
            command=self._plot_master_curve_comparison,
            width=10
        ).pack(side=tk.RIGHT)

        # 3. Settings
        settings_frame = ttk.LabelFrame(left_frame, text="설정", padding=5)
        settings_frame.pack(fill=tk.X, pady=2, padx=3)

        # Reference temperature
        row1 = ttk.Frame(settings_frame)
        row1.pack(fill=tk.X, pady=2)
        ttk.Label(row1, text="기준 온도 Tref (°C):", font=('Segoe UI', 17)).pack(side=tk.LEFT)
        self.mc_tref_var = tk.StringVar(value="20.0")
        ttk.Entry(row1, textvariable=self.mc_tref_var, width=10).pack(side=tk.RIGHT)

        # bT mode
        row2 = ttk.Frame(settings_frame)
        row2.pack(fill=tk.X, pady=2)
        ttk.Label(row2, text="bT 계산 방법:", font=('Segoe UI', 17)).pack(side=tk.LEFT)
        self.mc_bt_mode_var = tk.StringVar(value="optimize")
        bt_combo = ttk.Combobox(
            row2, textvariable=self.mc_bt_mode_var,
            values=["optimize", "theoretical"],
            width=12, state="readonly", font=self.FONTS['body']
        )
        bt_combo.pack(side=tk.RIGHT)

        # Apply bT checkbox
        self.mc_use_bt_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            settings_frame,
            text="수직 이동 bT 적용 (Apply Vertical Shift)",
            variable=self.mc_use_bt_var
        ).pack(anchor=tk.W, pady=2)

        ttk.Label(settings_frame,
                  text="optimize: 수치 최적화 / theoretical: T/Tref 공식",
                  font=('Segoe UI', 16), foreground='#64748B').pack(anchor=tk.W)

        # Optimization target selection
        row3 = ttk.Frame(settings_frame)
        row3.pack(fill=tk.X, pady=2)
        ttk.Label(row3, text="최적화 대상:", font=('Segoe UI', 17)).pack(side=tk.LEFT)
        self.mc_target_var = tk.StringVar(value="E_storage")
        target_combo = ttk.Combobox(
            row3, textvariable=self.mc_target_var,
            values=["E_storage", "E_loss", "tan_delta"],
            width=12, state="readonly", font=self.FONTS['body']
        )
        target_combo.pack(side=tk.RIGHT)
        ttk.Label(settings_frame,
                  text="E_storage: E' / E_loss: E'' / tan_delta: tanδ",
                  font=('Segoe UI', 16), foreground='#64748B').pack(anchor=tk.W)

        # Smoothing control for master curve
        smooth_frame = ttk.LabelFrame(settings_frame, text="마스터 커브 스무딩", padding=3)
        smooth_frame.pack(fill=tk.X, pady=3)

        self.mc_smooth_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            smooth_frame,
            text="스무딩 적용",
            variable=self.mc_smooth_var
        ).pack(anchor=tk.W)

        # Smoothing window slider
        slider_frame = ttk.Frame(smooth_frame)
        slider_frame.pack(fill=tk.X, pady=2)
        ttk.Label(slider_frame, text="스무딩 강도:", font=('Segoe UI', 17)).pack(side=tk.LEFT)
        self.mc_smooth_window_var = tk.IntVar(value=23)
        self.mc_smooth_slider = ttk.Scale(
            slider_frame,
            from_=5, to=51,
            orient=tk.HORIZONTAL,
            variable=self.mc_smooth_window_var,
            command=lambda v: self.mc_smooth_label.config(text=f"{int(float(v))}")
        )
        self.mc_smooth_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.mc_smooth_label = ttk.Label(slider_frame, text="23", width=3)
        self.mc_smooth_label.pack(side=tk.RIGHT)

        # bT comparison checkbox
        self.mc_compare_bt_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            smooth_frame,
            text="bT 적용/미적용 비교 보기",
            variable=self.mc_compare_bt_var,
            command=self._toggle_bt_comparison
        ).pack(anchor=tk.W, pady=2)

        # Persson 마스터 커브 스무딩 적용 버튼
        ttk.Separator(smooth_frame, orient='horizontal').pack(fill=tk.X, pady=3)
        persson_smooth_row = ttk.Frame(smooth_frame)
        persson_smooth_row.pack(fill=tk.X, pady=2)
        ttk.Button(
            persson_smooth_row,
            text="스무딩 처리하기",
            command=self._apply_smoothing_to_persson,
            width=14,
            style='Outline.TButton'
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(
            persson_smooth_row,
            text="원본 복원",
            command=self._reset_persson_to_original,
            width=10
        ).pack(side=tk.RIGHT, padx=2)

        # 4. Calculate button
        calc_frame = ttk.Frame(settings_frame)
        calc_frame.pack(fill=tk.X, pady=5)

        self.mc_calc_btn = ttk.Button(
            calc_frame,
            text="마스터 커브 생성",
            command=self._generate_master_curve
        )
        self.mc_calc_btn.pack(fill=tk.X)

        # Progress bar
        self.mc_progress_var = tk.IntVar()
        self.mc_progress_bar = ttk.Progressbar(
            calc_frame, variable=self.mc_progress_var, maximum=100
        )
        self.mc_progress_bar.pack(fill=tk.X, pady=2)

        # 5. Results Summary
        results_frame = ttk.LabelFrame(left_frame, text="결과 요약", padding=5)
        results_frame.pack(fill=tk.X, pady=2, padx=3)

        self.mc_result_text = tk.Text(results_frame, height=10, font=("Courier", 15), wrap=tk.WORD)
        self.mc_result_text.pack(fill=tk.X)

        # 6. Shift Factor Table
        table_frame = ttk.LabelFrame(left_frame, text="Shift Factor 테이블", padding=5)
        table_frame.pack(fill=tk.BOTH, expand=True, pady=2, padx=3)

        # Create treeview for shift factors
        columns = ('T', 'aT', 'bT', 'log_aT')
        self.mc_shift_table = ttk.Treeview(table_frame, columns=columns, show='headings', height=8)
        self.mc_shift_table.heading('T', text='T (°C)')
        self.mc_shift_table.heading('aT', text='aT')
        self.mc_shift_table.heading('bT', text='bT')
        self.mc_shift_table.heading('log_aT', text='log(aT)')

        self.mc_shift_table.column('T', width=60, anchor='center')
        self.mc_shift_table.column('aT', width=80, anchor='center')
        self.mc_shift_table.column('bT', width=60, anchor='center')
        self.mc_shift_table.column('log_aT', width=70, anchor='center')

        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.mc_shift_table.yview)
        self.mc_shift_table.configure(yscrollcommand=scrollbar.set)

        self.mc_shift_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 7. Export buttons - IMPORTANT: This must be visible via scrolling
        export_frame = ttk.LabelFrame(left_frame, text="★ 내보내기 ★", padding=5)
        export_frame.pack(fill=tk.X, pady=5, padx=3)

        ttk.Button(
            export_frame, text="마스터 커브 CSV 내보내기",
            command=self._export_master_curve
        ).pack(fill=tk.X, pady=2)

        # Make this button more prominent
        finalize_btn = ttk.Button(
            export_frame, text="▶ 마스터 커브 확정 → Tab 3 (계산설정)",
            command=self._finalize_master_curve_to_tab3
        )
        finalize_btn.pack(fill=tk.X, pady=2)

        # (Persson 확정 버튼은 '데이터 로드' 섹션으로 이동됨)

        # Force update of scroll region after all widgets are added
        left_frame.update_idletasks()
        mc_canvas.configure(scrollregion=mc_canvas.bbox("all"))

        # ============== Right Panel: Plots ==============

        right_panel = ttk.Frame(main_container)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        plot_frame = ttk.LabelFrame(right_panel, text="시각화", padding=5)
        plot_frame.pack(fill=tk.BOTH, expand=True)

        # Create figure with 2x2 subplots
        self.fig_mc = Figure(figsize=(11, 8), dpi=100)

        # Top-left: Raw data (multi-temperature)
        self.ax_mc_raw = self.fig_mc.add_subplot(221)
        self.ax_mc_raw.set_title('원본 데이터 (온도별)', fontweight='bold', fontsize=15)
        self.ax_mc_raw.set_xlabel('주파수 f (Hz)', fontsize=13)
        self.ax_mc_raw.set_ylabel('E\', E\'\' (MPa)', fontsize=13)
        self.ax_mc_raw.set_xscale('log')
        self.ax_mc_raw.set_yscale('log')
        self.ax_mc_raw.grid(True, alpha=0.3)

        # Top-right: Master curve
        self.ax_mc_master = self.fig_mc.add_subplot(222)
        self.ax_mc_master.set_title('마스터 커브 (Tref)', fontweight='bold', fontsize=15)
        self.ax_mc_master.set_xlabel('Reduced Frequency (Hz)', fontsize=13)
        self.ax_mc_master.set_ylabel('E\', E\'\' (MPa)', fontsize=13)
        self.ax_mc_master.set_xscale('log')
        self.ax_mc_master.set_yscale('log')
        self.ax_mc_master.grid(True, alpha=0.3)

        # Bottom-left: aT vs Temperature
        self.ax_mc_aT = self.fig_mc.add_subplot(223)
        self.ax_mc_aT.set_title('수평 이동 계수 aT', fontweight='bold', fontsize=15)
        self.ax_mc_aT.set_xlabel('온도 T (°C)', fontsize=13)
        self.ax_mc_aT.set_ylabel('log10(aT)', fontsize=13)
        self.ax_mc_aT.grid(True, alpha=0.3)

        # Bottom-right: bT vs Temperature
        self.ax_mc_bT = self.fig_mc.add_subplot(224)
        self.ax_mc_bT.set_title('수직 이동 계수 bT', fontweight='bold', fontsize=15)
        self.ax_mc_bT.set_xlabel('온도 T (°C)', fontsize=13)
        self.ax_mc_bT.set_ylabel('bT', fontsize=13)
        self.ax_mc_bT.grid(True, alpha=0.3)

        self.fig_mc.tight_layout()

        self.canvas_mc = FigureCanvasTkAgg(self.fig_mc, plot_frame)
        self.canvas_mc.draw()
        self.canvas_mc.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas_mc, plot_frame)
        toolbar.update()

        # Initialize master curve generator
        self.master_curve_gen = None
        self.mc_raw_df = None

    def _load_multi_temp_dma(self):
        """Load multi-temperature DMA data for master curve generation."""
        filename = filedialog.askopenfilename(
            title="다중 온도 DMA 데이터 파일 선택",
            filetypes=[
                ("Excel files", "*.xlsx *.xls"),
                ("CSV files", "*.csv"),
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ]
        )

        if not filename:
            return

        try:
            import pandas as pd

            # Load data
            if filename.endswith('.xlsx') or filename.endswith('.xls'):
                self.mc_raw_df = pd.read_excel(filename, skiprows=1)
            else:
                # Try different delimiters
                try:
                    self.mc_raw_df = pd.read_csv(filename, skiprows=1, sep='\t')
                    if len(self.mc_raw_df.columns) < 4:
                        self.mc_raw_df = pd.read_csv(filename, skiprows=1, sep=',')
                except:
                    self.mc_raw_df = pd.read_csv(filename, skiprows=1, delim_whitespace=True)

            # Standardize column names based on number of columns
            n_cols = len(self.mc_raw_df.columns)

            if n_cols >= 7:
                # Full format with |E*|: f, T, f_reduced, Amplitude, E', E'', |E*|
                col_names = ['f', 'T', 'f_reduced', 'Amplitude', "E'", "E''", "E_star"]
                # Extend with generic names if more columns exist
                while len(col_names) < n_cols:
                    col_names.append(f"col_{len(col_names)}")
                self.mc_raw_df.columns = col_names[:n_cols]
            elif n_cols == 6:
                # Full format: f, T, f_reduced, Amplitude, E', E''
                self.mc_raw_df.columns = ['f', 'T', 'f_reduced', 'Amplitude', "E'", "E''"]
            elif n_cols == 5:
                # Missing E'': f, T, f_reduced, Amplitude, E'
                self.mc_raw_df.columns = ['f', 'T', 'f_reduced', 'Amplitude', "E'"]
                # E'' is missing - show warning
                self._show_status("E'' (손실 탄성률) 컬럼이 없습니다.\n"
                    "마스터 커브 생성은 E'만 사용합니다.\n\n"
                    "완전한 분석을 위해 6개 컬럼 데이터를 권장합니다:\n"
                    "f(Hz), T(°C), f_reduced, Amplitude, E'(MPa), E''(MPa)", 'warning')
                # Estimate E'' as 10% of E' (rough estimate for demonstration)
                self.mc_raw_df["E''"] = self.mc_raw_df["E'"] * 0.1
            elif n_cols == 4:
                # Minimal: f, T, E', E''
                self.mc_raw_df.columns = ['f', 'T', "E'", "E''"]
            elif n_cols == 3:
                # Very minimal: f, T, E' (no E'')
                self.mc_raw_df.columns = ['f', 'T', "E'"]
                self.mc_raw_df["E''"] = self.mc_raw_df["E'"] * 0.1
            else:
                raise ValueError(f"데이터 컬럼 수가 부족합니다 ({n_cols}개). 최소 3개 컬럼이 필요합니다.")

            # Convert to numeric and drop NaN
            for col in self.mc_raw_df.columns:
                self.mc_raw_df[col] = pd.to_numeric(self.mc_raw_df[col], errors='coerce')
            self.mc_raw_df = self.mc_raw_df.dropna()

            # Get unique temperatures
            temps = self.mc_raw_df['T'].unique()
            n_temps = len(temps)
            n_points = len(self.mc_raw_df)

            self.mc_data_info_var.set(f"로드됨: {n_points}개 데이터, {n_temps}개 온도")

            # Plot raw data
            self._plot_mc_raw_data()

            self._show_status(f"데이터 로드 완료:\n"
                f"  - 총 {n_points}개 데이터 포인트\n"
                f"  - {n_temps}개 온도: {temps.min():.1f}°C ~ {temps.max():.1f}°C\n"
                f"  - 주파수 범위: {self.mc_raw_df['f'].min():.2f} ~ {self.mc_raw_df['f'].max():.2f} Hz", 'success')

        except Exception as e:
            import traceback
            messagebox.showerror("오류", f"데이터 로드 실패:\n{str(e)}\n\n{traceback.format_exc()}")

    def _plot_mc_raw_data(self):
        """Plot raw multi-temperature DMA data."""
        if self.mc_raw_df is None:
            return

        self.ax_mc_raw.clear()
        self.ax_mc_raw.set_title('원본 데이터 (온도별)', fontweight='bold', fontsize=15)
        self.ax_mc_raw.set_xlabel('주파수 f (Hz)', fontsize=13)
        self.ax_mc_raw.set_ylabel('E\', E\'\' (MPa)', fontsize=13)
        self.ax_mc_raw.set_xscale('log')
        self.ax_mc_raw.set_yscale('log')
        self.ax_mc_raw.grid(True, alpha=0.3)

        temps = np.sort(self.mc_raw_df['T'].unique())
        colors = plt.cm.coolwarm(np.linspace(0, 1, len(temps)))

        for i, T in enumerate(temps):
            mask = self.mc_raw_df['T'] == T
            data = self.mc_raw_df[mask]

            self.ax_mc_raw.plot(data['f'], data["E'"], 'o-', color=colors[i],
                               markersize=3, linewidth=1, alpha=0.7,
                               label=f"E' ({T:.0f}°C)")
            self.ax_mc_raw.plot(data['f'], data["E''"], 's--', color=colors[i],
                               markersize=2, linewidth=0.8, alpha=0.5)

        # Add legend with limited entries
        if len(temps) <= 8:
            self.ax_mc_raw.legend(fontsize=12, loc='upper left', ncol=2)

        self.fig_mc.tight_layout()
        self.canvas_mc.draw()

    def _load_persson_master_curve(self):
        """Load pre-generated master curve in Persson format (f, E', E'')."""
        filename = filedialog.askopenfilename(
            title="Persson 정품 마스터 커브 파일 선택",
            filetypes=[
                ("Text files", "*.txt"),
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )

        if not filename:
            return

        try:
            import pandas as pd

            # Try different delimiters
            for sep in ['\t', ',', ' ', ';']:
                try:
                    df = pd.read_csv(filename, sep=sep, skipinitialspace=True,
                                     comment='#', header=None)
                    if len(df.columns) >= 3:
                        break
                except:
                    continue

            if df is None or len(df.columns) < 3:
                raise ValueError("파일에서 3개 이상의 컬럼을 찾을 수 없습니다.")

            # Assume columns: f (Hz), E' (MPa), E'' (MPa)
            # Drop any header rows (non-numeric)
            df = df.apply(pd.to_numeric, errors='coerce').dropna()

            if len(df) < 2:
                raise ValueError("유효한 데이터 행이 부족합니다.")

            f = df.iloc[:, 0].values  # Frequency (Hz)
            E_storage = df.iloc[:, 1].values  # E' (MPa)
            E_loss = df.iloc[:, 2].values  # E'' (MPa)

            # Convert Hz to rad/s
            omega = 2 * np.pi * f

            # Convert MPa to Pa (if values seem to be in MPa range)
            if E_storage.max() < 1e6:  # Likely in MPa
                E_storage = E_storage * 1e6
                E_loss = E_loss * 1e6
                unit_str = "MPa → Pa 변환됨"
            else:
                unit_str = "Pa (변환 없음)"

            # Store the Persson master curve separately for comparison
            self.persson_master_curve = {
                'f': f.copy(),
                'omega': omega.copy(),
                'E_storage': E_storage.copy(),
                'E_loss': E_loss.copy(),
                'filename': os.path.basename(filename)
            }

            # Also create material from this data
            from persson_model.utils.data_loader import create_material_from_dma
            self.material_persson = create_material_from_dma(
                omega=omega,
                E_storage=E_storage,
                E_loss=E_loss,
                material_name=f"Persson ({os.path.basename(filename)})",
                reference_temp=20.0
            )

            # Update info display
            tan_delta_avg = np.mean(E_loss / E_storage)
            self.mc_data_info_var.set(
                f"★ Persson 정품: {len(f)} pts, "
                f"f={f.min():.1e}~{f.max():.1e} Hz, "
                f"tan δ 평균={tan_delta_avg:.3f}"
            )

            # Plot the loaded data
            self._plot_persson_master_curve()

            self._show_status(f"Persson 정품 마스터 커브 로드 완료\n\n"
                f"파일: {os.path.basename(filename)}\n"
                f"데이터 포인트: {len(f)}\n"
                f"주파수 범위: {f.min():.2e} ~ {f.max():.2e} Hz\n"
                f"E' 범위: {E_storage.min()/1e6:.2f} ~ {E_storage.max()/1e6:.2f} MPa\n"
                f"E'' 범위: {E_loss.min()/1e6:.2f} ~ {E_loss.max()/1e6:.2f} MPa\n"
                f"tan δ 평균: {tan_delta_avg:.3f}\n\n"
                f"단위: {unit_str}", 'success')

            self.status_var.set("Persson 정품 마스터 커브 로드 완료")

        except Exception as e:
            import traceback
            messagebox.showerror("오류", f"마스터 커브 로드 실패:\n{str(e)}\n\n{traceback.format_exc()}")

    def _load_persson_aT(self):
        """Load aT (and bT) shift factor data for temperature shifting.

        Supports formats:
        - 2 columns: T(°C), aT
        - 3 columns: T(°C), aT, bT (Persson format)
        """
        filename = filedialog.askopenfilename(
            title="aT 시프트 팩터 파일 선택",
            filetypes=[
                ("Text files", "*.txt"),
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )

        if not filename:
            return

        try:
            import pandas as pd

            # Try different delimiters (handle Fortran-style spacing)
            df = None
            for sep in [r'\s+', '\t', ',', ';']:
                try:
                    df = pd.read_csv(filename, sep=sep, skipinitialspace=True,
                                     comment='#', header=None, engine='python')
                    if len(df.columns) >= 2:
                        break
                except:
                    continue

            if df is None or len(df.columns) < 2:
                raise ValueError("파일에서 2개 이상의 컬럼을 찾을 수 없습니다.\n예상 형식: T(°C), aT [, bT]")

            # Drop any header rows (non-numeric)
            df = df.apply(pd.to_numeric, errors='coerce').dropna()

            if len(df) < 2:
                raise ValueError("유효한 데이터 행이 부족합니다.")

            T = df.iloc[:, 0].values  # Temperature (°C)
            aT = df.iloc[:, 1].values  # aT (always linear in Persson format)

            # Check for bT column
            has_bT = len(df.columns) >= 3
            if has_bT:
                bT = df.iloc[:, 2].values
            else:
                bT = np.ones_like(T)  # Default bT = 1

            # Persson format: aT is linear (not log), detect by checking values
            # aT values span many orders of magnitude (e.g., 1e-2 to 1e7)
            log_aT = np.log10(np.maximum(aT, 1e-20))  # Avoid log(0)
            format_str = "Persson 형식 (T, aT, bT)" if has_bT else "T, aT 형식"

            # Sort by temperature
            sort_idx = np.argsort(T)
            T = T[sort_idx]
            aT = aT[sort_idx]
            log_aT = log_aT[sort_idx]
            bT = bT[sort_idx]

            # Find reference temperature (where aT ≈ 1)
            ref_idx = np.argmin(np.abs(aT - 1.0))
            T_ref = T[ref_idx]

            # Store aT/bT data
            self.persson_aT_data = {
                'T': T.copy(),
                'aT': aT.copy(),
                'log_aT': log_aT.copy(),
                'bT': bT.copy(),
                'T_ref': T_ref,
                'has_bT': has_bT,
                'filename': os.path.basename(filename)
            }

            # Create interpolation functions
            from scipy.interpolate import interp1d
            self.persson_aT_interp = interp1d(
                T, log_aT,
                kind='linear',
                bounds_error=False,
                fill_value='extrapolate'
            )
            self.persson_bT_interp = interp1d(
                T, bT,
                kind='linear',
                bounds_error=False,
                fill_value='extrapolate'
            )

            # Update info display
            bT_info = f", bT={bT.min():.2f}~{bT.max():.2f}" if has_bT else ""
            self.mc_aT_info_var.set(
                f"aT: {len(T)} pts, T={T.min():.0f}~{T.max():.0f}°C, Tref={T_ref:.0f}°C{bT_info}"
            )

            # Plot aT (and bT) on the bottom-right plot
            self._plot_persson_aT()

            self._show_status(f"시프트 팩터 로드 완료\n\n"
                f"파일: {os.path.basename(filename)}\n"
                f"데이터 포인트: {len(T)}\n"
                f"온도 범위: {T.min():.1f} ~ {T.max():.1f} °C\n"
                f"기준 온도 (Tref): {T_ref:.1f} °C (aT=1)\n"
                f"aT 범위: {aT.min():.2e} ~ {aT.max():.2e}\n"
                f"bT 포함: {'예' if has_bT else '아니오'}\n"
                f"형식: {format_str}\n\n"
                f"이제 Tab 5에서 온도를 변경하여\n"
                f"다른 온도에서의 μ_visc를 계산할 수 있습니다.", 'success')

            self.status_var.set("시프트 팩터 (aT, bT) 로드 완료")

        except Exception as e:
            import traceback
            messagebox.showerror("오류", f"aT 로드 실패:\n{str(e)}\n\n{traceback.format_exc()}")

    def _plot_persson_aT(self):
        """Plot loaded aT (and bT) shift factor data."""
        if not hasattr(self, 'persson_aT_data') or self.persson_aT_data is None:
            return

        data = self.persson_aT_data

        # Use bottom-right plot (bT plot)
        self.ax_mc_bT.clear()

        # Remove any existing twin axis
        if hasattr(self, '_ax_bT_twin') and self._ax_bT_twin is not None:
            try:
                self._ax_bT_twin.remove()
            except:
                pass
            self._ax_bT_twin = None

        T = data['T']
        log_aT = data['log_aT']
        T_ref = data['T_ref']
        has_bT = data.get('has_bT', False)

        # Plot log₁₀(aT) on primary axis
        line1, = self.ax_mc_bT.plot(T, log_aT, 'bo-', linewidth=2, markersize=4, label=r'log$_{10}$(aT)')
        self.ax_mc_bT.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        self.ax_mc_bT.axvline(x=T_ref, color='red', linestyle='--', alpha=0.5, label=f'Tref={T_ref:.0f}°C')

        self.ax_mc_bT.set_xlabel('온도 T (°C)', fontsize=13)
        self.ax_mc_bT.set_ylabel(r'log$_{10}$(aT)', color='blue', fontsize=13)
        self.ax_mc_bT.tick_params(axis='y', labelcolor='blue')
        self.ax_mc_bT.grid(True, alpha=0.3)

        # If bT data exists, plot on secondary y-axis
        if has_bT and 'bT' in data:
            bT = data['bT']
            self._ax_bT_twin = self.ax_mc_bT.twinx()
            line2, = self._ax_bT_twin.plot(T, bT, 'r^-', linewidth=2, markersize=4, label='bT')
            self._ax_bT_twin.set_ylabel('bT (수직 시프트)', color='red', fontsize=13)
            self._ax_bT_twin.tick_params(axis='y', labelcolor='red')

            # Combined legend
            lines = [line1, line2]
            labels = [l.get_label() for l in lines]
            self.ax_mc_bT.legend(lines, labels, loc='upper right', fontsize=12)
            self.ax_mc_bT.set_title('시프트 팩터 aT & bT (Persson)', fontweight='bold', fontsize=15)
        else:
            self.ax_mc_bT.legend(loc='upper right', fontsize=12)
            self.ax_mc_bT.set_title('시프트 팩터 aT (Persson)', fontweight='bold', fontsize=15)

        self.fig_mc.tight_layout()
        self.canvas_mc.draw()

    def _load_psd_direct(self):
        """Load PSD data directly (q, C(q) format)."""
        filename = filedialog.askopenfilename(
            title="PSD 파일 선택 (q, C(q))",
            filetypes=[
                ("Text files", "*.txt"),
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )

        if not filename:
            return

        try:
            import pandas as pd

            # Try different delimiters
            df = None
            for sep in ['\t', ',', ' ', ';']:
                try:
                    df = pd.read_csv(filename, sep=sep, skipinitialspace=True,
                                     comment='#', header=None)
                    if len(df.columns) >= 2:
                        break
                except:
                    continue

            if df is None or len(df.columns) < 2:
                raise ValueError("파일에서 2개 이상의 컬럼을 찾을 수 없습니다.\n예상 형식: q (1/m), C(q) (m^4)")

            # Drop any header rows (non-numeric)
            df = df.apply(pd.to_numeric, errors='coerce').dropna()

            if len(df) < 2:
                raise ValueError("유효한 데이터 행이 부족합니다.")

            q = df.iloc[:, 0].values  # q (1/m)
            C_q = df.iloc[:, 1].values  # C(q) (m^4)

            # Check if data is in log scale (all values negative or small)
            if np.all(q < 100) and np.all(C_q < 0):
                # Likely log10 values
                q = 10**q
                C_q = 10**C_q
                format_str = "log10 형식 → 선형 변환됨"
            elif np.all(q > 0) and np.all(C_q > 0):
                format_str = "선형 형식"
            else:
                # Mixed - try to use as is
                format_str = "혼합 형식"

            # Validate
            if np.any(q <= 0) or np.any(C_q <= 0):
                raise ValueError("q와 C(q)는 모두 양수여야 합니다.")

            # Sort by q
            sort_idx = np.argsort(q)
            q = q[sort_idx]
            C_q = C_q[sort_idx]

            # Create PSD model
            from persson_model.utils.data_loader import create_psd_from_data
            self.psd_model = create_psd_from_data(q, C_q, interpolation_kind='log-log')

            # Store raw data
            self.psd_direct_data = {
                'q': q.copy(),
                'C_q': C_q.copy(),
                'filename': os.path.basename(filename)
            }

            # Mark Tab 0 as finalized (PSD ready)
            self.tab0_finalized = True
            self.psd_source = f"직접 로드: {os.path.basename(filename)}"

            # Update q_min (q_max는 기본값 1.0e+6 유지)
            self.q_min_var.set(f"{q.min():.2e}")

            # Update info display in Tab 0
            if hasattr(self, 'psd_direct_info_var'):
                self.psd_direct_info_var.set(
                    f"★ PSD 로드됨: {len(q)} pts, q={q.min():.1e}~{q.max():.1e} 1/m"
                )

            # Set the apply type to "direct" for easy finalization
            if hasattr(self, 'apply_psd_type_var'):
                self.apply_psd_type_var.set("direct")

            # Plot PSD on Tab 0's 2D PSD plot (bottom-right)
            self._plot_psd_direct_on_tab0()

            self._show_status(f"PSD 직접 로드 완료\n\n"
                f"파일: {os.path.basename(filename)}\n"
                f"데이터 포인트: {len(q)}\n"
                f"q 범위: {q.min():.2e} ~ {q.max():.2e} 1/m\n"
                f"C(q) 범위: {C_q.min():.2e} ~ {C_q.max():.2e} m⁴\n"
                f"형식: {format_str}\n\n"
                f"'▶ PSD 확정 → 계산에 사용' 버튼을 클릭하여 확정하세요.", 'success')

            self.status_var.set("PSD 직접 로드 완료")

        except Exception as e:
            import traceback
            messagebox.showerror("오류", f"PSD 로드 실패:\n{str(e)}\n\n{traceback.format_exc()}")

    def _plot_psd_direct_on_tab0(self):
        """Plot directly loaded PSD data on Tab 0's 2D PSD plot."""
        if not hasattr(self, 'psd_direct_data') or self.psd_direct_data is None:
            return

        data = self.psd_direct_data
        q = data['q']
        C_q = data['C_q']

        # Use Tab 0's 2D PSD plot (bottom-right: ax_psd_2d)
        if hasattr(self, 'ax_psd_2d'):
            self.ax_psd_2d.clear()
            self.ax_psd_2d.loglog(q, C_q, 'b-', linewidth=2, label='C(q) 직접 로드')
            self.ax_psd_2d.set_xlabel('파수 q (1/m)', fontsize=13)
            self.ax_psd_2d.set_ylabel(r'C(q) (m$^4$)', fontsize=13)
            self.ax_psd_2d.set_title(f"★ PSD 직접 로드: {data['filename']}", fontweight='bold', fontsize=15)
            self.ax_psd_2d.legend(loc='upper right', fontsize=12)
            self.ax_psd_2d.grid(True, alpha=0.3, which='both')
            self.fig_psd_profile.subplots_adjust(left=0.12, right=0.95, top=0.94, bottom=0.10, hspace=0.42, wspace=0.35)
            self.canvas_psd_profile.draw()

    def _plot_persson_master_curve(self):
        """Plot the loaded Persson master curve."""
        if not hasattr(self, 'persson_master_curve') or self.persson_master_curve is None:
            return

        data = self.persson_master_curve

        # Use the master curve plot (top-right)
        self.ax_mc_master.clear()

        f = data['f']
        E_storage = data['E_storage'] / 1e6  # Convert to MPa for display
        E_loss = data['E_loss'] / 1e6
        tan_delta = E_loss / E_storage

        # E' and E'' on log-log scale
        self.ax_mc_master.loglog(f, E_storage, 'b-', linewidth=2, label="E' (Persson)")
        self.ax_mc_master.loglog(f, E_loss, 'r-', linewidth=2, label="E'' (Persson)")

        self.ax_mc_master.set_xlabel('주파수 f (Hz)', fontsize=13)
        self.ax_mc_master.set_ylabel("E', E'' (MPa)", fontsize=13)
        self.ax_mc_master.set_title(f"★ Persson 정품 마스터 커브: {data['filename']}", fontweight='bold', fontsize=15)
        self.ax_mc_master.legend(loc='upper left', fontsize=12)
        self.ax_mc_master.grid(True, alpha=0.3)

        # tan δ on the aT plot (bottom-left)
        self.ax_mc_aT.clear()
        self.ax_mc_aT.semilogx(f, tan_delta, 'g-', linewidth=2, label='tan δ (Persson)')
        self.ax_mc_aT.set_xlabel('주파수 f (Hz)', fontsize=13)
        self.ax_mc_aT.set_ylabel('tan δ = E\'\'/E\'', fontsize=13)
        self.ax_mc_aT.set_title('tan δ (Persson 정품)', fontweight='bold', fontsize=15)
        self.ax_mc_aT.legend(loc='upper right', fontsize=12)
        self.ax_mc_aT.grid(True, alpha=0.3)
        self.ax_mc_aT.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

        self.fig_mc.tight_layout()
        self.canvas_mc.draw()

    def _apply_smoothing_to_persson(self):
        """Apply smoothing to loaded Persson master curve."""
        if not hasattr(self, 'persson_master_curve') or self.persson_master_curve is None:
            self._show_status("먼저 Persson 정품 마스터 커브를 로드하세요.", 'warning')
            return

        try:
            from scipy.signal import savgol_filter

            data = self.persson_master_curve

            # Get smoothing window
            window = self.mc_smooth_window_var.get()
            if window % 2 == 0:
                window += 1  # Savitzky-Golay requires odd window

            # Store original data if not already stored
            if 'E_storage_orig' not in data:
                data['E_storage_orig'] = data['E_storage'].copy()
                data['E_loss_orig'] = data['E_loss'].copy()

            # Apply smoothing in log space for better results
            log_E_storage = np.log10(data['E_storage_orig'])
            log_E_loss = np.log10(data['E_loss_orig'])

            # Apply Savitzky-Golay filter
            poly_order = min(3, window - 2)  # Polynomial order must be less than window
            log_E_storage_smooth = savgol_filter(log_E_storage, window, poly_order)
            log_E_loss_smooth = savgol_filter(log_E_loss, window, poly_order)

            # Convert back from log space
            E_storage_smooth = 10**log_E_storage_smooth
            E_loss_smooth = 10**log_E_loss_smooth

            # Update data
            data['E_storage'] = E_storage_smooth
            data['E_loss'] = E_loss_smooth
            data['smoothed'] = True
            data['smooth_window'] = window

            # Update the material object
            from persson_model.utils.data_loader import create_material_from_dma
            self.material_persson = create_material_from_dma(
                omega=data['omega'],
                E_storage=E_storage_smooth,
                E_loss=E_loss_smooth,
                material_name=f"Persson Smoothed (w={window})",
                reference_temp=20.0
            )

            # Update info display
            tan_delta_avg = np.mean(E_loss_smooth / E_storage_smooth)
            self.mc_data_info_var.set(
                f"★ Persson 정품 (스무딩 w={window}): {len(data['f'])} pts, "
                f"tan δ 평균={tan_delta_avg:.3f}"
            )

            # Replot
            self._plot_persson_master_curve_with_original()

            self._show_status(f"Persson 마스터 커브 스무딩 적용 완료\n\n"
                f"윈도우 크기: {window}\n"
                f"tan δ 평균: {tan_delta_avg:.3f}", 'success')

            self.status_var.set(f"Persson 마스터 커브 스무딩 적용 (w={window})")

        except Exception as e:
            import traceback
            messagebox.showerror("오류", f"스무딩 적용 실패:\n{str(e)}\n\n{traceback.format_exc()}")

    def _reset_persson_to_original(self):
        """Reset Persson master curve to original (before smoothing)."""
        if not hasattr(self, 'persson_master_curve') or self.persson_master_curve is None:
            self._show_status("Persson 마스터 커브가 로드되지 않았습니다.", 'warning')
            return

        data = self.persson_master_curve

        if 'E_storage_orig' not in data:
            self._show_status("이미 원본 상태입니다.", 'success')
            return

        # Restore original data
        data['E_storage'] = data['E_storage_orig'].copy()
        data['E_loss'] = data['E_loss_orig'].copy()
        data['smoothed'] = False
        if 'smooth_window' in data:
            del data['smooth_window']

        # Update the material object
        from persson_model.utils.data_loader import create_material_from_dma
        self.material_persson = create_material_from_dma(
            omega=data['omega'],
            E_storage=data['E_storage'],
            E_loss=data['E_loss'],
            material_name=f"Persson ({data['filename']})",
            reference_temp=20.0
        )

        # Update info display
        tan_delta_avg = np.mean(data['E_loss'] / data['E_storage'])
        self.mc_data_info_var.set(
            f"★ Persson 정품: {len(data['f'])} pts, "
            f"tan δ 평균={tan_delta_avg:.3f}"
        )

        # Replot (original only)
        self._plot_persson_master_curve()

        self.status_var.set("Persson 마스터 커브 원본 복원 완료")

    def _plot_persson_master_curve_with_original(self):
        """Plot Persson master curve with original (before smoothing) overlay."""
        if not hasattr(self, 'persson_master_curve') or self.persson_master_curve is None:
            return

        data = self.persson_master_curve

        # Use the master curve plot (top-right)
        self.ax_mc_master.clear()
        self.ax_mc_aT.clear()

        f = data['f']
        E_storage = data['E_storage'] / 1e6  # Convert to MPa for display
        E_loss = data['E_loss'] / 1e6
        tan_delta = E_loss / E_storage

        # Plot smoothed data
        self.ax_mc_master.loglog(f, E_storage, 'b-', linewidth=2, label="E' (스무딩)")
        self.ax_mc_master.loglog(f, E_loss, 'r-', linewidth=2, label="E'' (스무딩)")

        # Plot original if available
        if 'E_storage_orig' in data:
            E_storage_orig = data['E_storage_orig'] / 1e6
            E_loss_orig = data['E_loss_orig'] / 1e6
            tan_delta_orig = E_loss_orig / E_storage_orig

            self.ax_mc_master.loglog(f, E_storage_orig, 'b:', linewidth=1, alpha=0.5, label="E' (원본)")
            self.ax_mc_master.loglog(f, E_loss_orig, 'r:', linewidth=1, alpha=0.5, label="E'' (원본)")

            # tan δ comparison
            self.ax_mc_aT.semilogx(f, tan_delta, 'g-', linewidth=2, label='tan δ (스무딩)')
            self.ax_mc_aT.semilogx(f, tan_delta_orig, 'g:', linewidth=1, alpha=0.5, label='tan δ (원본)')
        else:
            self.ax_mc_aT.semilogx(f, tan_delta, 'g-', linewidth=2, label='tan δ')

        self.ax_mc_master.set_xlabel('주파수 f (Hz)', fontsize=13)
        self.ax_mc_master.set_ylabel("E', E'' (MPa)", fontsize=13)
        smooth_info = f" (w={data.get('smooth_window', 'N/A')})" if data.get('smoothed') else ""
        self.ax_mc_master.set_title(f"★ Persson 마스터 커브{smooth_info}", fontweight='bold', fontsize=15)
        self.ax_mc_master.legend(loc='upper left', fontsize=12)
        self.ax_mc_master.grid(True, alpha=0.3)

        self.ax_mc_aT.set_xlabel('주파수 f (Hz)', fontsize=13)
        self.ax_mc_aT.set_ylabel('tan δ = E\'\'/E\'', fontsize=13)
        self.ax_mc_aT.set_title('tan δ 비교', fontweight='bold', fontsize=15)
        self.ax_mc_aT.legend(loc='upper right', fontsize=12)
        self.ax_mc_aT.grid(True, alpha=0.3)
        self.ax_mc_aT.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

        self.fig_mc.tight_layout()
        self.canvas_mc.draw()

    def _plot_master_curve_comparison(self):
        """Plot comparison between Persson master curve and generated master curve."""
        has_persson = hasattr(self, 'persson_master_curve') and self.persson_master_curve is not None
        has_generated = hasattr(self, 'master_curve_gen') and self.master_curve_gen is not None and \
                        self.master_curve_gen.master_f is not None

        if not has_persson and not has_generated:
            self._show_status("비교할 마스터 커브가 없습니다.\n\n"
                                   "1. Persson 정품 마스터 커브를 로드하거나\n"
                                   "2. 마스터 커브를 생성하세요.", 'warning')
            return

        # Clear all plots for comparison
        self.ax_mc_raw.clear()
        self.ax_mc_master.clear()
        self.ax_mc_aT.clear()
        self.ax_mc_bT.clear()

        # == Plot 1: E' comparison ==
        if has_persson:
            p_data = self.persson_master_curve
            f_p = p_data['f']
            E_p = p_data['E_storage'] / 1e6
            E_pp = p_data['E_loss'] / 1e6
            self.ax_mc_raw.loglog(f_p, E_p, 'b-', linewidth=2.5, label="E' (Persson 정품)", alpha=0.9)
            self.ax_mc_master.loglog(f_p, E_pp, 'r-', linewidth=2.5, label="E'' (Persson 정품)", alpha=0.9)

        if has_generated:
            f_g = self.master_curve_gen.master_f
            E_g = self.master_curve_gen.master_E_storage
            E_gg = self.master_curve_gen.master_E_loss
            self.ax_mc_raw.loglog(f_g, E_g, 'b--', linewidth=2, label="E' (생성)", alpha=0.7)
            self.ax_mc_master.loglog(f_g, E_gg, 'r--', linewidth=2, label="E'' (생성)", alpha=0.7)

        # Configure E' plot
        self.ax_mc_raw.set_xlabel('주파수 f (Hz)', fontsize=13)
        self.ax_mc_raw.set_ylabel("E' (MPa)", fontsize=13)
        self.ax_mc_raw.set_title("E' (저장 탄성률) 비교", fontweight='bold', fontsize=15)
        self.ax_mc_raw.legend(loc='lower right', fontsize=12)
        self.ax_mc_raw.grid(True, alpha=0.3)

        # Configure E'' plot
        self.ax_mc_master.set_xlabel('주파수 f (Hz)', fontsize=13)
        self.ax_mc_master.set_ylabel("E'' (MPa)", fontsize=13)
        self.ax_mc_master.set_title("E'' (손실 탄성률) 비교", fontweight='bold', fontsize=15)
        self.ax_mc_master.legend(loc='lower right', fontsize=12)
        self.ax_mc_master.grid(True, alpha=0.3)

        # == Plot 3: tan δ comparison ==
        if has_persson:
            tan_p = E_pp / E_p
            self.ax_mc_aT.semilogx(f_p, tan_p, 'g-', linewidth=2.5, label='tan δ (Persson 정품)', alpha=0.9)

        if has_generated:
            tan_g = E_gg / E_g
            self.ax_mc_aT.semilogx(f_g, tan_g, 'g--', linewidth=2, label='tan δ (생성)', alpha=0.7)

        self.ax_mc_aT.set_xlabel('주파수 f (Hz)', fontsize=13)
        self.ax_mc_aT.set_ylabel('tan δ = E\'\'/E\'', fontsize=13)
        self.ax_mc_aT.set_title('tan δ (손실 탄젠트) 비교', fontweight='bold', fontsize=15)
        self.ax_mc_aT.legend(loc='upper right', fontsize=12)
        self.ax_mc_aT.grid(True, alpha=0.3)
        self.ax_mc_aT.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='tan δ = 1')

        # == Plot 4: E'' ratio or difference ==
        if has_persson and has_generated:
            # Interpolate to compare at same frequencies
            from scipy.interpolate import interp1d

            # Use overlapping frequency range
            f_min = max(f_p.min(), f_g.min())
            f_max = min(f_p.max(), f_g.max())
            f_common = np.logspace(np.log10(f_min), np.log10(f_max), 50)

            # Interpolate in log-log space
            log_E_pp_interp = interp1d(np.log10(f_p), np.log10(E_pp), bounds_error=False, fill_value='extrapolate')
            log_E_gg_interp = interp1d(np.log10(f_g), np.log10(E_gg), bounds_error=False, fill_value='extrapolate')

            E_pp_common = 10**log_E_pp_interp(np.log10(f_common))
            E_gg_common = 10**log_E_gg_interp(np.log10(f_common))

            ratio = E_pp_common / E_gg_common

            self.ax_mc_bT.semilogx(f_common, ratio, 'm-', linewidth=2, label="E''정품 / E''생성")
            self.ax_mc_bT.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
            self.ax_mc_bT.set_xlabel('주파수 f (Hz)', fontsize=13)
            self.ax_mc_bT.set_ylabel("E'' 비율", fontsize=13)
            self.ax_mc_bT.set_title("E'' 비율 (정품/생성)", fontweight='bold', fontsize=15)
            self.ax_mc_bT.legend(loc='upper right', fontsize=12)
            self.ax_mc_bT.grid(True, alpha=0.3)

            # Print comparison summary
            print("\n" + "="*60)
            print("마스터 커브 비교 결과")
            print("="*60)
            print(f"E'' 비율 (정품/생성) = {ratio.mean():.2f} ± {ratio.std():.2f}")
            print(f"tan δ 평균 (정품): {tan_p.mean():.3f}")
            print(f"tan δ 평균 (생성): {tan_g.mean():.3f}")
            print(f"tan δ 비율: {tan_p.mean()/tan_g.mean():.2f}x")
            print("="*60)
        else:
            self.ax_mc_bT.text(0.5, 0.5, "비교 데이터 없음\n(정품과 생성 둘 다 필요)",
                              ha='center', va='center', fontsize=14, transform=self.ax_mc_bT.transAxes)
            self.ax_mc_bT.set_title("E'' 비율 비교", fontweight='bold', fontsize=15)

        self.fig_mc.tight_layout()
        self.canvas_mc.draw()

        self.status_var.set("마스터 커브 비교 플롯 완료")

    def _use_persson_master_curve_for_calc(self):
        """Use loaded Persson master curve for friction calculation."""
        if not hasattr(self, 'material_persson') or self.material_persson is None:
            self._show_status("먼저 Persson 정품 마스터 커브를 로드하세요.", 'warning')
            return

        # Replace current material with Persson material
        self.material = self.material_persson
        self.material_source = f"Persson 정품: {self.persson_master_curve['filename']}"
        self.tab1_finalized = True

        # Update displays
        data = self.persson_master_curve
        tan_delta_avg = np.mean(data['E_loss'] / data['E_storage'])

        self._show_status(f"Persson 정품 마스터 커브가 계산에 사용됩니다.\n\n"
            f"파일: {data['filename']}\n"
            f"tan δ 평균: {tan_delta_avg:.3f}\n\n"
            f"Tab 3, Tab 5에서 이 데이터로 μ_visc 계산이 가능합니다.", 'success')

        self.status_var.set("Persson 정품 마스터 커브 → 계산용 확정")

    def _generate_master_curve(self):
        """Generate master curve using TTS."""
        if self.mc_raw_df is None:
            self._show_status("먼저 다중 온도 DMA 데이터를 로드하세요.", 'warning')
            return

        try:
            self.mc_calc_btn.config(state='disabled')
            self.mc_progress_var.set(10)
            self.root.update_idletasks()

            # Get settings
            T_ref = float(self.mc_tref_var.get())
            use_bT = self.mc_use_bt_var.get()
            bT_mode = self.mc_bt_mode_var.get()
            target = self.mc_target_var.get()

            # Create master curve generator
            self.master_curve_gen = MasterCurveGenerator(T_ref=T_ref)

            # Load data
            self.master_curve_gen.load_data(
                self.mc_raw_df,
                T_col='T', f_col='f',
                E_storage_col="E'", E_loss_col="E''"
            )

            self.mc_progress_var.set(30)
            self.root.update_idletasks()

            # Optimize shift factors
            self.master_curve_gen.optimize_shift_factors(
                use_bT=use_bT,
                bT_mode=bT_mode,
                target=target,
                verbose=False
            )

            self.mc_progress_var.set(60)
            self.root.update_idletasks()

            # Generate master curve with smoothing settings
            smooth_enabled = self.mc_smooth_var.get()
            smooth_window = self.mc_smooth_window_var.get()
            # Ensure window is odd
            if smooth_window % 2 == 0:
                smooth_window += 1

            master_curve = self.master_curve_gen.generate_master_curve(
                n_points=300, smooth=smooth_enabled, window_length=smooth_window
            )

            self.mc_progress_var.set(80)
            self.root.update_idletasks()

            # Fit WLF
            wlf_result = self.master_curve_gen.fit_wlf()

            self.mc_progress_var.set(90)
            self.root.update_idletasks()

            # Update plots
            self._update_mc_plots(master_curve, wlf_result, target)

            # Update results text
            self._update_mc_results(wlf_result, target)

            # Update shift factor table
            self._update_mc_shift_table()

            self.mc_progress_var.set(100)
            self.status_var.set("마스터 커브 생성 완료")

        except Exception as e:
            import traceback
            messagebox.showerror("오류", f"마스터 커브 생성 실패:\n{str(e)}\n\n{traceback.format_exc()}")
        finally:
            self.mc_calc_btn.config(state='normal')

    def _update_mc_plots(self, master_curve, wlf_result, target='E_storage'):
        """Update master curve plots."""
        # Clear all axes except raw data
        self.ax_mc_master.clear()
        self.ax_mc_aT.clear()
        self.ax_mc_bT.clear()

        # Get data
        temps = np.sort(self.master_curve_gen.temperatures)
        aT = self.master_curve_gen.aT
        bT = self.master_curve_gen.bT
        T_ref = self.master_curve_gen.T_ref

        # Target display name
        target_names = {
            'E_storage': "E'",
            'E_loss': "E''",
            'tan_delta': "tanδ"
        }
        target_display = target_names.get(target, target)

        # Colors for temperatures
        colors = plt.cm.coolwarm(np.linspace(0, 1, len(temps)))

        # Plot 1: Master curve with shifted data
        self.ax_mc_master.set_title(f'마스터 커브 (Tref={T_ref}°C, 최적화: {target_display})', fontweight='bold', fontsize=15)
        self.ax_mc_master.set_xlabel('Reduced Frequency (Hz)', fontsize=13)
        self.ax_mc_master.set_ylabel('E\', E\'\' (MPa)', fontsize=13)
        self.ax_mc_master.set_xscale('log')
        self.ax_mc_master.set_yscale('log')
        self.ax_mc_master.grid(True, alpha=0.3)

        # Plot shifted data points
        for i, T in enumerate(temps):
            shifted = self.master_curve_gen.get_shifted_data(T)
            self.ax_mc_master.scatter(shifted['f_reduced'], shifted['E_storage_shifted'],
                                     c=[colors[i]], s=10, alpha=0.5, marker='o')
            self.ax_mc_master.scatter(shifted['f_reduced'], shifted['E_loss_shifted'],
                                     c=[colors[i]], s=8, alpha=0.3, marker='s')

        # Plot master curve
        self.ax_mc_master.plot(master_curve['f'], master_curve['E_storage'],
                              'k-', linewidth=2, label="E' (Master)")
        self.ax_mc_master.plot(master_curve['f'], master_curve['E_loss'],
                              'k--', linewidth=2, label="E'' (Master)")
        self.ax_mc_master.legend(fontsize=12)

        # Plot 2: aT vs Temperature
        self.ax_mc_aT.set_title('수평 이동 계수 aT', fontweight='bold', fontsize=15)
        self.ax_mc_aT.set_xlabel('온도 T (°C)', fontsize=13)
        self.ax_mc_aT.set_ylabel('log10(aT)', fontsize=13)
        self.ax_mc_aT.grid(True, alpha=0.3)

        log_aT = [np.log10(aT[T]) for T in temps]
        self.ax_mc_aT.scatter(temps, log_aT, c='blue', s=50, zorder=3, label='Measured')

        # WLF fit line
        if wlf_result['C1'] is not None:
            T_fit = np.linspace(temps.min(), temps.max(), 100)
            log_aT_fit = -wlf_result['C1'] * (T_fit - T_ref) / (wlf_result['C2'] + (T_fit - T_ref))
            self.ax_mc_aT.plot(T_fit, log_aT_fit, 'r-', linewidth=2,
                              label=f"WLF (C1={wlf_result['C1']:.2f}, C2={wlf_result['C2']:.1f})")
        self.ax_mc_aT.legend(fontsize=12)
        self.ax_mc_aT.axhline(0, color='gray', linestyle=':', alpha=0.5)
        self.ax_mc_aT.axvline(T_ref, color='green', linestyle='--', alpha=0.5, label=f'Tref={T_ref}°C')

        # Plot 3: bT vs Temperature
        self.ax_mc_bT.set_title('수직 이동 계수 bT', fontweight='bold', fontsize=15)
        self.ax_mc_bT.set_xlabel('온도 T (°C)', fontsize=13)
        self.ax_mc_bT.set_ylabel('bT', fontsize=13)
        self.ax_mc_bT.grid(True, alpha=0.3)

        bT_values = [bT[T] for T in temps]
        self.ax_mc_bT.scatter(temps, bT_values, c='blue', s=50, zorder=3, label='Measured')

        # Theoretical line T/Tref
        T_ref_K = T_ref + 273.15
        bT_theory = (temps + 273.15) / T_ref_K
        self.ax_mc_bT.plot(temps, bT_theory, 'r--', linewidth=1.5, label='T/Tref (이론)')

        self.ax_mc_bT.axhline(1, color='gray', linestyle=':', alpha=0.5)
        self.ax_mc_bT.axvline(T_ref, color='green', linestyle='--', alpha=0.5)
        self.ax_mc_bT.legend(fontsize=12)

        self.fig_mc.tight_layout()
        self.canvas_mc.draw()

        # Auto-register graph data for master curve
        if master_curve is not None and 'f' in master_curve:
            self._register_graph_data(
                "MasterCurve_E_storage",
                master_curve['f'], master_curve['E_storage'],
                "f(Hz)\tE_storage(MPa)",
                f"Master Curve E' (Tref={T_ref}C)")
            self._register_graph_data(
                "MasterCurve_E_loss",
                master_curve['f'], master_curve['E_loss'],
                "f(Hz)\tE_loss(MPa)",
                f"Master Curve E'' (Tref={T_ref}C)")

    def _update_mc_results(self, wlf_result, target='E_storage'):
        """Update master curve results text."""
        self.mc_result_text.delete('1.0', tk.END)

        T_ref = self.master_curve_gen.T_ref
        temps = self.master_curve_gen.temperatures

        # Target display name
        target_names = {
            'E_storage': "E' (Storage Modulus)",
            'E_loss': "E'' (Loss Modulus)",
            'tan_delta': "tanδ (Loss Factor)"
        }
        target_display = target_names.get(target, target)

        text = f"=== 마스터 커브 생성 결과 ===\n\n"
        text += f"기준 온도 Tref: {T_ref}°C\n"
        text += f"최적화 대상: {target_display}\n"
        text += f"온도 범위: {temps.min():.1f} ~ {temps.max():.1f}°C\n"
        text += f"온도 개수: {len(temps)}개\n\n"

        if wlf_result['C1'] is not None:
            text += f"=== WLF 파라미터 ===\n"
            text += f"C1 = {wlf_result['C1']:.4f}\n"
            text += f"C2 = {wlf_result['C2']:.4f}\n"
            text += f"R² = {wlf_result['r_squared']:.4f}\n\n"

        text += f"마스터 커브 주파수 범위:\n"
        text += f"  {self.master_curve_gen.master_f.min():.2e} ~ "
        text += f"{self.master_curve_gen.master_f.max():.2e} Hz\n"

        self.mc_result_text.insert(tk.END, text)

    def _toggle_bt_comparison(self):
        """Toggle bT comparison view in master curve plot."""
        if self.master_curve_gen is None or self.master_curve_gen.master_f is None:
            return

        show_comparison = self.mc_compare_bt_var.get()

        # Clear and redraw master curve plot
        self.ax_mc_master.clear()

        temps = np.sort(self.master_curve_gen.temperatures)
        T_ref = self.master_curve_gen.T_ref
        colors = plt.cm.coolwarm(np.linspace(0, 1, len(temps)))

        self.ax_mc_master.set_xlabel('Reduced Frequency (Hz)')
        self.ax_mc_master.set_ylabel('E\', E\'\' (MPa)')
        self.ax_mc_master.set_xscale('log')
        self.ax_mc_master.set_yscale('log')
        self.ax_mc_master.grid(True, alpha=0.3)

        if show_comparison:
            # Show both with and without bT
            self.ax_mc_master.set_title(f'마스터 커브 비교: bT 적용 vs 미적용 (Tref={T_ref}°C)', fontweight='bold')

            # Plot WITH bT (solid lines)
            for i, T in enumerate(temps):
                shifted = self.master_curve_gen.get_shifted_data(T)
                self.ax_mc_master.scatter(shifted['f_reduced'], shifted['E_storage_shifted'],
                                         c=[colors[i]], s=12, alpha=0.7, marker='o')

            # Plot master curve with bT
            self.ax_mc_master.plot(self.master_curve_gen.master_f, self.master_curve_gen.master_E_storage,
                                  'k-', linewidth=2.5, label="E' (bT 적용)", zorder=10)
            self.ax_mc_master.plot(self.master_curve_gen.master_f, self.master_curve_gen.master_E_loss,
                                  'k--', linewidth=2.5, label="E'' (bT 적용)", zorder=10)

            # Plot WITHOUT bT (dashed, lighter)
            for i, T in enumerate(temps):
                f_reduced = self.master_curve_gen.raw_data[T]['f'] * self.master_curve_gen.aT[T]
                E_storage_no_bt = self.master_curve_gen.raw_data[T]['E_storage']  # No bT division
                E_loss_no_bt = self.master_curve_gen.raw_data[T]['E_loss']
                self.ax_mc_master.scatter(f_reduced, E_storage_no_bt,
                                         c=[colors[i]], s=8, alpha=0.3, marker='x')

            # Generate master curve without bT for comparison
            all_f = []
            all_E_storage_no_bt = []
            all_E_loss_no_bt = []
            for T in temps:
                f_reduced = self.master_curve_gen.raw_data[T]['f'] * self.master_curve_gen.aT[T]
                all_f.extend(f_reduced)
                all_E_storage_no_bt.extend(self.master_curve_gen.raw_data[T]['E_storage'])
                all_E_loss_no_bt.extend(self.master_curve_gen.raw_data[T]['E_loss'])

            all_f = np.array(all_f)
            all_E_storage_no_bt = np.array(all_E_storage_no_bt)
            all_E_loss_no_bt = np.array(all_E_loss_no_bt)
            sort_idx = np.argsort(all_f)

            self.ax_mc_master.plot(all_f[sort_idx], all_E_storage_no_bt[sort_idx],
                                  'b:', linewidth=1.5, alpha=0.7, label="E' (bT 미적용)")
            self.ax_mc_master.plot(all_f[sort_idx], all_E_loss_no_bt[sort_idx],
                                  'r:', linewidth=1.5, alpha=0.7, label="E'' (bT 미적용)")

        else:
            # Normal view with bT
            target = self.mc_target_var.get()
            target_names = {'E_storage': "E'", 'E_loss': "E''", 'tan_delta': "tanδ"}
            target_display = target_names.get(target, target)
            self.ax_mc_master.set_title(f'마스터 커브 (Tref={T_ref}°C, 최적화: {target_display})', fontweight='bold')

            for i, T in enumerate(temps):
                shifted = self.master_curve_gen.get_shifted_data(T)
                self.ax_mc_master.scatter(shifted['f_reduced'], shifted['E_storage_shifted'],
                                         c=[colors[i]], s=10, alpha=0.5, marker='o')
                self.ax_mc_master.scatter(shifted['f_reduced'], shifted['E_loss_shifted'],
                                         c=[colors[i]], s=8, alpha=0.3, marker='s')

            self.ax_mc_master.plot(self.master_curve_gen.master_f, self.master_curve_gen.master_E_storage,
                                  'k-', linewidth=2, label="E' (Master)")
            self.ax_mc_master.plot(self.master_curve_gen.master_f, self.master_curve_gen.master_E_loss,
                                  'k--', linewidth=2, label="E'' (Master)")

        self.ax_mc_master.legend(fontsize=11, loc='best')
        self.fig_mc.tight_layout()
        self.canvas_mc.draw()

    def _update_mc_shift_table(self):
        """Update shift factor table."""
        # Clear existing entries
        for item in self.mc_shift_table.get_children():
            self.mc_shift_table.delete(item)

        # Add new entries
        temps = np.sort(self.master_curve_gen.temperatures)
        for T in temps:
            aT = self.master_curve_gen.aT[T]
            bT = self.master_curve_gen.bT[T]
            log_aT = np.log10(aT)

            self.mc_shift_table.insert('', 'end', values=(
                f'{T:.1f}',
                f'{aT:.4e}',
                f'{bT:.4f}',
                f'{log_aT:.4f}'
            ))

    def _export_master_curve(self):
        """Export master curve data to CSV."""
        if self.master_curve_gen is None or self.master_curve_gen.master_f is None:
            self._show_status("먼저 마스터 커브를 생성하세요.", 'warning')
            return

        filename = filedialog.asksaveasfilename(
            title="마스터 커브 저장",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if not filename:
            return

        try:
            import pandas as pd

            # Master curve data
            mc_data = pd.DataFrame({
                'f (Hz)': self.master_curve_gen.master_f,
                'omega (rad/s)': 2 * np.pi * self.master_curve_gen.master_f,
                "E' (MPa)": self.master_curve_gen.master_E_storage,
                "E'' (MPa)": self.master_curve_gen.master_E_loss,
                'tan_delta': self.master_curve_gen.master_E_loss / self.master_curve_gen.master_E_storage
            })

            # Shift factors
            shift_data = self.master_curve_gen.get_shift_factor_table()

            # Save to CSV with both tables
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                f.write(f"# Master Curve (Tref = {self.master_curve_gen.T_ref}°C)\n")
                mc_data.to_csv(f, index=False)
                f.write("\n# Shift Factors\n")
                shift_data.to_csv(f, index=False)

            self._show_status(f"마스터 커브 저장 완료:\n{filename}", 'success')

        except Exception as e:
            messagebox.showerror("오류", f"저장 실패:\n{str(e)}")

    def _apply_master_curve_to_verification(self):
        """Apply generated master curve to Tab 1 for friction calculation."""
        if self.master_curve_gen is None or self.master_curve_gen.master_f is None:
            self._show_status("먼저 마스터 커브를 생성하세요.", 'warning')
            return

        try:
            # Get master curve data
            omega = 2 * np.pi * self.master_curve_gen.master_f
            E_storage = self.master_curve_gen.master_E_storage * 1e6  # MPa to Pa
            E_loss = self.master_curve_gen.master_E_loss * 1e6  # MPa to Pa
            T_ref = self.master_curve_gen.T_ref

            # Create material from master curve
            self.material = create_material_from_dma(
                omega=omega,
                E_storage=E_storage,
                E_loss=E_loss,
                material_name=f"Master Curve (Tref={T_ref}°C)",
                reference_temp=T_ref
            )

            # Store raw data for plotting (in omega units)
            self.raw_dma_data = {
                'omega': omega,
                'E_storage': E_storage,
                'E_loss': E_loss
            }

            # Store shift factor data for temperature conversion
            self.master_curve_shift_factors = {
                'aT': self.master_curve_gen.aT.copy(),
                'bT': self.master_curve_gen.bT.copy(),
                'T_ref': T_ref,
                'C1': self.master_curve_gen.C1,
                'C2': self.master_curve_gen.C2,
                'temperatures': list(self.master_curve_gen.temperatures)
            }

            # Update temperature entry with reference temperature
            self.temperature_var.set(str(T_ref))

            # Update status label in Tab 1
            f_min = self.master_curve_gen.master_f.min()
            f_max = self.master_curve_gen.master_f.max()
            self.dma_import_status_var.set(f"Master Curve (Tref={T_ref}°C, {f_min:.1e}~{f_max:.1e} Hz)")

            # Update material display
            self._update_material_display()

            # Switch to calculation settings tab
            self.notebook.select(2)  # Tab 2: 계산 설정

            # Build info message
            info_msg = f"마스터 커브가 적용되었습니다.\n\n"
            info_msg += f"기준 온도 (Tref): {self.master_curve_gen.T_ref}°C\n"
            info_msg += f"주파수 범위: {self.master_curve_gen.master_f.min():.2e} ~ "
            info_msg += f"{self.master_curve_gen.master_f.max():.2e} Hz\n\n"

            if self.master_curve_gen.C1 is not None:
                info_msg += f"WLF 파라미터:\n"
                info_msg += f"  C1 = {self.master_curve_gen.C1:.2f}\n"
                info_msg += f"  C2 = {self.master_curve_gen.C2:.2f}°C\n"

            self._show_status(info_msg, 'success')

        except Exception as e:
            import traceback
            messagebox.showerror("오류", f"적용 실패:\n{str(e)}\n\n{traceback.format_exc()}")

    def _finalize_master_curve_to_tab3(self):
        """Finalize master curve and send to Tab 3 (Calculation Settings)."""
        if self.master_curve_gen is None or self.master_curve_gen.master_f is None:
            self._show_status("먼저 마스터 커브를 생성하세요.", 'warning')
            return

        try:
            # Get master curve data
            omega = 2 * np.pi * self.master_curve_gen.master_f
            E_storage = self.master_curve_gen.master_E_storage * 1e6  # MPa to Pa
            E_loss = self.master_curve_gen.master_E_loss * 1e6  # MPa to Pa
            T_ref = self.master_curve_gen.T_ref

            # Create material from master curve
            self.material = create_material_from_dma(
                omega=omega,
                E_storage=E_storage,
                E_loss=E_loss,
                material_name=f"Master Curve (Tref={T_ref}°C)",
                reference_temp=T_ref
            )

            # Store raw data
            self.raw_dma_data = {
                'omega': omega,
                'E_storage': E_storage,
                'E_loss': E_loss
            }

            # Store shift factor data
            self.master_curve_shift_factors = {
                'aT': self.master_curve_gen.aT.copy(),
                'bT': self.master_curve_gen.bT.copy(),
                'T_ref': T_ref,
                'C1': self.master_curve_gen.C1,
                'C2': self.master_curve_gen.C2,
                'temperatures': list(self.master_curve_gen.temperatures)
            }

            # Store finalized master curve data
            self.finalized_master_curve = {
                'omega': omega,
                'E_storage': E_storage,
                'E_loss': E_loss,
                'T_ref': T_ref,
                'f_min': self.master_curve_gen.master_f.min(),
                'f_max': self.master_curve_gen.master_f.max(),
                'C1': self.master_curve_gen.C1,
                'C2': self.master_curve_gen.C2
            }

            # Register finalized master curve
            f = self.master_curve_gen.master_f
            self._register_graph_data(
                "MasterCurve_E_storage_Finalized", f, self.master_curve_gen.master_E_storage,
                "f(Hz)\tE_storage(MPa)", f"Finalized Master Curve E' (Tref={T_ref}C)")
            self._register_graph_data(
                "MasterCurve_E_loss_Finalized", f, self.master_curve_gen.master_E_loss,
                "f(Hz)\tE_loss(MPa)", f"Finalized Master Curve E'' (Tref={T_ref}C)")

            # Update temperature in Tab 3
            self.temperature_var.set(str(T_ref))

            # Mark Tab 1 as finalized
            self.tab1_finalized = True
            self.material_source = f"Tab 1 마스터 커브 (Tref={T_ref}°C)"

            # Update Tab 2 state
            self._update_tab2_state()

            # Build info message
            f_min = self.master_curve_gen.master_f.min()
            f_max = self.master_curve_gen.master_f.max()

            info_msg = f"마스터 커브 확정 → Tab 3 전송 완료\n\n"
            info_msg += f"기준 온도 (Tref): {T_ref}°C\n"
            info_msg += f"주파수 범위: {f_min:.2e} ~ {f_max:.2e} Hz\n"
            info_msg += f"데이터 점 수: {len(omega)}\n\n"

            if self.master_curve_gen.C1 is not None:
                info_msg += f"WLF 파라미터:\n"
                info_msg += f"  C1 = {self.master_curve_gen.C1:.2f}\n"
                info_msg += f"  C2 = {self.master_curve_gen.C2:.2f}°C\n\n"

            info_msg += "Tab 3 (계산 설정)에서 계속하세요."

            self._show_status(info_msg, 'success')

            # Switch to Tab 3
            self.notebook.select(3)

            self.status_var.set(f"마스터 커브 확정: Tref={T_ref}°C, {f_min:.1e}~{f_max:.1e} Hz")

        except Exception as e:
            import traceback
            messagebox.showerror("오류", f"확정 실패:\n{str(e)}\n\n{traceback.format_exc()}")

    def _create_parameters_tab(self, parent):
        """Create calculation parameters tab."""
        # Instruction
        instruction = ttk.LabelFrame(parent, text="탭 설명", padding=10)
        instruction.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(instruction, text=
            "계산 매개변수를 설정합니다: 압력, 속도 범위 (로그 스케일), 온도.\n"
            "속도 범위: 0.0001~10 m/s (로그 간격)로 주파수 스윕을 수행합니다.",
            font=('Segoe UI', 17)
        ).pack()

        # Create main container with 2 columns
        main_container = ttk.Frame(parent)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Left panel for inputs (fixed width, scrollable)
        left_frame = ttk.Frame(main_container, width=getattr(self, '_left_panel_width', 600))
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        left_frame.pack_propagate(False)

        # Logo at bottom
        self._add_logo_to_panel(left_frame)

        # Toolbar with G(q,v) 계산 button (fixed at top, always accessible)
        toolbar = self._create_panel_toolbar(left_frame, buttons=[])

        # G(q,v) 계산 button - assigned to self.calc_button for state management
        self.calc_button = ttk.Button(toolbar, text="G(q,v) 계산 실행",
                                      command=self._run_calculation,
                                      style='Accent.TButton')
        self.calc_button.pack(side=tk.LEFT, padx=3, pady=1)

        # Progress bar in toolbar
        self.progress_var = tk.IntVar()
        self.progress_bar = ttk.Progressbar(
            toolbar, variable=self.progress_var, maximum=100, length=200
        )
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4, pady=1)

        # Right panel for visualization
        right_panel = ttk.Frame(main_container)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # ===== Scrollable left panel =====
        param_canvas = tk.Canvas(left_frame, highlightthickness=0)
        param_scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=param_canvas.yview)
        left_panel = ttk.Frame(param_canvas)

        left_panel.bind(
            "<Configure>",
            lambda e: param_canvas.configure(scrollregion=param_canvas.bbox("all"))
        )

        param_canvas.create_window((0, 0), window=left_panel, anchor="nw", width=getattr(self, '_left_panel_width', 600) - 20)
        param_canvas.configure(yscrollcommand=param_scrollbar.set)

        param_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        param_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Mousewheel scroll (cross-platform)
        def _on_mw_param(event):
            if event.delta:
                param_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            elif event.num == 4:
                param_canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                param_canvas.yview_scroll(1, "units")

        def _bind_mw_param(event):
            param_canvas.bind_all("<MouseWheel>", _on_mw_param)
            param_canvas.bind_all("<Button-4>", _on_mw_param)
            param_canvas.bind_all("<Button-5>", _on_mw_param)
        def _unbind_mw_param(event):
            param_canvas.unbind_all("<MouseWheel>")
            param_canvas.unbind_all("<Button-4>")
            param_canvas.unbind_all("<Button-5>")
        param_canvas.bind("<Enter>", _bind_mw_param)
        param_canvas.bind("<Leave>", _unbind_mw_param)

        # Input panel in left column (scrollable)
        input_frame = ttk.LabelFrame(left_panel, text="계산 매개변수", padding=10)
        input_frame.pack(fill=tk.X, pady=(0, 5))

        # Create input fields
        row = 0

        # Nominal pressure
        ttk.Label(input_frame, text="공칭 압력 (MPa):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.sigma_0_var = tk.StringVar(value="0.3")
        ttk.Entry(input_frame, textvariable=self.sigma_0_var, width=15).grid(row=row, column=1, pady=5)

        # Velocity range
        row += 1
        ttk.Label(input_frame, text="속도 범위:").grid(row=row, column=0, sticky=tk.W, pady=5)
        ttk.Label(input_frame, text="로그 스케일: 1e-7~10000 m/s").grid(row=row, column=1, sticky=tk.W, pady=5)

        row += 1
        ttk.Label(input_frame, text="  최소 속도 v_min (m/s):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.v_min_var = tk.StringVar(value="0.00001")
        ttk.Entry(input_frame, textvariable=self.v_min_var, width=15).grid(row=row, column=1, pady=5)

        row += 1
        ttk.Label(input_frame, text="  최대 속도 v_max (m/s):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.v_max_var = tk.StringVar(value="1000")
        ttk.Entry(input_frame, textvariable=self.v_max_var, width=15).grid(row=row, column=1, pady=5)

        row += 1
        ttk.Label(input_frame, text="  속도 포인트 수:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.n_velocity_var = tk.StringVar(value="30")
        ttk.Entry(input_frame, textvariable=self.n_velocity_var, width=15).grid(row=row, column=1, pady=5)

        # Temperature
        row += 1
        ttk.Label(input_frame, text="온도 (°C):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.temperature_var = tk.StringVar(value="20")
        ttk.Entry(input_frame, textvariable=self.temperature_var, width=15).grid(row=row, column=1, pady=5)

        # Poisson ratio
        row += 1
        ttk.Label(input_frame, text="푸아송 비:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.poisson_var = tk.StringVar(value="0.49")
        ttk.Entry(input_frame, textvariable=self.poisson_var, width=15).grid(row=row, column=1, pady=5)

        # Wavenumber range
        row += 1
        ttk.Label(input_frame, text="최소 파수 q_min (1/m):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.q_min_var = tk.StringVar(value="2.00e-01")
        ttk.Entry(input_frame, textvariable=self.q_min_var, width=15).grid(row=row, column=1, pady=5)

        # Surface type q1 presets (user-defined)
        row += 1
        ttk.Label(input_frame, text="q_max/q1 프리셋:").grid(row=row, column=0, sticky=tk.W, pady=5)
        preset_q1_frame = ttk.Frame(input_frame)
        preset_q1_frame.grid(row=row, column=1, pady=5, sticky=tk.W)
        self.surface_q1_var = tk.StringVar(value='(선택...)')
        self.surface_q1_combo = ttk.Combobox(preset_q1_frame, textvariable=self.surface_q1_var,
                                              state='readonly', width=18, font=self.FONTS['body'])
        self.surface_q1_combo.pack(side=tk.LEFT)
        ttk.Button(preset_q1_frame, text="로드", command=self._load_preset_surface_q1, width=4,
                   style='Outline.TButton').pack(side=tk.LEFT, padx=1)
        ttk.Button(preset_q1_frame, text="삭제", command=self._delete_preset_surface_q1, width=4).pack(side=tk.LEFT, padx=1)
        ttk.Button(preset_q1_frame, text="현재값 저장", command=self._add_preset_surface_q1, width=8).pack(side=tk.LEFT, padx=1)
        self._refresh_preset_surface_q1_list()

        row += 1
        ttk.Label(input_frame, text="최대 파수 q_max (1/m):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.q_max_var = tk.StringVar(value="1.0e+6")
        # q_max 입력 필드 (강조)
        qmax_entry_frame = tk.Frame(input_frame, bg=self.COLORS['primary'], padx=1, pady=1)
        qmax_entry_frame.grid(row=row, column=1, pady=5)
        ttk.Entry(qmax_entry_frame, textvariable=self.q_max_var, width=15).pack()

        row += 1
        ttk.Label(input_frame, text="파수 포인트 수:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.n_q_var = tk.StringVar(value="100")
        ttk.Entry(input_frame, textvariable=self.n_q_var, width=15).grid(row=row, column=1, pady=5)

        # Phi integration points for G(q) calculation
        row += 1
        ttk.Label(input_frame, text="G(q) φ 적분 포인트:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.n_phi_gq_var = tk.StringVar(value="36")
        ttk.Entry(input_frame, textvariable=self.n_phi_gq_var, width=15).grid(row=row, column=1, pady=5)

        # G 보정 계수 (Norm Factor)
        row += 1
        ttk.Label(input_frame, text="G 보정 계수 (Norm Factor):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.g_norm_factor_var = tk.StringVar(value="1.5625")
        norm_entry_frame = tk.Frame(input_frame, bg=self.COLORS['warning'], padx=1, pady=1)
        norm_entry_frame.grid(row=row, column=1, pady=5, sticky=tk.W)
        ttk.Entry(norm_entry_frame, textvariable=self.g_norm_factor_var, width=8).pack(side=tk.LEFT)
        ttk.Label(input_frame, text="G(q) = ∫ / (8 × NF), 기본값 1.5625",
                  font=('Segoe UI', 16), foreground='#64748B').grid(row=row+1, column=0, columnspan=2, sticky=tk.W)
        row += 1

        # ===== h'rms (ξ) / q1 모드 선택 섹션 =====
        # h'rms = ξ = RMS slope (경사), NOT h_rms (height)
        row += 1
        # 모드 선택 섹션 (강조)
        mode_wrapper = tk.Frame(input_frame, bg=self.COLORS['primary'], padx=1, pady=1)
        mode_wrapper.grid(row=row, column=0, columnspan=2, sticky=tk.EW, pady=10)
        mode_frame = ttk.LabelFrame(mode_wrapper, text="h'rms (ξ, slope) / q1 결정 모드", padding=5)
        mode_frame.pack(fill=tk.X)

        # 설명 라벨
        desc_label = ttk.Label(mode_frame,
            text="※ h'rms = ξ = RMS slope (경사), ξ² = 2π∫k³C(k)dk",
            font=('Segoe UI', 16), foreground='#64748B')
        desc_label.pack(fill=tk.X, pady=(0, 5))

        # 모드 선택 라디오 버튼
        self.hrms_q1_mode_var = tk.StringVar(value="q1_to_hrms")  # 기본값: q1 → h'rms(ξ) (모드 2)

        mode_row1 = ttk.Frame(mode_frame)
        mode_row1.pack(fill=tk.X, pady=2)
        ttk.Radiobutton(mode_row1, text="모드 1: h'rms (ξ) → q1 계산",
                       variable=self.hrms_q1_mode_var, value="hrms_to_q1",
                       command=self._on_hrms_q1_mode_changed).pack(side=tk.LEFT)

        mode_row2 = ttk.Frame(mode_frame)
        mode_row2.pack(fill=tk.X, pady=2)
        ttk.Radiobutton(mode_row2, text="모드 2: q1 → h'rms (ξ) 계산",
                       variable=self.hrms_q1_mode_var, value="q1_to_hrms",
                       command=self._on_hrms_q1_mode_changed).pack(side=tk.LEFT)

        # 구분선
        ttk.Separator(mode_frame, orient='horizontal').pack(fill=tk.X, pady=5)

        # h'rms(ξ) 입력 (모드 1용)
        self.hrms_input_frame = ttk.Frame(mode_frame)
        self.hrms_input_frame.pack(fill=tk.X, pady=2)
        ttk.Label(self.hrms_input_frame, text="목표 h'rms (ξ):").pack(side=tk.LEFT)
        self.target_hrms_slope_var = tk.StringVar(value="4.0520")
        self.hrms_entry = ttk.Entry(self.hrms_input_frame, textvariable=self.target_hrms_slope_var, width=12)
        self.hrms_entry.pack(side=tk.LEFT, padx=5)

        # q1 입력 (모드 2용)
        self.q1_input_frame = ttk.Frame(mode_frame)
        self.q1_input_frame.pack(fill=tk.X, pady=2)
        ttk.Label(self.q1_input_frame, text="목표 q1 (1/m):").pack(side=tk.LEFT)
        self.input_q1_var = tk.StringVar(value="1.0e+06")
        self.q1_entry = ttk.Entry(self.q1_input_frame, textvariable=self.input_q1_var, width=12)
        self.q1_entry.pack(side=tk.LEFT, padx=5)

        # Add trace to sync target_hrms_slope_var with Tab 1's psd_xi_var and Tab 4's display
        self.target_hrms_slope_var.trace_add('write', self._on_target_hrms_changed)

        # 계산 버튼
        calc_btn_frame = ttk.Frame(mode_frame)
        calc_btn_frame.pack(fill=tk.X, pady=5)
        # h'rms/q1 계산 버튼
        self.hrms_q1_calc_btn = ttk.Button(calc_btn_frame, text="h'rms/q1 계산",
                                           command=self._calculate_hrms_q1,
                                           style='Accent.TButton')
        self.hrms_q1_calc_btn.pack(side=tk.LEFT, padx=5)

        # 초기 모드 UI 상태 적용 (모드2: q1 활성, h'rms 비활성) - 버튼 생성 후 호출
        self._on_hrms_q1_mode_changed()

        ttk.Button(calc_btn_frame, text="Tab 4로 전달",
                  command=self._send_hrms_q1_to_tab4).pack(side=tk.LEFT, padx=5)

        # 구분선
        ttk.Separator(mode_frame, orient='horizontal').pack(fill=tk.X, pady=5)

        # 결과 표시 영역
        result_frame = ttk.Frame(mode_frame)
        result_frame.pack(fill=tk.X, pady=2)

        # 계산된 q1 표시 (모드 1 결과)
        q1_result_row = ttk.Frame(result_frame)
        q1_result_row.pack(fill=tk.X, pady=2)
        ttk.Label(q1_result_row, text="→ 계산된 q1:").pack(side=tk.LEFT)
        self.calculated_q1_var = tk.StringVar(value="(계산 후 표시)")
        self.calculated_q1_label = ttk.Label(q1_result_row, textvariable=self.calculated_q1_var,
                                             font=('Arial', 12, 'bold'), foreground='#2563EB')
        self.calculated_q1_label.pack(side=tk.LEFT, padx=5)
        ttk.Label(q1_result_row, text="(1/m)").pack(side=tk.LEFT)

        # 계산된 h'rms(ξ) 표시 (모드 2 결과)
        hrms_result_row = ttk.Frame(result_frame)
        hrms_result_row.pack(fill=tk.X, pady=2)
        ttk.Label(hrms_result_row, text="→ 계산된 h'rms (ξ):").pack(side=tk.LEFT)
        self.calculated_hrms_var = tk.StringVar(value="(계산 후 표시)")
        self.calculated_hrms_label = ttk.Label(hrms_result_row, textvariable=self.calculated_hrms_var,
                                               font=('Arial', 12, 'bold'), foreground='#059669')
        self.calculated_hrms_label.pack(side=tk.LEFT, padx=5)
        ttk.Label(hrms_result_row, text="(무차원)").pack(side=tk.LEFT)

        # 초기 모드에 따른 UI 상태 설정
        self._on_hrms_q1_mode_changed()

        # PSD type
        row += 1
        ttk.Label(input_frame, text="PSD 유형:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.psd_type_var = tk.StringVar(value="measured")
        ttk.Combobox(
            input_frame,
            textvariable=self.psd_type_var,
            values=["measured", "fractal"],
            state="readonly",
            width=12,
            font=self.FONTS['body']
        ).grid(row=row, column=1, pady=5)

        # Calculation visualization area in right panel
        viz_frame = ttk.LabelFrame(right_panel, text="계산 과정 시각화", padding=10)
        viz_frame.pack(fill=tk.BOTH, expand=True)

        # Status display for current calculation state
        status_display_frame = ttk.Frame(viz_frame)
        status_display_frame.pack(fill=tk.X, pady=(0, 5))

        self.calc_status_label = ttk.Label(
            status_display_frame,
            text="대기 중 | v = - m/s | q 범위 = - ~ - (1/m) | f 범위 = - ~ - (Hz)",
            font=('Arial', 13, 'bold'),
            foreground='#2563EB'
        )
        self.calc_status_label.pack()

        # Create figure for calculation progress with 2x2 subplots
        self.fig_calc_progress = Figure(figsize=(12, 10), dpi=100)

        # Top-left: PSD(q) with integration progress
        self.ax_psd_q = self.fig_calc_progress.add_subplot(221)

        # Top-right: DMA master curve with frequency window
        self.ax_dma_progress = self.fig_calc_progress.add_subplot(222)

        # Bottom-left: G(q) curves accumulating per velocity (KEY)
        self.ax_gq_live = self.fig_calc_progress.add_subplot(223)

        # Bottom-right: Contact area A(q)/A₀ evolution
        self.ax_contact_live = self.fig_calc_progress.add_subplot(224)

        self.canvas_calc_progress = FigureCanvasTkAgg(self.fig_calc_progress, viz_frame)
        self.canvas_calc_progress.draw()
        self.canvas_calc_progress.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Initialize PSD(q) plot - top-left
        self.ax_psd_q.set_xlabel('파수 q (1/m)', fontweight='bold', fontsize=13)
        self.ax_psd_q.set_ylabel(r'PSD C(q) (m$^4$)', fontweight='bold', fontsize=13)
        self.ax_psd_q.set_xscale('log')
        self.ax_psd_q.set_yscale('log')
        self.ax_psd_q.grid(True, alpha=0.3)
        self.ax_psd_q.set_title('PSD C(q)', fontweight='bold', fontsize=15)

        # Initialize DMA plot - top-right
        self.ax_dma_progress.set_xlabel('주파수 f (Hz)', fontweight='bold', fontsize=13)
        self.ax_dma_progress.set_ylabel('탄성률 (Pa)', fontweight='bold', fontsize=13)
        self.ax_dma_progress.set_xscale('log')
        self.ax_dma_progress.set_yscale('log')
        self.ax_dma_progress.grid(True, alpha=0.3)
        self.ax_dma_progress.set_title('DMA 마스터 곡선', fontweight='bold', fontsize=15)

        # Initialize G(q) live plot - bottom-left
        self.ax_gq_live.set_xlabel('파수 q (1/m)', fontweight='bold', fontsize=13)
        self.ax_gq_live.set_ylabel('G(q)', fontweight='bold', fontsize=13)
        self.ax_gq_live.set_xscale('log')
        self.ax_gq_live.set_yscale('log')
        self.ax_gq_live.grid(True, alpha=0.3)
        self.ax_gq_live.set_title('실시간 G(q) 적분 누적', fontweight='bold', fontsize=15)
        self.ax_gq_live.text(0.5, 0.5, '계산 대기 중...',
                             transform=self.ax_gq_live.transAxes,
                             ha='center', va='center', fontsize=16,
                             color='#94A3B8', style='italic')

        # Initialize contact area live plot - bottom-right
        self.ax_contact_live.set_xlabel('파수 q (1/m)', fontweight='bold', fontsize=13)
        self.ax_contact_live.set_ylabel(r'A(q)/A$_0$', fontweight='bold', fontsize=13)
        self.ax_contact_live.set_xscale('log')
        self.ax_contact_live.grid(True, alpha=0.3)
        self.ax_contact_live.set_title(r'접촉 면적비 A(q)/A$_0$ 변화', fontweight='bold', fontsize=15)
        self.ax_contact_live.text(0.5, 0.5, '계산 대기 중...',
                                  transform=self.ax_contact_live.transAxes,
                                  ha='center', va='center', fontsize=16,
                                  color='#94A3B8', style='italic')

        self.fig_calc_progress.tight_layout(pad=2.5)

        # Save button for calculation progress plot in left panel
        save_btn_frame = ttk.Frame(left_panel)
        save_btn_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Button(
            save_btn_frame,
            text="계산 과정 그래프 저장",
            command=lambda: self._save_plot(self.fig_calc_progress, "calculation_progress")
        ).pack(fill=tk.X)

    def _on_target_hrms_changed(self, *args):
        """Callback when target h'rms value is changed in Tab 2.
        Syncs the value to Tab 1's psd_xi_var and Tab 4's display."""
        try:
            new_value = self.target_hrms_slope_var.get()
            # Validate it's a number
            float(new_value)

            # Sync to Tab 1's psd_xi_var
            if hasattr(self, 'psd_xi_var'):
                self.psd_xi_var.set(new_value)

            # Update Tab 4's display if it exists
            if hasattr(self, 'rms_target_xi_display'):
                self.rms_target_xi_display.set(f"{float(new_value):.4f}")

            # Update target_xi for calculations
            self.target_xi = float(new_value)
        except (ValueError, AttributeError):
            # Invalid value or variables not yet initialized
            pass

    def _refresh_preset_surface_q1_list(self):
        """Refresh the preset surface q1 list in the combobox."""
        try:
            preset_dir = self._get_preset_data_dir('surface_q1')
            files = [f.replace('.txt', '') for f in os.listdir(preset_dir) if f.endswith('.txt')]
            if files:
                self.surface_q1_combo['values'] = sorted(files)
            else:
                self.surface_q1_combo['values'] = ['(데이터 없음)']
        except Exception as e:
            print(f"[q1 프리셋] 목록 로드 오류: {e}")
            self.surface_q1_combo['values'] = ['(오류)']

    def _load_preset_surface_q1(self):
        """Load a preset surface q1 file and apply q_max, q1 values."""
        selected = self.surface_q1_var.get()
        if not selected or selected.startswith('('):
            self._show_status("로드할 프리셋을 선택하세요.", 'warning')
            return

        try:
            preset_dir = self._get_preset_data_dir('surface_q1')
            filepath = os.path.join(preset_dir, selected + '.txt')

            q_max_val = None
            q1_val = None
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('#') or not line:
                        continue
                    if line.startswith('q_max='):
                        q_max_val = line.split('=', 1)[1].strip()
                    elif line.startswith('q1='):
                        q1_val = line.split('=', 1)[1].strip()

            if q_max_val is None or q1_val is None:
                messagebox.showerror("오류", "프리셋 파일 형식이 올바르지 않습니다.")
                return

            # Apply q_max to the field
            self.q_max_var.set(q_max_val)
            # Apply q1 to the target q1 field
            self.input_q1_var.set(q1_val)
            # Switch to q1→h'rms mode
            self.hrms_q1_mode_var.set("q1_to_hrms")
            self._on_hrms_q1_mode_changed()
            # Auto-trigger q1→h'rms calculation
            try:
                self._calculate_hrms_q1()
            except Exception as e:
                print(f"[q1 프리셋] h'rms 자동 계산 건너뜀: {e}")
            self.status_var.set(f"프리셋 로드: {selected}, q_max={q_max_val}, q1={q1_val}")

        except Exception as e:
            messagebox.showerror("오류", f"프리셋 로드 실패:\n{str(e)}")

    def _delete_preset_surface_q1(self):
        """Delete selected preset surface q1 file."""
        selected = self.surface_q1_var.get()
        if not selected or selected.startswith('('):
            self._show_status("삭제할 프리셋을 선택하세요.", 'warning')
            return

        if not messagebox.askyesno("확인", f"'{selected}' 프리셋을 삭제하시겠습니까?"):
            return

        try:
            preset_dir = self._get_preset_data_dir('surface_q1')
            filepath = os.path.join(preset_dir, selected + '.txt')
            os.remove(filepath)
            self._refresh_preset_surface_q1_list()
            self.surface_q1_var.set("(선택...)")
            self._show_status(f"삭제 완료: {selected}", 'success')
        except Exception as e:
            messagebox.showerror("오류", f"삭제 실패:\n{str(e)}")

    def _add_preset_surface_q1(self):
        """Save current q_max and q1 values as a preset."""
        q_max_val = self.q_max_var.get().strip()
        q1_val = self.input_q1_var.get().strip()

        if not q_max_val or not q1_val:
            self._show_status("q_max와 목표 q1 값을 먼저 입력하세요.", 'warning')
            return

        # Validate that they are valid numbers
        try:
            float(q_max_val)
            float(q1_val)
        except ValueError:
            self._show_status("q_max와 q1에 유효한 숫자를 입력하세요.", 'warning')
            return

        from tkinter import simpledialog
        name = simpledialog.askstring("프리셋 저장", f"프리셋 이름을 입력하세요:\n(현재 q_max={q_max_val}, q1={q1_val})")
        if not name:
            return

        try:
            preset_dir = self._get_preset_data_dir('surface_q1')
            filepath = os.path.join(preset_dir, name + '.txt')

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# q_max/q1 프리셋: {name}\n")
                f.write(f"q_max={q_max_val}\n")
                f.write(f"q1={q1_val}\n")

            self._refresh_preset_surface_q1_list()
            self.surface_q1_var.set(name)
            self._show_status(f"프리셋 저장 완료: {name}\nq_max={q_max_val}, q1={q1_val}", 'success')

        except Exception as e:
            messagebox.showerror("오류", f"프리셋 저장 실패:\n{str(e)}")

    def _on_hrms_q1_mode_changed(self):
        """모드 변경 시 UI 상태 업데이트."""
        mode = self.hrms_q1_mode_var.get()
        if mode == "hrms_to_q1":
            # 모드 1: h'rms 입력 활성화, q1 입력 비활성화
            self.hrms_entry.config(state='normal')
            self.q1_entry.config(state='disabled')
            self.hrms_q1_calc_btn.config(text="h'rms → q1 계산")
        else:
            # 모드 2: q1 입력 활성화, h'rms 입력 비활성화
            self.hrms_entry.config(state='disabled')
            self.q1_entry.config(state='normal')
            self.hrms_q1_calc_btn.config(text="q1 → h'rms 계산")

    def _calculate_hrms_q1(self):
        """선택된 모드에 따라 h'rms(ξ) 또는 q1 계산.

        h'rms = ξ = RMS slope (경사), 무차원
        ξ²(q) = 2π ∫[q0→q] k³ C(k) dk
        """
        if self.psd_model is None:
            self._show_status("PSD 데이터를 먼저 로드해주세요!", 'warning')
            return

        try:
            mode = self.hrms_q1_mode_var.get()

            # PSD 데이터에서 q 범위 결정 - 실제 데이터 사용
            if hasattr(self.psd_model, 'q_data') and self.psd_model.q_data is not None:
                q_data = self.psd_model.q_data.copy()
                C_data = self.psd_model.C_data.copy()
                q_source = "실제 PSD 데이터"
            else:
                q_min = float(self.q_min_var.get())
                q_max = float(self.q_max_var.get())
                q_data = np.logspace(np.log10(q_min), np.log10(q_max), 500)
                C_data = self.psd_model(q_data)
                q_source = f"생성된 PSD (q: {q_min:.1e} ~ {q_max:.1e})"

            # 유효한 데이터만 사용
            valid = (q_data > 0) & (C_data > 0) & np.isfinite(q_data) & np.isfinite(C_data)
            q_data = q_data[valid]
            C_data = C_data[valid]

            if len(q_data) < 10:
                messagebox.showerror("오류", "유효한 PSD 데이터가 부족합니다.")
                return

            # 누적 h'rms(ξ) 계산: ξ²(q) = 2π∫[q0 to q] k³C(k)dk
            xi_squared_cumulative = np.zeros(len(q_data))
            for i in range(len(q_data)):
                q_int = q_data[:i+1]
                C_int = C_data[:i+1]
                xi_squared_cumulative[i] = 2 * np.pi * np.trapezoid(q_int**3 * C_int, q_int)
            xi_cumulative = np.sqrt(np.maximum(xi_squared_cumulative, 0))

            # 최대 도달 가능한 h'rms
            max_xi = xi_cumulative[-1]
            max_q = q_data[-1]
            min_q = q_data[0]

            if mode == "hrms_to_q1":
                # 모드 1: 주어진 h'rms(ξ)로 q1 계산
                target_xi = float(self.target_hrms_slope_var.get())

                # ξ 값이 도달 가능한지 확인
                if target_xi > max_xi:
                    self._show_status(f"목표 ξ ({target_xi:.4f})가 최대 도달 가능한 값 ({max_xi:.4f})보다 큽니다.\n"
                        f"PSD q 범위: {min_q:.2e} ~ {max_q:.2e} (1/m)\n\n"
                        f"q 범위를 늘리거나 목표 ξ를 줄이세요.", 'warning')
                    return

                if target_xi < xi_cumulative[0]:
                    self._show_status(f"목표 ξ ({target_xi:.4f})가 최소값 ({xi_cumulative[0]:.4f})보다 작습니다.", 'warning')
                    return

                # 목표 ξ에 해당하는 q1 찾기 (보간 사용)
                from scipy.interpolate import interp1d
                f_interp = interp1d(xi_cumulative, q_data, kind='linear', fill_value='extrapolate')
                q1_calculated = float(f_interp(target_xi))

                # 역검증: 계산된 q1에서 실제 xi 확인
                f_verify = interp1d(q_data, xi_cumulative, kind='linear', fill_value='extrapolate')
                xi_verified = float(f_verify(q1_calculated))

                # 결과 표시 - 둘 다 업데이트
                self.calculated_q1_var.set(f"{q1_calculated:.3e}")
                self.calculated_hrms_var.set(f"{xi_verified:.4f} (검증)")
                self.calculated_q1 = q1_calculated
                self.target_xi = target_xi

                self.status_var.set(f"계산 완료: ξ={target_xi:.4f} → q1={q1_calculated:.3e} (1/m)")
                self._show_status(f"모드 1: h'rms (ξ) → q1 계산\n\n"
                    f"[입력]\n"
                    f"  목표 ξ (h'rms): {target_xi:.4f}\n\n"
                    f"[출력]\n"
                    f"  계산된 q1: {q1_calculated:.3e} (1/m)\n"
                    f"  검증 ξ: {xi_verified:.4f}\n\n"
                    f"[PSD 정보]\n"
                    f"  {q_source}\n"
                    f"  q 범위: {min_q:.2e} ~ {max_q:.2e}\n"
                    f"  최대 가능 ξ: {max_xi:.4f}\n\n"
                    f"※ ξ² = 2π∫k³C(k)dk", 'success')

            else:
                # 모드 2: 주어진 q1로 h'rms(ξ) 계산
                target_q1 = float(self.input_q1_var.get())

                # q1이 범위 내에 있는지 확인
                if target_q1 < min_q or target_q1 > max_q:
                    self._show_status(f"입력 q1 ({target_q1:.3e})이 PSD 데이터 범위 밖입니다.\n"
                        f"범위: {min_q:.3e} ~ {max_q:.3e} (1/m)", 'warning')
                    return

                # q1에 해당하는 ξ 찾기 (보간 사용)
                from scipy.interpolate import interp1d
                f_interp = interp1d(q_data, xi_cumulative, kind='linear', fill_value='extrapolate')
                xi_calculated = float(f_interp(target_q1))

                # 결과 표시 - 둘 다 업데이트
                self.calculated_hrms_var.set(f"{xi_calculated:.4f}")
                self.calculated_q1_var.set(f"{target_q1:.3e} (입력)")
                self.calculated_q1 = target_q1
                self.target_xi = xi_calculated

                # ξ 입력란에도 반영
                self.target_hrms_slope_var.set(f"{xi_calculated:.4f}")

                self.status_var.set(f"계산 완료: q1={target_q1:.3e} → ξ={xi_calculated:.4f}")
                self._show_status(f"모드 2: q1 → h'rms (ξ) 계산\n\n"
                    f"[입력]\n"
                    f"  목표 q1: {target_q1:.3e} (1/m)\n\n"
                    f"[출력]\n"
                    f"  계산된 ξ (h'rms): {xi_calculated:.4f}\n\n"
                    f"[PSD 정보]\n"
                    f"  {q_source}\n"
                    f"  q 범위: {min_q:.2e} ~ {max_q:.2e}\n"
                    f"  최대 가능 ξ: {max_xi:.4f}\n\n"
                    f"※ ξ² = 2π∫k³C(k)dk", 'success')

        except ValueError as e:
            messagebox.showerror("오류", f"입력값이 유효하지 않습니다: {e}")
        except Exception as e:
            messagebox.showerror("오류", f"계산 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()

    def _send_hrms_q1_to_tab4(self):
        """계산된 h'rms(ξ)와 q1을 Tab 4로 전달."""
        try:
            # 계산된 q1이 있으면 표시만 (q_max는 사용자가 직접 설정하므로 덮어쓰지 않음)
            # if hasattr(self, 'calculated_q1') and self.calculated_q1 is not None:
            #     self.rms_q_max_var.set(f"{self.calculated_q1:.3e}")

            # target_xi를 Tab 4에 전달
            if self.target_xi is not None:
                if hasattr(self, 'rms_target_xi_display'):
                    self.rms_target_xi_display.set(f"{self.target_xi:.4f}")
                if hasattr(self, 'psd_xi_var'):
                    self.psd_xi_var.set(f"{self.target_xi:.4f}")

            # Tab 4로 전환
            self.notebook.select(4)

            self.status_var.set(f"Tab 4로 전달 완료: ξ={self.target_xi:.4f}, q1={self.calculated_q1:.3e}")
            self._show_status(f"Tab 4로 전달되었습니다.\n\n"
                f"ξ (h'rms): {self.target_xi:.4f}\n"
                f"q1: {self.calculated_q1:.3e} (1/m)\n\n"
                f"Tab 4에서 'h'rms slope 계산' 버튼을 클릭하세요.", 'success')

        except Exception as e:
            messagebox.showerror("오류", f"Tab 4로 전달 중 오류: {e}")

    def _create_results_tab(self, parent):
        """Create G(q,v) results tab."""
        # Toolbar
        self._create_panel_toolbar(parent, buttons=[
            ("결과 그래프 저장", lambda: self._save_plot(self.fig_results, "results_plot"), 'TButton'),
        ])

        # Instruction
        instruction = ttk.LabelFrame(parent, text="탭 설명", padding=10)
        instruction.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(instruction, text=
            "G(q,v) 2D 행렬 계산 결과: 다중 속도 G(q) 곡선, 히트맵, 접촉 면적.\n"
            "모든 속도가 컬러 코딩되어 하나의 그래프에 표시됩니다.",
            font=('Segoe UI', 17)
        ).pack()

        # Plot area
        plot_frame = ttk.Frame(parent)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.fig_results = Figure(figsize=(16, 10), dpi=100)
        self.canvas_results = FigureCanvasTkAgg(self.fig_results, plot_frame)
        self.canvas_results.draw()
        self.canvas_results.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas_results, plot_frame)
        toolbar.update()

        # Save button
        save_btn_frame = ttk.Frame(parent)
        save_btn_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(
            save_btn_frame,
            text="결과 그래프 저장",
            command=lambda: self._save_plot(self.fig_results, "results_plot")
        ).pack(side=tk.LEFT, padx=5)

    def _create_log_panel(self):
        """Create an activity log panel at the top of the main window."""
        C = self.COLORS
        log_container = tk.Frame(self.root, bg=C['sidebar'], height=120)
        log_container.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(4, 0))
        log_container.pack_propagate(False)

        # Title bar
        title_bar = tk.Frame(log_container, bg=C['sidebar'], height=22)
        title_bar.pack(fill=tk.X, side=tk.TOP)
        title_bar.pack_propagate(False)
        tk.Label(title_bar, text="\u25A0 \uc791\uc5c5 \ub85c\uadf8",
                 bg=C['sidebar'], fg='#94A3B8',
                 font=('Segoe UI', 14, 'bold')).pack(side=tk.LEFT, padx=6)

        # Toggle button to expand/collapse
        self._log_expanded = True
        self._log_container = log_container

        def _toggle_log():
            if self._log_expanded:
                log_container.config(height=22)
                toggle_btn.config(text="\u25BC")
                self._log_expanded = False
            else:
                log_container.config(height=120)
                toggle_btn.config(text="\u25B2")
                self._log_expanded = True

        toggle_btn = tk.Button(title_bar, text="\u25B2", bg=C['sidebar'], fg='#94A3B8',
                               font=('Segoe UI', 10), bd=0, command=_toggle_log,
                               activebackground=C['sidebar'], activeforeground='#E2E8F0',
                               cursor='hand2')
        toggle_btn.pack(side=tk.RIGHT, padx=6)

        # Clear button
        def _clear_log():
            self._log_text.config(state='normal')
            self._log_text.delete('1.0', tk.END)
            self._log_text.config(state='disabled')

        clear_btn = tk.Button(title_bar, text="\uc9c0\uc6b0\uae30", bg=C['sidebar'], fg='#94A3B8',
                              font=('Segoe UI', 12), bd=0, command=_clear_log,
                              activebackground=C['sidebar'], activeforeground='#E2E8F0',
                              cursor='hand2')
        clear_btn.pack(side=tk.RIGHT, padx=4)

        # Log text area
        log_frame = tk.Frame(log_container, bg=C['sidebar'])
        log_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))

        self._log_text = tk.Text(log_frame, bg='#0F172A', fg='#CBD5E1',
                                 font=('Consolas', 14), wrap=tk.WORD,
                                 bd=0, highlightthickness=0,
                                 state='disabled', cursor='arrow')
        log_scrollbar = ttk.Scrollbar(log_frame, orient='vertical',
                                      command=self._log_text.yview)
        self._log_text.config(yscrollcommand=log_scrollbar.set)

        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Tag colors for different log levels
        self._log_text.tag_configure('info', foreground='#93C5FD')
        self._log_text.tag_configure('success', foreground='#6EE7B7')
        self._log_text.tag_configure('warning', foreground='#FCD34D')
        self._log_text.tag_configure('error', foreground='#FCA5A5')
        self._log_text.tag_configure('timestamp', foreground='#475569')

    def _append_log(self, message, level='info'):
        """Append a message to the activity log panel."""
        if not hasattr(self, '_log_text'):
            return
        import datetime
        ts = datetime.datetime.now().strftime('%H:%M:%S')
        level_prefix = {'info': 'INFO', 'success': '\u2714 OK',
                        'warning': '\u26A0 WARN', 'error': '\u2716 ERR'}
        prefix = level_prefix.get(level, 'INFO')
        clean_msg = message.replace('\n', ' | ').strip()

        self._log_text.config(state='normal')
        self._log_text.insert(tk.END, f"[{ts}] ", 'timestamp')
        self._log_text.insert(tk.END, f"{prefix}: {clean_msg}\n", level)
        self._log_text.see(tk.END)
        self._log_text.config(state='disabled')

    def _create_status_bar(self):
        """Create modern status bar with colored level indicators."""
        C = self.COLORS
        self.status_var = tk.StringVar(value="Ready")
        self._status_clear_id = None  # For auto-clear timer
        self._status_var_from_show = False  # Flag to avoid double logging

        # Trace status_var writes to auto-log direct .set() calls
        def _on_status_var_write(*args):
            if self._status_var_from_show:
                return
            msg = self.status_var.get()
            if msg and msg != "Ready":
                self._append_log(msg, 'info')
        self.status_var.trace_add('write', _on_status_var_write)

        status_frame = tk.Frame(self.root, bg=C['statusbar_bg'], height=36)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        status_frame.pack_propagate(False)

        # Status indicator dot (color changes by level)
        self._status_dot = tk.Label(status_frame, text="\u25CF", bg=C['statusbar_bg'],
                 fg=C['success'], font=('Segoe UI', 16))
        self._status_dot.pack(side=tk.LEFT, padx=(12, 4))

        # Status text (color changes by level)
        self._status_label = tk.Label(status_frame, textvariable=self.status_var,
                 bg=C['statusbar_bg'], fg=C['statusbar_fg'],
                 font=self.FONTS['small'], anchor=tk.W)
        self._status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Version badge
        tk.Label(status_frame, text="Persson Model v3.0",
                 bg=C['statusbar_bg'], fg='#475569',
                 font=self.FONTS['tiny']).pack(side=tk.RIGHT, padx=12)

    def _show_status(self, message, level='info', duration=8000):
        """Show a status message in the status bar and activity log.

        Args:
            message: Status message text (newlines replaced with ' | ')
            level: 'info', 'success', 'warning', or 'error'
            duration: Auto-clear duration in ms (0 = no auto-clear)
        """
        # Color mapping for status levels
        level_colors = {
            'info':    ('#3B82F6', '#93C5FD'),   # blue dot, light blue text
            'success': ('#059669', '#6EE7B7'),   # green dot, light green text
            'warning': ('#D97706', '#FCD34D'),   # orange dot, yellow text
            'error':   ('#DC2626', '#FCA5A5'),   # red dot, light red text
        }

        dot_color, text_color = level_colors.get(level, level_colors['info'])

        # Clean up message (replace newlines with separator)
        clean_msg = message.replace('\n', ' | ').strip()

        # Update status bar (flag to prevent double logging from trace)
        self._status_var_from_show = True
        self.status_var.set(clean_msg)
        self._status_var_from_show = False
        if hasattr(self, '_status_dot'):
            self._status_dot.config(fg=dot_color)
        if hasattr(self, '_status_label'):
            self._status_label.config(fg=text_color)

        # Append to activity log panel
        self._append_log(message, level)

        # Cancel previous auto-clear timer
        if self._status_clear_id is not None:
            try:
                self.root.after_cancel(self._status_clear_id)
            except Exception:
                pass

        # Auto-clear after duration
        if duration > 0:
            def _clear():
                self.status_var.set("Ready")
                if hasattr(self, '_status_dot'):
                    self._status_dot.config(fg=self.COLORS['success'])
                if hasattr(self, '_status_label'):
                    self._status_label.config(fg=self.COLORS['statusbar_fg'])
                self._status_clear_id = None
            self._status_clear_id = self.root.after(duration, _clear)

    def _update_material_display(self):
        """Update material information (if needed)."""
        pass  # Simplified for now

    def _load_material(self):
        """Load DMA data from file."""
        filename = filedialog.askopenfilename(
            title="Select DMA Data File",
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if filename:
            try:
                omega_raw, E_storage_raw, E_loss_raw = load_dma_from_file(
                    filename, skip_header=1, freq_unit='Hz', modulus_unit='MPa'
                )

                # Apply smoothing/fitting
                smoothed = smooth_dma_data(omega_raw, E_storage_raw, E_loss_raw)

                # Store raw data for visualization
                self.raw_dma_data = {
                    'omega': omega_raw,
                    'E_storage': E_storage_raw,
                    'E_loss': E_loss_raw
                }

                # Create material from smoothed data
                self.material = create_material_from_dma(
                    omega=smoothed['omega'],
                    E_storage=smoothed['E_storage_smooth'],
                    E_loss=smoothed['E_loss_smooth'],
                    material_name=os.path.splitext(os.path.basename(filename))[0] + " (smoothed)",
                    reference_temp=float(self.temperature_var.get())
                )

                self._show_status(f"DMA data loaded and smoothed: {len(omega_raw)} points", 'success')

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load DMA data:\n{str(e)}")

    def _load_psd_data(self):
        """Load PSD data from file."""
        filename = filedialog.askopenfilename(
            title="Select PSD Data File",
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if filename:
            try:
                # Auto-detect format (is_log_data=None)
                q, C_q = load_psd_from_file(filename, skip_header=1, is_log_data=None)

                # Validate the loaded data
                if len(q) == 0:
                    raise ValueError("No valid data points found in file")

                if np.any(q <= 0) or np.any(C_q <= 0):
                    raise ValueError("Invalid data: q and C must be positive after conversion")

                # Store raw PSD data for comparison plotting
                self.raw_psd_data = {
                    'q': q.copy(),
                    'C_q': C_q.copy()
                }

                self.psd_model = create_psd_from_data(q, C_q, interpolation_kind='log-log')

                # q_min/q_max는 사용자가 의도적으로 설정할 수 있으므로 자동 덮어쓰기 하지 않음
                # (초기값이 기본값인 경우에만 업데이트)
                self.psd_type_var.set("measured")

                # Show info about loaded data
                self._show_status(f"PSD data loaded: {len(q)} points\n"
                    f"q 범위: {q[0]:.2e} ~ {q[-1]:.2e} 1/m\n"
                    f"C(q) 범위: {C_q.min():.2e} ~ {C_q.max():.2e} m⁴", 'success')

            except Exception as e:
                import traceback
                messagebox.showerror("Error", f"Failed to load PSD data:\n{str(e)}\n\n{traceback.format_exc()}")

    def _import_from_master_curve(self):
        """Import DMA data from master curve tab (Tab 0)."""
        if self.master_curve_gen is None or self.master_curve_gen.master_f is None:
            self._show_status("먼저 Tab 0에서 마스터 커브를 생성하세요.", 'warning')
            return

        try:
            # Get master curve data
            omega = 2 * np.pi * self.master_curve_gen.master_f
            E_storage = self.master_curve_gen.master_E_storage * 1e6  # MPa to Pa
            E_loss = self.master_curve_gen.master_E_loss * 1e6  # MPa to Pa
            T_ref = self.master_curve_gen.T_ref

            # Create material from master curve
            self.material = create_material_from_dma(
                omega=omega,
                E_storage=E_storage,
                E_loss=E_loss,
                material_name=f"Master Curve (Tref={T_ref}°C)",
                reference_temp=T_ref
            )

            # Store raw data for plotting
            self.raw_dma_data = {
                'omega': omega,
                'E_storage': E_storage,
                'E_loss': E_loss
            }

            # Store shift factor data
            self.master_curve_shift_factors = {
                'aT': self.master_curve_gen.aT.copy(),
                'bT': self.master_curve_gen.bT.copy(),
                'T_ref': T_ref,
                'C1': self.master_curve_gen.C1,
                'C2': self.master_curve_gen.C2,
                'temperatures': list(self.master_curve_gen.temperatures)
            }

            # Update temperature entry
            self.temperature_var.set(str(T_ref))

            # Update status
            f_min = self.master_curve_gen.master_f.min()
            f_max = self.master_curve_gen.master_f.max()
            self.dma_import_status_var.set(f"Master Curve (Tref={T_ref}°C, {f_min:.1e}~{f_max:.1e} Hz)")

            # Update plots
            self._update_material_display()

            self.status_var.set(f"마스터 커브 가져오기 완료")

        except Exception as e:
            messagebox.showerror("Error", f"스무딩 적용 실패:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def _apply_dma_smoothing_extrapolation(self):
        """Apply smoothing and/or extrapolation to DMA data in verification tab."""
        if self.raw_dma_data is None:
            self._show_status("먼저 DMA 데이터를 불러오세요.", 'warning')
            return

        try:
            from scipy.signal import savgol_filter
            from scipy.interpolate import interp1d

            # Get raw data
            omega_raw = self.raw_dma_data['omega'].copy()
            E_storage_raw = self.raw_dma_data['E_storage'].copy()
            E_loss_raw = self.raw_dma_data['E_loss'].copy()

            # Sort by omega
            sort_idx = np.argsort(omega_raw)
            omega_sorted = omega_raw[sort_idx]
            E_storage_sorted = E_storage_raw[sort_idx]
            E_loss_sorted = E_loss_raw[sort_idx]

            # Apply smoothing if enabled
            if self.verify_smooth_var.get():
                window = self.verify_smooth_window_var.get()
                if window % 2 == 0:
                    window += 1  # Must be odd
                window = min(window, len(omega_sorted) - 1)
                if window < 5:
                    window = 5
                if len(omega_sorted) > window:
                    # Use log scale for smoothing
                    log_E_storage = np.log10(np.maximum(E_storage_sorted, 1e-10))
                    log_E_loss = np.log10(np.maximum(E_loss_sorted, 1e-10))
                    log_E_storage_smooth = savgol_filter(log_E_storage, window, 3)
                    log_E_loss_smooth = savgol_filter(log_E_loss, window, 3)
                    E_storage_smooth = 10**log_E_storage_smooth
                    E_loss_smooth = 10**log_E_loss_smooth
                else:
                    E_storage_smooth = E_storage_sorted
                    E_loss_smooth = E_loss_sorted
            else:
                E_storage_smooth = E_storage_sorted
                E_loss_smooth = E_loss_sorted

            # Apply extrapolation if enabled
            if self.verify_extrap_var.get():
                # Get user-specified frequency range
                f_min = float(self.dma_extrap_fmin_var.get())
                f_max = float(self.dma_extrap_fmax_var.get())
                omega_min_target = 2 * np.pi * f_min
                omega_max_target = 2 * np.pi * f_max

                # Create extended omega array
                omega_extended = np.logspace(
                    np.log10(omega_min_target),
                    np.log10(omega_max_target),
                    500
                )

                # Create interpolators (log-log space)
                log_omega = np.log10(omega_sorted)
                log_E_storage = np.log10(np.maximum(E_storage_smooth, 1e-10))
                log_E_loss = np.log10(np.maximum(E_loss_smooth, 1e-10))

                interp_storage = interp1d(log_omega, log_E_storage, kind='linear',
                                         fill_value='extrapolate')
                interp_loss = interp1d(log_omega, log_E_loss, kind='linear',
                                      fill_value='extrapolate')

                # Extrapolate
                log_omega_ext = np.log10(omega_extended)
                log_E_storage_ext = interp_storage(log_omega_ext)
                log_E_loss_ext = interp_loss(log_omega_ext)

                # Linear extrapolation in log-log space using edge slopes
                # For low frequencies: extrapolate using slope from first few points
                low_mask = log_omega_ext < log_omega.min()
                if np.any(low_mask):
                    # Use first 10 points (or less if not enough data) to estimate slope
                    n_fit = min(10, len(log_omega) // 4, len(log_omega) - 1)
                    if n_fit >= 2:
                        slope_storage_low = (log_E_storage[n_fit] - log_E_storage[0]) / (log_omega[n_fit] - log_omega[0])
                        slope_loss_low = (log_E_loss[n_fit] - log_E_loss[0]) / (log_omega[n_fit] - log_omega[0])
                        delta_omega = log_omega_ext[low_mask] - log_omega[0]
                        log_E_storage_ext[low_mask] = log_E_storage[0] + slope_storage_low * delta_omega
                        log_E_loss_ext[low_mask] = log_E_loss[0] + slope_loss_low * delta_omega

                # For high frequencies: extrapolate using slope from last few points
                high_mask = log_omega_ext > log_omega.max()
                if np.any(high_mask):
                    # Use last 10 points (or less if not enough data) to estimate slope
                    n_fit = min(10, len(log_omega) // 4, len(log_omega) - 1)
                    if n_fit >= 2:
                        slope_storage_high = (log_E_storage[-1] - log_E_storage[-n_fit-1]) / (log_omega[-1] - log_omega[-n_fit-1])
                        slope_loss_high = (log_E_loss[-1] - log_E_loss[-n_fit-1]) / (log_omega[-1] - log_omega[-n_fit-1])
                        delta_omega = log_omega_ext[high_mask] - log_omega[-1]
                        log_E_storage_ext[high_mask] = log_E_storage[-1] + slope_storage_high * delta_omega
                        log_E_loss_ext[high_mask] = log_E_loss[-1] + slope_loss_high * delta_omega

                omega_final = omega_extended
                E_storage_final = 10**log_E_storage_ext
                E_loss_final = 10**log_E_loss_ext
            else:
                omega_final = omega_sorted
                E_storage_final = E_storage_smooth
                E_loss_final = E_loss_smooth

            # Update material
            self.material = create_material_from_dma(
                omega=omega_final,
                E_storage=E_storage_final,
                E_loss=E_loss_final,
                material_name=self.material.name if self.material else "Processed DMA",
                reference_temp=float(self.temperature_var.get())
            )

            # Update status
            f_min = omega_final.min() / (2 * np.pi)
            f_max = omega_final.max() / (2 * np.pi)
            smooth_str = f"스무딩(w={self.verify_smooth_window_var.get()})" if self.verify_smooth_var.get() else ""
            extrap_str = "외삽" if self.verify_extrap_var.get() else ""
            process_str = "+".join(filter(None, [smooth_str, extrap_str])) or "원본"
            self.dma_import_status_var.set(f"처리됨 [{process_str}] ({f_min:.1e}~{f_max:.1e} Hz)")

            # Update status
            self.status_var.set("DMA 스무딩/외삽 적용 완료")

            self._show_status(f"DMA 데이터 처리 완료\n- 스무딩: {'적용' if self.verify_smooth_var.get() else '미적용'}\n- 외삽: {'적용' if self.verify_extrap_var.get() else '미적용'}\n- 주파수 범위: {f_min:.1e} ~ {f_max:.1e} Hz", 'success')

        except Exception as e:
            messagebox.showerror("Error", f"스무딩/외삽 적용 실패:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def _calc_Cq0_from_xi(self):
        """Calculate C(q0) from target h'rms (ξ).

        Formula derivation:
        ξ² = 2π ∫[q0→q1] q³ C(q) dq
        For power-law PSD: C(q) = C(q0) * (q/q0)^(-2(H+1))

        ξ² = 2π * C(q0) * q0^(2(H+1)) * [q^(2-2H) / (2-2H)]_{q0}^{q1}
           = 2π * C(q0) * q0^(2(H+1)) * (q1^(2-2H) - q0^(2-2H)) / (2-2H)

        Therefore:
        C(q0) = ξ² * (2-2H) / (2π * q0^(2(H+1)) * (q1^(2-2H) - q0^(2-2H)))
        """
        try:
            q0 = float(self.psd_q0_var.get())
            q1 = float(self.psd_q1_var.get())
            H = float(self.psd_H_var.get())
            xi_target = float(self.psd_xi_var.get())

            if q1 <= q0:
                messagebox.showerror("Error", "q1 must be greater than q0")
                return

            # Calculate C(q0) from target ξ
            exp_factor = 2 - 2 * H
            if abs(exp_factor) < 1e-10:
                # Special case H ≈ 1
                integral_factor = np.log(q1 / q0)
            else:
                integral_factor = (q1**exp_factor - q0**exp_factor) / exp_factor

            # ξ² = 2π * C(q0) * q0^(2(H+1)) * integral_factor
            # C(q0) = ξ² / (2π * q0^(2(H+1)) * integral_factor)
            C_q0 = xi_target**2 / (2 * np.pi * q0**(2*(H+1)) * integral_factor)

            # Update C(q0) entry
            self.psd_Cq0_var.set(f"{C_q0:.3e}")
            self.status_var.set(f"C(q0) = {C_q0:.3e} calculated for ξ = {xi_target}")

        except Exception as e:
            messagebox.showerror("Error", f"C(q0) 계산 실패:\n{str(e)}")

    def _apply_psd_settings(self):
        """Apply PSD power-law settings from user input."""
        try:
            from scipy.interpolate import interp1d

            # Get user parameters
            q0 = float(self.psd_q0_var.get())
            q1 = float(self.psd_q1_var.get())
            H = float(self.psd_H_var.get())
            C_q0 = float(self.psd_Cq0_var.get())

            if q1 <= q0:
                messagebox.showerror("Error", "q1 must be greater than q0")
                return

            if H < 0 or H > 1:
                self._show_status("Hurst exponent H should be between 0 and 1", 'warning')

            # Create power-law PSD: C(q) = C(q0) * (q/q0)^(-2(H+1))
            # Power law exponent: -2(H+1)
            exponent = -2 * (H + 1)

            # Create q array for power law region (q0 to q1)
            q_powerlaw = np.logspace(np.log10(q0), np.log10(q1), 500)
            C_powerlaw = C_q0 * (q_powerlaw / q0) ** exponent

            # Determine minimum q for plateau region
            # Use raw PSD data range if available, otherwise use q0/100
            if self.raw_psd_data is not None:
                q_min_plateau = min(self.raw_psd_data['q'])
            else:
                q_min_plateau = q0 / 100

            # Create plateau region (q < q0) with constant C(q0)
            if q_min_plateau < q0:
                q_plateau = np.logspace(np.log10(q_min_plateau), np.log10(q0), 100)[:-1]  # exclude q0 to avoid duplicate
                C_plateau = np.full_like(q_plateau, C_q0)

                # Combine plateau and power law regions
                q_array = np.concatenate([q_plateau, q_powerlaw])
                C_array = np.concatenate([C_plateau, C_powerlaw])
            else:
                q_array = q_powerlaw
                C_array = C_powerlaw

            # Create interpolator for PSD model
            log_q = np.log10(q_array)
            log_C = np.log10(C_array)

            # Store q0 and C_q0 for the model function
            _q0 = q0
            _C_q0 = C_q0
            _exponent = exponent

            def psd_model(q_input):
                """Power-law PSD model with plateau for q < q0."""
                q_input = np.atleast_1d(q_input)

                C_out = np.empty_like(q_input)

                # q < q0: plateau at C(q0)
                mask_plateau = q_input < _q0
                C_out[mask_plateau] = _C_q0

                # q >= q0: power law
                mask_powerlaw = ~mask_plateau
                C_out[mask_powerlaw] = _C_q0 * (q_input[mask_powerlaw] / _q0) ** _exponent

                return C_out

            # Store the PSD model with q_data and C_data attributes for RMS slope calculation
            self.psd_model = psd_model
            # Add attributes to function object so RMS slope calculation uses correct q range
            # IMPORTANT: q_data/C_data now include plateau region (q < q0) for complete integration
            self.psd_model.q_data = q_array.copy()  # Full range including plateau
            self.psd_model.C_data = C_array.copy()  # Full range including plateau
            # Also store full range for plotting (same as q_data/C_data now)
            self.psd_model.q_full = q_array.copy()
            self.psd_model.C_full = C_array.copy()
            # Store power law region separately for reference
            self.psd_model.q_powerlaw = q_powerlaw.copy()
            self.psd_model.C_powerlaw = C_powerlaw.copy()
            self.psd_model.q0 = q0
            self.psd_model.C_q0 = C_q0

            # q_min/q_max는 사용자가 의도적으로 설정할 수 있으므로 자동 덮어쓰기 하지 않음
            # self.q_min_var.set(str(q0))
            # self.q_max_var.set(str(q1))

            # Calculate actual RMS slope from the PSD for verification
            # ξ² = 2π ∫ q³ C(q) dq
            q_calc = np.logspace(np.log10(q0), np.log10(q1), 1000)
            C_calc = psd_model(q_calc)
            integrand = q_calc**3 * C_calc
            xi_squared = 2 * np.pi * np.trapezoid(integrand, q_calc)
            xi_actual = np.sqrt(xi_squared)

            # Use user's input ξ value from Tab 2 directly as target_xi
            # (The user specified ξ, calculated C(q0) from it, so target ξ is the input value)
            try:
                xi_user_input = float(self.psd_xi_var.get())
            except:
                xi_user_input = xi_actual

            # Store user's target xi for consistency with Tab 4 (RMS Slope)
            self.target_xi = xi_user_input
            self.psd_model.target_xi = xi_user_input

            # Update status
            self.status_var.set(f"PSD applied: ξ(target)={xi_user_input:.3f}, ξ(calc)={xi_actual:.3f}, H={H:.2f}")

            # Show both target and calculated ξ for transparency
            xi_diff_pct = abs(xi_user_input - xi_actual) / xi_user_input * 100 if xi_user_input > 0 else 0
            xi_info = f"- Target h'rms ξ = {xi_user_input:.4f}\n- Calculated h'rms ξ = {xi_actual:.4f}"
            if xi_diff_pct > 1:
                xi_info += f"\n  (차이: {xi_diff_pct:.1f}% - 수치 적분 오차)"

            self._show_status(f"PSD model applied:\n"
                              f"- q range: {q0:.1e} ~ {q1:.1e} 1/m\n"
                              f"- Hurst exponent H: {H:.3f}\n"
                              f"- C(q0): {C_q0:.1e} m^4\n"
                              f"- Power law: C(q) = C(q0)*(q/q0)^{exponent:.2f}\n"
                              f"{xi_info}", 'success')

        except Exception as e:
            messagebox.showerror("Error", f"PSD settings failed:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def _get_mc_prefix(self):
        """Return a filename prefix based on the loaded master curve file name."""
        if hasattr(self, 'persson_master_curve') and self.persson_master_curve is not None:
            fname = self.persson_master_curve.get('filename', '')
            if fname:
                base = os.path.splitext(fname)[0]
                # Sanitize for filename use
                base = base.replace(' ', '_').replace('/', '_').replace('\\', '_')
                return base
        if hasattr(self, 'material_source') and self.material_source:
            src = str(self.material_source)
            # Extract short name from source string
            for prefix in ['Persson 정품: ', 'Persson 정품 (', '기본 파일 (', 'Tab 1 마스터 커브']:
                if src.startswith(prefix):
                    src = src[len(prefix):].rstrip(')')
                    break
            src = src.replace(' ', '_').replace('/', '_').replace('\\', '_')
            if len(src) > 30:
                src = src[:30]
            return src
        return ""

    def _make_export_filename(self, base_name, ext=".csv"):
        """Create export filename with master curve prefix included."""
        mc = self._get_mc_prefix()
        if mc:
            return f"{mc}_{base_name}{ext}"
        return f"{base_name}{ext}"

    def _save_plot(self, fig, default_name):
        """Save matplotlib figure to file."""
        mc_name = self._make_export_filename(default_name, ext=".png")
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=mc_name,
            filetypes=[
                ("PNG files", "*.png"),
                ("PDF files", "*.pdf"),
                ("SVG files", "*.svg"),
                ("JPEG files", "*.jpg"),
                ("All files", "*.*")
            ]
        )

        if filename:
            try:
                fig.savefig(filename, dpi=300, bbox_inches='tight')
                self._show_status(f"그래프가 저장되었습니다:\n{filename}", 'success')
                self.status_var.set(f"그래프 저장 완료: {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"그래프 저장 실패:\n{str(e)}")

    def _run_calculation(self):
        """Run G(q,v) 2D calculation."""
        # Check if PSD has been set from Tab 0
        tab0_ready = getattr(self, 'tab0_finalized', False)
        if not tab0_ready or self.psd_model is None:
            self._show_status("PSD 데이터가 설정되지 않았습니다!\n\n"
                "Tab 0 (PSD 생성)에서 PSD를 확정한 후\n"
                "'PSD 확정 → Tab 3' 버튼을 클릭하세요.", 'warning')
            return

        # Check if Master Curve has been set from Tab 1
        tab1_ready = getattr(self, 'tab1_finalized', False)
        if not tab1_ready or self.material is None:
            self._show_status("마스터 커브 데이터가 설정되지 않았습니다!\n\n"
                "Tab 1 (마스터 커브 생성)에서 마스터 커브를 확정한 후\n"
                "'마스터 커브 확정 → Tab 3' 버튼을 클릭하세요.", 'warning')
            return

        try:
            self.status_var.set("Calculating G(q,v)...")
            self.calc_button.config(state='disabled')
            self.root.update()

            # Get parameters
            sigma_0 = float(self.sigma_0_var.get()) * 1e6  # MPa to Pa
            v_min = float(self.v_min_var.get())
            v_max = float(self.v_max_var.get())
            n_v = int(self.n_velocity_var.get())
            q_min = float(self.q_min_var.get())
            q_max = float(self.q_max_var.get())
            n_q = int(self.n_q_var.get())
            poisson = float(self.poisson_var.get())
            temperature = float(self.temperature_var.get())

            # Create arrays
            v_array = np.logspace(np.log10(v_min), np.log10(v_max), n_v)
            q_array = np.logspace(np.log10(q_min), np.log10(q_max), n_q)

            # Create G calculator (φ 적분 포인트는 계산 설정 탭에서 가져옴)
            n_phi_gq = int(self.n_phi_gq_var.get()) if hasattr(self, 'n_phi_gq_var') else 36
            self.g_calculator = GCalculator(
                psd_func=self.psd_model,
                modulus_func=lambda w: self.material.get_modulus(w, temperature=temperature),
                sigma_0=sigma_0,
                velocity=v_array[0],  # Initial velocity
                poisson_ratio=poisson,
                n_angle_points=n_phi_gq,
                integration_method='trapz'
            )

            # GUI에서 설정한 norm factor 적용
            try:
                self.g_calculator.PSD_NORMALIZATION_FACTOR = float(self.g_norm_factor_var.get())
            except (ValueError, AttributeError):
                self.g_calculator.PSD_NORMALIZATION_FACTOR = 1.5625

            # Initialize calculation progress plots (2x2 layout)
            try:
                # Clear all four subplots
                self.ax_psd_q.clear()
                self.ax_dma_progress.clear()
                self.ax_gq_live.clear()
                self.ax_contact_live.clear()

                # Colour map for velocity-indexed curves
                _n_v_total = len(v_array)
                _cmap = plt.get_cmap('plasma')

                # TOP-LEFT: PSD(q) with integration range
                if self.psd_model is not None:
                    if hasattr(self.psd_model, 'q_data') and len(self.psd_model.q_data) > 0:
                        q_plot_min = min(self.psd_model.q_data)
                        q_plot_max = max(self.psd_model.q_data[self.psd_model.q_data <= q_max]) if np.any(self.psd_model.q_data <= q_max) else q_max
                        q_plot_max = max(q_plot_max, q_max)
                    elif hasattr(self.psd_model, 'q0'):
                        q_plot_min = self.psd_model.q0 / 10
                        q_plot_max = q_max
                    else:
                        q_plot_min = q_min
                        q_plot_max = q_max

                    q_plot = np.logspace(np.log10(q_plot_min), np.log10(q_plot_max), 300)
                    C_q = self.psd_model(q_plot)
                    self.ax_psd_q.loglog(q_plot, C_q, color='#3B82F6', linewidth=2.5, label='C(q)')

                    if hasattr(self.psd_model, 'q0'):
                        q0_psd = self.psd_model.q0
                        if q_plot_min < q0_psd:
                            self.ax_psd_q.axvspan(q_plot_min, q0_psd, alpha=0.10, facecolor='#F59E0B',
                                                 label=r'플래토 (q<q$_0$)')
                            self.ax_psd_q.axvline(x=q0_psd, color='#F59E0B', linestyle='--', linewidth=1.2, alpha=0.6)

                    # Integration range shading
                    self.ax_psd_q.axvspan(q_min, q_max, alpha=0.08, facecolor='#06B6D4',
                                         edgecolor='#0891B2', linewidth=1.5, label='적분 q 범위')
                    self.ax_psd_q.set_xlabel('파수 q (1/m)', fontweight='bold')
                    self.ax_psd_q.set_ylabel(r'C(q) (m$^4$)', fontweight='bold')
                    self.ax_psd_q.set_xscale('log'); self.ax_psd_q.set_yscale('log')
                    self.ax_psd_q.grid(True, alpha=0.25, linewidth=0.5)
                    self.ax_psd_q.legend(loc='upper right', fontsize=12)
                    self.ax_psd_q.set_title('PSD C(q) — 적분 범위 표시', fontweight='bold')

                # TOP-RIGHT: DMA master curve
                if self.material is not None:
                    omega_plot = np.logspace(-2, 8, 200)
                    f_plot = omega_plot / (2 * np.pi)
                    E_prime = self.material.get_storage_modulus(omega_plot)
                    E_double_prime = self.material.get_loss_modulus(omega_plot)
                    self.ax_dma_progress.plot(f_plot, E_prime, color='#3B82F6', linewidth=2.5, label="E'")
                    self.ax_dma_progress.plot(f_plot, E_double_prime, color='#EF4444', linewidth=2, linestyle='--', label="E''")
                    self.ax_dma_progress.set_xlabel('주파수 f (Hz)', fontweight='bold')
                    self.ax_dma_progress.set_ylabel('탄성률 (Pa)', fontweight='bold')
                    self.ax_dma_progress.set_xscale('log'); self.ax_dma_progress.set_yscale('log')
                    self.ax_dma_progress.grid(True, alpha=0.25, linewidth=0.5)
                    self.ax_dma_progress.legend(loc='best')
                    self.ax_dma_progress.set_title('DMA 마스터 곡선 — 주파수 윈도우', fontweight='bold')

                # BOTTOM-LEFT: G(q) live accumulation placeholder
                self.ax_gq_live.set_xlabel('파수 q (1/m)', fontweight='bold')
                self.ax_gq_live.set_ylabel('G(q)', fontweight='bold')
                self.ax_gq_live.set_xscale('log'); self.ax_gq_live.set_yscale('log')
                self.ax_gq_live.grid(True, alpha=0.25, linewidth=0.5)
                self.ax_gq_live.set_title('실시간 G(q) 적분 누적', fontweight='bold', fontsize=15)
                self.ax_gq_live.text(0.5, 0.5, '적분 시작 대기 중 …',
                                     transform=self.ax_gq_live.transAxes,
                                     ha='center', va='center', fontsize=16,
                                     color='#94A3B8', style='italic')

                # BOTTOM-RIGHT: Contact area placeholder
                self.ax_contact_live.set_xlabel('파수 q (1/m)', fontweight='bold', fontsize=13)
                self.ax_contact_live.set_ylabel(r'A(q)/A$_0$', fontweight='bold', fontsize=13)
                self.ax_contact_live.set_xscale('log')
                self.ax_contact_live.grid(True, alpha=0.25, linewidth=0.5)
                self.ax_contact_live.set_title(r'접촉 면적비 A(q)/A$_0$ 변화', fontweight='bold', fontsize=15)
                self.ax_contact_live.text(0.5, 0.5, '적분 시작 대기 중 …',
                                          transform=self.ax_contact_live.transAxes,
                                          ha='center', va='center', fontsize=16,
                                          color='#94A3B8', style='italic')

                self.fig_calc_progress.tight_layout(pad=2.5)
                self.canvas_calc_progress.draw()
            except Exception as e:
                print(f"Error initializing plots: {e}")
                import traceback
                traceback.print_exc()

            # ── Real-time visualisation callback ──────────────────────
            # Track whether we've cleared the placeholder text
            _placeholder_cleared = {'gq': False, 'contact': False}
            # Keep track of previously swept DMA bands for trail effect
            _prev_dma_bands = []

            def progress_callback(percent, v_idx=None, current_v=None,
                                  G_col=None, P_col=None):
                self.progress_var.set(percent)

                # Derive v_idx / current_v from percent if caller sent
                # only percent (legacy fallback)
                if v_idx is None:
                    v_idx = max(0, int(percent / 100 * _n_v_total) - 1)
                if current_v is None:
                    current_v = v_array[min(v_idx, len(v_array) - 1)]

                omega_min_cb = q_array[0] * current_v
                omega_max_cb = q_array[-1] * current_v
                f_min_cb = omega_min_cb / (2 * np.pi)
                f_max_cb = omega_max_cb / (2 * np.pi)

                # ── Status labels ─────────────────────────────
                self.status_var.set(
                    f"적분 중… v={current_v:.2e} m/s  "
                    f"({v_idx+1}/{_n_v_total})  "
                    f"f: {f_min_cb:.1e}–{f_max_cb:.1e} Hz"
                )
                self.calc_status_label.config(
                    text=(f"적분 진행 {percent:.0f}%  |  "
                          f"v = {current_v:.2e} m/s  ({v_idx+1}/{_n_v_total})  |  "
                          f"f = {f_min_cb:.1e} ~ {f_max_cb:.1e} Hz"),
                    foreground='#DC2626'
                )

                _color = _cmap(v_idx / max(1, _n_v_total - 1))

                # ── DMA frequency window (top-right) ──────────
                try:
                    # Fade previous bands
                    for old_band in _prev_dma_bands:
                        try:
                            old_band.set_alpha(max(0.03, old_band.get_alpha() * 0.5))
                        except Exception:
                            pass
                    # Remove any current highlight
                    for artist in self.ax_dma_progress.collections[:]:
                        if hasattr(artist, '_is_highlight_current'):
                            artist.remove()
                    # Draw new active band with velocity colour
                    band = self.ax_dma_progress.axvspan(
                        f_min_cb, f_max_cb,
                        alpha=0.25, facecolor=_color,
                        edgecolor=(*_color[:3], 0.8), linewidth=1.5,
                        zorder=0)
                    band._is_highlight_current = True
                    _prev_dma_bands.append(band)
                except Exception:
                    pass

                # ── G(q) live accumulation (bottom-left) ──────
                if G_col is not None:
                    try:
                        if not _placeholder_cleared['gq']:
                            self.ax_gq_live.clear()
                            self.ax_gq_live.set_xlabel('파수 q (1/m)', fontweight='bold', fontsize=13)
                            self.ax_gq_live.set_ylabel('G(q)', fontweight='bold', fontsize=13)
                            self.ax_gq_live.set_xscale('log')
                            self.ax_gq_live.set_yscale('log')
                            self.ax_gq_live.grid(True, alpha=0.25, linewidth=0.5)
                            self.ax_gq_live.set_title('실시간 G(q) 적분 누적', fontweight='bold', fontsize=15)
                            _placeholder_cleared['gq'] = True

                        # Fade already-plotted curves
                        for line in self.ax_gq_live.lines:
                            line.set_alpha(max(0.12, line.get_alpha() * 0.7))

                        # Plot fresh G(q) curve
                        valid = G_col > 0
                        if np.any(valid):
                            self.ax_gq_live.loglog(
                                q_array[valid], G_col[valid],
                                color=_color, linewidth=2.2, alpha=0.92,
                                label=f'v={current_v:.1e}')

                            # Fill under the latest curve for emphasis
                            self.ax_gq_live.fill_between(
                                q_array[valid], G_col[valid],
                                alpha=0.08, color=_color)

                        # Legend only every few curves to avoid clutter
                        if v_idx % max(1, _n_v_total // 6) == 0 or v_idx == _n_v_total - 1:
                            handles = [l for l in self.ax_gq_live.lines
                                       if l.get_alpha() and l.get_alpha() > 0.5]
                            if handles:
                                self.ax_gq_live.legend(
                                    handles=handles[-6:], loc='upper left',
                                    fontsize=12, framealpha=0.7)
                    except Exception:
                        pass

                # ── Contact area live (bottom-right) ──────────
                if P_col is not None:
                    try:
                        if not _placeholder_cleared['contact']:
                            self.ax_contact_live.clear()
                            self.ax_contact_live.set_xlabel('파수 q (1/m)', fontweight='bold', fontsize=13)
                            self.ax_contact_live.set_ylabel(r'A(q)/A$_0$', fontweight='bold', fontsize=13)
                            self.ax_contact_live.set_xscale('log')
                            self.ax_contact_live.grid(True, alpha=0.25, linewidth=0.5)
                            self.ax_contact_live.set_title(r'접촉 면적비 A(q)/A$_0$ 변화', fontweight='bold', fontsize=15)
                            _placeholder_cleared['contact'] = True

                        # Fade previous curves
                        for line in self.ax_contact_live.lines:
                            line.set_alpha(max(0.12, line.get_alpha() * 0.7))

                        valid = np.isfinite(P_col) & (P_col > 0)
                        if np.any(valid):
                            self.ax_contact_live.plot(
                                q_array[valid], P_col[valid],
                                color=_color, linewidth=2.2, alpha=0.92,
                                label=f'v={current_v:.1e}')

                            self.ax_contact_live.fill_between(
                                q_array[valid], P_col[valid],
                                alpha=0.06, color=_color)

                        if v_idx % max(1, _n_v_total // 6) == 0 or v_idx == _n_v_total - 1:
                            handles = [l for l in self.ax_contact_live.lines
                                       if l.get_alpha() and l.get_alpha() > 0.5]
                            if handles:
                                self.ax_contact_live.legend(
                                    handles=handles[-6:], loc='upper right',
                                    fontsize=12, framealpha=0.7)
                    except Exception:
                        pass

                # Redraw canvas once per callback
                try:
                    self.fig_calc_progress.tight_layout(pad=2.5)
                    self.canvas_calc_progress.draw()
                except Exception:
                    pass

                self.root.update()

            results_2d = self.g_calculator.calculate_G_multi_velocity(
                q_array, v_array, q_min=q_min, progress_callback=progress_callback
            )

            # ── Post-calculation: clean up and finalise plots ─────
            try:
                # Remove transient DMA bands
                for artist in self.ax_dma_progress.collections[:]:
                    if hasattr(artist, '_is_highlight_current') or hasattr(artist, '_is_highlight'):
                        artist.remove()
                for band in _prev_dma_bands:
                    try:
                        band.remove()
                    except Exception:
                        pass

                # Final G(q) and Contact area — keep accumulated curves as-is,
                # but ensure last curves are fully opaque
                for line in self.ax_gq_live.lines[-1:]:
                    line.set_alpha(1.0)
                    line.set_linewidth(2.5)
                for line in self.ax_contact_live.lines[-1:]:
                    line.set_alpha(1.0)
                    line.set_linewidth(2.5)

                self.calc_status_label.config(
                    text=f"계산 완료!  |  총 {len(v_array)}개 속도 × {len(q_array)}개 파수",
                    foreground='#059669'
                )

                self.fig_calc_progress.tight_layout(pad=2.5)
                self.canvas_calc_progress.draw()
            except Exception as e:
                print(f"Error clearing highlights: {e}")

            # Calculate detailed results for selected velocities (for visualization)
            # Select 5-8 velocities spanning the range
            n_detail_v = min(8, len(v_array))
            detail_v_indices = np.linspace(0, len(v_array)-1, n_detail_v, dtype=int)
            detailed_results_multi_v = []

            for v_idx in detail_v_indices:
                self.g_calculator.velocity = v_array[v_idx]
                detailed = self.g_calculator.calculate_G_with_details(
                    q_array, q_min=q_min, store_inner_integral=False
                )
                detailed['velocity'] = v_array[v_idx]
                detailed_results_multi_v.append(detailed)

            self.results = {
                '2d_results': results_2d,
                'detailed_results_multi_v': detailed_results_multi_v,
                'sigma_0': sigma_0,
                'temperature': temperature,
                'poisson': poisson
            }

            # Plot results
            self._plot_g_results()

            self.status_var.set("Calculation complete!")
            self.calc_button.config(state='normal')
            self._show_status(f"G(q,v) calculated for {n_v} velocities and {n_q} wavenumbers", 'success')

            # G(q,v) 계산 완료 후 Tab 4의 h'rms slope 자동 계산
            try:
                self._calculate_rms_slope()
            except Exception as e_rms:
                print(f"Auto h'rms calculation skipped: {e_rms}")

        except Exception as e:
            self.calc_button.config(state='normal')
            import traceback
            tb = traceback.format_exc()
            print(f"G calculation error:\n{tb}")
            messagebox.showerror("Error", f"Calculation failed:\n{str(e)}\n\n{tb[-500:]}")

    def _plot_g_results(self):
        """Plot G(q,v) 2D results with enhanced visualizations."""
        self.fig_results.clear()

        results_2d = self.results['2d_results']
        q = results_2d['q']
        v = results_2d['v']
        G_matrix = results_2d['G_matrix']
        P_matrix = results_2d['P_matrix']

        # Get detailed results if available
        has_detailed = 'detailed_results_multi_v' in self.results
        if has_detailed:
            detailed_multi_v = self.results['detailed_results_multi_v']

        # Create 2x3 subplot layout
        ax1 = self.fig_results.add_subplot(2, 3, 1)
        ax2 = self.fig_results.add_subplot(2, 3, 2)
        ax3 = self.fig_results.add_subplot(2, 3, 3)
        ax4 = self.fig_results.add_subplot(2, 3, 4)
        ax5 = self.fig_results.add_subplot(2, 3, 5)
        ax6 = self.fig_results.add_subplot(2, 3, 6)

        # Standard font settings for all plots
        TITLE_FONT = 16
        LABEL_FONT = 14
        LEGEND_FONT = 12
        TITLE_PAD = 10

        # Plot 1: Multi-velocity G(q) curves (다중 속도 G(q) 곡선)
        cmap = plt.get_cmap('viridis')
        colors = [cmap(i / len(v)) for i in range(len(v))]

        for j, (v_val, color) in enumerate(zip(v, colors)):
            if j % max(1, len(v) // 10) == 0:  # Plot every 10th curve
                ax1.loglog(q, G_matrix[:, j], color=color, linewidth=1.5,
                          label=f'v={v_val:.4f} m/s')

        ax1.set_xlabel('파수 q (1/m)', fontweight='bold', fontsize=LABEL_FONT, labelpad=3)
        ax1.set_ylabel('G(q)', fontweight='bold', fontsize=LABEL_FONT, rotation=90, labelpad=5)
        ax1.set_title('(a) 다중 속도에서의 G(q)', fontweight='bold', fontsize=TITLE_FONT, pad=TITLE_PAD)
        ax1.legend(fontsize=LEGEND_FONT, ncol=2)
        ax1.grid(True, alpha=0.3)

        # Fix axis formatter to use superscript notation
        from matplotlib.ticker import FuncFormatter
        def log_tick_formatter(val, pos=None):
            if val <= 0:
                return ''
            exponent = int(np.floor(np.log10(val)))
            if abs(val - 10**exponent) < 1e-10:
                return f'$10^{{{exponent}}}$'
            else:
                mantissa = val / (10**exponent)
                if abs(mantissa - 1.0) < 0.01:
                    return f'$10^{{{exponent}}}$'
                else:
                    return f'${mantissa:.1f} \\times 10^{{{exponent}}}$'
        ax1.xaxis.set_major_formatter(FuncFormatter(log_tick_formatter))
        ax1.yaxis.set_major_formatter(FuncFormatter(log_tick_formatter))

        # Plot 2: Local stress probability distribution P(σ,q) for multiple wavenumbers
        # CHANGED: Plot vs wavenumber (q) instead of velocity, with v fixed at 1 m/s
        # Persson theory: P(σ,q) = 1/√(4πG_stress(q)) [exp(-(σ-σ0)²/4G_stress) - exp(-(σ+σ0)²/4G_stress)]
        # where G_stress(q) = (π/4) * (E*/sigma_0)² * ∫[q0→q] k³C(k)dk  [dimensionless]
        sigma_0_Pa = self.results['sigma_0']  # Pa
        sigma_0_MPa = sigma_0_Pa / 1e6  # Convert Pa to MPa

        # Calculate G_stress(q) at FIXED velocity v = 1 m/s
        poisson = float(self.poisson_var.get())
        temperature = float(self.temperature_var.get())
        q_min = float(self.q_min_var.get())
        q_max = float(self.q_max_var.get())

        # Filter q array to user-specified range for stress distribution calculation
        # Use separate variable to avoid affecting other plots
        q_mask = (q >= q_min) & (q <= q_max)
        q_stress = q[q_mask]
        C_q_stress = self.psd_model(q_stress)

        print(f"Filtered q range for stress dist: {q_stress[0]:.2e} ~ {q_stress[-1]:.2e} (1/m), {len(q_stress)} points")

        # Fixed velocity for wavenumber analysis
        v_fixed = 0.01  # m/s (lower velocity to see clearer peak at sigma_0)

        # Calculate G_stress(q) at fixed velocity
        omega_low = q_min * v_fixed
        E_low = self.material.get_storage_modulus(np.array([omega_low]))[0]
        E_star_low = E_low / (1 - poisson**2)

        integrand = q_stress**3 * C_q_stress
        E_normalized = E_star_low / sigma_0_Pa

        # Calculate G_stress(q) array
        G_stress_array = np.zeros_like(q_stress)
        for i in range(1, len(q_stress)):
            integrand_partial = q_stress[:i+1]**3 * C_q_stress[:i+1]
            G_stress_array[i] = (np.pi / 4) * E_normalized**2 * np.trapezoid(integrand_partial, q_stress[:i+1])

        # Set x-axis range based on MAXIMUM G to show full Gaussian shapes
        # Use the maximum G value to ensure all curves fit within the plot
        G_max = G_stress_array[-1]  # Use maximum G to capture widest distribution
        std_max = np.sqrt(G_max) * sigma_0_MPa
        sigma_max = sigma_0_MPa + 6 * std_max  # Use 6σ to show complete Gaussian tail
        sigma_min = -sigma_0_MPa - 6 * std_max  # Extended negative side for full mirror image

        # Ensure minimum x-axis range
        if sigma_max < 2 * sigma_0_MPa:
            sigma_max = 2 * sigma_0_MPa
            sigma_min = -sigma_max
        sigma_array = np.linspace(sigma_min, sigma_max, 800)  # Increased points for smoother curves

        # Select 10 wavenumbers to plot (uniformly spaced in log)
        n_q_selected = min(10, len(q_stress))
        q_indices = np.linspace(0, len(q_stress)-1, n_q_selected, dtype=int)

        # Create color map for wavenumbers (viridis - 같은 탭 다른 그래프와 색조 통일)
        cmap_q = plt.get_cmap('viridis')
        colors_q = [cmap_q(i / max(n_q_selected-1, 1)) for i in range(n_q_selected)]

        # σ0 주변으로 확대할 범위 계산 (최소 5개 파수가 보이도록)
        G_valid_sorted = sorted([G_stress_array[idx] for idx in q_indices if G_stress_array[idx] > 1e-10])
        if len(G_valid_sorted) >= 5:
            # 5번째로 작은 G 기준 → 상위 5개 좁은 분포가 보임
            G_zoom = G_valid_sorted[4]
        elif G_valid_sorted:
            G_zoom = G_valid_sorted[-1]
        else:
            G_zoom = G_max
        std_zoom = np.sqrt(G_zoom) * sigma_0_MPa
        half_range = max(4 * std_zoom, sigma_0_MPa * 0.5)

        # Track maximum values for axis scaling
        max_P_sigma = 0

        # Plot stress distributions for selected wavenumbers (순수 P(σ)만)
        for i, q_idx in enumerate(q_indices):
            color = colors_q[i]
            q_val = q_stress[q_idx]
            G_norm_q = G_stress_array[q_idx]

            # Calculate stress distribution at this wavenumber
            if G_norm_q > 1e-10:
                # Normalize σ by sigma_0 for calculation
                sigma_norm = sigma_array / sigma_0_MPa

                # Calculate P(σ) = term1 - term2
                normalization = 1 / (sigma_0_MPa * np.sqrt(4 * np.pi * G_norm_q))
                term1 = normalization * np.exp(-(sigma_norm - 1)**2 / (4 * G_norm_q))
                term2 = normalization * np.exp(-(sigma_norm + 1)**2 / (4 * G_norm_q))

                # Final P(σ) - clip to ensure non-negative
                P_sigma = np.maximum(0, term1 - term2)

                # Track maximum values for axis scaling
                max_P_sigma = max(max_P_sigma, np.max(P_sigma))

                # Calculate P(σ > 0): probability of positive stress
                positive_indices = sigma_array > 0
                P_positive = np.trapezoid(P_sigma[positive_indices], sigma_array[positive_indices])
                P_positive_percent = P_positive * 100

                # Plot only pure P(σ) solid line
                ax2.plot(sigma_array, P_sigma, color=color, linewidth=1.8,
                        label=f'q={q_val:.1e}', alpha=0.9)

        # Add vertical line for nominal pressure
        ax2.axvline(sigma_0_MPa, color='black', linestyle='--', linewidth=2,
                   label=f'σ0 = {sigma_0_MPa:.2f} MPa', alpha=0.7)

        ax2.set_xlabel('응력 σ (MPa)', fontweight='bold', fontsize=LABEL_FONT, labelpad=3)
        ax2.set_ylabel('응력 분포 P(σ)', fontweight='bold', fontsize=LABEL_FONT, rotation=90, labelpad=5)
        ax2.set_title(f'(b) 파수별 국소 응력 확률 분포 (v={v_fixed:.2f} m/s 고정)', fontweight='bold', fontsize=TITLE_FONT, pad=TITLE_PAD)
        ax2.legend(fontsize=LEGEND_FONT, ncol=2, loc='upper right')
        ax2.grid(True, alpha=0.3)
        # X축: σ0 주변 확대 (5개+ 파수 분포 경향 확인용)
        ax2.set_xlim(sigma_0_MPa - half_range, sigma_0_MPa + half_range)
        ax2.set_ylim(0, max_P_sigma * 1.15 if max_P_sigma > 0 else 1.0)

        # Plot 3: Contact Area P(q,v) (접촉 면적)
        for j, (v_val, color) in enumerate(zip(v, colors)):
            if j % max(1, len(v) // 10) == 0:
                # Filter out values very close to 1.0 for better visualization
                P_curve = P_matrix[:, j].copy()
                # Clip very small differences from 1.0
                P_curve = np.clip(P_curve, 0, 0.999)

                ax3.semilogx(q, P_curve, color=color, linewidth=1.5,
                            label=f'v={v_val:.4f} m/s')

        ax3.set_xlabel('파수 q (1/m)', fontweight='bold', fontsize=LABEL_FONT, labelpad=3)
        ax3.set_ylabel('접촉 면적 비율 P(q)', fontweight='bold', fontsize=LABEL_FONT, rotation=90, labelpad=5)
        ax3.set_title('(c) 다중 속도에서의 접촉 면적', fontweight='bold', fontsize=TITLE_FONT, pad=TITLE_PAD)
        ax3.legend(fontsize=LEGEND_FONT, ncol=2)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1.0)  # Set y-axis limit for better visualization

        # Fix axis formatter
        ax3.xaxis.set_major_formatter(FuncFormatter(log_tick_formatter))

        # Plot 4: Final contact area vs velocity (속도에 따른 최종 접촉 면적)
        P_final = P_matrix[-1, :]

        # Create gradient color for all velocity points
        scatter = ax4.scatter(v, P_final, c=np.arange(len(v)), cmap='viridis',
                             s=50, zorder=3, edgecolors='black', linewidth=0.5)

        # Add connecting line with gradient using line segments
        if len(v) >= 2:
            from matplotlib.collections import LineCollection
            points = np.array([v, P_final]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm = plt.Normalize(0, len(v)-1)
            lc = LineCollection(segments, cmap='viridis', norm=norm, linewidth=2, alpha=0.6)
            lc.set_array(np.arange(len(v)))
            ax4.add_collection(lc)

        ax4.set_xscale('log')
        ax4.set_xlabel('속도 v (m/s)', fontweight='bold', fontsize=LABEL_FONT, labelpad=3)
        ax4.set_ylabel('최종 접촉 면적 P(q_max)', fontweight='bold', fontsize=LABEL_FONT, rotation=90, labelpad=5)
        ax4.set_title('(d) 속도에 따른 접촉 면적', fontweight='bold', fontsize=TITLE_FONT, pad=TITLE_PAD)
        ax4.grid(True, alpha=0.3)

        # Fix axis formatter
        ax4.xaxis.set_major_formatter(FuncFormatter(log_tick_formatter))

        # Plot 5: Inner integral vs q for multiple velocities (다중 속도에서의 내부 적분)
        # Physical meaning: ∫dφ |E(qvcosφ)|² - 슬립 방향 성분의 점탄성 응답
        if has_detailed:
            for i, detail_result in enumerate(detailed_multi_v):
                v_val = detail_result['velocity']
                # Find matching velocity index in v array to use consistent color
                v_idx = np.argmin(np.abs(v - v_val))
                color = colors[v_idx]  # Use same color scheme as other plots
                ax5.loglog(detail_result['q'], detail_result['avg_modulus_term'],
                          color=color, linewidth=1.5, label=f'v={v_val:.4f} m/s', alpha=0.8)

            ax5.set_xlabel('파수 q (1/m)', fontweight='bold', fontsize=LABEL_FONT, labelpad=3)
            ax5.set_ylabel(r'각도 적분 $\int d\phi\,|E/(1-\nu^2)\sigma_0|^2$', fontweight='bold', fontsize=LABEL_FONT, rotation=90, labelpad=5)
            ax5.set_title('(e) 내부 적분: 상대적 강성비', fontweight='bold', fontsize=TITLE_FONT, pad=TITLE_PAD)
            ax5.legend(fontsize=LEGEND_FONT, ncol=2)
            ax5.grid(True, alpha=0.3)

            # Add physical interpretation text box
            textstr = ('물리적 의미: ∫dφ|E*/(1-ν²)σ0|²\n'
                      '= 외부 압력 대비 고무의 단단함을 나타내는 척도\n'
                      'E*: 복소 탄성률 (주파수 ω=qv cosφ)\n'
                      '(1-ν²)σ0: 평면 변형률 보정 + 명목 압력\n'
                      '높을수록 → 고무가 단단 → 응력 불균일 증가\n'
                      '낮을수록 → 고무가 말랑 → 완전 접촉에 가까움')
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
            ax5.text(0.98, 0.02, textstr, transform=ax5.transAxes, fontsize=11,
                    verticalalignment='bottom', horizontalalignment='right', bbox=props)

            # Fix axis formatter
            ax5.xaxis.set_major_formatter(FuncFormatter(log_tick_formatter))
            ax5.yaxis.set_major_formatter(FuncFormatter(log_tick_formatter))
        else:
            ax5.text(0.5, 0.5, '내부 적분 데이터 없음',
                    ha='center', va='center', transform=ax5.transAxes, fontsize=LABEL_FONT)
            ax5.set_title('(e) 내부 적분', fontweight='bold', fontsize=TITLE_FONT, pad=TITLE_PAD)

        # Plot 6: Parseval theorem - Cumulative RMS slope with q1 determination
        # Slope²(q) = 2π∫[0 to q] k³C(k)dk
        if self.psd_model is not None:
            # Calculate cumulative RMS slope
            q_parse = np.logspace(np.log10(q[0]), np.log10(q[-1]), 1000)
            C_q_parse = self.psd_model(q_parse)

            # Cumulative integration using correct formula
            # Slope²(q) = 2π∫[qmin to q] k³C(k)dk
            slope_squared_cumulative = np.zeros_like(q_parse)

            for i in range(len(q_parse)):
                q_int = q_parse[:i+1]
                C_int = C_q_parse[:i+1]
                # Integrate q³C(q) with 2π factor
                slope_squared_cumulative[i] = 2 * np.pi * np.trapezoid(q_int**3 * C_int, q_int)

            slope_rms_cumulative = np.sqrt(slope_squared_cumulative)

            # Find q1 where slope_rms = target (from parameter settings)
            target_slope_rms = float(self.target_hrms_slope_var.get())
            q1_idx = np.argmax(slope_rms_cumulative >= target_slope_rms)

            if q1_idx > 0:
                # Use interpolation for more accurate q1
                from scipy.interpolate import interp1d
                # Safe slice bounds (prevent negative index wrap-around)
                sl_start = max(0, q1_idx - 10)
                sl_end = min(len(slope_rms_cumulative), q1_idx + 10)
                if sl_end - sl_start >= 2:
                    f_interp = interp1d(slope_rms_cumulative[sl_start:sl_end],
                                       q_parse[sl_start:sl_end],
                                       kind='linear', fill_value='extrapolate')
                    q1_determined = float(f_interp(target_slope_rms))
                else:
                    q1_determined = float(q_parse[q1_idx])
            else:
                # If target not reached, use extrapolation with Hurst exponent
                # Fit power law to last portion of data
                log_q_fit = np.log10(q_parse[-50:])
                log_C_fit = np.log10(C_q_parse[-50:])
                slope_fit = np.polyfit(log_q_fit, log_C_fit, 1)[0]
                H = -slope_fit / 2 - 1  # Hurst exponent

                # Extrapolate C(q) = A * q^(-2(H+1))
                A = C_q_parse[-1] / (q_parse[-1]**(-2*(H+1)))

                # Find q1 by solving integral equation
                # This is approximate; could use root finding for precision
                q1_determined = q_parse[-1] * 1.5  # Placeholder
                self._show_status(f"Target slope {target_slope_rms} not reached. Extrapolating with H={H:.3f}", 'success')

            # Plot cumulative h'rms (RMS slope)
            ax6.semilogx(q_parse, slope_rms_cumulative, 'b-', linewidth=2.5, label="누적 h'rms")

            # Add horizontal line at target h'rms
            ax6.axhline(target_slope_rms, color='red', linestyle='--', linewidth=2,
                       label=f"목표 h'rms = {target_slope_rms}", alpha=0.7, zorder=5)

            # Add vertical line at q1
            if q1_idx > 0:
                ax6.axvline(q1_determined, color='green', linestyle='--', linewidth=2,
                           label=f'결정된 q1 = {q1_determined:.2e} (1/m)', alpha=0.7, zorder=5)

                # Mark intersection point
                ax6.plot(q1_determined, target_slope_rms, 'ro', markersize=12,
                        markeredgecolor='black', markeredgewidth=2, zorder=10,
                        label='교차점')

                # Update calculated q1 display in Tab 3
                self.calculated_q1_var.set(f"{q1_determined:.3e}")

                # q_max는 사용자가 직접 설정하므로 자동 덮어쓰기 하지 않음
                # self.rms_q_max_var.set(f"{q1_determined:.3e}")

                # Store calculated q1 for other uses
                self.calculated_q1 = q1_determined

            ax6.set_xlabel('파수 q (1/m)', fontweight='bold', fontsize=LABEL_FONT, labelpad=3)
            ax6.set_ylabel("누적 h'rms (기울기)", fontweight='bold', fontsize=LABEL_FONT, rotation=90, labelpad=5)
            ax6.set_title(f"(f) Parseval 정리: q1 자동 결정 (목표 h'rms={target_slope_rms})", fontweight='bold', fontsize=TITLE_FONT, pad=TITLE_PAD)

            # Legend with better positioning
            ax6.legend(fontsize=LEGEND_FONT, loc='lower right', framealpha=0.9)

            ax6.grid(True, alpha=0.3)

            # Add annotation box
            if q1_idx > 0:
                textstr = (f"파서벌 정리:\nh'rms = √(2π∫k³C(k)dk)\n\n"
                          f"결정된 q1 = {q1_determined:.2e} 1/m\n"
                          f"해당 h'rms = {target_slope_rms:.2f}")
            else:
                textstr = (f"파서벌 정리:\nh'rms = √(2π∫k³C(k)dk)\n\n"
                          f"최종 h'rms = {slope_rms_cumulative[-1]:.3f}\n"
                          f"(목표 {target_slope_rms} 미달)")

            props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, edgecolor='black')
            ax6.text(0.02, 0.98, textstr, transform=ax6.transAxes, fontsize=12,
                    verticalalignment='top', bbox=props)

            # Fix axis formatter
            ax6.xaxis.set_major_formatter(FuncFormatter(log_tick_formatter))
        else:
            ax6.text(0.5, 0.5, 'PSD 데이터 없음',
                    ha='center', va='center', transform=ax6.transAxes, fontsize=LABEL_FONT)
            ax6.set_title('(f) Parseval 정리', fontweight='bold', fontsize=TITLE_FONT, pad=TITLE_PAD)

        self.fig_results.suptitle('G(q,v) 2D 행렬 계산 결과', fontweight='bold', fontsize=16, y=0.99)
        self.fig_results.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.08, hspace=0.45, wspace=0.38)
        self.canvas_results.draw()

    def _save_detailed_csv(self):
        """Save detailed CSV results."""
        if not self.results:
            self._show_status("No results to save. Run calculation first!", 'warning')
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if filename:
            # Implementation here
            self._show_status("CSV save functionality to be implemented", 'success')

    def _export_all_results(self):
        """Export all results."""
        if not self.results:
            self._show_status("No results to export. Run calculation first!", 'warning')
            return

        output_dir = filedialog.askdirectory(title="Select output directory")
        if output_dir:
            self._show_status("Export functionality to be implemented", 'success')

    def _register_graph_data(self, name: str, x_data, y_data, header: str, description: str):
        """
        Register graph data to the automatic data list.
        Called whenever a graph is drawn with significant data.

        Parameters
        ----------
        name : str
            Unique identifier for this data (e.g., 'PSD_Full_2D', 'MasterCurve_E_storage')
        x_data : array-like
            X-axis data (q, f, v, etc.)
        y_data : array-like
            Y-axis data (C(q), E', mu, etc.)
        header : str
            Header line for export file (e.g., 'q(1/m)\\tC(q)(m^4)')
        description : str
            Human-readable description (e.g., 'Full PSD (2D isotropic)')
        """
        from datetime import datetime
        import numpy as np

        x_arr = np.asarray(x_data)
        y_arr = np.asarray(y_data)

        # Create timestamp for tracking
        timestamp = datetime.now().strftime("%H:%M:%S")

        self.graph_data_registry[name] = {
            'x': x_arr,
            'y': y_arr,
            'header': header,
            'description': description,
            'timestamp': timestamp
        }

        # Update the graph data listbox if it exists and is visible
        if hasattr(self, 'graph_data_listbox') and self.graph_data_listbox.winfo_exists():
            self._refresh_graph_data_listbox()

    def _refresh_graph_data_listbox(self):
        """Refresh the graph data listbox with current registry contents."""
        if not hasattr(self, 'graph_data_listbox') or not self.graph_data_listbox.winfo_exists():
            return

        # Store current selection
        current_selection = self.graph_data_listbox.curselection()

        # Clear and repopulate
        self.graph_data_listbox.delete(0, tk.END)

        # Combine registry with legacy data
        combined_data = dict(self.graph_data_registry)  # Start with registry data

        # Add legacy data collection (same as before)
        self._collect_legacy_graph_data(combined_data)

        self.available_graph_data = combined_data

        for name, data in combined_data.items():
            timestamp = data.get('timestamp', '')
            ts_str = f" [{timestamp}]" if timestamp else ""
            self.graph_data_listbox.insert(tk.END, f"{name} - {data['description']}{ts_str}")

        if not combined_data:
            self.graph_data_listbox.insert(tk.END, "(No graph data available)")

        # Restore selection if possible
        for idx in current_selection:
            if idx < self.graph_data_listbox.size():
                self.graph_data_listbox.selection_set(idx)

    def _collect_legacy_graph_data(self, data_dict):
        """Collect graph data from legacy class attributes (backward compatibility)."""
        # Tab 0: PSD data (if not already in registry)
        if hasattr(self, 'profile_psd_analyzer') and self.profile_psd_analyzer is not None:
            if self.profile_psd_analyzer.q is not None and self.profile_psd_analyzer.C_full_2d is not None:
                if "PSD_Full_2D" not in data_dict:
                    data_dict["PSD_Full_2D"] = {
                        'x': self.profile_psd_analyzer.q,
                        'y': self.profile_psd_analyzer.C_full_2d,
                        'header': 'q(1/m)\tC(q)(m^4)',
                        'description': 'Full PSD (2D isotropic)'
                    }
            if self.profile_psd_analyzer.C_top_2d is not None:
                if "PSD_Top_2D" not in data_dict:
                    data_dict["PSD_Top_2D"] = {
                        'x': self.profile_psd_analyzer.q,
                        'y': self.profile_psd_analyzer.C_top_2d,
                        'header': 'q(1/m)\tC(q)(m^4)',
                        'description': 'Top PSD (2D isotropic)'
                    }

        # Parameter PSD
        if hasattr(self, 'param_psd_data') and self.param_psd_data is not None:
            if "PSD_Param" not in data_dict:
                data_dict["PSD_Param"] = {
                    'x': self.param_psd_data['q'],
                    'y': self.param_psd_data['C'],
                    'header': 'q(1/m)\tC(q)(m^4)',
                    'description': f"Param PSD (H={self.param_psd_data['H']:.4f})"
                }

        # Finalized PSD
        if hasattr(self, 'finalized_psd') and self.finalized_psd is not None:
            if "PSD_Finalized" not in data_dict:
                data_dict["PSD_Finalized"] = {
                    'x': self.finalized_psd['q'],
                    'y': self.finalized_psd['C'],
                    'header': 'q(1/m)\tC(q)(m^4)',
                    'description': f"Finalized PSD ({self.finalized_psd['type']})"
                }

        # Tab 1: Master Curve
        if hasattr(self, 'master_curve_gen') and self.master_curve_gen is not None:
            if self.master_curve_gen.master_f is not None:
                if "MasterCurve_E_storage" not in data_dict:
                    data_dict["MasterCurve_E_storage"] = {
                        'x': self.master_curve_gen.master_f,
                        'y': self.master_curve_gen.master_E_storage,
                        'header': 'f(Hz)\tE_storage(MPa)',
                        'description': f"Master Curve E' (Tref={self.master_curve_gen.T_ref}C)"
                    }
                if "MasterCurve_E_loss" not in data_dict:
                    data_dict["MasterCurve_E_loss"] = {
                        'x': self.master_curve_gen.master_f,
                        'y': self.master_curve_gen.master_E_loss,
                        'header': 'f(Hz)\tE_loss(MPa)',
                        'description': f"Master Curve E'' (Tref={self.master_curve_gen.T_ref}C)"
                    }

        # Mu_visc results
        if hasattr(self, 'mu_visc_results') and self.mu_visc_results is not None:
            if 'v' in self.mu_visc_results and 'mu_visc' in self.mu_visc_results:
                if "Friction_mu_vs_v" not in data_dict:
                    data_dict["Friction_mu_vs_v"] = {
                        'x': self.mu_visc_results['v'],
                        'y': self.mu_visc_results['mu_visc'],
                        'header': 'v(m/s)\tmu_visc',
                        'description': 'Friction coefficient vs velocity'
                    }

    def _show_graph_data_export_popup(self):
        """Show popup window for exporting all graph data to txt files."""
        popup = tk.Toplevel(self.root)
        popup.title("Graph Data Export")
        popup.geometry("600x500")
        popup.transient(self.root)

        # Main frame
        main_frame = ttk.Frame(popup, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        ttk.Label(main_frame, text="Available Graph Data", font=('Arial', 15, 'bold')).pack(anchor=tk.W)
        ttk.Label(main_frame, text="Select data to export as txt files", font=('Segoe UI', 17)).pack(anchor=tk.W)

        # Listbox with scrollbar
        list_frame = ttk.Frame(main_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.graph_data_listbox = tk.Listbox(list_frame, selectmode=tk.MULTIPLE,
                                              yscrollcommand=scrollbar.set, height=15,
                                              font=('Segoe UI', 17))
        self.graph_data_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.graph_data_listbox.yview)

        # Collect and display available data using the registry system
        self.available_graph_data = dict(self.graph_data_registry)  # Start with registry
        self._collect_legacy_graph_data(self.available_graph_data)  # Add legacy sources

        # Also add DMA raw data if available
        if hasattr(self, 'raw_dma_data') and self.raw_dma_data is not None:
            omega = self.raw_dma_data['omega']
            if "DMA_E_storage" not in self.available_graph_data:
                self.available_graph_data["DMA_E_storage"] = {
                    'x': omega,
                    'y': self.raw_dma_data['E_storage'],
                    'header': 'omega(rad/s)\tE_storage(Pa)',
                    'description': 'DMA E_storage'
                }
            if "DMA_E_loss" not in self.available_graph_data:
                self.available_graph_data["DMA_E_loss"] = {
                    'x': omega,
                    'y': self.raw_dma_data['E_loss'],
                    'header': 'omega(rad/s)\tE_loss(Pa)',
                    'description': 'DMA E_loss'
                }

        # Add items to listbox with timestamp if available
        for name, data in self.available_graph_data.items():
            timestamp = data.get('timestamp', '')
            ts_str = f" [{timestamp}]" if timestamp else ""
            self.graph_data_listbox.insert(tk.END, f"{name} - {data['description']}{ts_str}")

        if not self.available_graph_data:
            self.graph_data_listbox.insert(tk.END, "(No graph data available)")

        # Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=5)

        ttk.Button(btn_frame, text="Select All", command=lambda: self.graph_data_listbox.select_set(0, tk.END)).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Clear Selection", command=lambda: self.graph_data_listbox.selection_clear(0, tk.END)).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Export Selected", command=lambda: self._export_selected_graph_data(popup)).pack(side=tk.RIGHT, padx=5)

        # Status
        self.export_status_var = tk.StringVar(value="Select data and click Export")
        ttk.Label(main_frame, textvariable=self.export_status_var, font=('Segoe UI', 17)).pack(anchor=tk.W)

    def _export_selected_graph_data(self, popup):
        """Export selected graph data to txt files."""
        selected_indices = self.graph_data_listbox.curselection()
        if not selected_indices:
            self._show_status("저장할 데이터를 선택하세요.", 'warning')
            return

        # Ask for output directory
        output_dir = filedialog.askdirectory(title="저장 폴더 선택")
        if not output_dir:
            return

        # Get list of data names
        data_names = list(self.available_graph_data.keys())

        saved_files = []
        for idx in selected_indices:
            if idx >= len(data_names):
                continue

            name = data_names[idx]
            data = self.available_graph_data[name]

            # Generate filename with MC prefix and timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            mc_prefix = self._get_mc_prefix()
            filename = f"{mc_prefix}_{name}_{timestamp}.txt" if mc_prefix else f"{name}_{timestamp}.txt"
            filepath = os.path.join(output_dir, filename)

            try:
                with open(filepath, 'w') as f:
                    # Write header
                    f.write(f"# {data['description']}\n")
                    f.write(f"# Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"# {data['header']}\n")

                    # Write data
                    if 'q' in data and 'C' in data:
                        for q, C in zip(data['q'], data['C']):
                            f.write(f"{q:.6e}\t{C:.6e}\n")
                    elif 'x' in data and 'y' in data:
                        for x, y in zip(data['x'], data['y']):
                            f.write(f"{x:.6e}\t{y:.6e}\n")

                saved_files.append(filename)

            except Exception as e:
                messagebox.showerror("오류", f"저장 실패 ({name}): {e}")

        if saved_files:
            self.export_status_var.set(f"Saved {len(saved_files)} files")
            self._show_status(f"총 {len(saved_files)}개 파일 저장 완료\n\n"
                f"저장 위치: {output_dir}\n\n"
                f"파일 목록:\n" + "\n".join(saved_files[:10]) +
                ("\n..." if len(saved_files) > 10 else ""), 'success')

    def _show_help(self):
        """Show help dialog as a popup window."""
        dialog = tk.Toplevel(self.root)
        dialog.title("사용자 가이드")
        dialog.resizable(True, True)
        dialog.transient(self.root)

        dlg_w, dlg_h = 750, 850
        x = self.root.winfo_x() + (self.root.winfo_width() - dlg_w) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - dlg_h) // 2
        dialog.geometry(f"{dlg_w}x{dlg_h}+{x}+{y}")

        C = self.COLORS

        # Title
        title_frame = tk.Frame(dialog, bg=C['sidebar'], padx=12, pady=8)
        title_frame.pack(fill=tk.X)
        tk.Label(title_frame, text="Persson 마찰 모델 v3.0 — 사용자 가이드",
                 bg=C['sidebar'], fg='white',
                 font=('Segoe UI', 18, 'bold')).pack(anchor=tk.W)

        # Scrollable text content
        text_frame = ttk.Frame(dialog)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        text_scroll = ttk.Scrollbar(text_frame, orient=tk.VERTICAL)
        text_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        text_widget = tk.Text(text_frame, wrap=tk.WORD, font=('Segoe UI', 14),
                              bg='white', relief='flat', borderwidth=0,
                              yscrollcommand=text_scroll.set, spacing1=2, spacing3=2)
        text_widget.pack(fill=tk.BOTH, expand=True)
        text_scroll.config(command=text_widget.yview)

        # Configure text tags for formatting
        text_widget.tag_configure('title', font=('Segoe UI', 16, 'bold'), foreground='#1B2A4A',
                                  spacing1=10, spacing3=4)
        text_widget.tag_configure('section', font=('Segoe UI', 15, 'bold'), foreground='#7C3AED',
                                  spacing1=12, spacing3=4)
        text_widget.tag_configure('body', font=('Segoe UI', 14), foreground='#1E293B',
                                  spacing1=1, spacing3=1, lmargin1=15, lmargin2=15)
        text_widget.tag_configure('indent', font=('Segoe UI', 13), foreground='#64748B',
                                  lmargin1=30, lmargin2=30, spacing1=1, spacing3=1)
        text_widget.tag_configure('note', font=('Segoe UI', 13, 'italic'), foreground='#DC2626',
                                  lmargin1=15, lmargin2=15, spacing1=2, spacing3=2)

        def add(text, tag='body'):
            text_widget.insert(tk.END, text + '\n', tag)

        add('프로그램 개요', 'title')
        add('Persson 점탄성 마찰 이론에 기반하여 고무-바닥 접촉의 마찰 계수(μ_visc)를 계산하는 프로그램입니다.')
        add('DMA 마스터 커브(재료 물성)와 PSD(표면 거칠기) 데이터를 입력하면,')
        add('G(q,v) → P(q) → μ_visc를 단계적으로 계산합니다.')
        add('')

        add('탭 1: PSD 생성', 'section')
        add('표면 프로파일 데이터(.txt)로부터 PSD C(q)를 계산합니다.')
        add('  • 프로파일 데이터 로드 후 "PSD 계산" 클릭', 'indent')
        add('  • 생성된 PSD를 검증 후 "PSD 확정 → 계산에 사용" 클릭', 'indent')
        add('  • 또는 기존 PSD 파일을 직접 로드할 수도 있습니다', 'indent')
        add('')

        add('탭 2: 마스터 커브', 'section')
        add('DMA 데이터로부터 마스터 커브를 생성하고, 시프트 인자(aT)를 확인합니다.')
        add('  • DMA 데이터(.txt): 주파수, E\', E\'\', 온도 데이터 포함', 'indent')
        add('  • 기준 온도(T_ref)를 설정하고 WLF/Arrhenius 시프트 적용', 'indent')
        add('  • 마스터 커브가 부드럽게 연결되는지 확인', 'indent')
        add('')

        add('탭 3: 계산 설정', 'section')
        add('G(q,v) 계산에 필요한 파라미터를 설정합니다.')
        add('  • σ₀: 공칭 접촉 압력 [Pa] — 하중/면적', 'indent')
        add('  • 속도 범위: 0.0001 ~ 10 m/s (로그 스케일)', 'indent')
        add('  • q 범위: PSD 적분에 사용할 파수 범위', 'indent')
        add('  • 설정 완료 후 "G(q,v) 계산 실행" 클릭', 'indent')
        add('')

        add('탭 4: G(q,v) 결과', 'section')
        add('계산된 G(q,v) 결과를 시각화합니다.')
        add('  • G(q) 곡선: 각 속도별 누적 탄성 에너지', 'indent')
        add('  • G(q,v) 히트맵: 파수-속도 평면에서의 G 분포', 'indent')
        add('  • P(q,v) 접촉 면적: 실접촉 면적 비율', 'indent')
        add('')

        add('탭 5: h\'rms / Strain', 'section')
        add('표면 거칠기의 RMS 기울기와 국소 변형률(strain)을 계산합니다.')
        add('  • h\'_rms(q): 파수까지의 누적 RMS 기울기', 'indent')
        add('  • ε(q) = α × h\'_rms: 국소 변형률 (비선형 보정에 필요)', 'indent')
        add('')

        add('탭 6: μ_visc 계산', 'section')
        add('최종 점탄성 마찰 계수를 계산합니다.')
        add('  • Strain Sweep 데이터로 f(ε), g(ε) 비선형 보정 가능', 'indent')
        add('  • 선형/비선형 모드 비교 지원', 'indent')
        add('  • 속도-마찰계수 곡선 (μ vs v) 출력', 'indent')
        add('')

        add('탭 7: 점탄성 설계', 'section')
        add('재료 설계 가이드라인을 제공합니다.')
        add('  • 주파수 감도 분석 W(f)', 'indent')
        add('  • 마찰 기여 주파수 대역 식별', 'indent')
        add('')

        add('탭 8: Strain Map', 'section')
        add('파수-속도 평면에서의 strain 분포를 히트맵으로 시각화합니다.')
        add('')

        add('탭 9: 피적분함수', 'section')
        add('G(q) 및 μ_visc 적분의 피적분함수(integrand)를 시각화하여 어떤 파수 대역이 계산에 가장 큰 기여를 하는지 분석합니다.')
        add('')

        add('탭 10-11: 수식 정리 / 변수 관계', 'section')
        add('이론 수식과 변수 관계를 참조할 수 있는 레퍼런스 탭입니다.')
        add('')

        add('탭 12: 디버그', 'section')
        add('계산 중간값과 진단 정보를 확인할 수 있습니다.')
        add('')

        add('탭 13: 영향 인자', 'section')
        add('마찰 계수에 영향을 미치는 주요 인자들의 감도 분석입니다.')
        add('')

        add('일반 사용 팁', 'section')
        add('  • File > Load DMA Data: DMA 마스터 커브 데이터 로드', 'indent')
        add('  • File > Save Results: 계산 결과를 CSV로 저장', 'indent')
        add('  • File > Graph Data Export: 그래프 데이터를 내보내기', 'indent')
        add('  • Settings > 레이아웃 설정: 글꼴 크기, 패널 폭, 창 크기 조절', 'indent')
        add('  • 각 탭 상단의 도구 모음에서 주요 동작 버튼을 사용하세요', 'indent')
        add('')

        add('※ 계산 순서: 탭 1~3 → 탭 4 (G 계산) → 탭 5 (h\'rms) → 탭 6 (μ_visc)', 'note')

        text_widget.config(state='disabled')

        # Close button
        btn_frame = ttk.Frame(dialog, padding=10)
        btn_frame.pack(fill=tk.X)
        ttk.Button(btn_frame, text="닫기", command=dialog.destroy, width=12).pack(side=tk.RIGHT, padx=5)

    def _apply_panel_width_recursive(self, widget, new_width):
        """Recursively find and resize left panel frames in a tab."""
        try:
            # Check if this is a direct child Frame with pack_propagate(False)
            # which indicates it's a fixed-width left panel
            info = widget.pack_info() if hasattr(widget, 'pack_info') else None
            if info and info.get('side') == 'left' and not info.get('expand'):
                # Check if this frame has pack_propagate disabled (fixed-width panel)
                if isinstance(widget, (ttk.Frame, tk.Frame)):
                    try:
                        current_w = widget.cget('width')
                        if current_w and 300 < int(current_w) < 1200:
                            widget.configure(width=new_width)
                    except (tk.TclError, ValueError):
                        pass
        except (tk.TclError, AttributeError):
            pass

        # Recurse into children
        try:
            for child in widget.winfo_children():
                self._apply_panel_width_recursive(child, new_width)
        except (tk.TclError, AttributeError):
            pass

    def _open_layout_settings(self):
        """Open global layout settings control panel."""
        dialog = tk.Toplevel(self.root)
        dialog.title("레이아웃 설정")
        dialog.resizable(True, True)
        dialog.transient(self.root)
        dialog.grab_set()

        dlg_w, dlg_h = 700, 850
        x = self.root.winfo_x() + (self.root.winfo_width() - dlg_w) // 2
        y = self.root.winfo_y() + max(0, (self.root.winfo_height() - dlg_h) // 2)
        dialog.geometry(f"{dlg_w}x{dlg_h}+{x}+{y}")
        dialog.minsize(600, 700)

        C = self.COLORS

        # Title
        title_frame = tk.Frame(dialog, bg=C['sidebar'], padx=12, pady=8)
        title_frame.pack(fill=tk.X)
        tk.Label(title_frame, text="레이아웃 제어판", bg=C['sidebar'], fg='white',
                 font=('Segoe UI', 18, 'bold')).pack(anchor=tk.W)

        # Scrollable content area to fit all settings
        outer_frame = ttk.Frame(dialog)
        outer_frame.pack(fill=tk.BOTH, expand=True)

        settings_canvas = tk.Canvas(outer_frame, highlightthickness=0)
        settings_scrollbar = ttk.Scrollbar(outer_frame, orient=tk.VERTICAL, command=settings_canvas.yview)
        content_frame = ttk.Frame(settings_canvas, padding=15)

        content_frame.bind("<Configure>",
                           lambda e: settings_canvas.configure(scrollregion=settings_canvas.bbox("all")))
        _settings_cw = settings_canvas.create_window((0, 0), window=content_frame, anchor="nw")
        settings_canvas.configure(yscrollcommand=settings_scrollbar.set)

        def _on_settings_canvas_configure(event):
            settings_canvas.itemconfig(_settings_cw, width=event.width)
        settings_canvas.bind('<Configure>', _on_settings_canvas_configure)

        settings_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        settings_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # ── Section 1: Font Settings ──
        font_frame = ttk.LabelFrame(content_frame, text="글꼴 설정", padding=10)
        font_frame.pack(fill=tk.X, pady=(0, 10))

        # UI Font Family
        row1 = ttk.Frame(font_frame)
        row1.pack(fill=tk.X, pady=3)
        ttk.Label(row1, text="UI 글꼴:", width=15).pack(side=tk.LEFT)
        ui_font_var = tk.StringVar(value=self.FONTS['body'][0])
        ui_font_combo = ttk.Combobox(row1, textvariable=ui_font_var, width=20,
                                      values=['Segoe UI', 'Malgun Gothic', 'NanumGothic',
                                              'Arial', 'Helvetica', 'Verdana', 'Consolas'])
        ui_font_combo.pack(side=tk.LEFT, padx=5)

        # UI Font Size
        row2 = ttk.Frame(font_frame)
        row2.pack(fill=tk.X, pady=3)
        ttk.Label(row2, text="UI 글꼴 크기:", width=15).pack(side=tk.LEFT)
        ui_size_var = tk.IntVar(value=self.FONTS['body'][1])
        ui_size_spin = ttk.Spinbox(row2, from_=10, to=30, textvariable=ui_size_var, width=6)
        ui_size_spin.pack(side=tk.LEFT, padx=5)
        ttk.Label(row2, text="pt", font=('Segoe UI', 13)).pack(side=tk.LEFT)

        # Plot Font Size
        row3 = ttk.Frame(font_frame)
        row3.pack(fill=tk.X, pady=3)
        ttk.Label(row3, text="그래프 글꼴 크기:", width=15).pack(side=tk.LEFT)
        plot_size_var = tk.IntVar(value=self.PLOT_FONTS['title'])
        plot_size_spin = ttk.Spinbox(row3, from_=8, to=28, textvariable=plot_size_var, width=6)
        plot_size_spin.pack(side=tk.LEFT, padx=5)
        ttk.Label(row3, text="pt (title 기준)", font=('Segoe UI', 13)).pack(side=tk.LEFT)

        # Mono Font
        row4 = ttk.Frame(font_frame)
        row4.pack(fill=tk.X, pady=3)
        ttk.Label(row4, text="코드 글꼴:", width=15).pack(side=tk.LEFT)
        mono_font_var = tk.StringVar(value=self.FONTS['mono'][0])
        mono_font_combo = ttk.Combobox(row4, textvariable=mono_font_var, width=20,
                                        values=['Consolas', 'Courier New', 'Lucida Console',
                                                'DejaVu Sans Mono', 'Source Code Pro'])
        mono_font_combo.pack(side=tk.LEFT, padx=5)

        # ── Section 2: Panel Width Settings ──
        panel_frame = ttk.LabelFrame(content_frame, text="패널 폭 설정", padding=10)
        panel_frame.pack(fill=tk.X, pady=(0, 10))

        # Get current left panel width (default 600)
        current_panel_width = getattr(self, '_left_panel_width', 600)

        row5 = ttk.Frame(panel_frame)
        row5.pack(fill=tk.X, pady=3)
        ttk.Label(row5, text="컨트롤 패널 폭:", width=15).pack(side=tk.LEFT)
        panel_width_var = tk.IntVar(value=current_panel_width)
        panel_width_scale = ttk.Scale(row5, from_=400, to=900, variable=panel_width_var,
                                       orient=tk.HORIZONTAL, length=250)
        panel_width_scale.pack(side=tk.LEFT, padx=5)
        panel_width_label = ttk.Label(row5, text=f"{current_panel_width}px", width=8)
        panel_width_label.pack(side=tk.LEFT)

        def _update_panel_width_label(*args):
            panel_width_label.config(text=f"{panel_width_var.get()}px")
        panel_width_var.trace_add('write', _update_panel_width_label)

        # ── Section 3: Window Settings ──
        window_frame = ttk.LabelFrame(content_frame, text="창 설정", padding=10)
        window_frame.pack(fill=tk.X, pady=(0, 10))

        row6 = ttk.Frame(window_frame)
        row6.pack(fill=tk.X, pady=3)
        ttk.Label(row6, text="창 크기:", width=15).pack(side=tk.LEFT)
        win_w_var = tk.IntVar(value=self.root.winfo_width())
        win_h_var = tk.IntVar(value=self.root.winfo_height())
        ttk.Spinbox(row6, from_=1000, to=3000, textvariable=win_w_var, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(row6, text="x").pack(side=tk.LEFT)
        ttk.Spinbox(row6, from_=600, to=2000, textvariable=win_h_var, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(row6, text="px", font=('Segoe UI', 13)).pack(side=tk.LEFT, padx=3)

        # ── Section 4: Plot Theme ──
        theme_frame = ttk.LabelFrame(content_frame, text="그래프 테마", padding=10)
        theme_frame.pack(fill=tk.X, pady=(0, 10))

        row7 = ttk.Frame(theme_frame)
        row7.pack(fill=tk.X, pady=3)
        ttk.Label(row7, text="수식 폰트:", width=15).pack(side=tk.LEFT)
        math_font_var = tk.StringVar(value=matplotlib.rcParams.get('mathtext.fontset', 'dejavusans'))
        math_combo = ttk.Combobox(row7, textvariable=math_font_var, width=20,
                                   values=['cm', 'stix', 'stixsans', 'dejavusans', 'dejavuserif'])
        math_combo.pack(side=tk.LEFT, padx=5)
        ttk.Label(row7, text="(cm=Cambria Math \uc2a4\ud0c0\uc77c)", font=('Segoe UI', 13),
                  foreground='#64748B').pack(side=tk.LEFT, padx=3)

        # ── Buttons ──
        btn_frame = ttk.Frame(dialog, padding=10)
        btn_frame.pack(fill=tk.X)

        def apply_settings():
            """Apply all layout settings."""
            import tkinter.font as tkfont

            # 1. Update FONTS dict
            new_family = ui_font_var.get()
            new_size = ui_size_var.get()
            mono_family = mono_font_var.get()

            self.FONTS = {
                'heading':   (new_family, new_size + 5, 'bold'),
                'subheading':(new_family, new_size + 3, 'bold'),
                'body':      (new_family, new_size),
                'body_bold': (new_family, new_size, 'bold'),
                'small':     (new_family, new_size - 1),
                'small_bold':(new_family, new_size - 1, 'bold'),
                'tiny':      (new_family, new_size - 2),
                'mono':      (mono_family, new_size),
                'mono_small':(mono_family, new_size - 1),
            }

            # Update tk default fonts
            for fname in ('TkDefaultFont', 'TkTextFont'):
                try:
                    f = tkfont.nametofont(fname)
                    f.configure(family=new_family, size=new_size)
                except Exception:
                    pass
            try:
                f = tkfont.nametofont('TkFixedFont')
                f.configure(family=mono_family, size=new_size)
            except Exception:
                pass

            # 2. Update plot font sizes
            new_plot_title = plot_size_var.get()
            scale = new_plot_title / 15.0  # 15 is default title size
            self.PLOT_FONTS = {
                'title': new_plot_title,
                'label': max(8, round(13 * scale)),
                'tick': max(8, round(12 * scale)),
                'legend': max(8, round(12 * scale)),
                'suptitle': max(10, round(16 * scale)),
                'annotation': max(8, round(12 * scale)),
                'title_sm': max(8, round(13 * scale)),
                'label_sm': max(8, round(12 * scale)),
                'legend_sm': max(7, round(10 * scale)),
            }

            # Update matplotlib rcParams
            matplotlib.rcParams.update({
                'font.size': self.PLOT_FONTS['label'],
                'axes.titlesize': self.PLOT_FONTS['title'],
                'axes.labelsize': self.PLOT_FONTS['label'],
                'xtick.labelsize': self.PLOT_FONTS['tick'],
                'ytick.labelsize': self.PLOT_FONTS['tick'],
                'legend.fontsize': self.PLOT_FONTS['legend'],
                'figure.titlesize': self.PLOT_FONTS['suptitle'],
                'mathtext.fontset': math_font_var.get(),
            })

            # 3. Update base font config for responsive scaling
            self._base_font_cfg = {
                'font.size':        self.PLOT_FONTS['label'],
                'axes.titlesize':   self.PLOT_FONTS['title'],
                'axes.labelsize':   self.PLOT_FONTS['label'],
                'xtick.labelsize':  self.PLOT_FONTS['tick'],
                'ytick.labelsize':  self.PLOT_FONTS['tick'],
                'legend.fontsize':  self.PLOT_FONTS['legend'],
                'figure.titlesize': self.PLOT_FONTS['suptitle'],
            }

            # 4. Update left panel width and apply to existing frames
            new_panel_w = panel_width_var.get()
            self._left_panel_width = new_panel_w

            # Apply width to all existing left panel frames in notebook tabs
            for tab_id in self.notebook.tabs():
                tab_widget = self.notebook.nametowidget(tab_id)
                self._apply_panel_width_recursive(tab_widget, new_panel_w)

            # 5. Update window size
            new_win_w = win_w_var.get()
            new_win_h = win_h_var.get()
            self.root.geometry(f"{new_win_w}x{new_win_h}")

            # 6. Re-apply theme with new fonts
            self._setup_modern_theme()

            # 7. Rescale all existing plot fonts
            self._font_scale = 0  # force rescale
            self._rescale_all_fonts()

            dialog.destroy()
            self._show_status("레이아웃 설정이 적용되었습니다.\n일부 변경은 탭 전환 시 반영됩니다.", 'success')

        def reset_defaults():
            """Reset to default settings."""
            ui_font_var.set('Segoe UI')
            ui_size_var.set(17)
            plot_size_var.set(15)
            mono_font_var.set('Consolas')
            panel_width_var.set(600)
            win_w_var.set(1600)
            win_h_var.set(1000)
            math_font_var.set('dejavusans')

        ttk.Button(btn_frame, text="적용", command=apply_settings, width=12).pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="초기화", command=reset_defaults, width=12).pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="취소", command=dialog.destroy, width=12).pack(side=tk.RIGHT, padx=5)

    def _show_about(self):
        """Show about dialog as a popup window."""
        dialog = tk.Toplevel(self.root)
        dialog.title("About")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        dlg_w, dlg_h = 480, 420
        x = self.root.winfo_x() + (self.root.winfo_width() - dlg_w) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - dlg_h) // 2
        dialog.geometry(f"{dlg_w}x{dlg_h}+{x}+{y}")

        C = self.COLORS

        # Header
        header = tk.Frame(dialog, bg=C['sidebar'], padx=20, pady=15)
        header.pack(fill=tk.X)
        tk.Label(header, text="NEXEN Rubber Friction Modelling Program",
                 bg=C['sidebar'], fg='white',
                 font=('Segoe UI', 18, 'bold')).pack(anchor=tk.W)
        tk.Label(header, text="v3.0  |  Based on Persson Contact Mechanics Theory",
                 bg=C['sidebar'], fg='#94A3B8',
                 font=('Segoe UI', 12)).pack(anchor=tk.W)

        # Content
        content = tk.Frame(dialog, bg='white', padx=25, pady=20)
        content.pack(fill=tk.BOTH, expand=True)

        def add_label(text, size=14, bold=False, fg='#1E293B', pady=(0, 2)):
            weight = 'bold' if bold else 'normal'
            tk.Label(content, text=text, bg='white', fg=fg,
                     font=('Segoe UI', size, weight),
                     anchor='w', justify=tk.LEFT).pack(anchor='w', pady=pady)

        add_label('이론적 기반', size=15, bold=True, fg='#7C3AED', pady=(5, 4))
        add_label('Persson, B.N.J. (2001, 2006)')
        add_label('Rubber friction and contact mechanics theory')
        add_label('')

        add_label('개발', size=15, bold=True, fg='#7C3AED', pady=(5, 4))
        add_label('NEXENTIRE Material Research Team')
        add_label('Baekhwan Kim (김백환)')
        add_label('')

        add_label('빌드 일자', size=15, bold=True, fg='#7C3AED', pady=(5, 4))
        add_label('2026.02.25')

        # Separator + Close button
        tk.Frame(dialog, bg='#CBD5E1', height=1).pack(fill=tk.X)
        btn_frame = tk.Frame(dialog, bg='white', padx=20, pady=10)
        btn_frame.pack(fill=tk.X)
        ttk.Button(btn_frame, text="닫기", command=dialog.destroy, width=12).pack(side=tk.RIGHT)

    def _create_equations_tab(self, parent):
        """Create equations reference tab - single unified scrollable layout with Cambria Math."""
        # Toolbar
        self._create_panel_toolbar(parent)

        # Single scrollable canvas for the entire tab
        canvas = tk.Canvas(parent, bg='white', highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='white')

        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Make scrollable_frame fill canvas width
        def _on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)
        canvas.bind('<Configure>', _on_canvas_configure)

        def _update_scrollregion(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))
        scrollable_frame.bind("<Configure>", _update_scrollregion)

        # Mouse wheel scrolling (cross-platform)
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        def _on_mousewheel_linux_up(event):
            canvas.yview_scroll(-3, "units")
        def _on_mousewheel_linux_down(event):
            canvas.yview_scroll(3, "units")

        def _bind_mousewheel(event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
            canvas.bind_all("<Button-4>", _on_mousewheel_linux_up)
            canvas.bind_all("<Button-5>", _on_mousewheel_linux_down)
        def _unbind_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")

        canvas.bind('<Enter>', _bind_mousewheel)
        canvas.bind('<Leave>', _unbind_mousewheel)

        # Pack scrollbar and canvas
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        # --- Helper: create a matplotlib figure for a LaTeX equation block ---
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        # Use Cambria Math for mathtext (good LaTeX rendering)
        math_fontfamily = 'cm'  # Computer Modern (built-in, best LaTeX look)

        def add_section_title(title_text, bg_color='#1B2A4A', fg_color='white'):
            """Add a colored section header."""
            frame = tk.Frame(scrollable_frame, bg=bg_color, padx=15, pady=12)
            frame.pack(fill=tk.X, padx=10, pady=(18, 4))
            tk.Label(frame, text=title_text, bg=bg_color, fg=fg_color,
                     font=('Segoe UI', 24, 'bold')).pack(anchor=tk.W)

        def add_text(text, font_size=19, fg='#1E293B', bold=False, padx=20, pady=4):
            """Add a plain text label."""
            weight = 'bold' if bold else 'normal'
            lbl = tk.Label(scrollable_frame, text=text, bg='white', fg=fg,
                           font=('Segoe UI', font_size, weight),
                           justify=tk.LEFT, anchor='w', wraplength=1800)
            lbl.pack(fill=tk.X, padx=padx, pady=pady, anchor='w')

        def add_equation(latex_str, fig_height=1.2, font_size=24):
            """Add a LaTeX equation rendered via matplotlib."""
            fig = Figure(figsize=(14, fig_height), facecolor='white')
            ax = fig.add_subplot(111)
            ax.axis('off')
            ax.text(0.02, 0.5, latex_str, transform=ax.transAxes,
                    fontsize=font_size, verticalalignment='center',
                    horizontalalignment='left', usetex=False,
                    math_fontfamily=math_fontfamily)
            fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.12)
            eq_canvas = FigureCanvasTkAgg(fig, master=scrollable_frame)
            eq_canvas.draw()
            eq_canvas.get_tk_widget().configure(height=int(fig_height * 80))
            eq_canvas.get_tk_widget().pack(fill=tk.X, padx=20, pady=(8, 8))

        def add_separator():
            tk.Frame(scrollable_frame, bg='#CBD5E1', height=2).pack(fill=tk.X, padx=10, pady=10)

        def add_graph(plot_func, fig_height=3.5):
            """Add an illustrative matplotlib graph."""
            import numpy as np
            fig = Figure(figsize=(12, fig_height), facecolor='#FAFBFC')
            ax = fig.add_subplot(111)
            ax.set_facecolor('#FAFBFC')
            plot_func(ax, np)
            ax.tick_params(labelsize=14)
            for spine in ax.spines.values():
                spine.set_color('#CBD5E1')
            fig.tight_layout(pad=1.5)
            graph_canvas = FigureCanvasTkAgg(fig, master=scrollable_frame)
            graph_canvas.draw()
            graph_canvas.get_tk_widget().configure(height=int(fig_height * 72))
            graph_canvas.get_tk_widget().pack(fill=tk.X, padx=30, pady=(6, 14))

        # === Title ===
        title_frame = tk.Frame(scrollable_frame, bg='white', pady=10)
        title_frame.pack(fill=tk.X, padx=10)
        tk.Label(title_frame, text='Persson 마찰 이론 - 계산 수식 정리',
                 bg='white', fg='#1B2A4A',
                 font=('Segoe UI', 28, 'bold')).pack(anchor='w', padx=10)

        # ═══════════════════════════════════════════════════════
        # Section 0: 기본 물리량 정의
        # ═══════════════════════════════════════════════════════
        add_section_title('0. 기본 물리량 정의')

        add_text('주파수 (고무가 느끼는 진동수):', bold=True, pady=(8, 0))
        add_equation(r'$\omega = q \cdot v \cdot \cos\phi$', fig_height=0.9)
        add_text('  q : 파수(wavenumber) — 표면 거칠기의 공간 진동수. 값이 클수록 더 미세한 요철을 의미', font_size=17, fg='#64748B')
        add_text('  v : 슬라이딩 속도 — 고무가 바닥 위를 미끄러지는 속도 [m/s]', font_size=17, fg='#64748B')
        add_text('  \u03c6 : 슬라이딩 방향과 파수 벡터 사이의 각도 (0~2\u03c0)', font_size=17, fg='#64748B')
        add_text('  \u03c9 : 고무가 표면 요철을 타고 넘으며 느끼는 진동 주파수. q가 클수록(미세 요철), v가 빠를수록 \u03c9 증가', font_size=17, fg='#64748B')

        def _plot_omega_vs_phi(ax, np):
            phi = np.linspace(0, 2*np.pi, 300)
            v = 0.01
            for q_val, c in [(1e3, '#2563EB'), (1e4, '#DC2626'), (1e5, '#059669')]:
                omega = q_val * v * np.cos(phi)
                ax.plot(np.degrees(phi), omega, linewidth=2.5, color=c,
                        label=f'q = {q_val:.0e} (v={v} m/s)')
            ax.set_xlabel(r'$\phi$ (degrees)', fontsize=16)
            ax.set_ylabel(r'$\omega$ (rad/s)', fontsize=16)
            ax.legend(fontsize=14, loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.set_title(r'$\omega = q \cdot v \cdot \cos\phi$ — 각도에 따른 진동수 변화', fontsize=16, pad=10)
        add_graph(_plot_omega_vs_phi)

        add_text('유효 탄성률 (평면 변형 상태):', bold=True, pady=(10, 0))
        add_equation(r'$E^*(\omega) = \frac{E(\omega)}{1-\nu^2}$', fig_height=1.0)
        add_text('  E(\u03c9) = E\'(\u03c9) + iE\'\'(\u03c9) : DMA 실험에서 측정한 복소 탄성률', font_size=17, fg='#64748B')
        add_text('    E\'(\u03c9) : 저장 탄성률 — 탄성 에너지를 저장하는 능력 (스프링 성분)', font_size=17, fg='#64748B')
        add_text('    E\'\'(\u03c9) : 손실 탄성률 — 에너지를 열로 소산하는 능력 (댐퍼 성분, 마찰의 원인)', font_size=17, fg='#64748B')
        add_text('  \u03bd : 푸아송 비 — 고무를 누를 때 옆으로 퍼지는 정도 (고무 \u2248 0.5, 거의 비압축성)', font_size=17, fg='#64748B')
        add_text('  (1-\u03bd\u00b2) 보정: 표면 접촉은 3차원 구속 상태이므로 단축 탄성률보다 더 뻣뻣하게 보정', font_size=17, fg='#64748B')

        def _plot_master_curve(ax, np):
            omega = np.logspace(-2, 10, 500)
            E_stor = 1e6 + (1e9 - 1e6) / (1 + (1e4 / omega)**0.6)
            E_loss = 0.35e9 * (omega / 1e4)**0.5 / (1 + (omega / 1e4)**0.85)
            ax.loglog(omega, E_stor, '-', linewidth=2.5, color='#2563EB', label="E' (저장 탄성률)")
            ax.loglog(omega, E_loss, '--', linewidth=2.5, color='#DC2626', label="E'' (손실 탄성률)")
            ax.set_xlabel(r'$\omega$ (rad/s)', fontsize=16)
            ax.set_ylabel('E (Pa)', fontsize=16)
            ax.legend(fontsize=14)
            ax.grid(True, alpha=0.3, which='both')
            ax.set_title("복소 탄성률 마스터 커브 (대표적 형상)", fontsize=16, pad=10)
        add_graph(_plot_master_curve)

        # ═══════════════════════════════════════════════════════
        # Section 1: G(q) 함수와 접촉 면적
        # ═══════════════════════════════════════════════════════
        add_section_title('1. G(q) 함수 — 거칠기에 의한 탄성 에너지 적분')

        add_text('A. 파워 스펙트럼 적분함수 G(q):', bold=True, pady=(8, 0))
        add_equation(
            r'$G(q) = \frac{1}{8} \int_{q_0}^{q} dq^{\prime}\, (q^{\prime})^3\, C(q^{\prime})'
            r' \int_{0}^{2\pi} d\phi\, \left| \frac{E(q^{\prime}v\cos\phi)}{(1-\nu^2)\sigma_0} \right|^2$',
            fig_height=1.3)
        add_text('물리적 의미:', font_size=17, bold=True, fg='#1E293B')
        add_text('  G(q)는 "파수 q₀부터 q까지의 거칠기 성분이 고무를 변형시키며 저장하는 탄성 에너지의 누적량"', font_size=17, fg='#64748B')
        add_text('  → G(q)가 크면: 고무가 요철을 따라가기 어려워 접촉 면적이 줄어듦', font_size=17, fg='#64748B')
        add_text('  → G(q)가 작으면: 고무가 요철에 잘 밀착하여 접촉 면적이 넓음', font_size=17, fg='#64748B')
        add_text('각 변수의 역할:', font_size=17, bold=True, fg='#1E293B')
        add_text('  q\'³ C(q\') : 파수 q\'에서의 거칠기 기여분. q\'³은 미세 요철일수록 기울기 기여가 큰 것을 반영', font_size=17, fg='#64748B')
        add_text('  |E/(σ₀(1-ν²))|² : 탄성률 대비 압력의 비율. 고무가 뻣뻣할수록(E↑) 변형 에너지가 커지고, 압력이 클수록(σ₀↑) 상대적으로 줄어듦', font_size=17, fg='#64748B')
        add_text('  단위: 무차원 (σ₀로 나누었으므로)', font_size=17, fg='#64748B')

        def _plot_G_vs_q(ax, np):
            q = np.logspace(2, 8, 500)
            G = 0.01 * (q / 1e2)**1.2 / (1 + (q / 1e7)**0.3)
            ax.loglog(q, G, '-', linewidth=2.5, color='#2563EB')
            ax.set_xlabel('q (1/m)', fontsize=16)
            ax.set_ylabel('G(q)', fontsize=16)
            ax.grid(True, alpha=0.3, which='both')
            ax.set_title('G(q) — 탄성 에너지 누적 적분', fontsize=16, pad=10)
            ax.annotate('G 증가 → 접촉면적 감소', xy=(1e6, 50), fontsize=14, color='#DC2626',
                        fontweight='bold')
        add_graph(_plot_G_vs_q)

        add_separator()

        # ── 각도 적분의 물리적 의미 ──
        add_text('각도 적분 ∫₀²π dφ 의 물리적 의미:', bold=True, fg='#7C3AED', pady=(8, 0))
        add_text('  실제 표면 거칠기는 2차원(x,y 평면)에 분포하지만, 슬라이딩은 한 방향(예: x축)으로 일어남', font_size=17, fg='#64748B')
        add_text('  → 파수 벡터 q = (qₓ, qᵧ)를 극좌표로 표현하면: qₓ = q·cos\u03c6, qᵧ = q·sin\u03c6', font_size=17, fg='#64748B')
        add_text('  → 슬라이딩 방향(x축)과 각도 \u03c6를 이루는 요철이 고무에 주는 진동 주파수는 \u03c9 = q·v·cos\u03c6', font_size=17, fg='#64748B')
        add_text('  → \u03c6 = 0° (슬라이딩 방향과 평행): 고무가 요철을 정면으로 타넘어 → 주파수 최대', font_size=17, fg='#64748B')
        add_text('  → \u03c6 = 90° (슬라이딩 방향과 수직): 고무가 요철과 나란히 미끄러져 → 주파수 0 (기여 없음)', font_size=17, fg='#64748B')
        add_text('  → 0~2\u03c0 적분 = 모든 방향의 요철 기여를 합산 (2D 표면의 등방 거칠기를 완전하게 반영)', font_size=17, fg='#64748B')

        add_separator()

        # ── P(q) = erf(...) 설명 ──
        add_text('B. 실접촉 면적 비율 P(q):', bold=True, pady=(6, 0))
        add_equation(
            r'$\frac{A(q)}{A_0} = P(q) \approx \mathrm{erf}\!\left( \frac{1}{2\sqrt{G(q)}} \right)$',
            fig_height=1.3)
        add_text('물리적 의미: 배율 q에서 바닥과 실제로 닿아있는 면적의 비율 (0 ≤ P ≤ 1)', font_size=17, fg='#64748B')

        def _plot_P_erf(ax, np):
            from scipy.special import erf
            G = np.linspace(0.01, 20, 500)
            P = erf(1 / (2 * np.sqrt(G)))
            ax.plot(G, P, '-', linewidth=2.5, color='#2563EB')
            ax.set_xlabel('G(q)', fontsize=16)
            ax.set_ylabel(r'P(q) = A(q)/A$_0$', fontsize=16)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.02, 1.05)
            ax.set_title('P(q) = erf(1/(2√G)) — G 증가에 따른 접촉면적 비율 감소', fontsize=16, pad=10)
            ax.annotate('G 작음 → 완전접촉', xy=(0.5, 0.92), fontsize=14, color='#059669', fontweight='bold')
            ax.annotate('G 큼 → 접촉감소', xy=(12, 0.15), fontsize=14, color='#DC2626', fontweight='bold')
        add_graph(_plot_P_erf)

        add_text('erf(x) 함수란?', bold=True, fg='#7C3AED', pady=(10, 0))
        add_text('  erf(x)는 오차 함수(error function)로, 가우시안 분포의 누적 확률을 나타냄:', font_size=17, fg='#64748B')
        add_equation(
            r'$\mathrm{erf}(x) = \frac{2}{\sqrt{\pi}} \int_{0}^{x} e^{-t^2}\, dt$',
            fig_height=1.2)
        add_text('  x = 0 → erf(0) = 0  |  x → ∞ → erf(∞) = 1  |  S자 형태로 0에서 1까지 증가', font_size=17, fg='#64748B')
        add_text('  직관: "가우시안 분포에서 평균 ± x 범위 안에 포함되는 비율"', font_size=17, fg='#64748B')

        def _plot_erf(ax, np):
            from scipy.special import erf
            x = np.linspace(-3, 3, 500)
            ax.plot(x, erf(x), '-', linewidth=2.5, color='#7C3AED')
            ax.set_xlabel('x', fontsize=16)
            ax.set_ylabel('erf(x)', fontsize=16)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.axhline(y=1, color='gray', linestyle=':', alpha=0.4)
            ax.axhline(y=-1, color='gray', linestyle=':', alpha=0.4)
            ax.set_title('erf(x) — 오차 함수', fontsize=16, pad=10)
        add_graph(_plot_erf)

        add_text('왜 A/A₀ = erf(1/(2√G)) 인가?', bold=True, fg='#7C3AED', pady=(10, 0))
        add_text('  Persson 이론에서 접촉 응력 σ는 가우시안 분포를 따름 (평균=σ₀, 분산∝G)', font_size=17, fg='#64748B')
        add_text('  실접촉 = 응력이 0보다 큰 영역 → σ > 0 인 확률을 적분', font_size=17, fg='#64748B')
        add_text('  가우시안의 σ > 0 누적확률을 계산하면 자연스럽게 erf 함수가 나옴:', font_size=17, fg='#64748B')
        add_equation(
            r'$P(q) = \int_{0}^{\infty} P(\sigma, q)\, d\sigma = \mathrm{erf}\!\left(\frac{\sigma_0}{2\sqrt{G(q)}\,\sigma_0}\right)'
            r' = \mathrm{erf}\!\left(\frac{1}{2\sqrt{G(q)}}\right)$',
            fig_height=1.3)
        add_text('  G(q) 작을 때: 분포가 좁음 → 거의 모든 점이 σ>0 → P ≈ 1 (완전 접촉)', font_size=17, fg='#64748B')
        add_text('  G(q) 클 때: 분포가 넓음 → σ<0인 영역 증가 → P → 0 (접촉 감소)', font_size=17, fg='#64748B')

        add_separator()

        # ── μ_visc ──
        add_text('C. 점탄성 마찰 계수 μ_visc:', bold=True, pady=(6, 0))
        add_equation(
            r'$\mu_{visc} \approx \frac{1}{2} \int_{q_0}^{q_1} dq\, q^3 C(q)\, S(q)\, P(q)'
            r' \int_{0}^{2\pi} d\phi\, \cos\phi\, \mathrm{Im}\!\left( \frac{E(qv\cos\phi)}{(1-\nu^2)\sigma_0} \right)$',
            fig_height=1.3)
        add_text('물리적 의미:', font_size=17, bold=True, fg='#1E293B')
        add_text('  고무가 거친 바닥 위를 미끄러질 때, 각 파수의 요철이 고무를 변형시키며 소산하는 에너지의 총합', font_size=17, fg='#64748B')
        add_text('각 항의 역할:', font_size=17, bold=True, fg='#1E293B')
        add_text('  q³C(q) : 파수 q에서의 거칠기 기울기 기여 (미세 요철일수록 기울기가 가파름)', font_size=17, fg='#64748B')
        add_text('  P(q) : 실접촉 면적 비율 — 닿아있는 면적만 마찰에 기여', font_size=17, fg='#64748B')
        add_text('  S(q) : 대변형 보정 — 접촉 면적이 줄어드는 효과를 보정', font_size=17, fg='#64748B')
        add_text('  Im[E(ω)] : 손실 탄성률 — 에너지 소산 (열로 변환)의 크기. 이것이 마찰력의 직접 원인', font_size=17, fg='#64748B')
        add_text('  cos\u03c6 : 슬라이딩 방향 성분만 마찰력에 기여 (수직 방향 요철은 마찰에 기여 안 함)', font_size=17, fg='#64748B')

        def _plot_mu_integrand(ax, np):
            q = np.logspace(2, 8, 500)
            integrand = q**3 * np.exp(-0.5 * ((np.log10(q) - 5) / 1.2)**2) * 1e-20
            ax.semilogx(q, integrand, '-', linewidth=2.5, color='#DC2626')
            ax.fill_between(q, integrand, alpha=0.15, color='#DC2626')
            ax.set_xlabel('q (1/m)', fontsize=16)
            ax.set_ylabel(r'$\mu$ 피적분함수', fontsize=16)
            ax.grid(True, alpha=0.3)
            ax.set_title(r'$\mu_{visc}$ 피적분함수 — 파수별 마찰 기여도', fontsize=16, pad=10)
            peak_idx = np.argmax(integrand)
            ax.annotate('마찰 기여 피크', xy=(q[peak_idx], integrand[peak_idx]),
                        fontsize=14, fontweight='bold', color='#DC2626',
                        xytext=(q[peak_idx]*5, integrand[peak_idx]*0.8),
                        arrowprops=dict(arrowstyle='->', color='#DC2626'))
        add_graph(_plot_mu_integrand)

        add_text('보정 계수 S(q):', bold=True, pady=(10, 0))
        add_equation(r'$S(q) = \gamma + (1-\gamma)\,P^2(q) \qquad (\gamma \approx 0.5)$', fig_height=0.9)
        add_text('  접촉 면적이 줄어들면 비접촉 영역의 고무도 변형에 참여 → 이를 보정하는 계수', font_size=17, fg='#64748B')

        def _plot_S_correction(ax, np):
            P = np.linspace(0, 1, 200)
            gamma = 0.5
            S = gamma + (1 - gamma) * P**2
            ax.plot(P, S, '-', linewidth=2.5, color='#059669')
            ax.set_xlabel('P(q)', fontsize=16)
            ax.set_ylabel('S(q)', fontsize=16)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0.4, 1.05)
            ax.set_title(r'S(q) = $\gamma$ + (1-$\gamma$)P²  (보정 계수, $\gamma$=0.5)', fontsize=16, pad=10)
            ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
            ax.annotate(r'$\gamma$ = 0.5 (최솟값)', xy=(0.05, 0.52), fontsize=14, color='#059669')
        add_graph(_plot_S_correction)

        # ═══════════════════════════════════════════════════════
        # Section 2: h_rms, h'_rms (RMS slope), Strain
        # ═══════════════════════════════════════════════════════
        add_section_title('2. 표면 거칠기 통계량: h_rms, h\'_rms, Strain')

        add_text('A. RMS 높이 h_rms (표면 거칠기의 크기):', bold=True, pady=(8, 0))
        add_equation(
            r'$h_{rms}^2(q) = 2\pi \int_{q_0}^{q} k\, C(k)\, dk$',
            fig_height=1.2)
        add_text('물리적 의미:', font_size=17, bold=True, fg='#1E293B')
        add_text('  h_rms는 표면 높이의 RMS(root mean square) 값으로, "표면이 평균으로부터 얼마나 위아래로 출렁이는가"', font_size=17, fg='#64748B')
        add_text('  피적분함수 k·C(k): 파수 k에서의 높이 기여분. 긴 파장(작은 q)의 거칠기가 h_rms에 주로 기여', font_size=17, fg='#64748B')
        add_text('  단위: [m] (미터)', font_size=17, fg='#64748B')

        def _plot_hrms(ax, np):
            q = np.logspace(2, 8, 500)
            hrms = 1e-5 * (1 - np.exp(-q / 1e4))
            ax.semilogx(q, hrms * 1e6, '-', linewidth=2.5, color='#2563EB')
            ax.set_xlabel('q (1/m)', fontsize=16)
            ax.set_ylabel(r'$h_{rms}$ ($\mu$m)', fontsize=16)
            ax.grid(True, alpha=0.3)
            ax.set_title(r'$h_{rms}(q)$ — 누적 RMS 높이 (큰 파장이 지배)', fontsize=16, pad=10)
            ax.annotate('긴 파장(작은 q)에서\n빠르게 포화', xy=(5e3, hrms[100]*1e6),
                        fontsize=14, color='#2563EB', fontweight='bold',
                        xytext=(1e5, hrms[100]*1e6*0.5),
                        arrowprops=dict(arrowstyle='->', color='#2563EB'))
        add_graph(_plot_hrms)

        add_separator()
        add_text('B. RMS 기울기 h\'_rms (표면 경사의 크기):', bold=True, pady=(6, 0))
        add_equation(
            r"$h_{rms}^{\prime\,2}(q) = \xi^2(q) = 2\pi \int_{q_0}^{q} k^3\, C(k)\, dk$",
            fig_height=1.2)
        add_text('물리적 의미:', font_size=17, bold=True, fg='#1E293B')
        add_text('  h\'_rms (= ξ)는 표면 기울기의 RMS 값으로, "표면이 얼마나 가파르게 경사져 있는가"', font_size=17, fg='#64748B')
        add_text('  피적분함수 k³·C(k): k³ 가중치로 인해 미세 요철(큰 q)일수록 기울기 기여가 매우 큼', font_size=17, fg='#64748B')
        add_text('  → h_rms는 긴 파장이 지배, h\'_rms는 짧은 파장이 지배 (같은 PSD에서 완전히 다른 특성)', font_size=17, fg='#64748B')
        add_text('  단위: [무차원] (길이/길이 = 기울기)', font_size=17, fg='#64748B')

        def _plot_hrms_slope(ax, np):
            q = np.logspace(2, 8, 500)
            hrms_slope = 0.001 * (q / 1e2)**0.8
            ax.loglog(q, hrms_slope, '-', linewidth=2.5, color='#DC2626')
            ax.set_xlabel('q (1/m)', fontsize=16)
            ax.set_ylabel(r"$h'_{rms} = \xi(q)$", fontsize=16)
            ax.grid(True, alpha=0.3, which='both')
            ax.set_title(r"$h'_{rms}(q)$ — 누적 RMS 기울기 (짧은 파장이 지배)", fontsize=16, pad=10)
            ax.annotate('짧은 파장(큰 q)에서\n계속 증가', xy=(1e6, 0.5),
                        fontsize=14, color='#DC2626', fontweight='bold')
        add_graph(_plot_hrms_slope)

        add_separator()
        add_text('C. 국소 변형률 ε(q) — h\'_rms로부터 정의:', bold=True, pady=(6, 0))
        add_equation(
            r"$\varepsilon(q) = \alpha \cdot h_{rms}^{\prime}(q) = \alpha \cdot \xi(q)$",
            fig_height=1.0)
        add_text('물리적 의미:', font_size=17, bold=True, fg='#1E293B')
        add_text('  고무가 거친 표면 요철을 따라 변형될 때, 접촉점 부근에서 고무가 받는 국소 변형률(strain)', font_size=17, fg='#64748B')
        add_text('  표면 기울기(h\'_rms)가 가파를수록 → 고무가 요철을 감싸기 위해 더 크게 변형 → ε 증가', font_size=17, fg='#64748B')
        add_text('  α : 비례 상수 (Persson 이론에서 α ≈ 0.5)', font_size=17, fg='#64748B')
        add_text('  비선형 보정에서의 역할:', font_size=17, bold=True, fg='#1E293B')
        add_text('  → ε(q)이 크면 고무의 Payne 효과(대변형 연화)가 발생', font_size=17, fg='#64748B')
        add_text('  → Strain Sweep 데이터에서 f(ε), g(ε) 함수를 구해 탄성률을 보정:', font_size=17, fg='#64748B')
        add_equation(
            r"$E'_{eff}(\omega) = E'(\omega) \times f(\varepsilon), \qquad E''_{eff}(\omega) = E''(\omega) \times g(\varepsilon)$",
            fig_height=1.0)
        add_text('  f(ε) ≤ 1 : 변형이 커지면 저장 탄성률 감소 (고무가 연화)', font_size=17, fg='#64748B')
        add_text('  g(ε) : 변형이 커지면 손실 탄성률이 먼저 증가했다 감소 (에너지 소산 패턴 변화)', font_size=17, fg='#64748B')

        def _plot_payne_effect(ax, np):
            eps = np.linspace(0, 50, 200)
            f_eps = 1 / (1 + 0.8 * (eps / 10)**0.9)
            g_eps = (1 + 2.5 * (eps / 10)) / (1 + 3 * (eps / 10)**1.4)
            ax.plot(eps, f_eps, '-', linewidth=2.5, color='#2563EB', label=r"f($\varepsilon$) — E' 감소율")
            ax.plot(eps, g_eps, '--', linewidth=2.5, color='#DC2626', label=r"g($\varepsilon$) — E'' 변화율")
            ax.set_xlabel(r'$\varepsilon$ (%)', fontsize=16)
            ax.set_ylabel('보정 계수', fontsize=16)
            ax.legend(fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.5)
            ax.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
            ax.set_title('Payne 효과 — 대변형 시 탄성률 보정 계수', fontsize=16, pad=10)
            ax.annotate('E\' 연화', xy=(30, f_eps[120]), fontsize=14, color='#2563EB', fontweight='bold')
            ax.annotate('E\'\' 피크 후 감소', xy=(15, max(g_eps)*0.95), fontsize=14, color='#DC2626', fontweight='bold')
        add_graph(_plot_payne_effect)

        # Bottom padding for scroll
        tk.Frame(scrollable_frame, bg='white', height=80).pack(fill=tk.X)

        # Force scrollregion update after all widgets are added
        self.root.after(100, _update_scrollregion)

    def _create_rms_slope_tab(self, parent):
        """Create h'rms / Local Strain calculation tab."""
        # Main container
        main_container = ttk.Frame(parent)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel for controls (fixed width)
        left_frame = ttk.Frame(main_container, width=getattr(self, '_left_panel_width', 600))
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        left_frame.pack_propagate(False)

        # Logo at bottom
        self._add_logo_to_panel(left_frame)

        # Toolbar (fixed at top, always accessible)
        self._create_panel_toolbar(left_frame, buttons=[
            ("h'rms / Local Strain 계산", self._calculate_rms_slope, 'Accent.TButton'),
        ])

        # ============== Left Panel: Controls ==============

        # 1. Description
        desc_frame = ttk.LabelFrame(left_frame, text="설명", padding=5)
        desc_frame.pack(fill=tk.X, pady=2, padx=3)

        desc_text = (
            "PSD 데이터로부터 h'rms(ξ)와\n"
            "Local Strain(ε)을 계산합니다.\n\n"
            "수식:\n"
            "  ξ²(q) = 2π ∫[q0→q] k³C(k)dk\n"
            "  ε(q) = factor × ξ(q)"
        )
        ttk.Label(desc_frame, text=desc_text, font=('Segoe UI', 17), justify=tk.LEFT).pack(anchor=tk.W)

        # 2. Calculation Settings
        settings_frame = ttk.LabelFrame(left_frame, text="계산 설정", padding=5)
        settings_frame.pack(fill=tk.X, pady=2, padx=3)

        # Strain factor
        row1 = ttk.Frame(settings_frame)
        row1.pack(fill=tk.X, pady=2)
        ttk.Label(row1, text="Strain Factor:", font=('Segoe UI', 17)).pack(side=tk.LEFT)
        self.strain_factor_var = tk.StringVar(value="0.5")
        ttk.Entry(row1, textvariable=self.strain_factor_var, width=8).pack(side=tk.RIGHT)

        ttk.Label(settings_frame, text="(ε = factor × ξ, Persson 권장: 0.5~1.0)",
                  font=('Segoe UI', 17), foreground='#64748B').pack(anchor=tk.W)

        # q range - Tab 3 (계산 설정)의 q_min/q_max 사용
        q_frame = ttk.LabelFrame(settings_frame, text="q 범위 (계산 설정 탭 연동)", padding=3)
        q_frame.pack(fill=tk.X, pady=3)

        row_q1 = ttk.Frame(q_frame)
        row_q1.pack(fill=tk.X, pady=1)
        ttk.Label(row_q1, text="q_min (1/m):", font=('Segoe UI', 17)).pack(side=tk.LEFT)
        self.rms_q_min_var = self.q_min_var  # 계산 설정 탭과 동일 변수 공유
        ttk.Label(row_q1, textvariable=self.rms_q_min_var, font=('Segoe UI', 17, 'bold'),
                  foreground='#2563EB').pack(side=tk.RIGHT)

        row_q2 = ttk.Frame(q_frame)
        row_q2.pack(fill=tk.X, pady=1)
        ttk.Label(row_q2, text="q_max (1/m):", font=('Segoe UI', 17)).pack(side=tk.LEFT)
        self.rms_q_max_var = self.q_max_var  # 계산 설정 탭과 동일 변수 공유
        ttk.Label(row_q2, textvariable=self.rms_q_max_var, font=('Segoe UI', 17, 'bold'),
                  foreground='#2563EB').pack(side=tk.RIGHT)

        ttk.Label(q_frame, text="※ 계산 설정 탭의 q 범위가 자동 적용됨",
                  font=('Segoe UI', 16), foreground='#64748B').pack(anchor=tk.W)

        # Target h'rms display (synced with Tab 2)
        target_frame = ttk.LabelFrame(settings_frame, text="목표 h'rms (Tab 2 연동)", padding=3)
        target_frame.pack(fill=tk.X, pady=3)

        row_target = ttk.Frame(target_frame)
        row_target.pack(fill=tk.X, pady=1)
        ttk.Label(row_target, text="목표 ξ:", font=('Segoe UI', 17)).pack(side=tk.LEFT)
        self.rms_target_xi_display = tk.StringVar(value="(Tab 2에서 설정)")
        self.rms_target_xi_label = ttk.Label(row_target, textvariable=self.rms_target_xi_display,
                                             font=('Arial', 12, 'bold'), foreground='#2563EB')
        self.rms_target_xi_label.pack(side=tk.RIGHT)

        # Refresh button for target value
        ttk.Button(target_frame, text="Tab 2 값 불러오기", command=self._sync_target_xi_from_tab2,
                   width=15).pack(pady=2)

        ttk.Label(target_frame, text="※ Tab 2에서 '목표 h'rms' 변경 시\n   이 버튼을 눌러 동기화하세요",
                  font=('Segoe UI', 16), foreground='#64748B').pack(anchor=tk.W)

        # Calculate button
        calc_frame = ttk.Frame(settings_frame)
        calc_frame.pack(fill=tk.X, pady=5)

        self.rms_calc_btn = ttk.Button(
            calc_frame,
            text="h'rms slope / Local Strain 계산",
            command=self._calculate_rms_slope
        )
        self.rms_calc_btn.pack(fill=tk.X)

        # Progress bar
        self.rms_progress_var = tk.IntVar()
        self.rms_progress_bar = ttk.Progressbar(
            calc_frame,
            variable=self.rms_progress_var,
            maximum=100
        )
        self.rms_progress_bar.pack(fill=tk.X, pady=2)

        # 3. Results Summary
        results_frame = ttk.LabelFrame(left_frame, text="결과 요약", padding=5)
        results_frame.pack(fill=tk.X, pady=2, padx=3)

        self.rms_result_text = tk.Text(results_frame, height=12, font=("Courier", 15), wrap=tk.WORD)
        self.rms_result_text.pack(fill=tk.X)

        # 4. Export / Apply buttons
        action_frame = ttk.LabelFrame(left_frame, text="작업", padding=5)
        action_frame.pack(fill=tk.X, pady=2, padx=3)

        self.apply_strain_btn = ttk.Button(
            action_frame,
            text="μ_visc 탭에 Local Strain 적용",
            command=self._apply_local_strain_to_mu_visc
        )
        self.apply_strain_btn.pack(fill=tk.X, pady=2)

        ttk.Button(
            action_frame,
            text="CSV 내보내기",
            command=self._export_rms_slope_data
        ).pack(fill=tk.X, pady=2)

        # ============== Right Panel: Plots ==============

        right_panel = ttk.Frame(main_container)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        plot_frame = ttk.LabelFrame(right_panel, text="그래프", padding=5)
        plot_frame.pack(fill=tk.BOTH, expand=True)

        # Create figure with 2x2 subplots
        self.fig_rms = Figure(figsize=(9, 7), dpi=100)

        # Top-left: h'rms vs q
        self.ax_rms_slope = self.fig_rms.add_subplot(221)
        self.ax_rms_slope.set_title("h'rms ξ(q)", fontweight='bold', fontsize=15)
        self.ax_rms_slope.set_xlabel('파수 q (1/m)', fontsize=13)
        self.ax_rms_slope.set_ylabel("ξ (h'rms)", fontsize=13)
        self.ax_rms_slope.set_xscale('log')
        self.ax_rms_slope.set_yscale('log')
        self.ax_rms_slope.grid(True, alpha=0.3)

        # Top-right: Local Strain vs q
        self.ax_local_strain = self.fig_rms.add_subplot(222)
        self.ax_local_strain.set_title('Local Strain ε(q)', fontweight='bold', fontsize=15)
        self.ax_local_strain.set_xlabel('파수 q (1/m)', fontsize=13)
        self.ax_local_strain.set_ylabel('ε (fraction)', fontsize=13)
        self.ax_local_strain.set_xscale('log')
        self.ax_local_strain.set_yscale('log')
        self.ax_local_strain.grid(True, alpha=0.3)

        # Bottom-left: RMS Height vs q
        self.ax_rms_height = self.fig_rms.add_subplot(223)
        self.ax_rms_height.set_title('RMS Height h_rms(q)', fontweight='bold', fontsize=15)
        self.ax_rms_height.set_xlabel('파수 q (1/m)', fontsize=13)
        self.ax_rms_height.set_ylabel('h_rms (m)', fontsize=13)
        self.ax_rms_height.set_xscale('log')
        self.ax_rms_height.set_yscale('log')
        self.ax_rms_height.grid(True, alpha=0.3)

        # Bottom-right: PSD (for reference)
        self.ax_psd_ref = self.fig_rms.add_subplot(224)
        self.ax_psd_ref.set_title('PSD C(q) (참조)', fontweight='bold', fontsize=15)
        self.ax_psd_ref.set_xlabel('파수 q (1/m)', fontsize=13)
        self.ax_psd_ref.set_ylabel(r'C(q) (m$^4$)', fontsize=13)
        self.ax_psd_ref.set_xscale('log')
        self.ax_psd_ref.set_yscale('log')
        self.ax_psd_ref.grid(True, alpha=0.3)

        self.fig_rms.subplots_adjust(left=0.14, right=0.95, top=0.94, bottom=0.10, hspace=0.42, wspace=0.38)

        self.canvas_rms = FigureCanvasTkAgg(self.fig_rms, plot_frame)
        self.canvas_rms.draw()
        self.canvas_rms.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas_rms, plot_frame)
        toolbar.update()

    def _sync_target_xi_from_tab2(self):
        """Sync target h'rms from Tab 2 to Tab 4 display and update target_xi."""
        try:
            # Get target h'rms from Tab 2
            target_xi_str = self.target_hrms_slope_var.get()
            target_xi = float(target_xi_str)

            # Update Tab 4 display
            self.rms_target_xi_display.set(f"{target_xi:.4f}")

            # Update target_xi for calculations
            self.target_xi = target_xi

            # Also sync Tab 1's psd_xi_var for consistency
            self.psd_xi_var.set(target_xi_str)

            self.status_var.set(f"목표 h'rms 동기화 완료: ξ = {target_xi:.4f}")
        except ValueError:
            self.rms_target_xi_display.set("(유효하지 않은 값)")
            self.status_var.set("오류: Tab 2의 목표 h'rms 값이 유효하지 않습니다.")

    def _calculate_rms_slope(self):
        """Calculate h'rms and Local Strain from PSD data."""
        # Check if PSD data is available from Tab 0
        tab0_ready = getattr(self, 'tab0_finalized', False)
        if not tab0_ready or self.psd_model is None:
            self._show_status("PSD 데이터가 설정되지 않았습니다!\n\n"
                "Tab 0 (PSD 생성)에서 PSD를 확정한 후\n"
                "'PSD 확정 → Tab 3' 버튼을 클릭하세요.", 'warning')
            return

        try:
            # Sync target_xi from Tab 2 before calculation
            self._sync_target_xi_from_tab2()

            self.rms_calc_btn.config(state='disabled')
            self.rms_progress_var.set(10)
            self.root.update_idletasks()

            # Get PSD data - MeasuredPSD uses q_data and C_data attributes
            if hasattr(self.psd_model, 'q_data'):
                q_array = self.psd_model.q_data
                C_q_array = self.psd_model.C_data
            elif hasattr(self.psd_model, 'q'):
                q_array = self.psd_model.q
                C_q_array = self.psd_model.C_iso if hasattr(self.psd_model, 'C_iso') else self.psd_model(self.psd_model.q)
            else:
                # Generate q array and call PSD model
                q_array = np.logspace(2, 8, 200)  # Default range
                C_q_array = self.psd_model(q_array)

            # Get strain factor
            strain_factor = float(self.strain_factor_var.get())

            # Get q range from 계산 설정 탭 (q_min_var, q_max_var)
            q_min = float(self.q_min_var.get())
            q_max = float(self.q_max_var.get())

            # Filter q range
            mask = (q_array >= q_min) & (q_array <= q_max)
            q_filtered = q_array[mask]
            C_filtered = C_q_array[mask]

            if len(q_filtered) < 3:
                messagebox.showerror("오류", "q 범위에 데이터 포인트가 부족합니다.")
                self.rms_calc_btn.config(state='normal')
                return

            self.rms_progress_var.set(30)
            self.root.update_idletasks()

            # Create RMS slope calculator
            self.rms_slope_calculator = RMSSlopeCalculator(
                q_filtered, C_filtered, strain_factor=strain_factor
            )

            self.rms_progress_var.set(60)
            self.root.update_idletasks()

            # Store profiles
            self.rms_slope_profiles = self.rms_slope_calculator.get_profiles()
            self.local_strain_array = self.rms_slope_profiles['strain'].copy()

            # Update plots
            self._update_rms_slope_plots()

            self.rms_progress_var.set(80)
            self.root.update_idletasks()

            # Update result text
            self._update_rms_result_text()

            self.rms_progress_var.set(100)
            self.status_var.set("h'rms slope / Local Strain 계산 완료")

            # Use target_xi from Tab 2 if available for consistency
            xi_max_display = self.target_xi if self.target_xi is not None else self.rms_slope_profiles['xi'][-1]
            self._show_status(f"h'rms slope / Local Strain 계산 완료!\n\n"
                f"ξ_max (h'rms) = {xi_max_display:.4f}\n"
                f"ε_max = {self.rms_slope_profiles['strain'][-1]*100:.2f}%\n"
                f"h_rms = {self.rms_slope_profiles['hrms'][-1]*1e6:.2f} μm", 'success')

        except Exception as e:
            messagebox.showerror("오류", f"계산 실패:\n{str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self.rms_calc_btn.config(state='normal')

    def _update_rms_slope_plots(self):
        """Update RMS slope plots."""
        if self.rms_slope_profiles is None:
            return

        profiles = self.rms_slope_profiles
        q = profiles['q']
        xi = profiles['xi']
        strain = profiles['strain']
        hrms = profiles['hrms']
        C_q = profiles['C_q']

        # Clear all subplots
        self.ax_rms_slope.clear()
        self.ax_local_strain.clear()
        self.ax_rms_height.clear()
        self.ax_psd_ref.clear()

        # Plot 1: h'rms
        valid_xi = xi > 0
        if np.any(valid_xi):
            self.ax_rms_slope.loglog(q[valid_xi], xi[valid_xi], 'b-', linewidth=2)
        self.ax_rms_slope.set_title("h'rms ξ(q)", fontweight='bold', fontsize=15)
        self.ax_rms_slope.set_xlabel('파수 q (1/m)', fontsize=13)
        self.ax_rms_slope.set_ylabel("ξ (h'rms)", fontsize=13)
        self.ax_rms_slope.grid(True, alpha=0.3)

        # Add final value annotation - use target_xi from Tab 2 if available
        if len(xi) > 0 and xi[-1] > 0:
            # Use target_xi from PSD settings (Tab 2) for consistency
            xi_max_display = self.target_xi if self.target_xi is not None else xi[-1]
            self.ax_rms_slope.axhline(y=xi_max_display, color='r', linestyle='--', alpha=0.5)
            self.ax_rms_slope.annotate(f'ξ_max={xi_max_display:.4f}',
                xy=(q[-1], xi_max_display), xytext=(0.7, 0.9),
                textcoords='axes fraction', fontsize=12,
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.5))

        # Plot 2: Local Strain
        valid_strain = strain > 0
        if np.any(valid_strain):
            self.ax_local_strain.loglog(q[valid_strain], strain[valid_strain]*100, 'r-', linewidth=2)
        self.ax_local_strain.set_title('Local Strain ε(q)', fontweight='bold', fontsize=15)
        self.ax_local_strain.set_xlabel('파수 q (1/m)', fontsize=13)
        self.ax_local_strain.set_ylabel('ε (%)', fontsize=13)
        self.ax_local_strain.grid(True, alpha=0.3)

        # Add strain thresholds
        self.ax_local_strain.axhline(y=1, color='g', linestyle=':', alpha=0.5, label='1%')
        self.ax_local_strain.axhline(y=10, color='orange', linestyle=':', alpha=0.5, label='10%')
        self.ax_local_strain.axhline(y=100, color='red', linestyle=':', alpha=0.5, label='100%')
        self.ax_local_strain.legend(loc='lower right', fontsize=12)

        if len(strain) > 0 and strain[-1] > 0:
            self.ax_local_strain.annotate(f'ε_max={strain[-1]*100:.2f}%',
                xy=(q[-1], strain[-1]*100), xytext=(0.7, 0.9),
                textcoords='axes fraction', fontsize=12,
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.5))

        # Plot 3: RMS Height
        valid_hrms = hrms > 0
        if np.any(valid_hrms):
            self.ax_rms_height.loglog(q[valid_hrms], hrms[valid_hrms]*1e6, 'g-', linewidth=2)
        self.ax_rms_height.set_title('RMS Height h_rms(q)', fontweight='bold', fontsize=15)
        self.ax_rms_height.set_xlabel('파수 q (1/m)', fontsize=13)
        self.ax_rms_height.set_ylabel('h_rms (μm)', fontsize=13)
        self.ax_rms_height.grid(True, alpha=0.3)

        if len(hrms) > 0 and hrms[-1] > 0:
            self.ax_rms_height.annotate(f'h_rms={hrms[-1]*1e6:.2f}μm',
                xy=(q[-1], hrms[-1]*1e6), xytext=(0.7, 0.9),
                textcoords='axes fraction', fontsize=12,
                arrowprops=dict(arrowstyle='->', color='green', alpha=0.5))

        # Plot 4: PSD Reference
        valid_C = C_q > 0
        if np.any(valid_C):
            self.ax_psd_ref.loglog(q[valid_C], C_q[valid_C], 'k-', linewidth=1.5)
        self.ax_psd_ref.set_title('PSD C(q) (참조)', fontweight='bold', fontsize=15)
        self.ax_psd_ref.set_xlabel('파수 q (1/m)', fontsize=13)
        self.ax_psd_ref.set_ylabel(r'C(q) (m$^4$)', fontsize=13)
        self.ax_psd_ref.grid(True, alpha=0.3)

        self.fig_rms.subplots_adjust(left=0.14, right=0.95, top=0.94, bottom=0.10, hspace=0.42, wspace=0.38)
        self.canvas_rms.draw()

    def _update_rms_result_text(self):
        """Update RMS slope result text."""
        self.rms_result_text.delete(1.0, tk.END)

        if self.rms_slope_calculator is None:
            return

        summary = self.rms_slope_calculator.get_summary()
        profiles = self.rms_slope_profiles

        self.rms_result_text.insert(tk.END, "=" * 35 + "\n")
        self.rms_result_text.insert(tk.END, "h'rms slope / Local Strain 결과\n")
        self.rms_result_text.insert(tk.END, "=" * 35 + "\n\n")

        self.rms_result_text.insert(tk.END, "[입력 데이터]\n")
        self.rms_result_text.insert(tk.END, f"  q_min: {summary['q_min']:.2e} 1/m\n")
        self.rms_result_text.insert(tk.END, f"  q_max: {summary['q_max']:.2e} 1/m\n")
        self.rms_result_text.insert(tk.END, f"  데이터 점: {summary['n_points']}\n")
        self.rms_result_text.insert(tk.END, f"  Strain Factor: {summary['strain_factor']}\n\n")

        self.rms_result_text.insert(tk.END, "[h'rms]\n")
        # Show both target ξ (user input from Tab 2) and calculated ξ
        if self.target_xi is not None:
            self.rms_result_text.insert(tk.END, f"  ξ_target (Tab 2 입력값): {self.target_xi:.4f}\n")
        self.rms_result_text.insert(tk.END, f"  ξ_calc (적분 계산값): {summary['xi_max']:.4f}\n")
        self.rms_result_text.insert(tk.END, f"  ξ(q_max): {summary['xi_at_qmax']:.4f}\n\n")

        self.rms_result_text.insert(tk.END, "[Local Strain]\n")
        self.rms_result_text.insert(tk.END, f"  ε_max: {summary['strain_max']*100:.2f}%\n")
        self.rms_result_text.insert(tk.END, f"  ε(q_max): {summary['strain_at_qmax']*100:.2f}%\n\n")

        self.rms_result_text.insert(tk.END, "[RMS Height]\n")
        self.rms_result_text.insert(tk.END, f"  h_rms: {summary['hrms_total']*1e6:.2f} μm\n\n")

        # Show strain at representative q values
        self.rms_result_text.insert(tk.END, "[파수별 Local Strain]\n")
        q = profiles['q']
        strain = profiles['strain']
        indices = np.linspace(0, len(q)-1, min(8, len(q)), dtype=int)
        for idx in indices:
            self.rms_result_text.insert(tk.END,
                f"  q={q[idx]:.2e}: ε={strain[idx]*100:.3f}%\n")

    def _apply_local_strain_to_mu_visc(self):
        """Apply calculated local strain to mu_visc calculation."""
        if self.local_strain_array is None or self.rms_slope_profiles is None:
            self._show_status("먼저 h'rms를 계산하세요.", 'warning')
            return

        # Store for use in mu_visc tab
        self.status_var.set("Local Strain이 μ_visc 탭에 적용 준비됨")

        self._show_status(f"Local Strain 데이터가 μ_visc 계산에 사용될 준비가 되었습니다.\n\n"
            f"데이터 점: {len(self.local_strain_array)}\n"
            f"ε 범위: {self.local_strain_array[0]*100:.4f}% ~ {self.local_strain_array[-1]*100:.2f}%\n\n"
            f"μ_visc 탭에서 '비선형 f,g 보정'을 활성화하고\n"
            f"Strain 추정 방법을 'rms_slope'로 설정하세요.", 'success')

    def _export_rms_slope_data(self):
        """Export h'rms data to CSV file."""
        if self.rms_slope_profiles is None:
            self._show_status("먼저 h'rms를 계산하세요.", 'warning')
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile=self._make_export_filename("hrms_slope_data"),
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if not filename:
            return

        try:
            import csv
            profiles = self.rms_slope_profiles

            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["# h'rms slope / Local Strain Data"])
                writer.writerow(["# q (1/m)", "C(q) (m^4)", "xi^2", "xi (h'rms)",
                                "strain (fraction)", "strain (%)", "h_rms^2 (m^2)", "h_rms (m)"])

                for i in range(len(profiles['q'])):
                    writer.writerow([
                        f"{profiles['q'][i]:.6e}",
                        f"{profiles['C_q'][i]:.6e}",
                        f"{profiles['xi_squared'][i]:.6e}",
                        f"{profiles['xi'][i]:.6e}",
                        f"{profiles['strain'][i]:.6e}",
                        f"{profiles['strain'][i]*100:.4f}",
                        f"{profiles['hrms_squared'][i]:.6e}",
                        f"{profiles['hrms'][i]:.6e}"
                    ])

            self._show_status(f"데이터 저장 완료:\n{filename}", 'success')
            self.status_var.set(f"h'rms 데이터 저장: {filename}")

        except Exception as e:
            messagebox.showerror("오류", f"저장 실패:\n{str(e)}")

    def _create_mu_visc_tab(self, parent):
        """Create enhanced Strain/mu_visc calculation tab with piecewise averaging."""
        # Create main container with 2 columns
        main_container = ttk.Frame(parent)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=2)

        # Left panel for inputs (scrollable) - fixed width
        left_frame = ttk.Frame(main_container, width=getattr(self, '_left_panel_width', 600))
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        left_frame.pack_propagate(False)  # Keep fixed width

        # Logo at bottom (pack before canvas so it stays at bottom)
        self._add_logo_to_panel(left_frame)

        # Toolbar (fixed at top, always accessible) - mu_visc 계산 button
        mu_toolbar = self._create_panel_toolbar(left_frame, buttons=[
            ("\u03bc_visc 계산", self._calculate_mu_visc, 'Accent.TButton'),
            ("f,g 계산", self._compute_fg_curves, 'TButton'),
        ])

        # Progress bar in toolbar (linked to same variable as scrollable area)
        self.mu_progress_var = tk.IntVar()
        self.mu_toolbar_progress = ttk.Progressbar(
            mu_toolbar, variable=self.mu_progress_var, maximum=100, length=150
        )
        self.mu_toolbar_progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4, pady=1)

        # Create canvas and scrollbar for left panel
        left_canvas = tk.Canvas(left_frame, highlightthickness=0)
        left_scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=left_canvas.yview)
        left_panel = ttk.Frame(left_canvas)

        left_panel.bind(
            "<Configure>",
            lambda e: left_canvas.configure(scrollregion=left_canvas.bbox("all"))
        )

        left_canvas.create_window((0, 0), window=left_panel, anchor="nw", width=getattr(self, '_left_panel_width', 600) - 20)
        left_canvas.configure(yscrollcommand=left_scrollbar.set)

        # Pack scrollbar and canvas
        left_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Mouse wheel scroll binding for the canvas
        def _on_mousewheel(event):
            left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _on_mousewheel_linux(event):
            if event.num == 4:
                left_canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                left_canvas.yview_scroll(1, "units")

        # Bind mouse wheel events
        left_canvas.bind("<MouseWheel>", _on_mousewheel)  # Windows/Mac
        left_canvas.bind("<Button-4>", _on_mousewheel_linux)  # Linux scroll up
        left_canvas.bind("<Button-5>", _on_mousewheel_linux)  # Linux scroll down

        # Also bind to the left_panel for when mouse is over widgets
        def _bind_mousewheel(widget):
            widget.bind("<MouseWheel>", _on_mousewheel)
            widget.bind("<Button-4>", _on_mousewheel_linux)
            widget.bind("<Button-5>", _on_mousewheel_linux)
            for child in widget.winfo_children():
                _bind_mousewheel(child)

        left_panel.bind("<Map>", lambda e: _bind_mousewheel(left_panel))

        # ============== Left Panel: Controls ==============

        # 1. Strain Data Loading
        strain_frame = ttk.LabelFrame(left_panel, text="1) Strain 데이터", padding=5)
        strain_frame.pack(fill=tk.X, pady=2, padx=3)

        # ===== 내장 Strain Sweep 선택 =====
        ttk.Label(strain_frame, text="-- 내장 Strain Sweep --",
                  font=('Segoe UI', 17, 'bold'), foreground='#059669').pack(anchor=tk.CENTER)

        preset_ss_frame = ttk.Frame(strain_frame)
        preset_ss_frame.pack(fill=tk.X, pady=1)
        self.preset_ss_var = tk.StringVar(value="(선택...)")
        self.preset_ss_combo = ttk.Combobox(preset_ss_frame, textvariable=self.preset_ss_var,
                                             state='readonly', width=20, font=self.FONTS['body'])
        self.preset_ss_combo.pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_ss_frame, text="로드", command=self._load_preset_strain_sweep, width=4).pack(side=tk.LEFT)
        ttk.Button(preset_ss_frame, text="삭제", command=self._delete_preset_strain_sweep, width=4).pack(side=tk.LEFT, padx=1)

        self._refresh_preset_strain_sweep_list()

        # Strain Sweep 직접 로드 버튼 (빨간 테두리) + 리스트에 추가
        ss_btn_frame = ttk.Frame(strain_frame)
        ss_btn_frame.pack(fill=tk.X, pady=1)

        ttk.Button(
            ss_btn_frame,
            text="Strain Sweep 로드",
            command=self._load_strain_sweep_data,
            style='Accent.TButton'
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Button(ss_btn_frame, text="-> 추가",
                   command=self._add_preset_strain_sweep, width=7).pack(side=tk.LEFT, padx=(3, 0))

        self.strain_file_label = ttk.Label(strain_frame, text="(파일 없음)", font=('Segoe UI', 17))
        self.strain_file_label.pack(anchor=tk.W)

        # ===== 내장 f,g 곡선 선택 =====
        ttk.Separator(strain_frame, orient='horizontal').pack(fill=tk.X, pady=3)
        ttk.Label(strain_frame, text="-- 내장 f,g 곡선 --",
                  font=('Segoe UI', 17, 'bold'), foreground='#059669').pack(anchor=tk.CENTER)

        preset_fg_frame = ttk.Frame(strain_frame)
        preset_fg_frame.pack(fill=tk.X, pady=1)
        self.preset_fg_var = tk.StringVar(value="(선택...)")
        self.preset_fg_combo = ttk.Combobox(preset_fg_frame, textvariable=self.preset_fg_var,
                                             state='readonly', width=20, font=self.FONTS['body'])
        self.preset_fg_combo.pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_fg_frame, text="로드", command=self._load_preset_fg, width=4).pack(side=tk.LEFT)
        ttk.Button(preset_fg_frame, text="삭제", command=self._delete_preset_fg, width=4).pack(side=tk.LEFT, padx=1)

        self._refresh_preset_fg_list()

        # f,g 직접 로드 버튼 + 리스트에 추가
        fg_btn_frame = ttk.Frame(strain_frame)
        fg_btn_frame.pack(fill=tk.X, pady=1)

        ttk.Button(
            fg_btn_frame,
            text="f,g 곡선 로드",
            command=self._load_fg_curve_data,
            width=15
        ).pack(side=tk.LEFT)

        ttk.Button(fg_btn_frame, text="-> 추가",
                   command=self._add_preset_fg, width=7).pack(side=tk.LEFT, padx=(3, 0))

        self.fg_file_label = ttk.Label(strain_frame, text="(파일 없음)", font=('Segoe UI', 17))
        self.fg_file_label.pack(anchor=tk.W)

        # 2. f,g Calculation Settings
        fg_settings_frame = ttk.LabelFrame(left_panel, text="2) f,g 계산", padding=5)
        fg_settings_frame.pack(fill=tk.X, pady=2, padx=3)

        # Target frequency and E0 points in one row
        row1 = ttk.Frame(fg_settings_frame)
        row1.pack(fill=tk.X, pady=1)
        ttk.Label(row1, text="주파수:", font=('Segoe UI', 17)).pack(side=tk.LEFT)
        self.fg_target_freq_var = tk.StringVar(value="1.0")
        ttk.Entry(row1, textvariable=self.fg_target_freq_var, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(row1, text="Hz  E0점:", font=('Segoe UI', 17)).pack(side=tk.LEFT)
        self.e0_points_var = tk.StringVar(value="1")
        ttk.Entry(row1, textvariable=self.e0_points_var, width=4).pack(side=tk.LEFT, padx=2)

        # Strain is percent and clip checkboxes
        self.strain_is_percent_var = tk.BooleanVar(value=True)
        self.clip_fg_var = tk.BooleanVar(value=True)
        check_row = ttk.Frame(fg_settings_frame)
        check_row.pack(fill=tk.X, pady=1)
        ttk.Checkbutton(check_row, text="% 단위", variable=self.strain_is_percent_var).pack(side=tk.LEFT)
        ttk.Checkbutton(check_row, text="Clip ≤1", variable=self.clip_fg_var).pack(side=tk.LEFT)

        # Grid max strain and Persson grid
        row2 = ttk.Frame(fg_settings_frame)
        row2.pack(fill=tk.X, pady=1)
        ttk.Label(row2, text="Grid Max(%):", font=('Segoe UI', 17)).pack(side=tk.LEFT)
        self.extend_strain_var = tk.StringVar(value="40")
        ttk.Entry(row2, textvariable=self.extend_strain_var, width=5).pack(side=tk.LEFT, padx=2)
        self.use_persson_grid_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row2, text="Persson Grid", variable=self.use_persson_grid_var).pack(side=tk.LEFT)

        # Compute f,g button
        ttk.Button(
            fg_settings_frame,
            text="f,g 계산",
            command=self._compute_fg_curves,
            style='Accent.TButton',
            width=15
        ).pack(anchor=tk.W, pady=2)

        # 3. Persson Average f,g (RANK 1 최적 가중치 자동 적용)
        persson_avg_frame = ttk.LabelFrame(left_panel, text="3) Persson average f,g", padding=5)
        persson_avg_frame.pack(fill=tk.X, pady=2, padx=3)

        # Split strain setting (RANK 1 default: 14.2%)
        split_row = ttk.Frame(persson_avg_frame)
        split_row.pack(fill=tk.X, pady=1)
        ttk.Label(split_row, text="Split(%):", font=('Segoe UI', 17)).pack(side=tk.LEFT)
        self.split_strain_var = tk.StringVar(value="14.2")
        ttk.Entry(split_row, textvariable=self.split_strain_var, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(split_row, text="RANK1 최적", font=('Segoe UI', 16), foreground='#2563EB').pack(side=tk.LEFT, padx=2)

        # RANK 1 weight info display
        info_frame = ttk.Frame(persson_avg_frame)
        info_frame.pack(fill=tk.X, pady=1)
        weight_info = (
            "f: low(0.02°C:10%,29.9°C:90%) high(29.9°C:30%,49.99°C:70%)\n"
            "g: low(0.02°C:35%,49.99°C:65%) high(29.9°C:55%,49.99°C:45%)"
        )
        ttk.Label(info_frame, text=weight_info, font=('Arial', 15),
                  foreground='#64748B', justify=tk.LEFT).pack(anchor=tk.W)

        # Persson average f,g 계산 button
        ttk.Button(
            persson_avg_frame,
            text="Persson average f,g 계산",
            command=self._persson_average_fg,
            style='Accent.TButton',
            width=25
        ).pack(anchor=tk.W, pady=3)

        # Status label for Persson average
        self.persson_avg_status_var = tk.StringVar(value="(미계산)")
        ttk.Label(persson_avg_frame, textvariable=self.persson_avg_status_var,
                  font=('Segoe UI', 17), foreground='#059669').pack(anchor=tk.W)

        # Hidden listboxes for internal compatibility (not displayed)
        _hidden_frame = ttk.Frame(left_panel)
        self.temp_listbox_A = tk.Listbox(_hidden_frame, height=0, selectmode=tk.MULTIPLE, exportselection=False)
        self.temp_listbox_B = tk.Listbox(_hidden_frame, height=0, selectmode=tk.MULTIPLE, exportselection=False)

        # Legacy simple averaging (keep for compatibility)
        self.temp_listbox_frame = ttk.Frame(left_panel)
        self.temp_listbox = tk.Listbox(
            self.temp_listbox_frame,
            height=0,
            selectmode=tk.MULTIPLE,
            exportselection=False
        )

        # 4. mu_visc Calculation Settings
        mu_settings_frame = ttk.LabelFrame(left_panel, text="4) μ_visc 계산", padding=5)
        mu_settings_frame.pack(fill=tk.X, pady=2, padx=3)

        # Nonlinear correction - single row (강조)
        nonlinear_wrapper = tk.Frame(mu_settings_frame, bg=self.COLORS['primary'], padx=2, pady=2)
        nonlinear_wrapper.pack(fill=tk.X, pady=1)
        nonlinear_row = ttk.Frame(nonlinear_wrapper)
        nonlinear_row.pack(fill=tk.X)
        self.use_fg_correction_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(nonlinear_row, text="비선형 f,g 보정", variable=self.use_fg_correction_var).pack(side=tk.LEFT)

        # Strain estimation in same frame
        strain_row = ttk.Frame(mu_settings_frame)
        strain_row.pack(fill=tk.X, pady=1)
        ttk.Label(strain_row, text="Strain:", font=('Segoe UI', 17)).pack(side=tk.LEFT)
        self.strain_est_method_var = tk.StringVar(value="rms_slope")
        strain_combo = ttk.Combobox(strain_row, textvariable=self.strain_est_method_var,
                     values=["rms_slope", "fixed", "persson", "simple"], width=10, state="readonly", font=self.FONTS['body'])
        strain_combo.pack(side=tk.LEFT, padx=2)

        self.fixed_strain_var = tk.StringVar(value="1.0")
        self.fixed_strain_entry = ttk.Entry(strain_row, textvariable=self.fixed_strain_var, width=5)
        self.fixed_strain_entry.pack(side=tk.LEFT)
        self.fixed_strain_label = ttk.Label(strain_row, text="%", font=('Segoe UI', 17))
        self.fixed_strain_label.pack(side=tk.LEFT)

        # Callback to show/hide fixed strain entry based on method
        def on_strain_method_change(*args):
            method = self.strain_est_method_var.get()
            if method == 'rms_slope':
                self.fixed_strain_entry.config(state='disabled')
                self.fixed_strain_label.config(foreground='#64748B')
            else:
                self.fixed_strain_entry.config(state='normal')
                self.fixed_strain_label.config(foreground='black')

        self.strain_est_method_var.trace_add('write', on_strain_method_change)
        on_strain_method_change()  # Initialize state

        # Integration parameters in single row
        integ_row = ttk.Frame(mu_settings_frame)
        integ_row.pack(fill=tk.X, pady=1)
        ttk.Label(integ_row, text="γ:", font=('Segoe UI', 17)).pack(side=tk.LEFT)
        self.gamma_var = tk.StringVar(value="0.55")
        ttk.Entry(integ_row, textvariable=self.gamma_var, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Label(integ_row, text="φ점:", font=('Segoe UI', 17)).pack(side=tk.LEFT)
        self.n_phi_var = tk.StringVar(value="14")
        ttk.Entry(integ_row, textvariable=self.n_phi_var, width=5).pack(side=tk.LEFT, padx=2)

        # Smoothing in single row
        smooth_row = ttk.Frame(mu_settings_frame)
        smooth_row.pack(fill=tk.X, pady=1)
        self.smooth_mu_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(smooth_row, text="스무딩", variable=self.smooth_mu_var).pack(side=tk.LEFT)
        self.smooth_window_var = tk.StringVar(value="5")
        ttk.Combobox(smooth_row, textvariable=self.smooth_window_var,
                     values=["3", "5", "7", "9", "11"], width=4, state="readonly", font=self.FONTS['body']).pack(side=tk.LEFT, padx=2)

        # ===== Temperature Shift Section =====
        ttk.Separator(mu_settings_frame, orient='horizontal').pack(fill=tk.X, pady=3)
        temp_frame = ttk.LabelFrame(mu_settings_frame, text="온도 시프트 (aT 적용)", padding=3)
        temp_frame.pack(fill=tk.X, pady=2)

        # Temperature input row (강조)
        temp_row1_wrapper = tk.Frame(temp_frame, bg=self.COLORS['primary'], padx=1, pady=1)
        temp_row1_wrapper.pack(fill=tk.X, pady=1)
        temp_row1 = ttk.Frame(temp_row1_wrapper)
        temp_row1.pack(fill=tk.X)
        ttk.Label(temp_row1, text="계산 온도:", font=('Segoe UI', 17)).pack(side=tk.LEFT)
        self.mu_calc_temp_var = tk.StringVar(value="20.0")
        self.mu_calc_temp_entry = ttk.Entry(temp_row1, textvariable=self.mu_calc_temp_var, width=8)
        self.mu_calc_temp_entry.pack(side=tk.LEFT, padx=2)
        ttk.Label(temp_row1, text="°C", font=('Segoe UI', 17)).pack(side=tk.LEFT)

        # aT status display
        self.mu_aT_status_var = tk.StringVar(value="aT: 미로드 (Tref에서 계산)")
        ttk.Label(temp_frame, textvariable=self.mu_aT_status_var,
                  font=('Segoe UI', 17), foreground='#64748B').pack(anchor=tk.W)

        # Temperature apply button
        temp_row2 = ttk.Frame(temp_frame)
        temp_row2.pack(fill=tk.X, pady=1)
        ttk.Button(temp_row2, text="온도 적용 & G 재계산",
                   command=self._apply_temperature_shift, width=20).pack(side=tk.LEFT)

        # G calculation status
        self.g_calc_status_var = tk.StringVar(value="")
        self.g_calc_status_label = ttk.Label(temp_frame, textvariable=self.g_calc_status_var,
                  font=('Segoe UI', 17), foreground='#2563EB')
        self.g_calc_status_label.pack(anchor=tk.W)

        # Calculate button and progress bar
        calc_row = ttk.Frame(mu_settings_frame)
        calc_row.pack(fill=tk.X, pady=2)
        # μ_visc 계산 버튼
        self.mu_calc_button = ttk.Button(calc_row, text="μ_visc 계산",
                                         command=self._calculate_mu_visc, width=15,
                                         style='Accent.TButton')
        self.mu_calc_button.pack(side=tk.LEFT, padx=2)

        self.mu_progress_bar = ttk.Progressbar(calc_row, variable=self.mu_progress_var, maximum=100, length=150)
        self.mu_progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        # 5. Results Display
        results_frame = ttk.LabelFrame(left_panel, text="5) 결과", padding=5)
        results_frame.pack(fill=tk.X, pady=2, padx=3)

        self.mu_result_text = tk.Text(results_frame, height=8, font=("Courier", 15), wrap=tk.WORD)
        self.mu_result_text.pack(fill=tk.X)

        # Export buttons
        export_btn_frame = ttk.Frame(results_frame)
        export_btn_frame.pack(fill=tk.X, pady=2)

        ttk.Button(export_btn_frame, text="μ CSV", command=self._export_mu_visc_results, width=10).pack(side=tk.LEFT, padx=1)
        ttk.Button(export_btn_frame, text="f,g CSV", command=self._export_fg_curves, width=10).pack(side=tk.LEFT, padx=1)
        ttk.Button(export_btn_frame, text="μ+A/A0 CSV", command=self._export_mu_and_area_csv, width=12).pack(side=tk.LEFT, padx=1)

        # Reference data comparison section
        ref_frame = ttk.LabelFrame(results_frame, text="참조 데이터 비교", padding=3)
        ref_frame.pack(fill=tk.X, pady=3)

        ref_row1 = ttk.Frame(ref_frame)
        ref_row1.pack(fill=tk.X, pady=1)

        self.show_ref_mu_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ref_row1, text="참조 μ_visc 표시", variable=self.show_ref_mu_var,
                       command=self._toggle_reference_mu).pack(side=tk.LEFT)

        ttk.Button(ref_row1, text="참조 편집", command=self._edit_reference_data,
                  width=8).pack(side=tk.LEFT, padx=2)
        ttk.Button(ref_row1, text="비교 분석", command=self._analyze_mu_comparison,
                  width=10).pack(side=tk.RIGHT, padx=2)

        # Row 2: Plot clear/reset buttons
        ref_row2 = ttk.Frame(ref_frame)
        ref_row2.pack(fill=tk.X, pady=1)

        def _clear_ref_plots():
            self.plotted_ref_datasets = []
            if hasattr(self, 'mu_visc_results') and self.mu_visc_results is not None:
                v = self.mu_visc_results.get('v')
                mu = self.mu_visc_results.get('mu')
                details = self.mu_visc_results.get('details')
                if v is not None and mu is not None and details is not None:
                    use_nl = self.mu_use_fg_var.get() if hasattr(self, 'mu_use_fg_var') else False
                    self._update_mu_visc_plots(v, mu, details, use_nonlinear=use_nl)
            else:
                self._reset_mu_visc_axes()
            self._show_status("참조 데이터 플롯을 지웠습니다.", 'info')

        def _reset_all_plots():
            self.plotted_ref_datasets = []
            self._reset_mu_visc_axes()
            self._show_status("그래프를 초기화했습니다.", 'info')

        ttk.Button(ref_row2, text="참조 플롯 지우기", command=_clear_ref_plots,
                   width=14).pack(side=tk.LEFT, padx=2)
        ttk.Button(ref_row2, text="플롯 초기화", command=_reset_all_plots,
                   width=10).pack(side=tk.LEFT, padx=2)

        # Gap display (A/A0 계산값 vs 참조값 차이)
        self.area_gap_var = tk.StringVar(value="A/A0 Gap: 계산 후 표시")
        ttk.Label(ref_frame, textvariable=self.area_gap_var,
                  font=('Segoe UI', 17), foreground='#2563EB').pack(anchor=tk.W, pady=2)

        # ============== Right Panel: Plots ==============

        right_panel = ttk.Frame(main_container)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        plot_frame = ttk.LabelFrame(right_panel, text="그래프", padding=5)
        plot_frame.pack(fill=tk.BOTH, expand=True)

        # Create figure with 2x2 subplots
        self.fig_mu_visc = Figure(figsize=(9, 7), dpi=100)

        # Top-left: f,g curves
        self.ax_fg_curves = self.fig_mu_visc.add_subplot(221)
        self.ax_fg_curves.set_title('f(ε), g(ε) 곡선', fontweight='bold', fontsize=15)
        self.ax_fg_curves.set_xlabel('변형률 ε (fraction)', fontsize=13)
        self.ax_fg_curves.set_ylabel('보정 계수', fontsize=13)
        self.ax_fg_curves.grid(True, alpha=0.3)

        # Top-right: mu_visc vs velocity
        self.ax_mu_v = self.fig_mu_visc.add_subplot(222)
        self.ax_mu_v.set_title('μ_visc(v) 곡선', fontweight='bold', fontsize=15)
        self.ax_mu_v.set_xlabel('속도 v (m/s)', fontsize=13)
        self.ax_mu_v.set_ylabel('마찰 계수 μ_visc', fontsize=13)
        self.ax_mu_v.set_xscale('log')
        self.ax_mu_v.grid(True, alpha=0.3)

        # Bottom-left: Contact Area Ratio vs Velocity
        self.ax_mu_cumulative = self.fig_mu_visc.add_subplot(223)
        self.ax_mu_cumulative.set_title('실접촉 면적비율 P(v)', fontweight='bold', fontsize=15)
        self.ax_mu_cumulative.set_xlabel('속도 v (m/s)', fontsize=13)
        self.ax_mu_cumulative.set_ylabel('평균 P(q)', fontsize=13)
        self.ax_mu_cumulative.set_xscale('log')
        self.ax_mu_cumulative.grid(True, alpha=0.3)

        # Bottom-right: P(q) and S(q)
        self.ax_ps = self.fig_mu_visc.add_subplot(224)
        self.ax_ps.set_title('P(q), S(q) 분포', fontweight='bold', fontsize=15)
        self.ax_ps.set_xlabel('파수 q (1/m)', fontsize=13)
        self.ax_ps.set_ylabel('P(q), S(q)', fontsize=13)
        self.ax_ps.set_xscale('log')
        self.ax_ps.grid(True, alpha=0.3)

        self.fig_mu_visc.subplots_adjust(left=0.12, right=0.90, top=0.94, bottom=0.10, hspace=0.42, wspace=0.38)

        self.canvas_mu_visc = FigureCanvasTkAgg(self.fig_mu_visc, plot_frame)
        self.canvas_mu_visc.draw()
        self.canvas_mu_visc.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas_mu_visc, plot_frame)
        toolbar.update()

        # Initialize piecewise result storage
        self.piecewise_result = None

    def _select_all_temps(self):
        """Select all temperatures in both Group A and B listboxes."""
        for i in range(self.temp_listbox_A.size()):
            self.temp_listbox_A.selection_set(i)
        for i in range(self.temp_listbox_B.size()):
            self.temp_listbox_B.selection_set(i)

    def _persson_average_fg(self):
        """Persson average f,g 계산: RANK 1 최적 가중치로 자동 계산.

        1) fg_by_T가 없으면 먼저 f,g 곡선 계산 (strain data 필요)
        2) DEFAULT_STRAIN_SPLIT (RANK 1) 가중치로 strain-split weighted average
        3) interpolator 생성 → mu_visc 계산에 바로 사용 가능
        """
        # Step 1: fg_by_T가 없으면 자동으로 계산
        if self.fg_by_T is None:
            if self.strain_data is None:
                self._show_status("Strain Sweep 데이터가 없습니다.\n"
                    "먼저 '1) Strain Sweep 로드'로 데이터를 로드하세요.", 'warning')
                return
            # 자동으로 f,g 곡선 계산
            self._compute_fg_curves()
            if self.fg_by_T is None:
                return  # 계산 실패

        try:
            # Step 2: RANK 1 최적 설정으로 strain-split 구성
            split_percent = float(self.split_strain_var.get())
            split_strain = split_percent / 100.0

            extend_percent = float(self.extend_strain_var.get())
            max_strain = extend_percent / 100.0

            use_persson = self.use_persson_grid_var.get()
            grid_strain = create_strain_grid(30, max_strain, use_persson_grid=use_persson)

            # DEFAULT_STRAIN_SPLIT (RANK 1) 가중치 + UI threshold 적용
            strain_split_cfg = dict(DEFAULT_STRAIN_SPLIT)
            strain_split_cfg['threshold'] = split_strain

            # 모든 온도 자동 사용 (Group A/B 수동 선택 불필요)
            all_temps = list(self.fg_by_T.keys())

            result = average_fg_curves(
                self.fg_by_T,
                all_temps,
                grid_strain,
                interp_kind='loglog_linear',
                avg_mode='mean',
                clip_leq_1=self.clip_fg_var.get(),
                strain_split=strain_split_cfg
            )

            if result is None:
                messagebox.showerror("오류",
                    "Persson average 실패: 데이터가 부족합니다.\n"
                    "DEFAULT_STRAIN_SPLIT 온도(0.02, 29.9, 49.99°C)와\n"
                    "데이터 온도가 매칭되지 않을 수 있습니다.")
                return

            f_stitched = result['f_avg']
            g_stitched = result['g_avg']
            n_eff_stitched = result['n_eff']

            # Extend to 100% strain with hold extrapolation
            max_data_strain = grid_strain[-1]
            original_len = len(grid_strain)
            if max_data_strain < 1.0:
                extend_strains = np.array([0.5, 0.7, 1.0])
                extend_strains = extend_strains[extend_strains > max_data_strain]
                if len(extend_strains) > 0:
                    grid_strain = np.concatenate([grid_strain, extend_strains])
                    f_stitched = np.concatenate([f_stitched, np.full(len(extend_strains), f_stitched[-1])])
                    g_stitched = np.concatenate([g_stitched, np.full(len(extend_strains), g_stitched[-1])])
                    n_eff_stitched = np.concatenate([n_eff_stitched, np.full(len(extend_strains), n_eff_stitched[-1])])

            # Store piecewise result (Persson average)
            self.piecewise_result = {
                'strain': grid_strain.copy(),
                'strain_original_len': original_len,
                'f_avg': f_stitched,
                'g_avg': g_stitched,
                'n_eff': n_eff_stitched,
                'split': split_strain,
                'temps_A': all_temps,
                'temps_B': all_temps,
                'strain_split_cfg': strain_split_cfg
            }

            # Set as main averaged result for mu_visc calculation
            self.fg_averaged = {
                'strain': grid_strain.copy(),
                'f_avg': f_stitched,
                'g_avg': g_stitched,
                'Ts_used': all_temps,
                'n_eff': n_eff_stitched
            }

            # Create interpolators with 'hold' extrapolation
            self.f_interpolator, self.g_interpolator = create_fg_interpolator(
                grid_strain, f_stitched, g_stitched,
                interp_kind='loglog_linear', extrap_mode='hold'
            )

            # Update plot
            self._update_fg_plot_persson_avg()

            # Update status
            n_temps = len(all_temps)
            temps_str = ", ".join(f"{t:.1f}" for t in sorted(all_temps))
            self.persson_avg_status_var.set(
                f"완료: Split={split_percent:.1f}%, {n_temps}개 온도 [{temps_str}°C]"
            )
            self.status_var.set(
                f"Persson average f,g 완료: RANK1 가중치, Split={split_percent:.1f}%, "
                f"{n_temps}개 온도"
            )

        except Exception as e:
            messagebox.showerror("오류", f"Persson average f,g 실패:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def _update_fg_plot_persson_avg(self):
        """Update f,g curves plot with Persson average visualization."""
        self.ax_fg_curves.clear()
        self.ax_fg_curves.set_title('f(ε), g(ε) Persson Average (RANK1)', fontweight='bold', fontsize=15)
        self.ax_fg_curves.set_xlabel('변형률 ε (fraction)', fontsize=13)
        self.ax_fg_curves.set_ylabel('보정 계수', fontsize=13)
        self.ax_fg_curves.grid(True, alpha=0.3)

        # Plot individual temperature curves (thin, low alpha)
        if self.fg_by_T is not None:
            for T, data in self.fg_by_T.items():
                s = data['strain']
                f = data['f']
                g = data['g']
                self.ax_fg_curves.plot(s, f, 'b-', alpha=0.15, linewidth=0.8)
                self.ax_fg_curves.plot(s, g, 'r-', alpha=0.15, linewidth=0.8)

        # Plot Persson average results
        if self.piecewise_result is not None:
            s = self.piecewise_result['strain']
            split = self.piecewise_result['split']

            f_final = self.piecewise_result['f_avg']
            g_final = self.piecewise_result['g_avg']
            self.ax_fg_curves.plot(s, f_final, 'b-', linewidth=3.5, label='f(ε) Persson Avg')
            self.ax_fg_curves.plot(s, g_final, 'r-', linewidth=3.5, label='g(ε) Persson Avg')

            # Split line
            self.ax_fg_curves.axvline(split, color='green', linewidth=2, linestyle=':', alpha=0.8,
                                      label=f'Split @ {split*100:.1f}%')

        elif self.fg_averaged is not None:
            s = self.fg_averaged['strain']
            f_avg = self.fg_averaged['f_avg']
            g_avg = self.fg_averaged['g_avg']
            self.ax_fg_curves.plot(s, f_avg, 'b-', linewidth=3, label='f(ε) 평균')
            self.ax_fg_curves.plot(s, g_avg, 'r-', linewidth=3, label='g(ε) 평균')

        self.ax_fg_curves.set_xlim(0, 1.0)
        self.ax_fg_curves.set_ylim(0, 1.1)
        self.ax_fg_curves.legend(loc='upper right', fontsize=12, ncol=2)

        self.canvas_mu_visc.draw()

    def _export_fg_curves(self):
        """Export f,g curves to CSV file with proper column separation."""
        if self.fg_averaged is None and self.piecewise_result is None:
            self._show_status("먼저 f,g 곡선을 계산하세요.", 'warning')
            return

        result = self.piecewise_result if self.piecewise_result is not None else self.fg_averaged

        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile=self._make_export_filename("fg_curves"),
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if not filename:
            return

        try:
            import csv
            from datetime import datetime

            # 평가 정보 수집
            try:
                load_mpa = float(self.sigma_0_var.get())
            except:
                load_mpa = 0.0
            try:
                calc_temp = float(self.mu_calc_temp_var.get())
            except:
                calc_temp = 20.0
            nonlinear_applied = self.use_fg_correction_var.get() if hasattr(self, 'use_fg_correction_var') else False

            with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)  # 쉼표 구분자 (기본값)
                # 헤더 정보
                writer.writerow(['# f,g 곡선 데이터'])
                writer.writerow(['# 생성일시', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
                writer.writerow(['# 공칭하중(MPa)', f'{load_mpa:.3f}'])
                writer.writerow(['# 계산온도(°C)', f'{calc_temp:.1f}'])
                writer.writerow(['# 비선형보정적용', '예' if nonlinear_applied else '아니오'])
                if self.piecewise_result is not None:
                    split = self.piecewise_result['split']
                    writer.writerow(['# Split Strain(%)', f'{split*100:.2f}'])
                    writer.writerow(['# Method', 'Persson Average (RANK1 weighted)'])
                    writer.writerow(['# Temps used', str(self.piecewise_result.get("temps_A", []))])
                writer.writerow([])  # 빈 줄
                writer.writerow(['strain_fraction', 'f_value', 'g_value', 'n_eff'])

                for i in range(len(result['strain'])):
                    writer.writerow([
                        f'{result["strain"][i]:.6e}',
                        f'{result["f_avg"][i]:.6f}',
                        f'{result["g_avg"][i]:.6f}',
                        f'{result["n_eff"][i]:.0f}'
                    ])

            self._show_status(f"f,g 곡선 저장 완료:\n{filename}", 'success')
            self.status_var.set(f"f,g 곡선 저장: {filename}")

        except Exception as e:
            messagebox.showerror("오류", f"저장 실패:\n{str(e)}")

    def _load_strain_sweep_data(self):
        """Load strain sweep data from file."""
        filename = filedialog.askopenfilename(
            title="Strain Sweep 파일 선택",
            filetypes=[
                ("All supported", "*.txt *.csv *.xlsx *.xls"),
                ("Text files", "*.txt"),
                ("CSV files", "*.csv"),
                ("Excel files", "*.xlsx *.xls"),
                ("All files", "*.*")
            ]
        )

        if not filename:
            return

        try:
            self.strain_data = load_strain_sweep_file(filename)

            if not self.strain_data:
                messagebox.showerror("오류", "유효한 데이터를 찾을 수 없습니다.")
                return

            # Update label
            self.strain_file_label.config(
                text=f"{os.path.basename(filename)} ({len(self.strain_data)}개 온도)"
            )

            # Populate temperature listboxes (Group A and B)
            temps = sorted(self.strain_data.keys())

            # Clear and populate Group A
            self.temp_listbox_A.delete(0, tk.END)
            for T in temps:
                self.temp_listbox_A.insert(tk.END, f"{T:.2f} °C")
                self.temp_listbox_A.selection_set(tk.END)

            # Clear and populate Group B
            self.temp_listbox_B.delete(0, tk.END)
            for T in temps:
                self.temp_listbox_B.insert(tk.END, f"{T:.2f} °C")
                self.temp_listbox_B.selection_set(tk.END)

            # Also update legacy listbox for compatibility
            self.temp_listbox.delete(0, tk.END)
            for T in temps:
                self.temp_listbox.insert(tk.END, f"{T:.2f} °C")
                self.temp_listbox.selection_set(tk.END)

            self.status_var.set(f"Strain 데이터 로드 완료: {len(self.strain_data)}개 온도")
            self._show_status(f"Strain 데이터 로드 완료\n온도 블록: {len(self.strain_data)}개", 'success')

        except Exception as e:
            messagebox.showerror("오류", f"Strain 데이터 로드 실패:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def _load_fg_curve_data(self):
        """Load pre-computed f,g curve data from file."""
        filename = filedialog.askopenfilename(
            title="f,g 곡선 파일 선택",
            filetypes=[
                ("Text/CSV files", "*.txt *.csv *.dat"),
                ("All files", "*.*")
            ]
        )

        if not filename:
            return

        try:
            fg_data = load_fg_curve_file(
                filename,
                strain_is_percent=self.strain_is_percent_var.get()
            )

            if fg_data is None:
                messagebox.showerror("오류", "유효한 f,g 데이터를 찾을 수 없습니다.")
                return

            # Create interpolators
            self.f_interpolator, self.g_interpolator = create_fg_interpolator(
                fg_data['strain'],
                fg_data['f'],
                fg_data['g'] if fg_data['g'] is not None else fg_data['f']
            )

            # Store for plotting
            self.fg_averaged = {
                'strain': fg_data['strain'],
                'f_avg': fg_data['f'],
                'g_avg': fg_data['g'] if fg_data['g'] is not None else fg_data['f']
            }

            # Update label
            self.fg_file_label.config(text=os.path.basename(filename))

            # Update plot
            self._update_fg_plot()

            self.status_var.set(f"f,g 곡선 로드 완료: {len(fg_data['strain'])}개 점")
            self._show_status(f"f,g 곡선 로드 완료\n데이터 포인트: {len(fg_data['strain'])}개", 'success')

        except Exception as e:
            messagebox.showerror("오류", f"f,g 곡선 로드 실패:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def _compute_fg_curves(self):
        """Compute f,g curves from strain sweep data."""
        if self.strain_data is None:
            self._show_status("먼저 Strain 데이터를 로드하세요.", 'warning')
            return

        try:
            target_freq = float(self.fg_target_freq_var.get())
            e0_points = int(self.e0_points_var.get())
            strain_is_percent = self.strain_is_percent_var.get()
            clip_fg = self.clip_fg_var.get()

            # Compute f,g curves
            self.fg_by_T = compute_fg_from_strain_sweep(
                self.strain_data,
                target_freq=target_freq,
                freq_mode='nearest',
                strain_is_percent=strain_is_percent,
                e0_n_points=e0_points,
                clip_leq_1=clip_fg
            )

            if not self.fg_by_T:
                messagebox.showerror("오류", "f,g 계산 실패: 유효한 데이터가 없습니다.")
                return

            temps = sorted(self.fg_by_T.keys())

            # Update Group A listbox
            self.temp_listbox_A.delete(0, tk.END)
            for T in temps:
                self.temp_listbox_A.insert(tk.END, f"{T:.2f} °C")
                self.temp_listbox_A.selection_set(tk.END)

            # Update Group B listbox
            self.temp_listbox_B.delete(0, tk.END)
            for T in temps:
                self.temp_listbox_B.insert(tk.END, f"{T:.2f} °C")
                self.temp_listbox_B.selection_set(tk.END)

            # Update legacy temperature listbox (for compatibility)
            self.temp_listbox.delete(0, tk.END)
            for T in temps:
                self.temp_listbox.insert(tk.END, f"{T:.2f} °C")
                self.temp_listbox.selection_set(tk.END)

            # Plot individual curves
            self._update_fg_plot()

            self.status_var.set(f"f,g 곡선 계산 완료: {len(self.fg_by_T)}개 온도")

        except Exception as e:
            messagebox.showerror("오류", f"f,g 계산 실패:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def _average_fg_curves(self):
        """Average f,g curves from selected temperatures."""
        if self.fg_by_T is None:
            self._show_status("먼저 f,g 곡선을 계산하세요.", 'warning')
            return

        try:
            # Get selected temperatures
            selections = self.temp_listbox.curselection()
            if not selections:
                self._show_status("최소 1개의 온도를 선택하세요.", 'warning')
                return

            temps = sorted(self.fg_by_T.keys())
            selected_temps = [temps[i] for i in selections]

            # Create strain grid
            max_strain = max(
                np.max(self.fg_by_T[T]['strain']) for T in selected_temps
            )
            grid_strain = create_strain_grid(30, max_strain, use_persson_grid=True)

            # Average curves
            self.fg_averaged = average_fg_curves(
                self.fg_by_T,
                selected_temps,
                grid_strain,
                interp_kind='loglog_linear',
                avg_mode='mean',
                clip_leq_1=self.clip_fg_var.get(),
                strain_split=DEFAULT_STRAIN_SPLIT
            )

            if self.fg_averaged is None:
                messagebox.showerror("오류", "평균화 실패")
                return

            # Create interpolators
            self.f_interpolator, self.g_interpolator = create_fg_interpolator(
                self.fg_averaged['strain'],
                self.fg_averaged['f_avg'],
                self.fg_averaged['g_avg']
            )

            # Update plot
            self._update_fg_plot()

            self.status_var.set(f"f,g 평균화 완료: {len(selected_temps)}개 온도 사용")

        except Exception as e:
            messagebox.showerror("오류", f"평균화 실패:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def _update_fg_plot(self):
        """Update f,g curves plot."""
        self.ax_fg_curves.clear()
        self.ax_fg_curves.set_title('f(ε), g(ε) 곡선', fontweight='bold', fontsize=15)
        self.ax_fg_curves.set_xlabel('변형률 ε (fraction)', fontsize=13)
        self.ax_fg_curves.set_ylabel('보정 계수', fontsize=13)
        self.ax_fg_curves.grid(True, alpha=0.3)

        # Plot individual temperature curves
        if self.fg_by_T is not None:
            for T, data in self.fg_by_T.items():
                s = data['strain']
                f = data['f']
                g = data['g']
                self.ax_fg_curves.plot(s, f, 'b-', alpha=0.3, linewidth=1)
                self.ax_fg_curves.plot(s, g, 'r-', alpha=0.3, linewidth=1)

        # Plot Persson average if available
        if self.piecewise_result is not None:
            s = self.piecewise_result['strain']
            split = self.piecewise_result['split']
            f_final = self.piecewise_result['f_avg']
            g_final = self.piecewise_result['g_avg']
            self.ax_fg_curves.plot(s, f_final, 'b-', linewidth=3.5, label='f(ε) Persson Avg')
            self.ax_fg_curves.plot(s, g_final, 'r-', linewidth=3.5, label='g(ε) Persson Avg')
            self.ax_fg_curves.axvline(split, color='green', linewidth=2, linestyle=':', alpha=0.8,
                                      label=f'Split @ {split*100:.1f}%')
            self.ax_fg_curves.legend(loc='upper right', fontsize=12, ncol=2)
        elif self.fg_averaged is not None:
            s = self.fg_averaged['strain']
            f_avg = self.fg_averaged['f_avg']
            g_avg = self.fg_averaged['g_avg']
            self.ax_fg_curves.plot(s, f_avg, 'b-', linewidth=3, label='f(ε) 평균')
            self.ax_fg_curves.plot(s, g_avg, 'r-', linewidth=3, label='g(ε) 평균')
            self.ax_fg_curves.legend(loc='upper right')

        self.ax_fg_curves.set_xlim(0, 1.0)
        self.ax_fg_curves.set_ylim(0, 1.1)

        self.canvas_mu_visc.draw()

    def _apply_temperature_shift(self):
        """Apply temperature shift to master curve using aT (and bT) and recalculate G."""
        try:
            # Get target temperature
            T_target = float(self.mu_calc_temp_var.get())

            # Check if aT data is loaded
            if not hasattr(self, 'persson_aT_interp') or self.persson_aT_interp is None:
                self._show_status("aT 시프트 팩터 데이터가 로드되지 않았습니다.\n\n"
                    "Tab 1에서 'aT 시프트 팩터 로드' 버튼을 사용하여\n"
                    "aT 데이터를 로드하세요.", 'warning')
                return

            # Check if Persson master curve is loaded
            if not hasattr(self, 'persson_master_curve') or self.persson_master_curve is None:
                self._show_status("Persson 정품 마스터 커브가 로드되지 않았습니다.\n\n"
                    "Tab 1에서 먼저 마스터 커브를 로드하세요.", 'warning')
                return

            # Get aT at target temperature
            T_ref = self.persson_aT_data['T_ref']
            log_aT = self.persson_aT_interp(T_target)
            aT = 10**log_aT

            # Get bT at target temperature (if available)
            has_bT = self.persson_aT_data.get('has_bT', False)
            if has_bT and hasattr(self, 'persson_bT_interp') and self.persson_bT_interp is not None:
                bT = float(self.persson_bT_interp(T_target))
            else:
                bT = 1.0  # No vertical shift if bT not available

            # Update status (bT is loaded but not applied to modulus for friction calc)
            status_msg = f"온도 시프트 계산 중... T={T_target}°C, aT={aT:.2e}"
            self.g_calc_status_var.set(status_msg)
            self.root.update_idletasks()

            # Get master curve data at reference temperature
            mc_data = self.persson_master_curve
            omega_ref = mc_data['omega'].copy()  # rad/s at Tref
            E_storage_ref = mc_data['E_storage'].copy()
            E_loss_ref = mc_data['E_loss'].copy()

            # Apply temperature shift:
            # Time-Temperature Superposition: E*(ω, T) = E*(ω·aT, T_ref)
            # To create shifted master curve: ω_shifted = ω_ref / aT
            # At higher T, aT < 1, so ω_shifted > ω_ref (frequency axis shifts UP)
            # → same ω maps to lower modulus on shifted curve → softer material ✓
            omega_shifted = omega_ref / aT

            # NOTE: bT (vertical shift) is NOT applied to modulus for friction calculation
            # bT is used only for master curve construction, not for subsequent friction calc
            # The modulus values remain unchanged from the reference temperature master curve
            E_storage_shifted = E_storage_ref
            E_loss_shifted = E_loss_ref

            # Update status
            self.g_calc_status_var.set(f"시프트된 마스터 커브 생성 중...")
            self.root.update_idletasks()

            # Create new material with shifted frequencies (bT not applied to modulus)
            from persson_model.utils.data_loader import create_material_from_dma
            material_name = f"Persson (T={T_target}°C, aT={aT:.2e})"

            self.material_shifted = create_material_from_dma(
                omega=omega_shifted,
                E_storage=E_storage_shifted,
                E_loss=E_loss_shifted,
                material_name=material_name,
                reference_temp=T_target
            )

            # Store shift info
            self.current_temp_shift = {
                'T_target': T_target,
                'T_ref': T_ref,
                'aT': aT,
                'log_aT': log_aT,
                'bT': bT,
                'has_bT': has_bT,
                'bT_applied': False  # bT is stored but not applied
            }

            # Update material for calculations
            self.material = self.material_shifted
            self.material_source = f"Persson 정품 (T={T_target}°C, aT={aT:.2e})"

            # Update aT status display
            status_str = f"T={T_target}°C, aT={aT:.2e} (Tref={T_ref}°C)"
            self.mu_aT_status_var.set(status_str)

            # Now recalculate G(q,v) with shifted material
            self.g_calc_status_var.set(f"G(q,v) 재계산 시작...")
            self.root.update_idletasks()

            # Call G recalculation (this will update Tab 3 data)
            self._recalculate_G_with_temperature()

            # Final status
            self.g_calc_status_var.set(f"완료: T={T_target}°C, aT={aT:.2e}, G 재계산됨")

            self.status_var.set(f"온도 시프트 적용 완료: T={T_target}°C")

        except ValueError as e:
            messagebox.showerror("오류", f"온도 값이 올바르지 않습니다: {e}")
        except Exception as e:
            import traceback
            messagebox.showerror("오류", f"온도 시프트 적용 실패:\n{str(e)}\n\n{traceback.format_exc()}")
            self.g_calc_status_var.set("오류 발생")

    def _recalculate_G_with_temperature(self):
        """Recalculate G(q,v) after temperature shift using existing g_calculator.

        Uses the same calculation method as Tab 3 to ensure consistency.
        Updates self.results['2d_results'] so mu_visc calculation uses new data.
        """
        try:
            # Check required data
            if self.psd_model is None:
                self.g_calc_status_var.set("PSD 데이터 없음 (Tab 0 확정 필요)")
                return

            if self.material is None:
                self.g_calc_status_var.set("마스터 커브 없음 (Tab 1 확정 필요)")
                return

            # Get parameters from Tab 3 settings
            sigma_0 = float(self.sigma_0_var.get()) * 1e6  # MPa to Pa
            temperature = float(self.temperature_var.get())
            poisson = float(self.poisson_var.get())
            n_q = int(self.n_q_var.get())
            n_phi = int(self.n_phi_var.get())

            # Get q range (use same variables as Tab 3)
            q_min = float(self.q_min_var.get())
            q_max = float(self.q_max_var.get())

            # Get velocity range
            v_min = float(self.v_min_var.get())
            v_max = float(self.v_max_var.get())
            n_v = int(self.n_velocity_var.get())

            q_array = np.logspace(np.log10(q_min), np.log10(q_max), n_q)
            v_array = np.logspace(np.log10(v_min), np.log10(v_max), n_v)

            # Update g_calculator with new (shifted) material
            from persson_model.core.g_calculator import GCalculator

            # Create complex modulus function from shifted material
            def modulus_func(omega):
                return self.material.get_modulus(omega, temperature=temperature)

            # Recreate g_calculator with shifted material (선형 계산만)
            self.g_calculator = GCalculator(
                psd_func=self.psd_model,
                modulus_func=modulus_func,
                sigma_0=sigma_0,
                velocity=v_array[0],
                poisson_ratio=poisson,
                n_angle_points=n_phi
            )

            # GUI에서 설정한 norm factor 적용
            try:
                self.g_calculator.PSD_NORMALIZATION_FACTOR = float(self.g_norm_factor_var.get())
            except (ValueError, AttributeError):
                self.g_calculator.PSD_NORMALIZATION_FACTOR = 1.5625

            # Progress callback
            def progress_callback(percent):
                self.g_calc_status_var.set(f"G(q,v) 재계산 중... {percent}%")
                self.root.update_idletasks()

            # Use the same calculation method as Tab 3
            results_2d = self.g_calculator.calculate_G_multi_velocity(
                q_array, v_array, q_min=q_min, progress_callback=progress_callback
            )

            # Update self.results so mu_visc calculation uses new data
            if not hasattr(self, 'results') or self.results is None:
                self.results = {}

            self.results['2d_results'] = results_2d

            # Also store temperature info
            self.results['temperature_shifted'] = True
            self.results['shift_info'] = self.current_temp_shift

            self.g_calc_status_var.set(f"G(q,v) 재계산 완료 (T={self.current_temp_shift['T_target']}°C)")

        except Exception as e:
            import traceback
            self.g_calc_status_var.set(f"G 계산 오류: {str(e)[:30]}")
            print(f"G recalculation error: {traceback.format_exc()}")

    def _update_G_display_after_temp_shift(self):
        """Update G(q,v) display after temperature shift."""
        if not hasattr(self, 'temp_shifted_G_data') or self.temp_shifted_G_data is None:
            return

        data = self.temp_shifted_G_data
        T = data['T']
        aT = data['aT']

        # Get data sources
        psd_src = getattr(self, 'psd_source', 'Unknown')
        mat_src = getattr(self, 'material_source', 'Unknown')

        # Update status text if Tab 3 result text exists
        if hasattr(self, 'g_result_text'):
            self.g_result_text.delete(1.0, tk.END)
            self.g_result_text.insert(tk.END, f"=== G(q,v) 온도 시프트 결과 ===\n\n")
            self.g_result_text.insert(tk.END, f"[데이터 소스]\n")
            self.g_result_text.insert(tk.END, f"  PSD (Tab 0): {psd_src}\n")
            self.g_result_text.insert(tk.END, f"  Master Curve (Tab 1): {mat_src}\n\n")
            self.g_result_text.insert(tk.END, f"계산 온도: T = {T}°C\n")
            self.g_result_text.insert(tk.END, f"시프트 팩터: aT = {aT:.4e}\n\n")

            G_final = data['G_qv'][:, -1]  # G at q_max
            self.g_result_text.insert(tk.END, f"G(q_max) 범위: {G_final.min():.4e} ~ {G_final.max():.4e}\n")

            P_final = data['P_q'][:, -1]
            self.g_result_text.insert(tk.END, f"P(q_max) 범위: {P_final.min():.4f} ~ {P_final.max():.4f}\n")

    def _calculate_mu_visc(self):
        """Calculate viscoelastic friction coefficient mu_visc.

        Implements the full Persson formula:
        μ_visc = (1/2) ∫[q0→q1] dq · q³ · C(q) · P(q) · S(q)
                 · ∫[0→2π] dφ · cosφ · Im[E(qv·cosφ, T)] / ((1-ν²)sigma_0)

        where:
        - P(q) = erf(1/(2√G(q))) : contact area ratio
        - S(q) = γ + (1-γ)P(q)² : contact correction factor
        - Im[E(ω,T)] : loss modulus (optionally corrected by g(strain))
        """
        # Check if data is from Tab 0 and Tab 1
        tab0_ready = getattr(self, 'tab0_finalized', False)
        tab1_ready = getattr(self, 'tab1_finalized', False)

        if not tab0_ready or self.psd_model is None:
            self._show_status("PSD 데이터가 설정되지 않았습니다!\n\n"
                "Tab 0 (PSD 생성)에서 PSD를 확정하세요.", 'warning')
            return

        # 마스터 커브: Tab 1 확정 또는 기본/예제 재료 허용
        if self.material is None:
            self._show_status("마스터 커브 데이터가 없습니다!\n\n"
                "Tab 1 (마스터 커브 생성)에서 마스터 커브를 확정하거나\n"
                "프로그램을 재시작하여 기본 재료를 로드하세요.", 'warning')
            return

        # 데이터 출처 정보 저장 (결과 표시에 사용)
        psd_src = getattr(self, 'psd_source', 'Unknown')
        mat_src = getattr(self, 'material_source', 'Unknown')
        self._current_calc_sources = {
            'psd': psd_src,
            'material': mat_src,
            'tab1_finalized': tab1_ready
        }

        if not self.results or '2d_results' not in self.results:
            self._show_status("먼저 G(q,v) 계산을 실행하세요 (탭 3).", 'warning')
            return

        # 자동 온도 시프트 & G 재계산 (aT 데이터가 있을 때)
        if hasattr(self, 'persson_aT_interp') and self.persson_aT_interp is not None:
            if hasattr(self, 'persson_master_curve') and self.persson_master_curve is not None:
                try:
                    self._apply_temperature_shift()
                except Exception as e_shift:
                    print(f"Auto temperature shift skipped: {e_shift}")

        try:
            self.status_var.set("μ_visc 계산 중...")
            self.mu_calc_button.config(state='disabled')
            self.mu_progress_var.set(0)  # Initialize progress bar
            self.root.update()

            # Get parameters
            sigma_0 = float(self.sigma_0_var.get()) * 1e6  # MPa to Pa
            temperature = float(self.temperature_var.get())
            poisson = float(self.poisson_var.get())
            gamma = float(self.gamma_var.get())
            n_phi = int(self.n_phi_var.get())
            use_fg = self.use_fg_correction_var.get()
            strain_est_method = self.strain_est_method_var.get()
            fixed_strain = float(self.fixed_strain_var.get()) / 100.0  # Convert % to fraction

            # Check h'rms data if using rms_slope method
            if strain_est_method == 'rms_slope':
                if self.rms_slope_calculator is None or self.rms_slope_profiles is None:
                    self._show_status("h'rms 데이터가 없습니다.\n\n"
                        "Tab 4 (h'rms/Local Strain)에서\n"
                        "'h'rms slope / Local Strain 계산' 버튼을 먼저 실행하세요.", 'warning')
                    self.mu_calc_button.config(state='normal')
                    return
                else:
                    # Show info about strain range being used
                    strain_min = self.rms_slope_profiles['strain'][0]
                    strain_max = self.rms_slope_profiles['strain'][-1]
                    self.status_var.set(f"h'rms 기반 strain 적용: {strain_min*100:.3f}% ~ {strain_max*100:.1f}%")
                    self.root.update()

            # Get G(q,v) results
            results_2d = self.results['2d_results']
            q = results_2d['q']
            v = results_2d['v']
            G_matrix = results_2d['G_matrix']

            # Get PSD values
            C_q = self.psd_model(q)

            # Precompute E' for strain estimation (using mid-frequency)
            omega_mid = 2 * np.pi * 1.0  # 1 Hz
            E_prime_ref = self.material.get_storage_modulus(omega_mid, temperature=temperature)

            # Prepare RMS slope strain interpolator if using that method
            rms_strain_interp = None
            if strain_est_method == 'rms_slope' and self.rms_slope_calculator is not None:
                from scipy.interpolate import interp1d
                rms_q = self.rms_slope_profiles['q']
                rms_strain = self.rms_slope_profiles['strain']
                # Use log-log interpolation for better accuracy
                log_q = np.log10(rms_q)
                log_strain = np.log10(np.maximum(rms_strain, 1e-10))
                rms_strain_interp = interp1d(log_q, log_strain, kind='linear',
                                             bounds_error=False, fill_value='extrapolate')

            # Create enhanced loss modulus function with strain-dependent correction
            def loss_modulus_func_enhanced(omega, T, q_val=None, G_val=None, C_val=None):
                """Loss modulus with optional nonlinear strain correction."""
                E_loss = self.material.get_loss_modulus(omega, temperature=T)

                if use_fg and self.g_interpolator is not None:
                    # Estimate local strain based on method
                    if strain_est_method == 'rms_slope' and rms_strain_interp is not None and q_val is not None:
                        # Use pre-calculated RMS slope based local strain
                        try:
                            strain_estimate = 10 ** rms_strain_interp(np.log10(q_val))
                            strain_estimate = np.clip(strain_estimate, 0.0, 1.0)
                        except:
                            strain_estimate = fixed_strain
                    elif strain_est_method == 'fixed':
                        strain_estimate = fixed_strain
                    elif strain_est_method == 'persson' and q_val is not None and C_val is not None:
                        # Persson's approach: strain ~ sqrt(C(q)*q^4) * sigma0/E'
                        from persson_model.core.friction import estimate_local_strain
                        E_prime = self.material.get_storage_modulus(omega, temperature=T)
                        strain_estimate = estimate_local_strain(
                            G_val if G_val is not None else 0.1,
                            C_val, q_val, sigma_0, E_prime, method='persson'
                        )
                    elif strain_est_method == 'simple' and G_val is not None:
                        # Simple estimate: strain ~ sqrt(G) * sigma0/E
                        E_prime = self.material.get_storage_modulus(omega, temperature=T)
                        strain_estimate = np.sqrt(max(G_val, 1e-10)) * sigma_0 / max(E_prime, 1e3)
                        strain_estimate = np.clip(strain_estimate, 0.0, 1.0)
                    else:
                        strain_estimate = fixed_strain

                    # Get g correction factor
                    g_val = self.g_interpolator(strain_estimate)
                    g_val = np.clip(g_val, 0.01, None)  # g can exceed 1.0
                    E_loss = E_loss * g_val

                return E_loss

            # Simple wrapper for FrictionCalculator compatibility
            def loss_modulus_func(omega, T):
                return self.material.get_loss_modulus(omega, temperature=T)

            # Set up g_interpolator for nonlinear correction
            g_interp = self.g_interpolator if use_fg else None

            # Create strain estimator function based on method
            def strain_estimator_func(q_arr, G_arr, velocity):
                """Return strain array for given q values."""
                n = len(q_arr)
                if strain_est_method == 'rms_slope' and rms_strain_interp is not None:
                    # Use pre-calculated RMS slope based local strain
                    strain_arr = np.zeros(n)
                    for i, qi in enumerate(q_arr):
                        try:
                            strain_arr[i] = 10 ** rms_strain_interp(np.log10(qi))
                        except:
                            strain_arr[i] = fixed_strain
                    return np.clip(strain_arr, 0.0, 1.0)
                elif strain_est_method == 'fixed':
                    return np.full(n, fixed_strain)
                elif strain_est_method == 'persson':
                    from persson_model.core.friction import estimate_local_strain
                    strain_arr = np.zeros(n)
                    C_q_local = self.psd_model(q_arr)
                    for i, (qi, Gi, Ci) in enumerate(zip(q_arr, G_arr, C_q_local)):
                        omega_i = qi * velocity
                        E_prime = self.material.get_storage_modulus(omega_i, temperature=temperature)
                        strain_arr[i] = estimate_local_strain(Gi, Ci, qi, sigma_0, E_prime, method='persson')
                    return np.clip(strain_arr, 0.0, 1.0)
                elif strain_est_method == 'simple':
                    strain_arr = np.sqrt(np.maximum(G_arr, 1e-10)) * sigma_0 / max(E_prime_ref, 1e3)
                    return np.clip(strain_arr, 0.0, 1.0)
                else:
                    return np.full(n, fixed_strain)

            # Apply nonlinear correction to G(q) if enabled
            # Recalculate G with f(ε), g(ε) applied INSIDE the integral:
            # G(q) = (1/8) ∫∫ q'³ C(q') |E_eff(q'v cosφ)|² / ((1-ν²)sigma_0)² dφ dq'
            # where |E_eff|² = (E'×f(ε))² + (E''×g(ε))²

            # PSD_NORMALIZATION_FACTOR: GUI 값 적용 (선형/비선형 동일)
            try:
                self.g_calculator.PSD_NORMALIZATION_FACTOR = float(self.g_norm_factor_var.get())
            except (ValueError, AttributeError):
                self.g_calculator.PSD_NORMALIZATION_FACTOR = 1.5625

            # ALWAYS recalculate G(q) with current normalization factor
            # This ensures Tab 2's G(q) graph and Tab 5's A/A0 use consistent values
            self.status_var.set("G(q) 재계산 중 (정규화 적용)...")
            self.root.update()

            q_min = float(self.q_min_var.get())
            G_matrix_corrected = np.zeros_like(G_matrix)
            P_matrix_corrected = np.zeros_like(G_matrix)
            G_integrand_corrected = np.zeros_like(G_matrix)

            for j, v_j in enumerate(v):
                self.g_calculator.velocity = v_j
                results_recalc = self.g_calculator.calculate_G_with_details(q, q_min=q_min)
                G_matrix_corrected[:, j] = results_recalc['G']
                P_matrix_corrected[:, j] = results_recalc['contact_area_ratio']
                G_integrand_corrected[:, j] = results_recalc['G_integrand']

                # Progress update (0-25%)
                if j % max(1, len(v) // 10) == 0:
                    progress = int((j + 1) / len(v) * 25)
                    self.mu_progress_var.set(progress)
                    self.root.update()

            # Update self.results['2d_results'] so Tab 2's G(q) graph shows correct values
            self.results['2d_results']['G_matrix'] = G_matrix_corrected
            self.results['2d_results']['P_matrix'] = P_matrix_corrected
            self.results['2d_results']['G_integrand_matrix'] = G_integrand_corrected

            self.status_var.set("G(q) 재계산 완료")
            self.root.update()

            if use_fg and self.f_interpolator is not None and self.g_interpolator is not None:
                self.status_var.set("비선형 G(q) 재계산 중 (적분 내 보정)...")
                self.root.update()

                # Get strain array for nonlinear correction
                if strain_est_method == 'rms_slope' and rms_strain_interp is not None:
                    strain_for_G = np.zeros(len(q))
                    for i, qi in enumerate(q):
                        try:
                            strain_for_G[i] = 10 ** rms_strain_interp(np.log10(qi))
                        except:
                            strain_for_G[i] = fixed_strain
                    strain_for_G = np.clip(strain_for_G, 0.0, 1.0)
                else:
                    strain_for_G = np.full(len(q), fixed_strain)

                # Set nonlinear correction on g_calculator
                # This applies f(ε), g(ε) INSIDE the angle integral
                self.g_calculator.storage_modulus_func = lambda w: self.material.get_storage_modulus(w, temperature=temperature)
                self.g_calculator.loss_modulus_func = lambda w: self.material.get_loss_modulus(w, temperature=temperature)
                self.g_calculator.set_nonlinear_correction(
                    f_interpolator=self.f_interpolator,
                    g_interpolator=self.g_interpolator,
                    strain_array=strain_for_G,
                    strain_q_array=q
                )

                # Recalculate G(q,v) with nonlinear correction inside integral
                for j, v_j in enumerate(v):
                    self.g_calculator.velocity = v_j
                    results_nl = self.g_calculator.calculate_G_with_details(q, q_min=q_min)
                    G_matrix_corrected[:, j] = results_nl['G']
                    P_matrix_corrected[:, j] = results_nl['contact_area_ratio']
                    G_integrand_corrected[:, j] = results_nl['G_integrand']

                    # Progress update (25-50%)
                    if j % max(1, len(v) // 10) == 0:
                        progress = 25 + int((j + 1) / len(v) * 25)
                        self.mu_progress_var.set(progress)
                        self.root.update()

                # Clear nonlinear correction after calculation
                self.g_calculator.clear_nonlinear_correction()

                # Update self.results['2d_results'] with nonlinear-corrected values
                self.results['2d_results']['G_matrix'] = G_matrix_corrected
                self.results['2d_results']['P_matrix'] = P_matrix_corrected
                self.results['2d_results']['G_integrand_matrix'] = G_integrand_corrected

                self.status_var.set("비선형 G(q) 재계산 완료 - μ_visc 계산 중...")
                self.root.update()

            # Create friction calculator with g_interpolator
            friction_calc = FrictionCalculator(
                psd_func=self.psd_model,
                loss_modulus_func=loss_modulus_func,
                sigma_0=sigma_0,
                velocity=v[0],
                temperature=temperature,
                poisson_ratio=poisson,
                gamma=gamma,
                n_angle_points=n_phi,
                g_interpolator=g_interp,
                strain_estimate=fixed_strain
            )

            # Calculate mu_visc for all velocities
            # Progress: Stage 1 (G recalc) 0-25%, Stage 2 (nonlinear, if used) 25-50%, Stage 3 (mu_visc) 50-100%
            def progress_callback(percent):
                if use_fg and self.f_interpolator is not None and self.g_interpolator is not None:
                    # With nonlinear: Stage 3 starts at 50%
                    scaled_percent = 50 + int(percent * 0.5)
                else:
                    # Without nonlinear: Stage 3 starts at 25% (after G recalc)
                    scaled_percent = 25 + int(percent * 0.75)
                self.mu_progress_var.set(scaled_percent)
                self.root.update()

            # Use strain_estimator if nonlinear correction is enabled
            strain_est = strain_estimator_func if use_fg else None

            # Use corrected G_matrix (will be same as original if nonlinear not applied)
            mu_array_raw, details = friction_calc.calculate_mu_visc_multi_velocity(
                q, G_matrix_corrected, v, C_q, progress_callback, strain_estimator=strain_est
            )

            # Apply smoothing if enabled
            smooth_mu = self.smooth_mu_var.get()
            if smooth_mu and len(mu_array_raw) >= 5:
                window = int(self.smooth_window_var.get())
                # Ensure window is odd and not larger than array
                window = min(window, len(mu_array_raw))
                if window % 2 == 0:
                    window -= 1
                window = max(3, window)

                # Apply Savitzky-Golay filter for smoothing
                mu_array = savgol_filter(mu_array_raw, window, 2)
            else:
                mu_array = mu_array_raw

            # Store results (both raw and smoothed)
            self.mu_visc_results = {
                'v': v,
                'mu': mu_array,
                'mu_raw': mu_array_raw,
                'details': details,
                'smoothed': smooth_mu
            }

            # Update plots
            self._update_mu_visc_plots(v, mu_array, details, use_nonlinear=use_fg)

            # Also refresh Tab 2's G(q) plot to show updated values
            try:
                self._plot_g_results()
            except Exception as e:
                print(f"Tab 2 G(q) 그래프 갱신 중 오류: {e}")

            # Update result text with detailed information
            self.mu_result_text.delete(1.0, tk.END)
            self.mu_result_text.insert(tk.END, "=" * 40 + "\n")
            self.mu_result_text.insert(tk.END, "μ_visc 계산 결과 (Persson 이론)\n")
            self.mu_result_text.insert(tk.END, "=" * 40 + "\n\n")

            # Display data sources
            sources = getattr(self, '_current_calc_sources', {})
            self.mu_result_text.insert(tk.END, "[데이터 출처]\n")
            self.mu_result_text.insert(tk.END, f"  PSD: {sources.get('psd', 'Unknown')}\n")
            self.mu_result_text.insert(tk.END, f"  마스터 커브: {sources.get('material', 'Unknown')}\n")
            if not sources.get('tab1_finalized', True):
                self.mu_result_text.insert(tk.END, "  ⚠ 주의: 기본/예제 마스터 커브 사용 중\n")
            self.mu_result_text.insert(tk.END, "\n")

            # Parameters used
            self.mu_result_text.insert(tk.END, "[계산 파라미터]\n")
            self.mu_result_text.insert(tk.END, f"  sigma_0 (공칭 압력): {sigma_0/1e6:.3f} MPa\n")
            self.mu_result_text.insert(tk.END, f"  T (온도): {temperature:.1f} °C\n")
            self.mu_result_text.insert(tk.END, f"  ν (푸아송비): {poisson:.2f}\n")
            self.mu_result_text.insert(tk.END, f"  γ (접촉 보정): {gamma:.2f}\n")
            self.mu_result_text.insert(tk.END, f"  각도 적분점: {n_phi}\n")
            norm_factor = self.g_calculator.PSD_NORMALIZATION_FACTOR
            self.mu_result_text.insert(tk.END, f"  PSD 정규화: {norm_factor:.3f}\n")

            # Smoothing info
            if smooth_mu:
                self.mu_result_text.insert(tk.END, f"  결과 스무딩: 적용 (윈도우={self.smooth_window_var.get()})\n")
            else:
                self.mu_result_text.insert(tk.END, "  결과 스무딩: 미적용\n")

            # f,g correction info - more prominent
            self.mu_result_text.insert(tk.END, "\n[비선형 보정]\n")
            if use_fg and self.f_interpolator is not None and self.g_interpolator is not None:
                self.mu_result_text.insert(tk.END, f"  상태: *** 적용됨 ***\n")
                self.mu_result_text.insert(tk.END, f"  Strain 추정: {strain_est_method}\n")
                if strain_est_method == 'fixed':
                    self.mu_result_text.insert(tk.END, f"  고정 Strain: {fixed_strain*100:.2f}%\n")
                if self.piecewise_result is not None:
                    split = self.piecewise_result['split']
                    self.mu_result_text.insert(tk.END, f"  Piecewise Split: {split*100:.1f}%\n")
                self.mu_result_text.insert(tk.END, "\n  [보정 적용 항목]\n")
                self.mu_result_text.insert(tk.END, "  • E'(ω) → E'(ω) × f(ε)  (저장탄성률)\n")
                self.mu_result_text.insert(tk.END, "  • E''(ω) → E''(ω) × g(ε) (손실탄성률)\n")
                self.mu_result_text.insert(tk.END, "  • G(q) 재계산: 적분 내부에서 보정된 E 사용\n")
                self.mu_result_text.insert(tk.END, "    G = (1/8)∫q³C(q)∫|E_eff/((1-ν²)σ₀)|²dφdq\n")
                self.mu_result_text.insert(tk.END, "  • P(q) = erf(1/(2√G)) : 재계산된 G 기반\n")
                self.mu_result_text.insert(tk.END, "  • S(q) = γ + (1-γ)P² : 재계산된 P 기반\n")
            elif use_fg and self.g_interpolator is not None:
                self.mu_result_text.insert(tk.END, f"  상태: *** 부분 적용 (g만) ***\n")
                self.mu_result_text.insert(tk.END, "  ※ f 곡선이 없어 손실탄성률만 보정됨\n")
            else:
                self.mu_result_text.insert(tk.END, "  상태: 미적용 (선형 계산)\n")
                if self.g_interpolator is None:
                    self.mu_result_text.insert(tk.END, "  ※ f,g 곡선이 없음\n")

            # Helper for smart formatting
            def smart_fmt(val, threshold=0.001):
                if abs(val) < threshold and val != 0:
                    return f'{val:.2e}'
                return f'{val:.4f}'

            self.mu_result_text.insert(tk.END, "\n[결과]\n")
            self.mu_result_text.insert(tk.END, f"  속도: {v[0]:.2e} ~ {v[-1]:.2e} m/s\n")
            mu_min, mu_max = np.min(mu_array), np.max(mu_array)
            self.mu_result_text.insert(tk.END, f"  μ_visc: {smart_fmt(mu_min)} ~ {smart_fmt(mu_max)}\n")

            # Find peak
            peak_idx = np.argmax(mu_array)
            peak_mu = mu_array[peak_idx]
            self.mu_result_text.insert(tk.END, f"  최대: μ={smart_fmt(peak_mu)} @ v={v[peak_idx]:.4f} m/s\n")

            # Show comprehensive diagnostic info
            if details and 'details' in details and len(details['details']) > 0:
                self.mu_result_text.insert(tk.END, f"\n[진단 정보 - 중간 속도]\n")
                mid_detail = details['details'][len(details['details']) // 2]
                mid_v = mid_detail.get('velocity', v[len(v)//2])
                self.mu_result_text.insert(tk.END, f"  속도: {mid_v:.2e} m/s\n")

                # G(q) values
                if 'G' in mid_detail:
                    G = mid_detail['G']
                    self.mu_result_text.insert(tk.END, f"  G(q) 범위: {np.min(G):.2e} ~ {np.max(G):.2e}\n")

                # P(q) - contact area ratio (critical for mu)
                if 'P' in mid_detail:
                    P = mid_detail['P']
                    P_mean = np.mean(P)
                    P_min, P_max = np.min(P), np.max(P)
                    self.mu_result_text.insert(tk.END, f"  P(q) 범위: {P_min:.4f} ~ {P_max:.4f} (평균: {P_mean:.4f})\n")
                    if P_max < 0.1:
                        self.mu_result_text.insert(tk.END, f"  ※ 경고: P(q)가 매우 작음 - G(q)가 너무 클 수 있음\n")
                        self.mu_result_text.insert(tk.END, f"     → sigma_0를 높이거나 표면 거칠기를 확인하세요\n")

                # S(q) - contact correction factor
                if 'S' in mid_detail:
                    S = mid_detail['S']
                    self.mu_result_text.insert(tk.END, f"  S(q) 범위: {np.min(S):.4f} ~ {np.max(S):.4f}\n")

                # Angle integral values
                if 'angle_integral' in mid_detail:
                    angle_int = mid_detail['angle_integral']
                    self.mu_result_text.insert(tk.END, f"  각도적분 범위: {np.min(angle_int):.2e} ~ {np.max(angle_int):.2e}\n")

                # Full integrand
                if 'integrand' in mid_detail:
                    integ = mid_detail['integrand']
                    self.mu_result_text.insert(tk.END, f"  피적분함수 max: {np.max(integ):.2e}\n")

                # q³C(q)P(q)S(q) term
                if 'q3CPS' in mid_detail:
                    q3CPS = mid_detail['q3CPS']
                    self.mu_result_text.insert(tk.END, f"  q³C(q)P(q)S(q) max: {np.max(q3CPS):.2e}\n")

                # === 추가: 각 항목별 기여도 분석 ===
                self.mu_result_text.insert(tk.END, f"\n[항목별 기여도 분석]\n")
                if 'q' in mid_detail and 'G' in mid_detail and 'P' in mid_detail:
                    q_diag = mid_detail['q']
                    G_diag = mid_detail['G']
                    P_diag = mid_detail['P']
                    S_diag = mid_detail.get('S', np.ones_like(P_diag))
                    angle_diag = mid_detail.get('angle_integral', np.ones_like(P_diag))
                    C_diag = mid_detail.get('C_q', self.psd_model(q_diag))

                    # 중간 q 인덱스
                    mid_idx = len(q_diag) // 2
                    q_mid = q_diag[mid_idx]

                    self.mu_result_text.insert(tk.END, f"  @ q = {q_mid:.2e} (1/m):\n")
                    self.mu_result_text.insert(tk.END, f"    • q³ = {q_mid**3:.2e}\n")
                    self.mu_result_text.insert(tk.END, f"    • C(q) = {C_diag[mid_idx]:.2e} m⁴\n")
                    self.mu_result_text.insert(tk.END, f"    • G(q) = {G_diag[mid_idx]:.2e}\n")
                    self.mu_result_text.insert(tk.END, f"    • P(q) = {P_diag[mid_idx]:.4f}  ← G가 크면 P→0!\n")
                    self.mu_result_text.insert(tk.END, f"    • S(q) = {S_diag[mid_idx]:.4f}\n")
                    self.mu_result_text.insert(tk.END, f"    • angle_int = {angle_diag[mid_idx]:.2e}\n")

                    # 곱의 결과
                    product = q_mid**3 * C_diag[mid_idx] * P_diag[mid_idx] * S_diag[mid_idx] * angle_diag[mid_idx]
                    self.mu_result_text.insert(tk.END, f"    • 곱 = {product:.2e}\n")

                    # P(q)가 작은 이유 분석
                    if P_diag[mid_idx] < 0.1:
                        self.mu_result_text.insert(tk.END, f"\n  *** P(q)가 작은 이유 ***\n")
                        self.mu_result_text.insert(tk.END, f"    P = erf(1/(2√G)) 에서:\n")
                        sqrt_G = np.sqrt(G_diag[mid_idx])
                        arg = 1.0 / (2.0 * sqrt_G) if sqrt_G > 0 else 10.0
                        self.mu_result_text.insert(tk.END, f"    √G = {sqrt_G:.2e}\n")
                        self.mu_result_text.insert(tk.END, f"    1/(2√G) = {arg:.4f}\n")
                        self.mu_result_text.insert(tk.END, f"    → G가 너무 크면 P→0, μ_visc→0\n")
                        self.mu_result_text.insert(tk.END, f"    → σ₀를 높이면 G가 작아짐 (G ∝ 1/σ₀²)\n")

                # C(q) - PSD values (critical for checking normalization)
                if 'C_q' in mid_detail:
                    C_q_diag = mid_detail['C_q']
                    self.mu_result_text.insert(tk.END, f"  C(q) 범위: {np.min(C_q_diag):.2e} ~ {np.max(C_q_diag):.2e} m⁴\n")
                    # Check typical values
                    if np.max(C_q_diag) < 1e-18:
                        self.mu_result_text.insert(tk.END, f"  ※ 경고: C(q)가 매우 작음 - PSD 정규화 확인 필요!\n")
                    elif np.max(C_q_diag) > 1e-8:
                        self.mu_result_text.insert(tk.END, f"  ※ 경고: C(q)가 매우 큼 - PSD 정규화 확인 필요!\n")

                # Loss modulus check - at mid frequency
                mid_q = mid_detail.get('q', q)[len(mid_detail.get('q', q)) // 2]
                mid_omega = mid_q * mid_v
                if hasattr(self, 'material') and self.material is not None:
                    E_loss_check = self.material.get_loss_modulus(mid_omega, temperature=temperature)
                    E_store_check = self.material.get_storage_modulus(mid_omega, temperature=temperature)
                    self.mu_result_text.insert(tk.END, f"\n[재료 특성 @ ω={mid_omega:.2e} rad/s]\n")
                    self.mu_result_text.insert(tk.END, f"  E'(저장탄성률): {E_store_check:.2e} Pa\n")
                    self.mu_result_text.insert(tk.END, f"  E''(손실탄성률): {E_loss_check:.2e} Pa\n")
                    self.mu_result_text.insert(tk.END, f"  tan(δ) = E''/E': {E_loss_check/max(E_store_check,1):.4f}\n")

            self.mu_result_text.insert(tk.END, "\n[속도별 μ_visc]\n")
            step = max(1, len(v) // 8)
            for i in range(0, len(v), step):
                self.mu_result_text.insert(tk.END, f"  v={v[i]:.2e}: μ={smart_fmt(mu_array[i])}\n")

            self.status_var.set("μ_visc 계산 완료")
            self.mu_calc_button.config(state='normal')

            self._show_status(f"μ_visc 계산 완료\n"
                               f"범위: {smart_fmt(mu_min)} ~ {smart_fmt(mu_max)}\n"
                               f"최대: μ={smart_fmt(peak_mu)} @ v={v[peak_idx]:.4f} m/s", 'success')

        except Exception as e:
            self.mu_calc_button.config(state='normal')
            messagebox.showerror("오류", f"μ_visc 계산 실패:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def _update_mu_visc_plots(self, v, mu_array, details, use_nonlinear=False):
        """Update mu_visc plots."""
        try:
            # Sanitize input arrays - replace NaN/Inf with safe values
            mu_array = np.nan_to_num(mu_array, nan=0.0, posinf=0.0, neginf=0.0)

            # Clear all subplots
            self.ax_mu_v.clear()
            self.ax_mu_cumulative.clear()

            # Remove any existing twin axes from ax_ps
            for ax in self.fig_mu_visc.axes:
                if ax is not self.ax_fg_curves and ax is not self.ax_mu_v and \
                   ax is not self.ax_mu_cumulative and ax is not self.ax_ps:
                    ax.remove()
            self.ax_ps.clear()

            # Helper function for smart formatting
            def smart_format(val, threshold=0.001):
                if abs(val) < threshold and val != 0:
                    return f'{val:.2e}'
                return f'{val:.4f}'

            # Plot 1: mu_visc vs velocity (handle NaN values)
            valid_mask = np.isfinite(mu_array)
            if np.any(valid_mask):
                self.ax_mu_v.semilogx(v[valid_mask], mu_array[valid_mask], 'b-', linewidth=2.5, marker='o', markersize=4)
            else:
                self.ax_mu_v.semilogx(v, np.zeros_like(v), 'b-', linewidth=2.5, marker='o', markersize=4)
            self.ax_mu_v.set_title('μ_visc(v) 곡선', fontweight='bold', fontsize=15)
            self.ax_mu_v.set_xlabel('속도 v (m/s)', fontsize=13)
            self.ax_mu_v.set_ylabel('마찰 계수 μ_visc', fontsize=13)
            self.ax_mu_v.grid(True, alpha=0.3)

            # Find peak (handle NaN values)
            mu_for_peak = np.where(np.isfinite(mu_array), mu_array, -np.inf)
            peak_idx = np.argmax(mu_for_peak)
            peak_mu = mu_array[peak_idx] if np.isfinite(mu_array[peak_idx]) else 0.0
            peak_v = v[peak_idx]
            self.ax_mu_v.plot(peak_v, peak_mu, 'r*', markersize=15,
                             label=f'최대값: μ={smart_format(peak_mu)} @ v={peak_v:.4f} m/s')

            # Find and mark μ at v=1 m/s (important reference point)
            if np.min(v) <= 1.0 <= np.max(v):
                from scipy.interpolate import interp1d
                # Interpolate to find μ at exactly 1 m/s
                valid_for_interp = np.isfinite(mu_array)
                if np.sum(valid_for_interp) >= 2:
                    f_interp = interp1d(np.log10(v[valid_for_interp]), mu_array[valid_for_interp],
                                       kind='linear', fill_value='extrapolate')
                    mu_at_1ms = float(f_interp(0))  # log10(1) = 0
                    self.ax_mu_v.plot(1.0, mu_at_1ms, 'go', markersize=12, markeredgecolor='black',
                                     markeredgewidth=1.5, zorder=10,
                                     label=f'v=1m/s: μ={smart_format(mu_at_1ms)}')
                    # Add vertical line at v=1 m/s
                    self.ax_mu_v.axvline(x=1.0, color='green', linestyle='--', alpha=0.5, linewidth=1)

            # Plot reference μ_visc data: single active + multiple overlay datasets
            ref_colors = ['#E53E3E', '#DD6B20', '#38A169', '#3182CE', '#805AD5',
                          '#D53F8C', '#718096', '#D69E2E', '#00B5D8', '#9F7AEA']
            try:
                if self.reference_mu_data is not None:
                    ref_v = self.reference_mu_data['v']
                    ref_mu = self.reference_mu_data['mu']
                    self.ax_mu_v.semilogx(ref_v, ref_mu, 'r-', linewidth=2, alpha=0.8,
                                         label='참조 (Persson)', zorder=5)
                # Plot multiple overlay reference datasets
                if hasattr(self, 'plotted_ref_datasets'):
                    for i, ds in enumerate(self.plotted_ref_datasets):
                        color = ref_colors[i % len(ref_colors)]
                        if ds['mu_log_v'] and ds['mu_vals']:
                            log_v_ref = np.array(ds['mu_log_v'])
                            mu_ref = np.array(ds['mu_vals'])
                            self.ax_mu_v.semilogx(10**log_v_ref, mu_ref, '-', color=color,
                                                  linewidth=1.8, alpha=0.8,
                                                  label=f'참조: {ds["name"]}', zorder=4)
            except Exception as e:
                print(f"[DEBUG] 참조 μ_visc 플롯 오류: {e}")

            self.ax_mu_v.legend(loc='upper left', fontsize=12)

            # Plot 2: Real Contact Area Ratio A/A0 = P(q_max) vs velocity
            P_qmax_array = np.zeros(len(v))

            for i, det in enumerate(details['details']):
                P = det.get('P', np.zeros(1))
                P_qmax_array[i] = P[-1] if len(P) > 0 else 0

            # Sanitize P_qmax_array
            P_qmax_array = np.nan_to_num(P_qmax_array, nan=0.0, posinf=1.0, neginf=0.0)

            # Store P_qmax_array in mu_visc_results for later export
            if hasattr(self, 'mu_visc_results') and self.mu_visc_results is not None:
                self.mu_visc_results['P_qmax'] = P_qmax_array

            # Calculate and display A/A0 gap with reference data
            self._update_area_gap_display(v, P_qmax_array)

            # Color based on nonlinear correction
            if use_nonlinear:
                label_str = 'A/A0 - 비선형 G(q)'
                color = 'r'
                title_suffix = ' (f,g 보정 적용)'
            else:
                label_str = 'A/A0 - 선형 G(q)'
                color = 'b'
                title_suffix = ''

            # Plot A/A0 = P(q_max)
            self.ax_mu_cumulative.semilogx(v, P_qmax_array, f'{color}-', linewidth=2,
                                            marker='s', markersize=4, label=label_str)

            # Overlay reference A/A0 data: single active + multiple overlay datasets
            try:
                if self.reference_area_data is not None:
                    ref_v = self.reference_area_data['v']
                    ref_area = self.reference_area_data['area']
                    self.ax_mu_cumulative.semilogx(ref_v, ref_area, 'r-', linewidth=2,
                                                    alpha=0.8, label='참조 A/A0 (Persson)', zorder=5)
                # Plot multiple overlay reference datasets
                if hasattr(self, 'plotted_ref_datasets'):
                    for i, ds in enumerate(self.plotted_ref_datasets):
                        clr = ref_colors[i % len(ref_colors)]
                        if ds['area_log_v'] and ds['area_vals']:
                            log_v_ref = np.array(ds['area_log_v'])
                            area_ref = np.array(ds['area_vals'])
                            self.ax_mu_cumulative.semilogx(10**log_v_ref, area_ref, '-', color=clr,
                                                            linewidth=1.8, alpha=0.8,
                                                            label=f'참조: {ds["name"]}', zorder=4)
            except Exception as e:
                print(f"[DEBUG] 참조 A/A0 플롯 오류: {e}")

            self.ax_mu_cumulative.set_title(f'실접촉 면적비율 A/A0{title_suffix}', fontweight='bold', fontsize=15)
            self.ax_mu_cumulative.set_xlabel('속도 v (m/s)', fontsize=13)
            self.ax_mu_cumulative.set_ylabel('A/A0 = P(q_max)', fontsize=13)
            self.ax_mu_cumulative.legend(loc='best', fontsize=12)
            self.ax_mu_cumulative.grid(True, alpha=0.3)

            # Set y-axis to show data with padding (auto-scale based on actual data)
            y_max_calc = np.max(P_qmax_array) * 1.2
            # Include reference data in y-axis range
            if self.reference_area_data is not None:
                y_max_calc = max(y_max_calc, np.max(self.reference_area_data['area']) * 1.2)
            # Include plotted overlay datasets in y-axis range
            if hasattr(self, 'plotted_ref_datasets'):
                for ds in self.plotted_ref_datasets:
                    if ds['area_vals']:
                        y_max_calc = max(y_max_calc, np.max(ds['area_vals']) * 1.2)
            y_max = y_max_calc if y_max_calc > 0 else 0.05
            if not np.isfinite(y_max):
                y_max = 1.0
            self.ax_mu_cumulative.set_ylim(0, y_max)

            # Plot 3: P(q), S(q) for middle velocity
            mid_idx = len(details['details']) // 2
            detail = details['details'][mid_idx]
            q = detail['q']
            P = detail['P']
            S = detail['S']
            cumulative = detail.get('cumulative_mu', np.zeros_like(q))

            # Handle NaN values in P, S, cumulative
            P = np.nan_to_num(P, nan=0.0, posinf=1.0, neginf=0.0)
            S = np.nan_to_num(S, nan=0.0, posinf=1.0, neginf=0.0)
            cumulative = np.nan_to_num(cumulative, nan=0.0, posinf=0.0, neginf=0.0)

            # Use twin axis for cumulative
            ax_twin = self.ax_ps.twinx()

            self.ax_ps.semilogx(q, P, 'b-', linewidth=1.5, label='P(q)')
            self.ax_ps.semilogx(q, S, 'r--', linewidth=1.5, label='S(q)')
            ax_twin.semilogx(q, cumulative, 'g-', linewidth=1.5, alpha=0.7, label='누적μ')

            self.ax_ps.set_title('P(q), S(q) / 누적 μ', fontweight='bold', fontsize=15)
            self.ax_ps.set_xlabel('파수 q (1/m)', fontsize=13)
            self.ax_ps.set_ylabel('P(q), S(q)', color='blue', fontsize=13)
            ax_twin.set_ylabel('누적 μ', color='green', fontsize=13)
            self.ax_ps.legend(loc='upper left', fontsize=12)
            ax_twin.legend(loc='upper right', fontsize=12)
            self.ax_ps.grid(True, alpha=0.3)
            self.ax_ps.set_ylim(0, 1.1)

            # Set twin axis limits safely
            cumulative_max = np.max(cumulative) if len(cumulative) > 0 else 0.1
            if not np.isfinite(cumulative_max) or cumulative_max <= 0:
                cumulative_max = 0.1
            ax_twin.set_ylim(0, cumulative_max * 1.2)

            self.fig_mu_visc.subplots_adjust(left=0.12, right=0.90, top=0.94, bottom=0.10, hspace=0.42, wspace=0.38)
            self.canvas_mu_visc.draw()

            # Auto-register graph data for friction results
            self._register_graph_data(
                "Friction_mu_vs_v", v, mu_array,
                "v(m/s)\tmu_visc", "Friction coefficient vs velocity")
            self._register_graph_data(
                "ContactArea_A_A0_vs_v", v, P_qmax_array,
                "v(m/s)\tA/A0", "Real contact area ratio vs velocity")

        except Exception as e:
            # Fallback: clear plots and show error message
            print(f"Plot update error: {e}")
            import traceback
            traceback.print_exc()
            self.ax_mu_v.clear()
            self.ax_mu_cumulative.clear()
            self.ax_ps.clear()
            self.ax_mu_v.text(0.5, 0.5, f'플롯 오류: {str(e)[:50]}',
                             ha='center', va='center', transform=self.ax_mu_v.transAxes)
            self.canvas_mu_visc.draw()

    def _toggle_reference_mu(self):
        """Toggle reference μ_visc display and redraw plot."""
        if self.mu_visc_results is not None:
            v = self.mu_visc_results['v']
            mu = self.mu_visc_results['mu']
            details = self.mu_visc_results['details']
            self._update_mu_visc_plots(v, mu, details)

    def _plot_ref_datasets_on_initial_axes(self):
        """Plot multiple reference datasets on initial (empty) axes without calculation results."""
        if not hasattr(self, 'plotted_ref_datasets') or not self.plotted_ref_datasets:
            return
        try:
            ref_colors = ['#E53E3E', '#DD6B20', '#38A169', '#3182CE', '#805AD5',
                          '#D53F8C', '#718096', '#D69E2E', '#00B5D8', '#9F7AEA']
            # Clear and re-plot on mu_v and mu_cumulative axes
            self.ax_mu_v.clear()
            self.ax_mu_v.set_title('μ_visc(v) 곡선', fontweight='bold', fontsize=15)
            self.ax_mu_v.set_xlabel('속도 v (m/s)', fontsize=13)
            self.ax_mu_v.set_ylabel('마찰 계수 μ_visc', fontsize=13)
            self.ax_mu_v.set_xscale('log')
            self.ax_mu_v.grid(True, alpha=0.3)

            self.ax_mu_cumulative.clear()
            self.ax_mu_cumulative.set_title('실접촉 면적비율 P(v)', fontweight='bold', fontsize=15)
            self.ax_mu_cumulative.set_xlabel('속도 v (m/s)', fontsize=13)
            self.ax_mu_cumulative.set_ylabel('평균 P(q)', fontsize=13)
            self.ax_mu_cumulative.set_xscale('log')
            self.ax_mu_cumulative.grid(True, alpha=0.3)

            for i, ds in enumerate(self.plotted_ref_datasets):
                color = ref_colors[i % len(ref_colors)]
                name = ds['name']
                if ds['mu_log_v'] and ds['mu_vals']:
                    log_v = np.array(ds['mu_log_v'])
                    mu_vals = np.array(ds['mu_vals'])
                    self.ax_mu_v.semilogx(10**log_v, mu_vals, '-', color=color,
                                          linewidth=1.8, alpha=0.8, label=f'참조: {name}')
                if ds['area_log_v'] and ds['area_vals']:
                    log_v = np.array(ds['area_log_v'])
                    area_vals = np.array(ds['area_vals'])
                    self.ax_mu_cumulative.semilogx(10**log_v, area_vals, '-', color=color,
                                                    linewidth=1.8, alpha=0.8, label=f'참조: {name}')

            self.ax_mu_v.legend(loc='upper left', fontsize=12)
            self.ax_mu_cumulative.legend(loc='best', fontsize=12)
            self.canvas_mu_visc.draw()
        except Exception as e:
            print(f"[DEBUG] _plot_ref_datasets_on_initial_axes error: {e}")

    def _reset_mu_visc_axes(self):
        """Reset all mu_visc plot axes to initial empty state."""
        try:
            pf = self.PLOT_FONTS
            # Top-left: f,g curves
            self.ax_fg_curves.clear()
            self.ax_fg_curves.set_title('f(\u03b5), g(\u03b5) 곡선', fontweight='bold', fontsize=pf['title'])
            self.ax_fg_curves.set_xlabel('변형률 \u03b5 (fraction)', fontsize=pf['label'])
            self.ax_fg_curves.set_ylabel('보정 계수', fontsize=pf['label'])
            self.ax_fg_curves.grid(True, alpha=0.3)

            # Top-right: mu_visc vs velocity
            self.ax_mu_v.clear()
            self.ax_mu_v.set_title('\u03bc_visc(v) 곡선', fontweight='bold', fontsize=pf['title'])
            self.ax_mu_v.set_xlabel('속도 v (m/s)', fontsize=pf['label'])
            self.ax_mu_v.set_ylabel('마찰 계수 \u03bc_visc', fontsize=pf['label'])
            self.ax_mu_v.set_xscale('log')
            self.ax_mu_v.grid(True, alpha=0.3)

            # Bottom-left: Contact Area Ratio vs Velocity
            self.ax_mu_cumulative.clear()
            self.ax_mu_cumulative.set_title('실접촉 면적비율 P(v)', fontweight='bold', fontsize=pf['title'])
            self.ax_mu_cumulative.set_xlabel('속도 v (m/s)', fontsize=pf['label'])
            self.ax_mu_cumulative.set_ylabel('평균 P(q)', fontsize=pf['label'])
            self.ax_mu_cumulative.set_xscale('log')
            self.ax_mu_cumulative.grid(True, alpha=0.3)

            # Bottom-right: P(q) and S(q)
            self.ax_ps.clear()
            self.ax_ps.set_title('P(q), S(q) 분포', fontweight='bold', fontsize=pf['title'])
            self.ax_ps.set_xlabel('파수 q (1/m)', fontsize=pf['label'])
            self.ax_ps.set_ylabel('P(q), S(q)', fontsize=pf['label'])
            self.ax_ps.set_xscale('log')
            self.ax_ps.grid(True, alpha=0.3)

            self.canvas_mu_visc.draw()
        except Exception as e:
            print(f"[DEBUG] _reset_mu_visc_axes error: {e}")

    def _analyze_mu_comparison(self):
        """Analyze difference between calculated and reference μ_visc and provide recommendations."""
        if self.mu_visc_results is None:
            self._show_status("먼저 μ_visc를 계산하세요.", 'warning')
            return

        if not hasattr(self, 'reference_mu_data') or self.reference_mu_data is None:
            self._show_status("참조 데이터가 없습니다.", 'warning')
            return

        # Get calculated and reference data
        calc_v = self.mu_visc_results['v']
        calc_mu = self.mu_visc_results['mu']
        ref_v = self.reference_mu_data['v']
        ref_mu = self.reference_mu_data['mu']

        # Interpolate to compare at common velocity points
        from scipy.interpolate import interp1d

        # Define comparison velocity range (overlap region)
        v_min = max(calc_v.min(), ref_v.min())
        v_max = min(calc_v.max(), ref_v.max())

        if v_min >= v_max:
            self._show_status("계산된 속도 범위와 참조 데이터 범위가 겹치지 않습니다.", 'warning')
            return

        # Create common velocity points in log space
        n_points = 50
        v_common = np.logspace(np.log10(v_min), np.log10(v_max), n_points)

        # Interpolate both datasets
        try:
            calc_interp = interp1d(np.log10(calc_v), calc_mu, kind='linear', fill_value='extrapolate')
            ref_interp = interp1d(np.log10(ref_v), ref_mu, kind='linear', fill_value='extrapolate')

            calc_at_common = calc_interp(np.log10(v_common))
            ref_at_common = ref_interp(np.log10(v_common))
        except Exception as e:
            messagebox.showerror("오류", f"보간 실패: {str(e)}")
            return

        # Calculate difference metrics
        diff = calc_at_common - ref_at_common
        abs_diff = np.abs(diff)
        rel_diff = diff / np.maximum(ref_at_common, 0.001) * 100  # percentage

        # Key velocity points
        v_points = [0.001, 0.01, 0.1, 1.0, 10.0]
        point_analysis = []

        for v_pt in v_points:
            if v_min <= v_pt <= v_max:
                calc_val = float(calc_interp(np.log10(v_pt)))
                ref_val = float(ref_interp(np.log10(v_pt)))
                diff_val = calc_val - ref_val
                rel_val = diff_val / max(ref_val, 0.001) * 100
                point_analysis.append((v_pt, calc_val, ref_val, diff_val, rel_val))

        # Define velocity regions
        low_v_mask = v_common < 0.01
        mid_v_mask = (v_common >= 0.01) & (v_common < 1.0)
        high_v_mask = v_common >= 1.0

        avg_diff_low = np.mean(diff[low_v_mask]) if np.any(low_v_mask) else 0
        avg_diff_mid = np.mean(diff[mid_v_mask]) if np.any(mid_v_mask) else 0
        avg_diff_high = np.mean(diff[high_v_mask]) if np.any(high_v_mask) else 0

        # Create analysis dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("μ_visc 비교 분석")
        dialog.geometry("650x700")
        dialog.transient(self.root)

        # Center dialog
        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - 650) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - 700) // 2
        dialog.geometry(f"+{x}+{y}")

        # Text widget for analysis
        text_frame = ttk.Frame(dialog, padding=10)
        text_frame.pack(fill=tk.BOTH, expand=True)

        text = tk.Text(text_frame, wrap=tk.WORD, font=('Courier', 15))
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text.yview)
        text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text.pack(fill=tk.BOTH, expand=True)

        # Build analysis report
        report = []
        report.append("=" * 60)
        report.append("         μ_visc 비교 분석 보고서")
        report.append("=" * 60)
        report.append("")

        report.append("[1. 전체 통계]")
        report.append(f"  분석 속도 범위: {v_min:.2e} ~ {v_max:.2e} m/s")
        report.append(f"  평균 차이 (계산-참조): {np.mean(diff):.4f}")
        report.append(f"  최대 차이: {np.max(abs_diff):.4f} (at v={v_common[np.argmax(abs_diff)]:.2e})")
        report.append(f"  평균 상대 차이: {np.mean(rel_diff):.1f}%")
        report.append("")

        report.append("[2. 속도 구간별 분석]")
        report.append(f"  저속 (v < 0.01 m/s): 평균 차이 = {avg_diff_low:+.4f}")
        report.append(f"  중속 (0.01 ≤ v < 1 m/s): 평균 차이 = {avg_diff_mid:+.4f}")
        report.append(f"  고속 (v ≥ 1 m/s): 평균 차이 = {avg_diff_high:+.4f}")
        report.append("")

        report.append("[3. 주요 속도에서의 비교]")
        report.append("  속도 (m/s)    계산값    참조값    차이      상대차이")
        report.append("  " + "-" * 55)
        for v_pt, calc_val, ref_val, diff_val, rel_val in point_analysis:
            report.append(f"  {v_pt:10.4f}  {calc_val:.4f}  {ref_val:.4f}  {diff_val:+.4f}  {rel_val:+.1f}%")
        report.append("")

        # Calculate slope (기울기) in log-log space
        slope_calc = np.polyfit(np.log10(v_common), calc_at_common, 1)[0]
        slope_ref = np.polyfit(np.log10(v_common), ref_at_common, 1)[0]
        slope_diff = slope_calc - slope_ref

        report.append("[4. 기울기 분석]")
        report.append("-" * 50)
        report.append(f"  계산 기울기 (dμ/d(log v)): {slope_calc:.4f}")
        report.append(f"  참조 기울기 (dμ/d(log v)): {slope_ref:.4f}")
        report.append(f"  기울기 차이: {slope_diff:+.4f}")
        report.append("")

        # Pattern analysis
        low_high_pattern = ""
        if avg_diff_low > 0 and avg_diff_high < 0:
            low_high_pattern = "저속↑ 고속↓"
            report.append(f"  패턴: {low_high_pattern} (기울기가 참조보다 낮음)")
        elif avg_diff_low < 0 and avg_diff_high > 0:
            low_high_pattern = "저속↓ 고속↑"
            report.append(f"  패턴: {low_high_pattern} (기울기가 참조보다 높음)")
        elif avg_diff_low > 0 and avg_diff_high > 0:
            low_high_pattern = "전체↑"
            report.append(f"  패턴: {low_high_pattern} (전체적으로 높음)")
        elif avg_diff_low < 0 and avg_diff_high < 0:
            low_high_pattern = "전체↓"
            report.append(f"  패턴: {low_high_pattern} (전체적으로 낮음)")
        report.append("")

        # Generate detailed recommendations based on pattern
        report.append("[5. 원인 분석 및 조언]")
        report.append("=" * 50)

        # Pattern: 저속↑ 고속↓ (기울기가 낮음)
        if avg_diff_low > 0 and avg_diff_high < 0:
            report.append("")
            report.append("▶ 패턴: 저속에서 높고, 고속에서 낮음 (기울기 부족)")
            report.append("")
            report.append("┌─────────────────────────────────────────────────────┐")
            report.append("│ [원인 1] DMA 마스터 커브 형태 차이                  │")
            report.append("├─────────────────────────────────────────────────────┤")
            report.append("│ • E'' 피크가 더 낮은 주파수에 위치                  │")
            report.append("│ • 전이 영역(transition region)이 더 넓음            │")
            report.append("│ • 저주파수 E''이 참조보다 높음                      │")
            report.append("│                                                     │")
            report.append("│ 해결: DMA 마스터 커브 비교 필요                     │")
            report.append("│       → Tab 1에서 마스터 커브 형태 확인            │")
            report.append("└─────────────────────────────────────────────────────┘")
            report.append("")
            report.append("┌─────────────────────────────────────────────────────┐")
            report.append("│ [원인 2] WLF 시프트 파라미터 차이                   │")
            report.append("├─────────────────────────────────────────────────────┤")
            report.append("│ • C1, C2 값이 다르면 주파수-온도 이동이 달라짐     │")
            report.append("│ • 참조 온도(T_ref)가 다를 수 있음                   │")
            report.append("│                                                     │")
            report.append("│ 해결: WLF 파라미터 확인                             │")
            report.append("│       → Persson 프로그램의 C1, C2, T_ref 확인      │")
            report.append("└─────────────────────────────────────────────────────┘")
            report.append("")
            report.append("┌─────────────────────────────────────────────────────┐")
            report.append("│ [원인 3] PSD 파라미터 차이                          │")
            report.append("├─────────────────────────────────────────────────────┤")
            report.append("│ • Hurst exponent (H)가 다름                         │")
            report.append("│   - H↑: 고주파수 기여↓ → 고속 μ↓                   │")
            report.append("│   - H↓: 고주파수 기여↑ → 고속 μ↑                   │")
            report.append("│ • q1 (상한 파수)이 다름                             │")
            report.append("│                                                     │")
            report.append("│ 해결: H 값 감소 시도 (예: 0.56 → 0.50)              │")
            report.append("│       또는 q1 증가 시도                             │")
            report.append("└─────────────────────────────────────────────────────┘")
            report.append("")
            report.append("┌─────────────────────────────────────────────────────┐")
            report.append("│ [원인 4] 저속 영역 추가 기여                        │")
            report.append("├─────────────────────────────────────────────────────┤")
            report.append("│ • Persson 프로그램이 저속에서 μ_adh 미포함         │")
            report.append("│ • 또는 P(q), S(q) 처리 방식 차이                    │")
            report.append("│                                                     │")
            report.append("│ 해결: γ 값 조정 (현재 → 0.5 이하로)                 │")
            report.append("└─────────────────────────────────────────────────────┘")

        # Pattern: 저속↓ 고속↑ (기울기가 높음)
        elif avg_diff_low < 0 and avg_diff_high > 0:
            report.append("")
            report.append("▶ 패턴: 저속에서 낮고, 고속에서 높음 (기울기 과다)")
            report.append("")
            report.append("┌─────────────────────────────────────────────────────┐")
            report.append("│ [원인 1] DMA 마스터 커브 형태 차이                  │")
            report.append("├─────────────────────────────────────────────────────┤")
            report.append("│ • E'' 피크가 더 높은 주파수에 위치                  │")
            report.append("│ • 저주파수 E''이 참조보다 낮음                      │")
            report.append("│                                                     │")
            report.append("│ 해결: DMA 마스터 커브 비교 필요                     │")
            report.append("└─────────────────────────────────────────────────────┘")
            report.append("")
            report.append("┌─────────────────────────────────────────────────────┐")
            report.append("│ [원인 2] PSD 파라미터 차이                          │")
            report.append("├─────────────────────────────────────────────────────┤")
            report.append("│ • Hurst exponent (H)가 너무 낮음                    │")
            report.append("│ • q1이 너무 높음                                    │")
            report.append("│                                                     │")
            report.append("│ 해결: H 값 증가 시도 또는 q1 감소                   │")
            report.append("└─────────────────────────────────────────────────────┘")

        # Pattern: 전체적으로 높거나 낮음
        elif low_high_pattern == "전체↑":
            report.append("")
            report.append("▶ 패턴: 전체적으로 참조보다 높음")
            report.append("")
            report.append("  가능한 원인:")
            report.append("  • P(q)=1, S(q)=1 근사 사용 (접촉 면적 보정 미적용)")
            report.append("  • C(q0) 값이 너무 큼")
            report.append("  • γ 값이 너무 높음")
            report.append("")
            report.append("  해결:")
            report.append("  • γ 값 감소 시도")
            report.append("  • C(q0) 값 감소 시도")

        elif low_high_pattern == "전체↓":
            report.append("")
            report.append("▶ 패턴: 전체적으로 참조보다 낮음")
            report.append("")
            report.append("  가능한 원인:")
            report.append("  • P(q), S(q) 보정이 과도하게 적용됨")
            report.append("  • f,g 비선형 보정이 과도함")
            report.append("  • E'' 마스터 커브가 전체적으로 낮음")
            report.append("")
            report.append("  해결:")
            report.append("  • γ 값 증가 시도")
            report.append("  • f,g 보정 해제 후 비교")

        report.append("")
        report.append("=" * 60)
        report.append("[6. 파라미터 조정 가이드]")
        report.append("=" * 60)
        report.append("")
        report.append("┌─────────────────────────────────────────────────────┐")
        report.append("│ 기울기를 높이려면 (고속 μ↑, 저속 μ↓):               │")
        report.append("├─────────────────────────────────────────────────────┤")
        report.append("│ • Hurst exponent (H) 감소                           │")
        report.append("│ • q1 (상한 파수) 증가                               │")
        report.append("│ • DMA: E'' 피크를 고주파수로 이동                   │")
        report.append("└─────────────────────────────────────────────────────┘")
        report.append("")
        report.append("┌─────────────────────────────────────────────────────┐")
        report.append("│ 기울기를 낮추려면 (고속 μ↓, 저속 μ↑):               │")
        report.append("├─────────────────────────────────────────────────────┤")
        report.append("│ • Hurst exponent (H) 증가                           │")
        report.append("│ • q1 (상한 파수) 감소                               │")
        report.append("│ • DMA: E'' 피크를 저주파수로 이동                   │")
        report.append("└─────────────────────────────────────────────────────┘")
        report.append("")
        report.append("┌─────────────────────────────────────────────────────┐")
        report.append("│ 전체 레벨 조정:                                     │")
        report.append("├─────────────────────────────────────────────────────┤")
        report.append("│ μ↑: γ 증가, f/g 보정 해제, C(q0) 증가              │")
        report.append("│ μ↓: γ 감소, f/g 보정 적용, C(q0) 감소              │")
        report.append("└─────────────────────────────────────────────────────┘")
        report.append("")

        # ============================================================
        # [7] 구체적인 파라미터 제안 계산
        # ============================================================
        report.append("=" * 60)
        report.append("[7. 구체적인 파라미터 제안]")
        report.append("=" * 60)
        report.append("")

        # Get current parameter values
        try:
            current_gamma = float(self.gamma_var.get())
        except:
            current_gamma = 0.6

        try:
            current_strain_factor = float(self.strain_factor_var.get())
        except:
            current_strain_factor = 0.5

        try:
            current_H = float(self.psd_H_var.get())
        except:
            current_H = 0.56

        try:
            current_q1 = float(self.psd_q1_var.get())
        except:
            current_q1 = 1e5

        try:
            current_Cq0 = float(self.psd_Cq0_var.get())
        except:
            current_Cq0 = 3.5e-13

        report.append("┌─────────────────────────────────────────────────────┐")
        report.append("│ 현재 파라미터 값                                    │")
        report.append("├─────────────────────────────────────────────────────┤")
        report.append(f"│ γ (gamma)      : {current_gamma:.3f}                              │")
        report.append(f"│ Strain Factor  : {current_strain_factor:.3f}                              │")
        report.append(f"│ H (Hurst)      : {current_H:.3f}                              │")
        report.append(f"│ q1             : {current_q1:.2e}                         │")
        report.append(f"│ C(q0)          : {current_Cq0:.2e}                         │")
        report.append("└─────────────────────────────────────────────────────┘")
        report.append("")

        # Calculate suggested parameters based on difference analysis
        mean_diff = np.mean(diff)
        mean_ref = np.mean(ref_at_common)
        rel_level_diff = mean_diff / mean_ref if mean_ref > 0 else 0

        # γ adjustment: μ ∝ (roughly) γ-dependent through S(q)
        # S(q) = γ + (1-γ)P² → higher γ → higher S → higher μ
        # Approximate: Δμ/μ ≈ k × Δγ/(1-γ) where k is sensitivity
        # Simplified: suggest γ change proportional to level difference
        if abs(rel_level_diff) > 0.01:
            # If μ is X% low, increase γ by ~X%
            suggested_gamma = current_gamma * (1 - rel_level_diff * 0.5)
            suggested_gamma = np.clip(suggested_gamma, 0.3, 0.9)
        else:
            suggested_gamma = current_gamma

        # Strain factor adjustment: higher strain → lower μ (through g(ε))
        # If μ is too high at low velocities, increase strain factor
        if avg_diff_low > 0.02:
            suggested_strain_factor = current_strain_factor * (1 + avg_diff_low * 0.5)
        elif avg_diff_low < -0.02:
            suggested_strain_factor = current_strain_factor * (1 + avg_diff_low * 0.5)
        else:
            suggested_strain_factor = current_strain_factor
        suggested_strain_factor = np.clip(suggested_strain_factor, 0.3, 1.0)

        # H adjustment: lower H → more high-frequency contribution → steeper slope
        # If slope is too low (저속↑ 고속↓), decrease H
        if slope_diff < -0.01:  # our slope is lower than reference
            # Need steeper slope → decrease H
            suggested_H = current_H + slope_diff * 2  # slope_diff is negative
            suggested_H = np.clip(suggested_H, 0.3, 0.9)
        elif slope_diff > 0.01:
            # Need less steep slope → increase H
            suggested_H = current_H + slope_diff * 2
            suggested_H = np.clip(suggested_H, 0.3, 0.9)
        else:
            suggested_H = current_H

        # q1 adjustment: higher q1 → more high-frequency contribution → higher μ at high v
        if avg_diff_high < -0.05:
            # μ is too low at high velocity → increase q1
            factor = 1 - avg_diff_high * 2  # avg_diff_high is negative
            suggested_q1 = current_q1 * factor
        elif avg_diff_high > 0.05:
            # μ is too high at high velocity → decrease q1
            factor = 1 - avg_diff_high * 2
            suggested_q1 = current_q1 * factor
        else:
            suggested_q1 = current_q1
        suggested_q1 = np.clip(suggested_q1, 1e4, 1e8)

        # C(q0) adjustment: higher C(q0) → higher μ overall
        if abs(rel_level_diff) > 0.05:
            suggested_Cq0 = current_Cq0 * (1 - rel_level_diff * 0.3)
            suggested_Cq0 = np.clip(suggested_Cq0, 1e-15, 1e-10)
        else:
            suggested_Cq0 = current_Cq0

        report.append("┌─────────────────────────────────────────────────────┐")
        report.append("│ ★ 제안 파라미터 값 ★                               │")
        report.append("├─────────────────────────────────────────────────────┤")

        # γ suggestion
        gamma_change = suggested_gamma - current_gamma
        gamma_arrow = "→" if abs(gamma_change) < 0.01 else ("↑" if gamma_change > 0 else "↓")
        report.append(f"│ γ (gamma)      : {suggested_gamma:.3f}  ({gamma_arrow} {abs(gamma_change):.3f})           │")

        # Strain factor suggestion
        sf_change = suggested_strain_factor - current_strain_factor
        sf_arrow = "→" if abs(sf_change) < 0.01 else ("↑" if sf_change > 0 else "↓")
        report.append(f"│ Strain Factor  : {suggested_strain_factor:.3f}  ({sf_arrow} {abs(sf_change):.3f})           │")

        # H suggestion
        h_change = suggested_H - current_H
        h_arrow = "→" if abs(h_change) < 0.01 else ("↑" if h_change > 0 else "↓")
        report.append(f"│ H (Hurst)      : {suggested_H:.3f}  ({h_arrow} {abs(h_change):.3f})           │")

        # q1 suggestion
        q1_ratio = suggested_q1 / current_q1
        q1_arrow = "→" if abs(q1_ratio - 1) < 0.05 else ("↑" if q1_ratio > 1 else "↓")
        report.append(f"│ q1             : {suggested_q1:.2e}  ({q1_arrow} ×{q1_ratio:.2f})     │")

        # C(q0) suggestion
        cq0_ratio = suggested_Cq0 / current_Cq0
        cq0_arrow = "→" if abs(cq0_ratio - 1) < 0.05 else ("↑" if cq0_ratio > 1 else "↓")
        report.append(f"│ C(q0)          : {suggested_Cq0:.2e}  ({cq0_arrow} ×{cq0_ratio:.2f})     │")

        report.append("└─────────────────────────────────────────────────────┘")
        report.append("")

        # Detailed reasoning
        report.append("┌─────────────────────────────────────────────────────┐")
        report.append("│ 제안 근거                                           │")
        report.append("├─────────────────────────────────────────────────────┤")
        report.append(f"│ • 전체 레벨 차이: {rel_level_diff*100:+.1f}%                          │")
        report.append(f"│ • 기울기 차이: {slope_diff:+.4f}                            │")
        report.append(f"│ • 저속 영역 차이: {avg_diff_low:+.4f}                          │")
        report.append(f"│ • 고속 영역 차이: {avg_diff_high:+.4f}                          │")
        report.append("└─────────────────────────────────────────────────────┘")
        report.append("")

        # Priority recommendations
        report.append("┌─────────────────────────────────────────────────────┐")
        report.append("│ 우선 조정 권장 순서                                 │")
        report.append("├─────────────────────────────────────────────────────┤")

        priorities = []
        if abs(slope_diff) > 0.02:
            priorities.append(("H (Hurst)", abs(slope_diff), "기울기 보정"))
        if abs(rel_level_diff) > 0.1:
            priorities.append(("γ (gamma)", abs(rel_level_diff), "전체 레벨 보정"))
        if abs(avg_diff_high) > 0.1:
            priorities.append(("q1", abs(avg_diff_high), "고속 영역 보정"))
        if abs(avg_diff_low) > 0.05:
            priorities.append(("Strain Factor", abs(avg_diff_low), "저속 영역 보정"))

        priorities.sort(key=lambda x: x[1], reverse=True)

        if priorities:
            for i, (param, impact, reason) in enumerate(priorities[:3], 1):
                report.append(f"│ {i}. {param:15s} - {reason:20s}    │")
        else:
            report.append("│ 현재 파라미터로 충분히 일치합니다.                 │")

        report.append("└─────────────────────────────────────────────────────┘")
        report.append("")

        # Insert report
        text.insert(tk.END, "\n".join(report))
        text.config(state=tk.DISABLED)

        # Close button
        ttk.Button(dialog, text="닫기", command=dialog.destroy).pack(pady=10)

    def _export_mu_visc_results(self):
        """Export mu_visc results to CSV files with selection dialog."""
        if self.mu_visc_results is None:
            self._show_status("먼저 μ_visc를 계산하세요.", 'warning')
            return

        # Create dialog for selecting data to export
        dialog = tk.Toplevel(self.root)
        dialog.title("CSV 내보내기 - μ_visc 데이터 선택")
        dialog.geometry("450x620")
        dialog.resizable(False, True)
        dialog.transient(self.root)
        dialog.grab_set()

        # Center the dialog
        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - 450) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - 620) // 2
        dialog.geometry(f"+{x}+{y}")

        # Description
        desc_frame = ttk.Frame(dialog, padding=10)
        desc_frame.pack(fill=tk.X)
        ttk.Label(desc_frame, text="내보낼 데이터를 선택하세요.\n각 데이터는 별도의 CSV 파일로 저장됩니다.",
                  font=('Segoe UI', 17)).pack(anchor=tk.W)

        # Export button frame - pack at bottom FIRST so it's always visible
        export_frame = ttk.Frame(dialog, padding=10)
        export_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # Select all / Deselect all buttons - pack at bottom before checkboxes
        btn_frame = ttk.Frame(dialog, padding=10)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # Checkbox frame - fills remaining space
        check_frame = ttk.LabelFrame(dialog, text="데이터 선택", padding=10)
        check_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Data options - main results
        main_label = ttk.Label(check_frame, text="[기본 결과 (v vs 값)]", font=('Arial', 12, 'bold'))
        main_label.pack(anchor=tk.W, pady=(0, 5))

        main_options = [
            ("μ_visc(v) - 마찰계수", "mu_v", True),
            ("μ_visc_raw(v) - 스무딩 전", "mu_raw_v", False),
        ]

        # q-dependent data options
        q_label = ttk.Label(check_frame, text="\n[q 의존성 데이터 (특정 속도)]", font=('Arial', 12, 'bold'))
        q_label.pack(anchor=tk.W, pady=(5, 5))

        q_options = [
            ("P(q) - 접촉 면적비", "P_q", False),
            ("S(q) - 접촉 보정 인자", "S_q", False),
            ("G(q) - 누적 G 값", "G_q", False),
            ("C(q) - PSD 값", "C_q", False),
            ("Integrand(q) - μ 피적분함수", "integrand_q", False),
            ("Cumulative μ(q) - 누적 기여", "cumulative_q", False),
            ("Angle Integral(q) - 각도 적분", "angle_int_q", False),
        ]

        # Create checkbox variables
        check_vars = {}

        for display_name, key, default in main_options:
            var = tk.BooleanVar(value=default)
            check_vars[key] = var
            cb = ttk.Checkbutton(check_frame, text=display_name, variable=var)
            cb.pack(anchor=tk.W, pady=1)

        for display_name, key, default in q_options:
            var = tk.BooleanVar(value=default)
            check_vars[key] = var
            cb = ttk.Checkbutton(check_frame, text=display_name, variable=var)
            cb.pack(anchor=tk.W, pady=1)

        # Velocity selection for q-dependent data
        v_frame = ttk.Frame(check_frame)
        v_frame.pack(fill=tk.X, pady=5)
        ttk.Label(v_frame, text="q 데이터 속도 인덱스:", font=('Segoe UI', 17)).pack(side=tk.LEFT)

        v_array = self.mu_visc_results['v']
        n_v = len(v_array)
        # Default to index closest to 1 m/s
        default_idx = np.argmin(np.abs(np.log10(v_array) - 0))  # log10(1) = 0
        self.export_v_idx_var = tk.StringVar(value=str(default_idx))
        v_spin = ttk.Spinbox(v_frame, from_=0, to=n_v-1, textvariable=self.export_v_idx_var, width=5)
        v_spin.pack(side=tk.LEFT, padx=5)

        # Show current velocity
        def update_v_label(*args):
            try:
                idx = int(self.export_v_idx_var.get())
                if 0 <= idx < n_v:
                    v_val = v_array[idx]
                    v_info_label.config(text=f"(v = {v_val:.2e} m/s)")
            except:
                pass

        v_info_label = ttk.Label(v_frame, text=f"(v = {v_array[default_idx]:.2e} m/s)", font=('Segoe UI', 17))
        v_info_label.pack(side=tk.LEFT)
        self.export_v_idx_var.trace('w', update_v_label)

        def select_all():
            for var in check_vars.values():
                var.set(True)

        def deselect_all():
            for var in check_vars.values():
                var.set(False)

        ttk.Button(btn_frame, text="전체 선택", command=select_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="전체 해제", command=deselect_all).pack(side=tk.LEFT, padx=5)

        def do_export():
            # Check if any data is selected
            selected = [key for key, var in check_vars.items() if var.get()]
            if not selected:
                self._show_status("내보낼 데이터를 선택하세요.", 'warning')
                return

            # Ask for save directory
            save_dir = filedialog.askdirectory(
                title="CSV 파일 저장 폴더 선택",
                parent=dialog
            )
            if not save_dir:
                return

            try:
                v = self.mu_visc_results['v']
                mu = self.mu_visc_results['mu']
                mu_raw = self.mu_visc_results.get('mu_raw', mu)
                details = self.mu_visc_results.get('details', {})

                # Get velocity index for q-dependent data
                v_idx = int(self.export_v_idx_var.get())
                v_idx = max(0, min(v_idx, len(v) - 1))

                exported_files = []

                # 평가 정보 수집
                from datetime import datetime
                try:
                    load_mpa = float(self.sigma_0_var.get())
                except:
                    load_mpa = 0.0
                try:
                    calc_temp = float(self.mu_calc_temp_var.get())
                except:
                    calc_temp = 20.0
                nonlinear_applied = self.use_fg_correction_var.get() if hasattr(self, 'use_fg_correction_var') else False

                # MC prefix for filenames
                mc_prefix = self._get_mc_prefix()
                def mc_fn(base):
                    return f"{mc_prefix}_{base}" if mc_prefix else base

                # 공통 헤더 정보 (쉼표 구분)
                def get_header_lines():
                    return [
                        f"# 생성일시,{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                        f"# 마스터커브,{mc_prefix if mc_prefix else 'N/A'}",
                        f"# 공칭하중(MPa),{load_mpa:.3f}",
                        f"# 계산온도(°C),{calc_temp:.1f}",
                        f"# 비선형보정적용,{'예' if nonlinear_applied else '아니오'}",
                        "#"
                    ]

                # Export main results (v vs value)
                if check_vars['mu_v'].get():
                    filename = mc_fn("mu_visc_vs_velocity.csv")
                    filepath = os.path.join(save_dir, filename)
                    lines = get_header_lines() + ["velocity [m/s],mu_visc"]
                    for vi, mui in zip(v, mu):
                        lines.append(f"{vi:.6e},{mui:.6f}")
                    with open(filepath, 'w', encoding='utf-8-sig') as f:
                        f.write("\n".join(lines))
                    exported_files.append(filename)

                if check_vars['mu_raw_v'].get():
                    filename = mc_fn("mu_visc_raw_vs_velocity.csv")
                    filepath = os.path.join(save_dir, filename)
                    lines = get_header_lines() + ["velocity [m/s],mu_visc_raw"]
                    for vi, mui in zip(v, mu_raw):
                        lines.append(f"{vi:.6e},{mui:.6f}")
                    with open(filepath, 'w', encoding='utf-8-sig') as f:
                        f.write("\n".join(lines))
                    exported_files.append(filename)

                # Export q-dependent data
                all_details = details.get('details', [])
                if v_idx < len(all_details):
                    det = all_details[v_idx]
                    q_arr = det.get('q', np.array([]))
                    v_selected = v[v_idx]

                    q_data_map = {
                        'P_q': ('P', 'P_q'),
                        'S_q': ('S', 'S_q'),
                        'G_q': ('G', 'G_q'),
                        'C_q': ('C_q', 'C_q'),
                        'integrand_q': ('integrand', 'integrand_q'),
                        'cumulative_q': ('cumulative_mu', 'cumulative_mu_q'),
                        'angle_int_q': ('angle_integral', 'angle_integral_q'),
                    }

                    for key, (data_key, file_suffix) in q_data_map.items():
                        if check_vars[key].get():
                            data = det.get(data_key)
                            if data is not None and len(data) == len(q_arr):
                                filename = mc_fn(f"{file_suffix}_v{v_idx}_{v_selected:.2e}.csv")
                                filepath = os.path.join(save_dir, filename)
                                lines = [f"# velocity = {v_selected:.6e} m/s", f"q [1/m],{data_key}"]
                                for qi, di in zip(q_arr, data):
                                    lines.append(f"{qi:.6e},{di:.6e}")
                                with open(filepath, 'w', encoding='utf-8') as f:
                                    f.write("\n".join(lines))
                                exported_files.append(filename)

                dialog.destroy()
                self._show_status(f"CSV 파일 내보내기 완료:\n\n" + "\n".join(exported_files) + f"\n\n저장 위치: {save_dir}", 'success')
                self.status_var.set(f"CSV 내보내기 완료: {len(exported_files)}개 파일")

            except Exception as e:
                import traceback
                messagebox.showerror("오류", f"내보내기 실패:\n{str(e)}\n\n{traceback.format_exc()}", parent=dialog)

        ttk.Button(export_frame, text="내보내기", command=do_export).pack(side=tk.RIGHT, padx=5)
        ttk.Button(export_frame, text="취소", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)

    def _update_area_gap_display(self, v, P_qmax_array):
        """Calculate and display A/A0 gap between calculated and reference values.

        Compares at v = 10^-4, 10^-2, 10^0 m/s as requested.
        """
        try:
            if not hasattr(self, 'area_gap_var'):
                return

            if self.reference_area_data is None:
                self.area_gap_var.set("A/A0 Gap: 참조 데이터 없음")
                return

            ref_v = self.reference_area_data['v']
            ref_area = self.reference_area_data['area']

            if len(v) < 2 or len(ref_v) < 2:
                self.area_gap_var.set("A/A0 Gap: 데이터 부족")
                return

            # Interpolate reference data to calculated velocities
            from scipy.interpolate import interp1d
            ref_interp = interp1d(np.log10(ref_v), ref_area, kind='linear',
                                  bounds_error=False, fill_value='extrapolate')

            # Target velocities: 10^-4, 10^-2, 10^0 m/s
            target_velocities = [1e-4, 1e-2, 1e0]
            gaps = []

            for target_v in target_velocities:
                # Find closest velocity index
                idx = np.argmin(np.abs(np.log10(v) - np.log10(target_v)))
                actual_v = v[idx]

                # Get calculated and reference values
                calc_val = P_qmax_array[idx]
                ref_val = ref_interp(np.log10(actual_v))

                # Calculate percentage gap
                if ref_val > 0:
                    gap = (calc_val - ref_val) / ref_val * 100
                else:
                    gap = 0
                gaps.append((actual_v, gap))

            # Format display: show gaps at 10^-4, 10^-2, 10^0
            gap_strs = [f"v={v_:.0e}: {g:+.1f}%" for v_, g in gaps]
            self.area_gap_var.set(f"A/A0 Gap: " + ", ".join(gap_strs))

        except Exception as e:
            print(f"[DEBUG] A/A0 gap 계산 오류: {e}")
            self.area_gap_var.set("A/A0 Gap: 계산 오류")

    def _export_mu_and_area_csv(self):
        """Export mu_visc and A/A0 data to a single CSV file with log10(v)."""
        if self.mu_visc_results is None:
            self._show_status("먼저 μ_visc를 계산하세요.", 'warning')
            return

        v = self.mu_visc_results.get('v')
        mu = self.mu_visc_results.get('mu')
        P_qmax = self.mu_visc_results.get('P_qmax')

        if v is None or mu is None:
            self._show_status("계산 결과가 없습니다.", 'warning')
            return

        if P_qmax is None:
            self._show_status("A/A0 데이터가 없습니다. 먼저 μ_visc를 계산하세요.", 'warning')
            return

        # File dialog
        file_path = filedialog.asksaveasfilename(
            title="μ_visc + A/A0 데이터 내보내기",
            defaultextension=".csv",
            filetypes=[("CSV 파일", "*.csv"), ("텍스트 파일", "*.txt"), ("모든 파일", "*.*")],
            initialfile=self._make_export_filename("mu_visc_and_area")
        )

        if not file_path:
            return

        try:
            from datetime import datetime

            # 평가 정보 수집
            try:
                load_mpa = float(self.sigma_0_var.get())
            except:
                load_mpa = 0.0
            try:
                calc_temp = float(self.mu_calc_temp_var.get())
            except:
                calc_temp = 20.0
            nonlinear_applied = self.use_fg_correction_var.get() if hasattr(self, 'use_fg_correction_var') else False

            with open(file_path, 'w', encoding='utf-8-sig') as f:
                # 헤더 정보 (주석은 별도 행으로)
                mc_prefix = self._get_mc_prefix()
                f.write("# mu_visc and A/A0 data\n")
                f.write(f"# 생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# 마스터커브: {mc_prefix if mc_prefix else 'N/A'}\n")
                f.write(f"# 공칭하중(MPa): {load_mpa:.3f}\n")
                f.write(f"# 계산온도(°C): {calc_temp:.1f}\n")
                f.write(f"# 비선형보정적용: {'예' if nonlinear_applied else '아니오'}\n")
                f.write("#\n")
                f.write("# Column 1: log10(v) [m/s]\n")
                f.write("# Column 2: mu_visc (friction coefficient)\n")
                f.write("# Column 3: A/A0 (real contact area ratio)\n")
                f.write("#\n")
                f.write("log10_v,mu_visc,A_A0\n")

                # Data - 쉼표로 구분 (진짜 CSV 형식)
                for i in range(len(v)):
                    log_v = np.log10(v[i])
                    f.write(f"{log_v:.6e},{mu[i]:.6e},{P_qmax[i]:.6e}\n")

            self._show_status(f"파일 저장 완료:\n{file_path}", 'success')
            self.status_var.set(f"CSV 내보내기 완료: {file_path}")

        except Exception as e:
            messagebox.showerror("오류", f"파일 저장 실패:\n{str(e)}")

    def _get_ref_datasets_path(self):
        """Return path to the saved reference datasets JSON file."""
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reference_datasets.json')

    def _load_saved_datasets(self):
        """Load saved reference datasets from JSON file."""
        path = self._get_ref_datasets_path()
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_datasets_to_file(self, datasets):
        """Save reference datasets to JSON file."""
        path = self._get_ref_datasets_path()
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(datasets, f, ensure_ascii=False, indent=2)

    def _generate_dataset_name(self):
        """Auto-generate dataset name from current calculation conditions.
        Format: CompoundName_Tref_Tfriction_NonLinear/Linear_SurfaceType
        Example: S100_40_20_NonLinear_IDIADA
        """
        parts = []

        # 1. Compound/Material name (short)
        compound = ""
        if hasattr(self, 'material') and self.material is not None and hasattr(self.material, 'name'):
            name = self.material.name
            # Extract short name from various formats
            if 'Persson' in name and '(' in name:
                # "Persson (filename)" -> extract filename without extension
                inner = name.split('(', 1)[1].rstrip(')')
                # Remove common suffixes
                for suffix in ['.txt', '.csv', '.dat', '_smoothed', ' (smoothed)']:
                    inner = inner.replace(suffix, '')
                # Try to extract compound code (e.g., S100, S120)
                import re
                match = re.search(r'[A-Z]\d{2,4}', inner)
                if match:
                    compound = match.group()
                else:
                    compound = inner.strip()[:20]
            elif 'Master Curve' in name:
                compound = "MC"
            elif 'SBR' in name:
                compound = "SBR"
            elif 'Measured' in name:
                compound = "Measured"
            else:
                compound = name[:15].replace(' ', '')
        elif hasattr(self, 'material_source') and self.material_source:
            src = str(self.material_source)
            import re
            match = re.search(r'[A-Z]\d{2,4}', src)
            if match:
                compound = match.group()
            elif 'SBR' in src:
                compound = "SBR"
            else:
                compound = src[:15].replace(' ', '')
        if compound:
            parts.append(compound)

        # 2. Rubber reference temperature (Tref)
        try:
            tref = self.mc_tref_var.get()
            tref_val = float(tref)
            parts.append(f"{tref_val:.0f}")
        except (AttributeError, ValueError):
            pass

        # 3. Friction/calculation temperature
        try:
            t_calc = self.mu_calc_temp_var.get()
            t_val = float(t_calc)
            parts.append(f"{t_val:.0f}")
        except (AttributeError, ValueError):
            try:
                t_calc = self.temperature_var.get()
                t_val = float(t_calc)
                parts.append(f"{t_val:.0f}")
            except (AttributeError, ValueError):
                pass

        # 4. NonLinear / Linear
        try:
            if self.use_fg_correction_var.get():
                parts.append("Non")
            else:
                parts.append("Lin")
        except (AttributeError):
            pass

        # 5. Surface type (from q1 preset name)
        try:
            surface = self.surface_q1_var.get()
            if surface and not surface.startswith('('):
                parts.append(surface)
        except (AttributeError):
            pass

        if parts:
            return "_".join(parts)
        else:
            from datetime import datetime
            return datetime.now().strftime("Data_%Y%m%d_%H%M")

    def _edit_reference_data(self):
        """Open dialog for editing reference mu_visc and A/A0 data."""
        dialog = tk.Toplevel(self.root)
        dialog.title("참조 데이터 편집")
        dialog.resizable(True, True)
        dialog.transient(self.root)
        dialog.grab_set()

        # Size dialog to 90% of root window, minimum 1100x800
        root_w = self.root.winfo_width()
        root_h = self.root.winfo_height()
        dlg_w = max(1100, int(root_w * 0.9))
        dlg_h = max(800, int(root_h * 0.9))
        x = self.root.winfo_x() + (root_w - dlg_w) // 2
        y = self.root.winfo_y() + (root_h - dlg_h) // 2
        dialog.geometry(f"{dlg_w}x{dlg_h}+{x}+{y}")
        dialog.minsize(1000, 700)

        # Instructions
        inst_frame = ttk.Frame(dialog, padding=10)
        inst_frame.pack(fill=tk.X)
        ttk.Label(inst_frame,
                  text="참조 데이터를 복사하여 붙여넣기 하세요.\n"
                       "형식: 각 줄에 'log10(v) [탭 or 공백] 값' (예: -5.0  0.59)",
                  font=('Segoe UI', 17)).pack(anchor=tk.W)

        # Main horizontal split: Left = text input, Right = saved datasets
        main_pane = ttk.PanedWindow(dialog, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # === LEFT: text input area with tabs ===
        left_frame = ttk.Frame(main_pane)
        main_pane.add(left_frame, weight=3)

        notebook = ttk.Notebook(left_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: mu_visc reference
        mu_frame = ttk.Frame(notebook, padding=5)
        notebook.add(mu_frame, text="  mu_visc 참조 데이터  ")

        ttk.Label(mu_frame, text="mu_visc 참조 데이터 (log10(v) \\t mu_visc):",
                  font=('Arial', 12, 'bold')).pack(anchor=tk.W)
        mu_text = tk.Text(mu_frame, height=20, font=("Courier", 15), wrap=tk.NONE)
        mu_scroll = ttk.Scrollbar(mu_frame, orient=tk.VERTICAL, command=mu_text.yview)
        mu_text.configure(yscrollcommand=mu_scroll.set)
        mu_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        mu_text.pack(fill=tk.BOTH, expand=True)

        # Pre-fill with existing data
        if hasattr(self, 'reference_mu_data') and self.reference_mu_data is not None:
            log_v = self.reference_mu_data.get('log_v')
            mu = self.reference_mu_data.get('mu')
            if log_v is not None and mu is not None:
                for lv, m in zip(log_v, mu):
                    mu_text.insert(tk.END, f"{lv:.6e}\t{m:.6e}\n")

        # Tab 2: A/A0 reference
        area_frame = ttk.Frame(notebook, padding=5)
        notebook.add(area_frame, text="  A/A0 참조 데이터  ")

        ttk.Label(area_frame, text="A/A0 참조 데이터 (log10(v) \\t A/A0):",
                  font=('Arial', 12, 'bold')).pack(anchor=tk.W)
        area_text = tk.Text(area_frame, height=20, font=("Courier", 15), wrap=tk.NONE)
        area_scroll = ttk.Scrollbar(area_frame, orient=tk.VERTICAL, command=area_text.yview)
        area_text.configure(yscrollcommand=area_scroll.set)
        area_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        area_text.pack(fill=tk.BOTH, expand=True)

        # Pre-fill with existing data
        if hasattr(self, 'reference_area_data') and self.reference_area_data is not None:
            log_v = self.reference_area_data.get('log_v')
            area = self.reference_area_data.get('area')
            if log_v is not None and area is not None:
                for lv, a in zip(log_v, area):
                    area_text.insert(tk.END, f"{lv:.6e}\t{a:.6e}\n")

        # Button to load calculation results into text areas
        calc_load_frame = ttk.Frame(left_frame)
        calc_load_frame.pack(fill=tk.X, pady=(3, 0))

        def load_calc_results():
            """Load current mu_visc calculation results into text areas."""
            if not hasattr(self, 'mu_visc_results') or self.mu_visc_results is None:
                self._show_status("먼저 μ_visc 계산을 실행하세요.", 'warning')
                return
            v = self.mu_visc_results.get('v')
            mu = self.mu_visc_results.get('mu')
            P_qmax = self.mu_visc_results.get('P_qmax')
            if v is None or mu is None:
                self._show_status("계산 결과에 속도/mu 데이터가 없습니다.", 'warning')
                return
            # Fill mu_visc text
            mu_text.delete("1.0", tk.END)
            for vi, mi in zip(v, mu):
                log_v = np.log10(vi) if vi > 0 else -99
                mu_text.insert(tk.END, f"{log_v:.6e}\t{mi:.6e}\n")
            # Fill A/A0 text
            if P_qmax is not None:
                area_text.delete("1.0", tk.END)
                for vi, ai in zip(v, P_qmax):
                    log_v = np.log10(vi) if vi > 0 else -99
                    area_text.insert(tk.END, f"{log_v:.6e}\t{ai:.6e}\n")
            n_mu = len(mu)
            n_area = len(P_qmax) if P_qmax is not None else 0
            self._show_status(f"계산 결과를 불러왔습니다.\n"
                                f"mu_visc: {n_mu}pts"
                                + (f", A/A0: {n_area}pts" if n_area > 0 else ""), 'success')

        calc_result_btn = ttk.Button(calc_load_frame, text="계산 결과 불러오기",
                                      command=load_calc_results)
        calc_result_btn.pack(side=tk.LEFT, padx=2, ipady=2)
        has_results = hasattr(self, 'mu_visc_results') and self.mu_visc_results is not None
        ttk.Label(calc_load_frame,
                  text="(계산 결과 있음)" if has_results else "(계산 결과 없음)",
                  foreground='#16A34A' if has_results else '#9CA3AF',
                  font=('Segoe UI', 14)).pack(side=tk.LEFT, padx=5)

        # === RIGHT: saved datasets panel ===
        right_frame = ttk.LabelFrame(main_pane, text="저장된 데이터셋", padding=5)
        main_pane.add(right_frame, weight=2)

        # Save controls: name entry + save button
        save_frame = ttk.Frame(right_frame)
        save_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(save_frame, text="이름:").pack(side=tk.LEFT, padx=(0, 3))
        dataset_name_var = tk.StringVar()
        name_entry = ttk.Entry(save_frame, textvariable=dataset_name_var, width=20)
        name_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        # Auto-name generation button
        def _auto_generate_name():
            auto_name = self._generate_dataset_name()
            dataset_name_var.set(auto_name)
        ttk.Button(save_frame, text="자동", command=_auto_generate_name, width=4).pack(side=tk.LEFT, padx=(0, 3))

        # Pre-fill with auto-generated name
        dataset_name_var.set(self._generate_dataset_name())

        # Dataset checkbox list (scrollable)
        list_frame = ttk.Frame(right_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)

        ds_canvas = tk.Canvas(list_frame, highlightthickness=0)
        ds_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=ds_canvas.yview)
        ds_inner_frame = ttk.Frame(ds_canvas)
        ds_canvas.configure(yscrollcommand=ds_scrollbar.set)
        ds_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        ds_canvas.pack(fill=tk.BOTH, expand=True)
        ds_canvas_window = ds_canvas.create_window((0, 0), window=ds_inner_frame, anchor='nw')

        def _on_ds_inner_configure(event=None):
            ds_canvas.configure(scrollregion=ds_canvas.bbox("all"))

        def _on_ds_canvas_configure(event=None):
            ds_canvas.itemconfig(ds_canvas_window, width=event.width)

        ds_inner_frame.bind('<Configure>', _on_ds_inner_configure)
        ds_canvas.bind('<Configure>', _on_ds_canvas_configure)

        ds_check_vars = {}  # {dataset_name: BooleanVar}

        # Load saved datasets
        saved_datasets = self._load_saved_datasets()

        def get_checked_names():
            """Return list of checked dataset names."""
            return [name for name, var in ds_check_vars.items() if var.get()]

        def _bind_scroll(widget):
            """Bind mouse wheel scroll events to a widget for the dataset canvas."""
            widget.bind('<Button-4>', lambda e: ds_canvas.yview_scroll(-1, "units"))
            widget.bind('<Button-5>', lambda e: ds_canvas.yview_scroll(1, "units"))
            widget.bind('<MouseWheel>', lambda e: ds_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

        _bind_scroll(ds_canvas)

        def refresh_checkbox_list():
            """Rebuild the checkbox list from saved_datasets."""
            for widget in ds_inner_frame.winfo_children():
                widget.destroy()
            ds_check_vars.clear()
            for ds_name in sorted(saved_datasets.keys()):
                ds = saved_datasets[ds_name]
                mu_count = len(ds.get('mu_log_v', []))
                area_count = len(ds.get('area_log_v', []))
                var = tk.BooleanVar(value=False)
                ds_check_vars[ds_name] = var
                row = ttk.Frame(ds_inner_frame)
                row.pack(fill=tk.X, pady=1, padx=2)
                cb = ttk.Checkbutton(row, variable=var,
                                     command=lambda n=ds_name: on_checkbox_toggle(n))
                cb.pack(side=tk.LEFT, padx=(2, 4))
                lbl = ttk.Label(row, text=f"{ds_name}  (mu:{mu_count}, A/A0:{area_count})",
                                font=('Segoe UI', 14))
                lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
                _bind_scroll(row)
                _bind_scroll(cb)
                _bind_scroll(lbl)

        refresh_checkbox_list()

        # Detail label
        detail_label = ttk.Label(right_frame, text="", font=('Segoe UI', 13),
                                 wraplength=280, foreground='#64748B')
        detail_label.pack(fill=tk.X, pady=3)

        def on_checkbox_toggle(ds_name):
            """Show detail for the toggled checkbox dataset."""
            ds = saved_datasets.get(ds_name)
            if ds:
                mu_n = len(ds.get('mu_log_v', []))
                area_n = len(ds.get('area_log_v', []))
                checked = get_checked_names()
                detail_label.config(
                    text=f"[{ds_name}] mu_visc: {mu_n}pts, A/A0: {area_n}pts\n"
                         f"저장일: {ds.get('saved_date', 'N/A')}\n"
                         f"선택됨: {len(checked)}개")

        # Buttons for dataset management
        ds_btn_frame = ttk.Frame(right_frame)
        ds_btn_frame.pack(fill=tk.X, pady=5)

        def save_dataset():
            """Save current text area data as a named dataset."""
            name = dataset_name_var.get().strip()
            if not name:
                self._show_status("데이터셋 이름을 입력하세요.", 'warning')
                return

            # Parse current text area data
            mu_content = mu_text.get("1.0", tk.END).strip()
            area_content = area_text.get("1.0", tk.END).strip()

            mu_log_v, mu_vals = [], []
            if mu_content:
                for line in mu_content.split('\n'):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            mu_log_v.append(float(parts[0]))
                            mu_vals.append(float(parts[1]))
                        except ValueError:
                            continue

            area_log_v, area_vals = [], []
            if area_content:
                for line in area_content.split('\n'):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            area_log_v.append(float(parts[0]))
                            area_vals.append(float(parts[1]))
                        except ValueError:
                            continue

            if not mu_log_v and not area_log_v:
                self._show_status("저장할 데이터가 없습니다.\n텍스트 영역에 데이터를 붙여넣기 하세요.", 'warning')
                return

            from datetime import datetime
            ds_entry = {
                'mu_log_v': mu_log_v,
                'mu_vals': mu_vals,
                'area_log_v': area_log_v,
                'area_vals': area_vals,
                'saved_date': datetime.now().strftime('%Y-%m-%d %H:%M')
            }

            if name in saved_datasets:
                if not messagebox.askyesno("덮어쓰기", f"'{name}' 데이터셋이 이미 존재합니다.\n덮어쓰시겠습니까?", parent=dialog):
                    return

            saved_datasets[name] = ds_entry
            self._save_datasets_to_file(saved_datasets)

            # Refresh checkbox list
            refresh_checkbox_list()

            self._show_status(f"'{name}' 데이터셋 저장 완료\n"
                                f"mu_visc: {len(mu_log_v)}pts, A/A0: {len(area_log_v)}pts", 'success')

        def load_dataset():
            """Load checked dataset into text areas."""
            checked = get_checked_names()
            if not checked:
                self._show_status("불러올 데이터셋을 체크하세요.", 'warning')
                return
            ds_name = checked[0]
            ds = saved_datasets[ds_name]

            # Fill mu_visc text
            mu_text.delete("1.0", tk.END)
            for lv, mv in zip(ds.get('mu_log_v', []), ds.get('mu_vals', [])):
                mu_text.insert(tk.END, f"{lv:.6e}\t{mv:.6e}\n")

            # Fill area text
            area_text.delete("1.0", tk.END)
            for lv, av in zip(ds.get('area_log_v', []), ds.get('area_vals', [])):
                area_text.insert(tk.END, f"{lv:.6e}\t{av:.6e}\n")

            dataset_name_var.set(ds_name)

        def delete_dataset():
            """Delete checked datasets."""
            checked = get_checked_names()
            if not checked:
                self._show_status("삭제할 데이터셋을 체크하세요.", 'warning')
                return
            names_str = ', '.join(checked)
            if not messagebox.askyesno("삭제 확인",
                                       f"선택한 {len(checked)}개 데이터셋을 삭제하시겠습니까?\n({names_str})",
                                       parent=dialog):
                return
            for ds_name in checked:
                del saved_datasets[ds_name]
            self._save_datasets_to_file(saved_datasets)
            refresh_checkbox_list()
            detail_label.config(text="")

        ttk.Button(save_frame, text="저장", command=save_dataset, width=6).pack(side=tk.LEFT)

        def plot_selected_datasets():
            """Plot checked datasets on the graph (multiple overlay)."""
            checked = get_checked_names()
            if not checked:
                self._show_status("플롯할 데이터셋을 체크하세요.", 'warning')
                return

            self.plotted_ref_datasets = []
            for ds_name in checked:
                ds = saved_datasets[ds_name]
                self.plotted_ref_datasets.append({
                    'name': ds_name,
                    'mu_log_v': ds.get('mu_log_v', []),
                    'mu_vals': ds.get('mu_vals', []),
                    'area_log_v': ds.get('area_log_v', []),
                    'area_vals': ds.get('area_vals', []),
                })

            # Refresh plots if mu_visc results exist
            if hasattr(self, 'mu_visc_results') and self.mu_visc_results is not None:
                v = self.mu_visc_results.get('v')
                mu = self.mu_visc_results.get('mu')
                details = self.mu_visc_results.get('details')
                if v is not None and mu is not None and details is not None:
                    use_nonlinear = self.mu_use_fg_var.get() if hasattr(self, 'mu_use_fg_var') else False
                    self._update_mu_visc_plots(v, mu, details, use_nonlinear=use_nonlinear)
            else:
                # Even without calculation results, plot reference data on initial axes
                self._plot_ref_datasets_on_initial_axes()

            self._show_status(f"선택한 {len(checked)}개 데이터셋을 그래프에 표시했습니다.\n"
                                f"데이터셋: {', '.join(checked)}", 'success')

        ttk.Button(ds_btn_frame, text="불러오기", command=load_dataset, width=10).pack(side=tk.LEFT, padx=3)
        ttk.Button(ds_btn_frame, text="삭제", command=delete_dataset, width=8).pack(side=tk.LEFT, padx=3)

        # Export to Excel button
        def export_to_excel():
            """Export checked datasets to a single Excel file."""
            checked = get_checked_names()
            if not checked:
                self._show_status("엑셀로 내보낼 데이터셋을 체크하세요.", 'warning')
                return

            file_path = filedialog.asksaveasfilename(
                parent=dialog,
                title="엑셀 파일로 내보내기",
                defaultextension=".xlsx",
                filetypes=[("Excel Files", "*.xlsx"), ("All Files", "*.*")],
                initialfile=f"reference_data_{checked[0]}.xlsx" if len(checked) == 1
                            else "reference_data_combined.xlsx"
            )
            if not file_path:
                return

            try:
                # Try openpyxl first, fall back to csv if not available
                try:
                    from openpyxl import Workbook
                    wb = Workbook()

                    # Sheet 1: mu_visc data
                    ws_mu = wb.active
                    ws_mu.title = "mu_visc"
                    # Header row
                    header = ["log10(v)"]
                    for ds_name in checked:
                        header.append(f"mu_visc ({ds_name})")
                    ws_mu.append(header)

                    # Collect all unique log_v values
                    all_log_v = set()
                    for ds_name in checked:
                        ds = saved_datasets[ds_name]
                        for lv in ds.get('mu_log_v', []):
                            all_log_v.add(round(lv, 8))
                    all_log_v = sorted(all_log_v)

                    # Write data rows
                    for lv in all_log_v:
                        row_data = [lv]
                        for ds_name in checked:
                            ds = saved_datasets[ds_name]
                            mu_lv = ds.get('mu_log_v', [])
                            mu_vals = ds.get('mu_vals', [])
                            val = None
                            for i, mlv in enumerate(mu_lv):
                                if abs(round(mlv, 8) - lv) < 1e-7:
                                    val = mu_vals[i]
                                    break
                            row_data.append(val if val is not None else "")
                        ws_mu.append(row_data)

                    # Sheet 2: A/A0 data
                    ws_area = wb.create_sheet("A_A0")
                    header_area = ["log10(v)"]
                    for ds_name in checked:
                        header_area.append(f"A/A0 ({ds_name})")
                    ws_area.append(header_area)

                    all_log_v_area = set()
                    for ds_name in checked:
                        ds = saved_datasets[ds_name]
                        for lv in ds.get('area_log_v', []):
                            all_log_v_area.add(round(lv, 8))
                    all_log_v_area = sorted(all_log_v_area)

                    for lv in all_log_v_area:
                        row_data = [lv]
                        for ds_name in checked:
                            ds = saved_datasets[ds_name]
                            area_lv = ds.get('area_log_v', [])
                            area_vals = ds.get('area_vals', [])
                            val = None
                            for i, alv in enumerate(area_lv):
                                if abs(round(alv, 8) - lv) < 1e-7:
                                    val = area_vals[i]
                                    break
                            row_data.append(val if val is not None else "")
                        ws_area.append(row_data)

                    # Auto-fit column widths
                    for ws in [ws_mu, ws_area]:
                        for col in ws.columns:
                            max_len = max(len(str(cell.value or "")) for cell in col)
                            ws.column_dimensions[col[0].column_letter].width = max(12, max_len + 2)

                    wb.save(file_path)
                    self._show_status(f"엑셀 파일 저장 완료: {os.path.basename(file_path)}\n"
                                        f"데이터셋: {', '.join(checked)}", 'success')

                except ImportError:
                    # Fallback: save as CSV if openpyxl not available
                    import csv
                    csv_path = file_path.replace('.xlsx', '.csv')
                    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
                        writer = csv.writer(f)
                        writer.writerow(["# Reference Data Export"])
                        writer.writerow([])
                        writer.writerow(["# mu_visc data"])
                        header = ["log10(v)"] + [f"mu_visc ({n})" for n in checked]
                        writer.writerow(header)
                        for ds_name in checked:
                            ds = saved_datasets[ds_name]
                            for lv, mv in zip(ds.get('mu_log_v', []), ds.get('mu_vals', [])):
                                writer.writerow([lv, mv])
                        writer.writerow([])
                        writer.writerow(["# A/A0 data"])
                        header_area = ["log10(v)"] + [f"A/A0 ({n})" for n in checked]
                        writer.writerow(header_area)
                        for ds_name in checked:
                            ds = saved_datasets[ds_name]
                            for lv, av in zip(ds.get('area_log_v', []), ds.get('area_vals', [])):
                                writer.writerow([lv, av])
                    self._show_status(f"CSV 파일 저장 완료 (openpyxl 미설치): {os.path.basename(csv_path)}", 'success')

            except Exception as e:
                messagebox.showerror("오류", f"엑셀 내보내기 실패:\n{str(e)}", parent=dialog)

        ttk.Button(ds_btn_frame, text="엑셀 내보내기", command=export_to_excel, width=12).pack(side=tk.LEFT, padx=3)

        # Plot button (separate row for visibility)
        plot_btn_frame = ttk.Frame(right_frame)
        plot_btn_frame.pack(fill=tk.X, pady=3)
        plot_btn = ttk.Button(plot_btn_frame, text="체크된 데이터 플롯", command=plot_selected_datasets)
        plot_btn.pack(fill=tk.X, padx=3, ipady=4)

        # Plot clear and reset buttons
        plot_ctrl_frame = ttk.Frame(right_frame)
        plot_ctrl_frame.pack(fill=tk.X, pady=3)

        def clear_plotted_datasets():
            """Clear all plotted reference datasets from graph."""
            self.plotted_ref_datasets = []
            # Refresh plots
            if hasattr(self, 'mu_visc_results') and self.mu_visc_results is not None:
                v = self.mu_visc_results.get('v')
                mu = self.mu_visc_results.get('mu')
                details = self.mu_visc_results.get('details')
                if v is not None and mu is not None and details is not None:
                    use_nonlinear = self.mu_use_fg_var.get() if hasattr(self, 'mu_use_fg_var') else False
                    self._update_mu_visc_plots(v, mu, details, use_nonlinear=use_nonlinear)
            else:
                # Reset axes to default empty state
                self._reset_mu_visc_axes()
            self._show_status("플롯된 참조 데이터를 모두 지웠습니다.", 'info')

        def reset_plots():
            """Reset all plots to initial empty state."""
            self.plotted_ref_datasets = []
            self._reset_mu_visc_axes()
            self._show_status("그래프를 초기화했습니다.", 'info')

        ttk.Button(plot_ctrl_frame, text="플롯 지우기", command=clear_plotted_datasets,
                   width=14).pack(side=tk.LEFT, padx=3, ipady=2)
        ttk.Button(plot_ctrl_frame, text="플롯 초기화", command=reset_plots,
                   width=14).pack(side=tk.LEFT, padx=3, ipady=2)

        # Button frame
        btn_frame = ttk.Frame(dialog, padding=10)
        btn_frame.pack(fill=tk.X)

        def apply_data():
            """Parse and apply the reference data."""
            try:
                # Parse mu_visc data
                mu_content = mu_text.get("1.0", tk.END).strip()
                if mu_content:
                    mu_lines = [l.strip() for l in mu_content.split('\n') if l.strip() and not l.startswith('#')]
                    if mu_lines:
                        mu_data = []
                        for line in mu_lines:
                            parts = line.split()
                            if len(parts) >= 2:
                                try:
                                    log_v = float(parts[0])
                                    mu_val = float(parts[1])
                                    mu_data.append((log_v, mu_val))
                                except ValueError:
                                    continue
                        if mu_data:
                            mu_data = np.array(mu_data)
                            log_v = mu_data[:, 0]
                            mu = mu_data[:, 1]
                            self.reference_mu_data = {
                                'v': 10**log_v,
                                'mu': mu,
                                'log_v': log_v,
                                'show': True
                            }
                            print(f"[참조 데이터] mu_visc 업데이트: {len(mu)} points")

                # Parse A/A0 data
                area_content = area_text.get("1.0", tk.END).strip()
                if area_content:
                    area_lines = [l.strip() for l in area_content.split('\n') if l.strip() and not l.startswith('#')]
                    if area_lines:
                        area_data = []
                        for line in area_lines:
                            parts = line.split()
                            if len(parts) >= 2:
                                try:
                                    log_v = float(parts[0])
                                    area_val = float(parts[1])
                                    area_data.append((log_v, area_val))
                                except ValueError:
                                    continue
                        if area_data:
                            area_data = np.array(area_data)
                            log_v = area_data[:, 0]
                            area = area_data[:, 1]
                            self.reference_area_data = {
                                'v': 10**log_v,
                                'area': area,
                                'log_v': log_v,
                                'show': True
                            }
                            print(f"[참조 데이터] A/A0 업데이트: {len(area)} points")

                # Refresh plots if mu_visc results exist
                if hasattr(self, 'mu_visc_results') and self.mu_visc_results is not None:
                    v = self.mu_visc_results.get('v')
                    mu = self.mu_visc_results.get('mu')
                    details = self.mu_visc_results.get('details')
                    if v is not None and mu is not None and details is not None:
                        use_nonlinear = self.mu_use_fg_var.get() if hasattr(self, 'mu_use_fg_var') else False
                        self._update_mu_visc_plots(v, mu, details, use_nonlinear=use_nonlinear)

                dialog.destroy()
                self._show_status("참조 데이터가 업데이트되었습니다.", 'success')

            except Exception as e:
                import traceback
                messagebox.showerror("오류", f"데이터 파싱 실패:\n{str(e)}\n\n{traceback.format_exc()}")

        ttk.Button(btn_frame, text="적용", command=apply_data, width=15).pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="취소", command=dialog.destroy, width=15).pack(side=tk.RIGHT, padx=5)

    def _create_strain_map_tab(self, parent):
        """Create Local Strain Map visualization tab."""
        # Toolbar
        self._create_panel_toolbar(parent, buttons=[
            ("계산 및 시각화", self._calculate_strain_map, 'Accent.TButton'),
        ])

        # Main container
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Top control panel
        control_frame = ttk.LabelFrame(main_frame, text="Local Strain Map 설정", padding=5)
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        # Description
        desc_text = (
            "각 파수 q와 슬립 속도 v에서의 국소 변형률 ε(q,v)와 감소된 모듈러스를 시각화합니다.\n"
            "ω = q·v·cos(φ) 로 주파수가 결정되며, 해당 주파수에서의 모듈러스가 변형률에 따라 감소합니다."
        )
        ttk.Label(control_frame, text=desc_text, font=('Segoe UI', 17)).pack(anchor=tk.W)

        # Control row
        ctrl_row = ttk.Frame(control_frame)
        ctrl_row.pack(fill=tk.X, pady=5)

        # Number of q points
        ttk.Label(ctrl_row, text="q 분할 수:").pack(side=tk.LEFT, padx=5)
        self.strain_map_nq_var = tk.StringVar(value="32")
        ttk.Entry(ctrl_row, textvariable=self.strain_map_nq_var, width=6).pack(side=tk.LEFT)

        # Number of v points
        ttk.Label(ctrl_row, text="  v 분할 수:").pack(side=tk.LEFT, padx=5)
        self.strain_map_nv_var = tk.StringVar(value="32")
        ttk.Entry(ctrl_row, textvariable=self.strain_map_nv_var, width=6).pack(side=tk.LEFT)

        # Strain estimation method - default to rms_slope
        ttk.Label(ctrl_row, text="  변형률 추정:").pack(side=tk.LEFT, padx=5)
        self.strain_map_method_var = tk.StringVar(value="rms_slope")
        method_combo = ttk.Combobox(
            ctrl_row, textvariable=self.strain_map_method_var,
            values=["rms_slope", "persson", "simple", "fixed"],
            width=10, state="readonly", font=self.FONTS['body']
        )
        method_combo.pack(side=tk.LEFT)

        # Fixed strain value
        ttk.Label(ctrl_row, text="  고정 ε (%):").pack(side=tk.LEFT, padx=5)
        self.strain_map_fixed_var = tk.StringVar(value="1.0")
        ttk.Entry(ctrl_row, textvariable=self.strain_map_fixed_var, width=6).pack(side=tk.LEFT)

        # Calculate button
        ttk.Button(
            ctrl_row, text="계산 및 시각화",
            command=self._calculate_strain_map
        ).pack(side=tk.LEFT, padx=20)

        # CSV Export button
        ttk.Button(
            ctrl_row, text="CSV 내보내기",
            command=self._export_strain_map_csv
        ).pack(side=tk.LEFT, padx=5)

        # Progress bar
        self.strain_map_progress = ttk.Progressbar(control_frame, mode='determinate')
        self.strain_map_progress.pack(fill=tk.X, pady=3)

        # Plot area - 3x4 grid for 10 heatmaps + f,g graph
        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.fig_strain_map = Figure(figsize=(18, 13), dpi=100)

        # 3x4 subplots layout:
        # Row 1: Local Strain | E' Storage (linear) | E'' Loss (linear) | E''*g Loss (nonlinear)
        # Row 2: f,g Factors  | E'*f Storage (nl)   | G(q) (lin)        | G(q) (nl)
        # Row 3: (empty)      | (empty)              | A/A0 (linear)     | A/A0 (nonlinear)
        self.ax_strain_contour = self.fig_strain_map.add_subplot(3, 4, 1)
        self.ax_E_storage = self.fig_strain_map.add_subplot(3, 4, 2)
        self.ax_E_loss_linear = self.fig_strain_map.add_subplot(3, 4, 3)
        self.ax_E_loss_nonlinear = self.fig_strain_map.add_subplot(3, 4, 4)
        self.ax_fg_factors = self.fig_strain_map.add_subplot(3, 4, 5)
        self.ax_E_storage_nonlinear = self.fig_strain_map.add_subplot(3, 4, 6)
        self.ax_G_integrand_linear = self.fig_strain_map.add_subplot(3, 4, 7)
        self.ax_G_integrand = self.fig_strain_map.add_subplot(3, 4, 8)
        self.ax_contact_linear = self.fig_strain_map.add_subplot(3, 4, 11)
        self.ax_contact_nonlinear = self.fig_strain_map.add_subplot(3, 4, 12)

        self.canvas_strain_map = FigureCanvasTkAgg(self.fig_strain_map, plot_frame)
        self.canvas_strain_map.draw()
        self.canvas_strain_map.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas_strain_map, plot_frame)
        toolbar.update()

        # Initialize plots
        self._init_strain_map_plots()

    def _init_strain_map_plots(self):
        """Initialize strain map plots with placeholder data."""
        heatmap_axes = [
            (self.ax_strain_contour, 'Local Strain [%]'),
            (self.ax_E_storage, "E' Storage [MPa]"),
            (self.ax_E_loss_linear, "E'' Loss [MPa]"),
            (self.ax_E_loss_nonlinear, "E''×g [MPa]"),
            (self.ax_E_storage_nonlinear, "E'×f [MPa]"),
            (self.ax_G_integrand_linear, "G(q) (lin)"),
            (self.ax_G_integrand, "G(q) (nl)"),
            (self.ax_contact_linear, "A/A0 (linear)"),
            (self.ax_contact_nonlinear, "A/A0 (nonlinear)")
        ]
        for ax, title in heatmap_axes:
            ax.set_title(title, fontweight='bold', fontsize=14)
            ax.set_xlabel('log10(v) [m/s]', fontsize=12)
            ax.set_ylabel('log10(q) [1/m]', fontsize=12)
            ax.text(0.5, 0.5, 'No data',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=14, color='gray')

        # f,g factor graph placeholder
        self.ax_fg_factors.set_title('f(ε), g(ε) Factors', fontweight='bold', fontsize=14)
        self.ax_fg_factors.set_xlabel('Strain', fontsize=12)
        self.ax_fg_factors.set_ylabel('Factor', fontsize=12)
        self.ax_fg_factors.text(0.5, 0.5, 'No data',
               ha='center', va='center', transform=self.ax_fg_factors.transAxes,
               fontsize=14, color='gray')

        self.fig_strain_map.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.08, hspace=0.40, wspace=0.32)
        self.canvas_strain_map.draw()

    def _calculate_strain_map(self):
        """Calculate and visualize local strain map."""
        # Check if data is from Tab 0 and Tab 1
        tab0_ready = getattr(self, 'tab0_finalized', False)
        tab1_ready = getattr(self, 'tab1_finalized', False)

        if not tab0_ready or self.psd_model is None:
            self._show_status("PSD 데이터가 설정되지 않았습니다!\n\n"
                "Tab 0 (PSD 생성)에서 PSD를 확정하세요.", 'warning')
            return

        if not tab1_ready or self.material is None:
            self._show_status("마스터 커브 데이터가 설정되지 않았습니다!\n\n"
                "Tab 1 (마스터 커브 생성)에서 마스터 커브를 확정하세요.", 'warning')
            return

        if not self.results or '2d_results' not in self.results:
            self._show_status("먼저 G(q,v) 계산을 실행하세요 (탭 2).", 'warning')
            return

        try:
            self.status_var.set("Local Strain Map 계산 중...")
            self.root.update()

            # Get parameters
            n_q = int(self.strain_map_nq_var.get())
            n_v = int(self.strain_map_nv_var.get())
            method = self.strain_map_method_var.get()
            fixed_strain = float(self.strain_map_fixed_var.get()) / 100.0

            sigma_0 = float(self.sigma_0_var.get()) * 1e6  # MPa to Pa
            temperature = float(self.temperature_var.get())
            poisson = float(self.poisson_var.get())

            # Get q and v ranges from G(q,v) results
            results_2d = self.results['2d_results']
            q_orig = results_2d['q']
            v_orig = results_2d['v']
            G_matrix_orig = results_2d['G_matrix']  # Tab 3에서 계산된 G(q,v)

            # Tab 2 설정의 유효 파수 범위 사용 (results 범위와 교집합)
            q_min_setting = float(self.q_min_var.get())
            q_max_setting = float(self.q_max_var.get())
            q_min = max(q_min_setting, q_orig.min())
            q_max = min(q_max_setting, q_orig.max())
            v_min = v_orig.min()
            v_max = v_orig.max()

            # Create q and v arrays for visualization
            q_array = np.logspace(np.log10(q_min), np.log10(q_max), n_q)
            v_array = np.logspace(np.log10(v_min), np.log10(v_max), n_v)

            # Create 2D interpolator for G(q, v) from Tab 3 results
            from scipy.interpolate import RegularGridInterpolator
            # Use log scale for better interpolation
            log_q_orig = np.log10(q_orig)
            log_v_orig = np.log10(v_orig)
            log_G_orig = np.log10(np.maximum(G_matrix_orig, 1e-30))  # Avoid log(0)

            G_interp = RegularGridInterpolator(
                (log_q_orig, log_v_orig), log_G_orig,
                method='linear', bounds_error=False, fill_value=None
            )


            # Get PSD values
            C_q = self.psd_model(q_array)

            # Prepare RMS slope interpolator if available
            rms_strain_interp = None
            if method == 'rms_slope' and hasattr(self, 'rms_slope_profiles') and self.rms_slope_profiles is not None:
                from scipy.interpolate import interp1d
                rms_q = self.rms_slope_profiles['q']
                rms_strain = self.rms_slope_profiles['strain']
                log_q = np.log10(rms_q)
                log_strain = np.log10(np.maximum(rms_strain, 1e-10))
                rms_strain_interp = interp1d(log_q, log_strain, kind='linear',
                                             bounds_error=False, fill_value='extrapolate')

            # Initialize matrices
            strain_matrix = np.zeros((n_q, n_v))
            E_storage_matrix = np.zeros((n_q, n_v))  # E' storage modulus
            E_loss_linear = np.zeros((n_q, n_v))
            E_loss_nonlinear = np.zeros((n_q, n_v))
            E_storage_nonlinear = np.zeros((n_q, n_v))  # E'·f(ε)

            # NEW: G integrand and contact area matrices
            G_integrand_linear = np.zeros((n_q, n_v))
            G_integrand_nonlinear = np.zeros((n_q, n_v))
            contact_linear = np.zeros((n_q, n_v))
            contact_nonlinear = np.zeros((n_q, n_v))

            # Calculate for each (q, v) pair
            total = n_q * n_v
            count = 0

            for i, q in enumerate(q_array):
                for j, v in enumerate(v_array):
                    # Characteristic frequency: ω = q * v (simplified, ignoring cos(φ))
                    omega = q * v

                    # Get linear E' and E'' at this frequency
                    E_loss = self.material.get_loss_modulus(np.array([omega]), temperature=temperature)[0]
                    E_storage = self.material.get_storage_modulus(np.array([omega]), temperature=temperature)[0]
                    E_loss_linear[i, j] = E_loss
                    E_storage_matrix[i, j] = E_storage

                    # Estimate local strain
                    if method == 'fixed':
                        strain = fixed_strain
                    elif method == 'rms_slope' and rms_strain_interp is not None:
                        try:
                            strain = 10 ** rms_strain_interp(np.log10(q))
                            # Fix NaN issue
                            if not np.isfinite(strain):
                                strain = fixed_strain
                        except:
                            strain = fixed_strain
                    elif method == 'persson':
                        # Persson approach: ε ~ sqrt(C(q) * q^4) * sigma_0 / E'
                        C_val = self.psd_model(q)
                        strain = np.sqrt(max(C_val * q**4, 1e-20)) * sigma_0 / max(E_storage, 1e3)
                    elif method == 'simple':
                        # Simple: ε ~ sigma_0 / E'
                        strain = sigma_0 / max(E_storage, 1e3)
                    else:
                        strain = fixed_strain

                    # Ensure finite value
                    if not np.isfinite(strain):
                        strain = fixed_strain
                    strain = np.clip(strain, 0.0, 1.0)
                    strain_matrix[i, j] = strain

                    # Apply f(ε), g(ε) correction for nonlinear E', E''
                    if self.g_interpolator is not None:
                        g_val = self.g_interpolator(strain)
                        g_val = np.clip(g_val, 0.01, None)  # g can exceed 1.0
                        E_loss_nonlinear[i, j] = E_loss * g_val
                    else:
                        E_loss_nonlinear[i, j] = E_loss

                    if self.f_interpolator is not None:
                        f_val = self.f_interpolator(strain)
                        f_val = np.clip(f_val, 0.0, 1.0)
                        E_storage_nonlinear[i, j] = E_storage * f_val
                    else:
                        f_val = 1.0
                        E_storage_nonlinear[i, j] = E_storage

                    # Get G(q) value from Tab 3 results (interpolated cumulative G)
                    # G_interp uses log scale
                    try:
                        log_G_val = G_interp((np.log10(q), np.log10(v)))
                        G_linear = 10 ** log_G_val if np.isfinite(log_G_val) else 1e-10
                    except:
                        G_linear = 1e-10

                    G_integrand_linear[i, j] = G_linear

                    # Calculate nonlinear G(q) with f(ε) and g(ε) correction
                    # G ~ |E_eff|² = (E'f)² + (E''g)²
                    # Correction ratio: ((E'f)² + (E''g)²) / (E'² + E''²)
                    E_sq = E_storage**2 + E_loss**2
                    if E_sq > 0 and (self.f_interpolator is not None or self.g_interpolator is not None):
                        g_val = 1.0
                        if self.g_interpolator is not None:
                            g_val = self.g_interpolator(strain)
                            g_val = np.clip(g_val, 0.01, None)
                        E_eff_sq = (E_storage * f_val)**2 + (E_loss * g_val)**2
                        nl_ratio = E_eff_sq / E_sq
                        G_nonlinear = G_linear * nl_ratio
                    else:
                        G_nonlinear = G_linear

                    G_integrand_nonlinear[i, j] = G_nonlinear

                    # Calculate contact area ratio A/A0 = P(q) = erf(1/(2*sqrt(G)))
                    from scipy.special import erf
                    G_linear_safe = max(G_linear, 1e-20)
                    G_nonlinear_safe = max(G_nonlinear, 1e-20)

                    arg_linear = 1.0 / (2.0 * np.sqrt(G_linear_safe))
                    arg_nonlinear = 1.0 / (2.0 * np.sqrt(G_nonlinear_safe))
                    contact_linear[i, j] = erf(min(arg_linear, 10.0))
                    contact_nonlinear[i, j] = erf(min(arg_nonlinear, 10.0))

                    count += 1
                    if count % (total // 20 + 1) == 0:
                        self.strain_map_progress['value'] = (count / total) * 100
                        self.root.update()

            # Store results
            self.strain_map_results = {
                'q': q_array,
                'v': v_array,
                'strain': strain_matrix,
                'C_q': C_q,
                'E_storage': E_storage_matrix,
                'E_storage_nonlinear': E_storage_nonlinear,
                'E_loss_linear': E_loss_linear,
                'E_loss_nonlinear': E_loss_nonlinear,
                'G_integrand_linear': G_integrand_linear,
                'G_integrand_nonlinear': G_integrand_nonlinear,
                'contact_linear': contact_linear,
                'contact_nonlinear': contact_nonlinear
            }

            # Update plots
            self._update_strain_map_plots()

            self.strain_map_progress['value'] = 100
            self.status_var.set("Local Strain Map 계산 완료")

        except Exception as e:
            import traceback
            messagebox.showerror("오류", f"계산 실패:\n{str(e)}\n\n{traceback.format_exc()}")
            self.status_var.set("오류 발생")

    def _update_strain_map_plots(self):
        """Update strain map heatmap plots (3x4 grid, 10 plots + f,g graph).

        Layout:
        Row 1: Local Strain | E' Storage | E'' Loss | E''×g
        Row 2: f,g Factors  | E'×f       | G(q) lin | G(q) nl
        Row 3: (empty)      | (empty)    | A/A0 lin | A/A0 nl

        - E' + E'' LogNorm 공유, E'×f + E''×g LogNorm 공유
        - G(q) lin/nl 공유 범위, Y축 유효영역 크롭
        """
        if not hasattr(self, 'strain_map_results') or self.strain_map_results is None:
            return

        q = self.strain_map_results['q']
        v = self.strain_map_results['v']
        strain = self.strain_map_results['strain']
        E_storage = self.strain_map_results['E_storage']
        E_storage_nl = self.strain_map_results['E_storage_nonlinear']
        E_loss_lin = self.strain_map_results.get('E_loss_linear')
        E_loss_nl = self.strain_map_results['E_loss_nonlinear']
        G_int_lin = self.strain_map_results.get('G_integrand_linear')
        G_int_nl = self.strain_map_results.get('G_integrand_nonlinear')
        contact_lin = self.strain_map_results.get('contact_linear')
        contact_nl = self.strain_map_results.get('contact_nonlinear')

        # Create meshgrid for pcolormesh
        log_v = np.log10(v)
        log_q = np.log10(q)
        V, Q = np.meshgrid(log_v, log_q)

        # Remove existing colorbars
        if hasattr(self, '_strain_map_colorbars'):
            for cbar in self._strain_map_colorbars:
                try:
                    cbar.remove()
                except:
                    pass
        self._strain_map_colorbars = []

        # Clear all axes
        all_axes = [self.ax_strain_contour, self.ax_E_storage,
                    self.ax_E_loss_linear, self.ax_E_loss_nonlinear,
                    self.ax_fg_factors, self.ax_E_storage_nonlinear,
                    self.ax_G_integrand_linear, self.ax_G_integrand,
                    self.ax_contact_linear, self.ax_contact_nonlinear]
        for ax in all_axes:
            ax.clear()
            ax.set_facecolor('#e0e0e0')  # 회색 배경 (마스킹된 영역)

        # ===== 유효 영역 마스크: A/A0 < 0.999 =====
        # A/A0 ≈ 1 영역은 접촉 감소가 시작되지 않은 trivial 영역
        AA0_THRESHOLD = 0.999
        if contact_nl is not None:
            valid_mask = contact_nl < AA0_THRESHOLD
        elif contact_lin is not None:
            valid_mask = contact_lin < AA0_THRESHOLD
        else:
            valid_mask = np.ones_like(strain, dtype=bool)

        # 유효 영역이 없으면 전체 표시 (fallback)
        if not np.any(valid_mask):
            valid_mask = np.ones_like(strain, dtype=bool)

        invalid_mask = ~valid_mask

        # ===== 색상 맵 =====
        strain_cmap = 'YlOrRd'
        E_storage_cmap = 'Blues'
        E_loss_cmap = 'Reds'
        contact_cmap = 'Greens'
        g_cmap = 'YlOrBr'

        # ===== E', E'' → MPa 단위 (linear) =====
        E_s_MPa = E_storage / 1e6
        E_snl_MPa = E_storage_nl / 1e6
        E_ll_MPa = E_loss_lin / 1e6 if E_loss_lin is not None else np.zeros_like(E_storage) / 1e6
        E_lnl_MPa = E_loss_nl / 1e6

        # Storage 쌍: E' + E'×f 공유 LogNorm (f 효과 비교)
        stor_all = np.concatenate([E_s_MPa.ravel(), E_snl_MPa.ravel()])
        stor_pos = stor_all[stor_all > 0]
        stor_vmin = np.percentile(stor_pos, 1) if len(stor_pos) > 0 else 1e-3
        stor_vmax = np.percentile(stor_pos, 99) if len(stor_pos) > 0 else 1.0
        stor_norm = LogNorm(vmin=stor_vmin, vmax=stor_vmax)

        # Loss 쌍: E'' + E''×g 공유 LogNorm (g 효과 비교)
        loss_all = np.concatenate([E_ll_MPa.ravel(), E_lnl_MPa.ravel()])
        loss_pos = loss_all[loss_all > 0]
        loss_vmin = np.percentile(loss_pos, 1) if len(loss_pos) > 0 else 1e-3
        loss_vmax = np.percentile(loss_pos, 99) if len(loss_pos) > 0 else 1.0
        loss_norm = LogNorm(vmin=loss_vmin, vmax=loss_vmax)

        # v=1 m/s 인덱스
        v_1ms_idx = int(np.argmin(np.abs(v - 1.0)))

        # 마스크 적용 헬퍼
        def _masked(data):
            return np.ma.array(data, mask=invalid_mask)

        # ===== Row 1 =====
        # Plot 1: Local Strain [%] — 마스크 미적용 (strain은 A/A0와 무관)
        strain_pct = np.nan_to_num(strain, nan=0.0) * 100
        im1 = self.ax_strain_contour.pcolormesh(V, Q, strain_pct, cmap=strain_cmap, shading='auto')
        self.ax_strain_contour.set_facecolor('white')
        self.ax_strain_contour.set_title('Local Strain [%]', fontweight='bold', fontsize=14)
        self.ax_strain_contour.set_xlabel('log10(v)', fontsize=12)
        self.ax_strain_contour.set_ylabel('log10(q)', fontsize=12)
        cbar1 = self.fig_strain_map.colorbar(im1, ax=self.ax_strain_contour)
        cbar1.set_label('%', fontsize=11)
        self._strain_map_colorbars.append(cbar1)
        try:
            cs = self.ax_strain_contour.contour(V, Q, strain_pct,
                                                 levels=[1, 5, 10], colors='k', linewidths=0.5)
            self.ax_strain_contour.clabel(cs, inline=True, fontsize=11, fmt='%.0f%%')
        except:
            pass
        strain_flat = strain.ravel()
        if len(strain_flat) > 0:
            self.ax_strain_contour.text(0.02, 0.98,
                f'Mean:{np.mean(strain_flat)*100:.1f}%\nMax:{np.max(strain_flat)*100:.1f}%',
                transform=self.ax_strain_contour.transAxes, fontsize=11, va='top',
                bbox=dict(boxstyle='round', fc='white', alpha=0.8))

        # Plot 2: E' Storage [MPa] — E' + E'×f 공유 LogNorm
        im2 = self.ax_E_storage.pcolormesh(V, Q, E_s_MPa, cmap=E_storage_cmap, shading='auto',
                                            norm=stor_norm)
        self.ax_E_storage.set_facecolor('white')
        self.ax_E_storage.set_title("E' Storage [MPa]", fontweight='bold', fontsize=14)
        self.ax_E_storage.set_xlabel('log10(v)', fontsize=12)
        self.ax_E_storage.set_ylabel('log10(q)', fontsize=12)
        cbar2 = self.fig_strain_map.colorbar(im2, ax=self.ax_E_storage)
        cbar2.set_label('MPa', fontsize=11)
        self._strain_map_colorbars.append(cbar2)
        E_s_at1 = E_s_MPa[:, v_1ms_idx]
        self.ax_E_storage.text(0.02, 0.98,
            f"v=1: {E_s_at1.min():.1f}~{E_s_at1.max():.1f} MPa",
            transform=self.ax_E_storage.transAxes, fontsize=11, va='top',
            bbox=dict(boxstyle='round', fc='white', alpha=0.8))

        # Plot 3: E'' Loss [MPa] — E'' + E''×g 공유 LogNorm
        if E_loss_lin is not None:
            im3 = self.ax_E_loss_linear.pcolormesh(V, Q, E_ll_MPa, cmap=E_loss_cmap, shading='auto',
                                                    norm=loss_norm)
            self.ax_E_loss_linear.set_facecolor('white')
            self.ax_E_loss_linear.set_title("E'' Loss [MPa]", fontweight='bold', fontsize=14)
            self.ax_E_loss_linear.set_xlabel('log10(v)', fontsize=12)
            self.ax_E_loss_linear.set_ylabel('log10(q)', fontsize=12)
            cbar3 = self.fig_strain_map.colorbar(im3, ax=self.ax_E_loss_linear)
            cbar3.set_label('MPa', fontsize=11)
            self._strain_map_colorbars.append(cbar3)
            E_ll_at1 = E_ll_MPa[:, v_1ms_idx]
            self.ax_E_loss_linear.text(0.02, 0.98,
                f"v=1: {E_ll_at1.min():.2f}~{E_ll_at1.max():.1f} MPa",
                transform=self.ax_E_loss_linear.transAxes, fontsize=11, va='top',
                bbox=dict(boxstyle='round', fc='white', alpha=0.8))

        # Plot 4: E''×g [MPa] — E'' + E''×g 공유 LogNorm
        im4 = self.ax_E_loss_nonlinear.pcolormesh(V, Q, E_lnl_MPa, cmap=E_loss_cmap, shading='auto',
                                                    norm=loss_norm)
        self.ax_E_loss_nonlinear.set_facecolor('white')
        self.ax_E_loss_nonlinear.set_title("E''×g [MPa]", fontweight='bold', fontsize=14)
        self.ax_E_loss_nonlinear.set_xlabel('log10(v)', fontsize=12)
        self.ax_E_loss_nonlinear.set_ylabel('log10(q)', fontsize=12)
        cbar4 = self.fig_strain_map.colorbar(im4, ax=self.ax_E_loss_nonlinear)
        cbar4.set_label('MPa', fontsize=11)
        self._strain_map_colorbars.append(cbar4)
        E_lnl_at1 = E_lnl_MPa[:, v_1ms_idx]
        self.ax_E_loss_nonlinear.text(0.02, 0.98,
            f"v=1: {E_lnl_at1.min():.2f}~{E_lnl_at1.max():.1f} MPa",
            transform=self.ax_E_loss_nonlinear.transAxes, fontsize=11, va='top',
            bbox=dict(boxstyle='round', fc='white', alpha=0.8))

        # ===== Row 2 =====
        # Plot 5: E'×f [MPa] — E' + E'×f 공유 LogNorm
        im5 = self.ax_E_storage_nonlinear.pcolormesh(V, Q, E_snl_MPa, cmap=E_storage_cmap, shading='auto',
                                                      norm=stor_norm)
        self.ax_E_storage_nonlinear.set_facecolor('white')
        self.ax_E_storage_nonlinear.set_title("E'×f [MPa]", fontweight='bold', fontsize=14)
        self.ax_E_storage_nonlinear.set_xlabel('log10(v)', fontsize=12)
        self.ax_E_storage_nonlinear.set_ylabel('log10(q)', fontsize=12)
        cbar5 = self.fig_strain_map.colorbar(im5, ax=self.ax_E_storage_nonlinear)
        cbar5.set_label('MPa', fontsize=11)
        self._strain_map_colorbars.append(cbar5)
        E_snl_at1 = E_snl_MPa[:, v_1ms_idx]
        self.ax_E_storage_nonlinear.text(0.02, 0.98,
            f"v=1: {E_snl_at1.min():.1f}~{E_snl_at1.max():.1f} MPa",
            transform=self.ax_E_storage_nonlinear.transAxes, fontsize=11, va='top',
            bbox=dict(boxstyle='round', fc='white', alpha=0.8))

        # ===== Y축 크롭: G integrand가 유의미해지는 q부터만 표시 =====
        # 제일 낮은 슬립속도(v 최소, 열 인덱스 0)에서 G값이 trivial하지 않은 행을 찾음
        q_crop_min = log_q[0]
        q_crop_max = log_q[-1]
        if G_int_nl is not None:
            # 가장 낮은 v 열에서 G_integrand가 의미 있는(> 1e-20) 행 찾기
            g_col_lowest_v = G_int_nl[:, 0]  # v 최소 열
            meaningful_rows = np.where(g_col_lowest_v > 1e-20)[0]
            if len(meaningful_rows) > 0:
                # 의미있는 첫 행부터 끝까지
                start_idx = max(meaningful_rows[0] - 1, 0)
                q_crop_min = log_q[start_idx]
                q_crop_max = log_q[-1]

        # G integrand, A/A0 에 Y축 크롭 적용할 axes 목록
        crop_axes = []

        # Plot 6: f(ε), g(ε) Factors graph
        self.ax_fg_factors.set_facecolor('white')
        if self.f_interpolator is not None:
            s_plot = np.linspace(0, 0.5, 200)
            f_vals = np.array([self.f_interpolator(s) for s in s_plot])
            self.ax_fg_factors.plot(s_plot * 100, f_vals, 'b-', linewidth=2, label='f(ε)')
        if self.g_interpolator is not None:
            s_plot = np.linspace(0, 0.5, 200)
            g_vals = np.array([self.g_interpolator(s) for s in s_plot])
            self.ax_fg_factors.plot(s_plot * 100, g_vals, 'r-', linewidth=2, label='g(ε)')
        self.ax_fg_factors.set_title('f(ε), g(ε) Factors', fontweight='bold', fontsize=14)
        self.ax_fg_factors.set_xlabel('Strain [%]', fontsize=12)
        self.ax_fg_factors.set_ylabel('Factor', fontsize=12)
        self.ax_fg_factors.set_ylim(0, 1.1)
        self.ax_fg_factors.legend(fontsize=12, loc='lower left')
        self.ax_fg_factors.grid(True, alpha=0.3)

        # G integrand 공유 범위 계산 (linear + nonlinear)
        g_lin_log = np.log10(np.maximum(G_int_lin, 1e-30)) if G_int_lin is not None else None
        g_nl_log = np.log10(np.maximum(G_int_nl, 1e-30)) if G_int_nl is not None else None
        g_vmin, g_vmax = -10, 0
        g_arrays = []
        if g_lin_log is not None:
            g_arrays.append(g_lin_log[valid_mask])
        if g_nl_log is not None:
            g_arrays.append(g_nl_log[valid_mask])
        if g_arrays:
            g_all_valid = np.concatenate(g_arrays)
            if len(g_all_valid) > 0:
                g_vmax = np.max(g_all_valid)
                g_vmin = max(np.percentile(g_all_valid, 5), g_vmax - 15)

        # Plot 7: G Integrand (linear) [log]
        if G_int_lin is not None:
            log_G_lin_masked = _masked(g_lin_log)
            im7a = self.ax_G_integrand_linear.pcolormesh(V, Q, log_G_lin_masked, cmap=g_cmap, shading='auto',
                                                          vmin=g_vmin, vmax=g_vmax)
            self.ax_G_integrand_linear.set_title('G(q) (lin) [log]', fontweight='bold', fontsize=14)
            self.ax_G_integrand_linear.set_xlabel('log10(v)', fontsize=12)
            self.ax_G_integrand_linear.set_ylabel('log10(q)', fontsize=12)
            cbar7a = self.fig_strain_map.colorbar(im7a, ax=self.ax_G_integrand_linear)
            cbar7a.set_label('log10(G)', fontsize=11)
            self._strain_map_colorbars.append(cbar7a)
            crop_axes.append(self.ax_G_integrand_linear)

        # Plot 7b: G Integrand (nonlinear) [log]
        if G_int_nl is not None:
            log_G_nl_masked = _masked(g_nl_log)
            im7b = self.ax_G_integrand.pcolormesh(V, Q, log_G_nl_masked, cmap=g_cmap, shading='auto',
                                                   vmin=g_vmin, vmax=g_vmax)
            self.ax_G_integrand.set_title('G(q) (nl) [log]', fontweight='bold', fontsize=14)
            self.ax_G_integrand.set_xlabel('log10(v)', fontsize=12)
            self.ax_G_integrand.set_ylabel('log10(q)', fontsize=12)
            cbar7b = self.fig_strain_map.colorbar(im7b, ax=self.ax_G_integrand)
            cbar7b.set_label('log10(G)', fontsize=11)
            self._strain_map_colorbars.append(cbar7b)
            crop_axes.append(self.ax_G_integrand)

        # Plot 9: A/A0 (linear) — 유효 영역만
        if contact_lin is not None:
            c_lin_masked = _masked(contact_lin)
            c_valid = contact_lin[valid_mask]
            if len(c_valid) > 0:
                c_vmin = max(np.min(c_valid) - 0.02, 0)
                c_vmax = min(np.max(c_valid) + 0.02, 1)
            else:
                c_vmin, c_vmax = 0, 1
            im9 = self.ax_contact_linear.pcolormesh(V, Q, c_lin_masked, cmap=contact_cmap, shading='auto',
                                                     vmin=c_vmin, vmax=c_vmax)
            self.ax_contact_linear.set_title('A/A0 (linear)', fontweight='bold', fontsize=14)
            self.ax_contact_linear.set_xlabel('log10(v)', fontsize=12)
            self.ax_contact_linear.set_ylabel('log10(q)', fontsize=12)
            cbar9 = self.fig_strain_map.colorbar(im9, ax=self.ax_contact_linear)
            cbar9.set_label('A/A0', fontsize=11)
            self._strain_map_colorbars.append(cbar9)
            crop_axes.append(self.ax_contact_linear)
            P_lin_at1 = contact_lin[:, v_1ms_idx]
            P_lin_valid = P_lin_at1[P_lin_at1 < AA0_THRESHOLD]
            if len(P_lin_valid) > 0:
                self.ax_contact_linear.text(0.02, 0.98,
                    f'v=1: {P_lin_valid.min():.3f}~{P_lin_valid.max():.3f}',
                    transform=self.ax_contact_linear.transAxes, fontsize=11, va='top',
                    bbox=dict(boxstyle='round', fc='white', alpha=0.8))

        # Plot 10: A/A0 (nonlinear) — 유효 영역만
        if contact_nl is not None:
            c_nl_masked = _masked(contact_nl)
            c_nl_valid = contact_nl[valid_mask]
            if len(c_nl_valid) > 0:
                c_nl_vmin = max(np.min(c_nl_valid) - 0.02, 0)
                c_nl_vmax = min(np.max(c_nl_valid) + 0.02, 1)
            else:
                c_nl_vmin, c_nl_vmax = 0, 1
            im10 = self.ax_contact_nonlinear.pcolormesh(V, Q, c_nl_masked, cmap=contact_cmap, shading='auto',
                                                         vmin=c_nl_vmin, vmax=c_nl_vmax)
            self.ax_contact_nonlinear.set_title('A/A0 (nonlinear)', fontweight='bold', fontsize=14)
            self.ax_contact_nonlinear.set_xlabel('log10(v)', fontsize=12)
            self.ax_contact_nonlinear.set_ylabel('log10(q)', fontsize=12)
            cbar10 = self.fig_strain_map.colorbar(im10, ax=self.ax_contact_nonlinear)
            cbar10.set_label('A/A0', fontsize=11)
            self._strain_map_colorbars.append(cbar10)
            crop_axes.append(self.ax_contact_nonlinear)
            P_nl_at1 = contact_nl[:, v_1ms_idx]
            P_nl_valid = P_nl_at1[P_nl_at1 < AA0_THRESHOLD]
            if len(P_nl_valid) > 0:
                self.ax_contact_nonlinear.text(0.02, 0.98,
                    f'v=1: {P_nl_valid.min():.3f}~{P_nl_valid.max():.3f}',
                    transform=self.ax_contact_nonlinear.transAxes, fontsize=11, va='top',
                    bbox=dict(boxstyle='round', fc='white', alpha=0.8))

        # ===== Y축 크롭 적용: G integrand, A/A0 — 유효 데이터 영역만 =====
        for ax in crop_axes:
            ax.set_ylim(q_crop_min, q_crop_max)
            ax.set_facecolor('white')  # 크롭 후 회색 배경 제거

        self.fig_strain_map.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.08, hspace=0.35, wspace=0.30)
        self.canvas_strain_map.draw()

    def _export_strain_map_csv(self):
        """Export Local Strain Map data to CSV files with selection dialog."""
        if not hasattr(self, 'strain_map_results') or self.strain_map_results is None:
            self._show_status("먼저 계산을 실행하세요.", 'warning')
            return

        # Create dialog for selecting data to export
        dialog = tk.Toplevel(self.root)
        dialog.title("CSV 내보내기 - 데이터 선택")
        dialog.geometry("400x600")
        dialog.resizable(False, True)
        dialog.transient(self.root)
        dialog.grab_set()

        # Center the dialog
        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - 400) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - 600) // 2
        dialog.geometry(f"+{x}+{y}")

        # Description
        desc_frame = ttk.Frame(dialog, padding=10)
        desc_frame.pack(fill=tk.X)
        ttk.Label(desc_frame, text="내보낼 데이터를 선택하세요.\n각 데이터는 별도의 CSV 파일로 저장됩니다.",
                  font=('Segoe UI', 17)).pack(anchor=tk.W)

        # Export button frame - pack at bottom FIRST so it's always visible
        export_frame = ttk.Frame(dialog, padding=10)
        export_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # Select all / Deselect all buttons - pack at bottom before checkboxes
        btn_frame = ttk.Frame(dialog, padding=10)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # Checkbox frame - fills remaining space
        check_frame = ttk.LabelFrame(dialog, text="데이터 선택", padding=10)
        check_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Data options with display names and keys
        data_options = [
            ("Local Strain [%]", "strain", True),
            ("E' Storage [Pa]", "E_storage", True),
            ("E'' Loss [Pa] (linear)", "E_loss_linear", True),
            ("E''×g Loss [Pa]", "E_loss_nonlinear", True),
            ("E'×f Storage [Pa]", "E_storage_nonlinear", True),
            ("G(q) (linear)", "G_integrand_linear", True),
            ("G(q) (nonlinear)", "G_integrand_nonlinear", True),
            ("A/A0 Contact (linear)", "contact_linear", True),
            ("A/A0 Contact (nonlinear)", "contact_nonlinear", True),
        ]

        # Create checkbox variables
        check_vars = {}
        for display_name, key, default in data_options:
            var = tk.BooleanVar(value=default)
            check_vars[key] = var
            cb = ttk.Checkbutton(check_frame, text=display_name, variable=var)
            cb.pack(anchor=tk.W, pady=2)

        def select_all():
            for var in check_vars.values():
                var.set(True)

        def deselect_all():
            for var in check_vars.values():
                var.set(False)

        ttk.Button(btn_frame, text="전체 선택", command=select_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="전체 해제", command=deselect_all).pack(side=tk.LEFT, padx=5)

        def do_export():
            # Check if any data is selected
            selected = [key for key, var in check_vars.items() if var.get()]
            if not selected:
                self._show_status("내보낼 데이터를 선택하세요.", 'warning')
                return

            # Ask for save directory
            from tkinter import filedialog
            save_dir = filedialog.askdirectory(
                title="CSV 파일 저장 폴더 선택",
                parent=dialog
            )
            if not save_dir:
                return

            try:
                q = self.strain_map_results['q']
                v = self.strain_map_results['v']

                exported_files = []

                for key in selected:
                    data = self.strain_map_results.get(key)
                    if data is None:
                        continue

                    # Special handling for strain (convert to %)
                    if key == "strain":
                        data = data * 100

                    # Create filename with MC prefix
                    mc_prefix = self._get_mc_prefix()
                    filename = f"{mc_prefix}_strain_map_{key}.csv" if mc_prefix else f"strain_map_{key}.csv"
                    filepath = os.path.join(save_dir, filename)

                    # Build CSV content
                    # First row: header with v values
                    # First column: q values
                    # Format: rows = q, columns = v
                    lines = []

                    # Header row: empty cell + v values
                    header = ["q \\ v"] + [f"{vi:.6e}" for vi in v]
                    lines.append(",".join(header))

                    # Data rows
                    for i, qi in enumerate(q):
                        row_data = [f"{qi:.6e}"] + [f"{data[i, j]:.6e}" for j in range(len(v))]
                        lines.append(",".join(row_data))

                    # Write file
                    with open(filepath, 'w', encoding='utf-8-sig', newline='') as f:
                        f.write("\n".join(lines) + "\n")

                    exported_files.append(filename)

                dialog.destroy()
                self._show_status(f"CSV 파일 내보내기 완료:\n\n" + "\n".join(exported_files) + f"\n\n저장 위치: {save_dir}", 'success')
                self.status_var.set(f"CSV 내보내기 완료: {len(exported_files)}개 파일")

            except Exception as e:
                import traceback
                messagebox.showerror("오류", f"내보내기 실패:\n{str(e)}\n\n{traceback.format_exc()}", parent=dialog)

        ttk.Button(export_frame, text="내보내기", command=do_export).pack(side=tk.RIGHT, padx=5)
        ttk.Button(export_frame, text="취소", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)

    def _create_integrand_tab(self, parent):
        """Create Integrand visualization tab for G(q) and μ_visc analysis."""
        # Main container
        main_container = ttk.Frame(parent)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel for controls
        left_frame = ttk.Frame(main_container, width=getattr(self, '_left_panel_width', 600))
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        left_frame.pack_propagate(False)

        # Logo at bottom
        self._add_logo_to_panel(left_frame)

        # Toolbar (fixed at top, always accessible)
        self._create_panel_toolbar(left_frame, buttons=[
            ("피적분함수 계산", self._calculate_integrand_visualization, 'Accent.TButton'),
        ])

        # ============== Left Panel: Controls ==============

        # 1. Description
        desc_frame = ttk.LabelFrame(left_frame, text="설명", padding=5)
        desc_frame.pack(fill=tk.X, pady=2, padx=3)

        desc_text = (
            "G(q) 및 μ_visc 계산의 피적분함수를\n"
            "시각화합니다.\n\n"
            "1. G 각도 적분: |E(qv cosφ)|² vs φ\n"
            "2. G(q) 피적분: q³C(q)×(각도적분) vs q\n"
            "3. μ_visc 각도 적분 vs 속도"
        )
        ttk.Label(desc_frame, text=desc_text, font=('Segoe UI', 17), justify=tk.LEFT).pack(anchor=tk.W)

        # 2. Calculation Settings
        settings_frame = ttk.LabelFrame(left_frame, text="계산 설정", padding=5)
        settings_frame.pack(fill=tk.X, pady=2, padx=3)

        # q value selection for angle integrand
        row1 = ttk.Frame(settings_frame)
        row1.pack(fill=tk.X, pady=2)
        ttk.Label(row1, text="q 값 (1/m):", font=('Segoe UI', 17)).pack(side=tk.LEFT)
        self.integrand_q_var = tk.StringVar(value="1e4, 1e5, 1e6")
        ttk.Entry(row1, textvariable=self.integrand_q_var, width=15).pack(side=tk.RIGHT)

        ttk.Label(settings_frame, text="(쉼표로 구분, 예: 1e4, 1e5, 1e6)",
                  font=('Segoe UI', 16), foreground='#64748B').pack(anchor=tk.W)

        # Velocity selection
        row2 = ttk.Frame(settings_frame)
        row2.pack(fill=tk.X, pady=2)
        ttk.Label(row2, text="속도 v (m/s):", font=('Segoe UI', 17)).pack(side=tk.LEFT)
        self.integrand_v_var = tk.StringVar(value="0.01")
        ttk.Entry(row2, textvariable=self.integrand_v_var, width=15).pack(side=tk.RIGHT)

        # Number of angle points
        row3 = ttk.Frame(settings_frame)
        row3.pack(fill=tk.X, pady=2)
        ttk.Label(row3, text="각도 분할 수:", font=('Segoe UI', 17)).pack(side=tk.LEFT)
        self.integrand_nangle_var = tk.StringVar(value="72")
        ttk.Entry(row3, textvariable=self.integrand_nangle_var, width=8).pack(side=tk.RIGHT)

        # Apply nonlinear correction checkbox
        self.integrand_use_fg_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            settings_frame,
            text="비선형 보정 f(ε), g(ε) 적용",
            variable=self.integrand_use_fg_var
        ).pack(anchor=tk.W, pady=2)

        # Calculate button
        calc_frame = ttk.Frame(settings_frame)
        calc_frame.pack(fill=tk.X, pady=5)

        self.integrand_calc_btn = ttk.Button(
            calc_frame,
            text="피적분함수 계산",
            command=self._calculate_integrand_visualization
        )
        self.integrand_calc_btn.pack(fill=tk.X)

        # Progress bar
        self.integrand_progress_var = tk.IntVar()
        self.integrand_progress_bar = ttk.Progressbar(
            calc_frame,
            variable=self.integrand_progress_var,
            maximum=100
        )
        self.integrand_progress_bar.pack(fill=tk.X, pady=2)

        # 3. Results Summary
        results_frame = ttk.LabelFrame(left_frame, text="결과 요약", padding=5)
        results_frame.pack(fill=tk.X, pady=2, padx=3)

        self.integrand_result_text = tk.Text(results_frame, height=16, font=("Courier", 15), wrap=tk.WORD)
        self.integrand_result_text.pack(fill=tk.X)

        # 4. Frequency range info
        freq_frame = ttk.LabelFrame(left_frame, text="주파수 범위 (ω = qv cosφ)", padding=5)
        freq_frame.pack(fill=tk.X, pady=2, padx=3)

        self.freq_range_text = tk.Text(freq_frame, height=6, font=("Courier", 15), wrap=tk.WORD)
        self.freq_range_text.pack(fill=tk.X)

        # ============== Right Panel: Plots ==============

        right_panel = ttk.Frame(main_container)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        plot_frame = ttk.LabelFrame(right_panel, text="피적분함수 그래프", padding=5)
        plot_frame.pack(fill=tk.BOTH, expand=True)

        # Create figure with 2x2 subplots
        self.fig_integrand = Figure(figsize=(10, 8), dpi=100)

        # Top-left: Angle integrand |E(qv cosφ)|² vs φ
        self.ax_angle_integrand = self.fig_integrand.add_subplot(221)
        self.ax_angle_integrand.set_title(r'G 각도 피적분함수: $|E(qv\cos\phi)|^2$ vs $\phi$', fontweight='bold', fontsize=14)
        self.ax_angle_integrand.set_xlabel(r'$\phi$ (rad)', fontsize=13)
        self.ax_angle_integrand.set_ylabel(r'$|E(\omega)/((1-\nu^2)\sigma_0)|^2$', fontsize=12)
        self.ax_angle_integrand.grid(True, alpha=0.3)

        # Top-right: G(q) integrand vs q
        self.ax_q_integrand = self.fig_integrand.add_subplot(222)
        self.ax_q_integrand.set_title(r'G(q) 피적분함수: $q^3 C(q) \times$ (각도적분) vs q', fontweight='bold', fontsize=14)
        self.ax_q_integrand.set_xlabel('q (1/m)', fontsize=13)
        self.ax_q_integrand.set_ylabel(r'$q^3 C(q) \times \int|E|^2 d\phi$', fontsize=13)
        self.ax_q_integrand.set_xscale('log')
        self.ax_q_integrand.set_yscale('log')
        self.ax_q_integrand.grid(True, alpha=0.3)

        # Bottom-left: μ_visc integrand vs φ (Im[E] × cosφ)
        self.ax_mu_integrand = self.fig_integrand.add_subplot(223)
        self.ax_mu_integrand.set_title(r'$\mu_{visc}$ 각도 피적분함수: $\cos\phi \times \mathrm{Im}[E]$ vs $\phi$', fontweight='bold', fontsize=14)
        self.ax_mu_integrand.set_xlabel(r'$\phi$ (rad)', fontsize=13)
        self.ax_mu_integrand.set_ylabel(r"$\cos\phi \times E''/((1-\nu^2)\sigma_0)$", fontsize=12)
        self.ax_mu_integrand.grid(True, alpha=0.3)

        # Bottom-right: Frequency range vs velocity
        self.ax_freq_range = self.fig_integrand.add_subplot(224)
        self.ax_freq_range.set_title('속도별 주파수 스캔 범위', fontweight='bold', fontsize=14)
        self.ax_freq_range.set_xlabel('속도 v (m/s)', fontsize=13)
        self.ax_freq_range.set_ylabel('주파수 ω (rad/s)', fontsize=13)
        self.ax_freq_range.set_xscale('log')
        self.ax_freq_range.set_yscale('log')
        self.ax_freq_range.grid(True, alpha=0.3)

        self.fig_integrand.subplots_adjust(left=0.12, right=0.95, top=0.94, bottom=0.10, hspace=0.42, wspace=0.35)

        self.canvas_integrand = FigureCanvasTkAgg(self.fig_integrand, plot_frame)
        self.canvas_integrand.draw()
        self.canvas_integrand.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas_integrand, plot_frame)
        toolbar.update()

    def _calculate_integrand_visualization(self):
        """Calculate and visualize integrands for G(q) and μ_visc."""
        # Check if data is available
        if self.psd_model is None or self.material is None:
            self._show_status("먼저 PSD와 DMA 데이터를 로드하세요.", 'warning')
            return

        try:
            self.integrand_calc_btn.config(state='disabled')
            self.integrand_progress_var.set(10)
            self.root.update_idletasks()

            # Parse q values
            q_str = self.integrand_q_var.get().strip()
            q_values = [float(x.strip()) for x in q_str.split(',')]

            # Get velocity
            v = float(self.integrand_v_var.get())

            # Get number of angle points
            n_angle = int(self.integrand_nangle_var.get())

            # Get parameters
            sigma_0 = float(self.sigma_0_var.get()) * 1e6  # MPa to Pa
            poisson_ratio = float(self.poisson_var.get())
            prefactor = 1.0 / ((1 - poisson_ratio**2) * sigma_0)

            # Check for nonlinear correction
            use_fg = self.integrand_use_fg_var.get()

            # Clear previous results
            self.integrand_result_text.delete('1.0', tk.END)
            self.freq_range_text.delete('1.0', tk.END)

            # Clear axes
            self.ax_angle_integrand.clear()
            self.ax_q_integrand.clear()
            self.ax_mu_integrand.clear()
            self.ax_freq_range.clear()

            # Set up axes labels and titles again
            self.ax_angle_integrand.set_title(r'G 각도 피적분함수: $|E(qv\cos\phi)|^2$ vs $\phi$', fontweight='bold', fontsize=14)
            self.ax_angle_integrand.set_xlabel(r'$\phi$ (rad)', fontsize=13)
            self.ax_angle_integrand.set_ylabel(r'$|E(\omega)/((1-\nu^2)\sigma_0)|^2$', fontsize=12)
            self.ax_angle_integrand.grid(True, alpha=0.3)

            self.ax_q_integrand.set_title(r'G(q) 피적분함수: $q^3 C(q) \times$ (각도적분) vs q', fontweight='bold', fontsize=14)
            self.ax_q_integrand.set_xlabel('q (1/m)', fontsize=13)
            self.ax_q_integrand.set_ylabel(r'$q^3 C(q) \times \int|E|^2 d\phi$', fontsize=13)
            self.ax_q_integrand.set_xscale('log')
            self.ax_q_integrand.set_yscale('log')
            self.ax_q_integrand.grid(True, alpha=0.3)

            self.ax_mu_integrand.set_title(r'$\mu_{visc}$ 각도 피적분함수: $\cos\phi \times \mathrm{Im}[E]$ vs $\phi$', fontweight='bold', fontsize=14)
            self.ax_mu_integrand.set_xlabel(r'$\phi$ (rad)', fontsize=13)
            self.ax_mu_integrand.set_ylabel(r"$\cos\phi \times E''/((1-\nu^2)\sigma_0)$", fontsize=12)
            self.ax_mu_integrand.grid(True, alpha=0.3)

            self.ax_freq_range.set_title('속도별 주파수 스캔 범위', fontweight='bold', fontsize=14)
            self.ax_freq_range.set_xlabel('속도 v (m/s)', fontsize=13)
            self.ax_freq_range.set_ylabel('주파수 ω (rad/s)', fontsize=13)
            self.ax_freq_range.set_xscale('log')
            self.ax_freq_range.set_yscale('log')
            self.ax_freq_range.grid(True, alpha=0.3)

            self.integrand_progress_var.set(20)
            self.root.update_idletasks()

            # Get f, g interpolators if using nonlinear correction
            f_interp = None
            g_interp = None
            if use_fg and hasattr(self, 'f_interpolator') and self.f_interpolator is not None:
                f_interp = self.f_interpolator
                g_interp = self.g_interpolator

            # Create angle array
            phi = np.linspace(0, 2 * np.pi, n_angle)

            colors = plt.cm.viridis(np.linspace(0, 0.9, len(q_values)))

            self.integrand_result_text.insert(tk.END, "=== G 각도 피적분함수 분석 ===\n\n")

            # Calculate angle integrand for each q value
            for idx, q in enumerate(q_values):
                # Calculate frequencies: ω = q * v * cos(φ)
                omega = q * v * np.cos(phi)
                omega_eval = np.abs(omega)
                omega_eval[omega_eval < 1e-10] = 1e-10

                # Get strain for nonlinear correction (if available)
                strain_at_q = 0.01  # default
                if use_fg and hasattr(self, 'rms_slope_profiles') and self.rms_slope_profiles is not None:
                    try:
                        from scipy.interpolate import interp1d
                        rms_q = self.rms_slope_profiles['q']
                        rms_strain = self.rms_slope_profiles['strain']
                        log_q_arr = np.log10(rms_q)
                        log_strain = np.log10(np.maximum(rms_strain, 1e-10))
                        interp_func = interp1d(log_q_arr, log_strain, kind='linear',
                                               bounds_error=False, fill_value='extrapolate')
                        strain_at_q = 10 ** interp_func(np.log10(q))
                        strain_at_q = np.clip(strain_at_q, 0.0, 1.0)
                    except:
                        pass

                # Calculate |E|² integrand for G
                integrand_G = np.zeros_like(phi)
                integrand_mu = np.zeros_like(phi)

                for i, w in enumerate(omega_eval):
                    E_prime = self.material.get_storage_modulus(np.array([w]))[0]
                    E_loss = self.material.get_loss_modulus(np.array([w]))[0]

                    if use_fg and f_interp is not None and g_interp is not None:
                        f_val = np.clip(f_interp(strain_at_q), 0.0, 1.0)
                        g_val = np.clip(g_interp(strain_at_q), 0.01, None)  # g can exceed 1.0
                        E_prime_eff = E_prime * f_val
                        E_loss_eff = E_loss * g_val
                    else:
                        E_prime_eff = E_prime
                        E_loss_eff = E_loss

                    # |E_eff|² for G integrand
                    E_star_sq = E_prime_eff**2 + E_loss_eff**2
                    integrand_G[i] = E_star_sq * prefactor**2

                    # cosφ × E'' for μ_visc integrand
                    integrand_mu[i] = np.cos(phi[i]) * E_loss_eff * prefactor

                # Plot G angle integrand
                self.ax_angle_integrand.plot(phi, integrand_G, '-', color=colors[idx],
                                             label=f'q = {q:.1e} 1/m', linewidth=1.5)

                # Plot μ_visc angle integrand
                self.ax_mu_integrand.plot(phi, integrand_mu, '-', color=colors[idx],
                                          label=f'q = {q:.1e} 1/m', linewidth=1.5)

                # Calculate angle integral result
                angle_integral_G = np.trapezoid(integrand_G, phi)
                angle_integral_mu = np.trapezoid(integrand_mu, phi)

                # Summary text
                self.integrand_result_text.insert(tk.END, f"q = {q:.2e} 1/m:\n")
                self.integrand_result_text.insert(tk.END, f"  ω_max = qv = {q*v:.2e} rad/s\n")
                self.integrand_result_text.insert(tk.END, f"  ∫|E|²dφ = {angle_integral_G:.4e}\n")
                self.integrand_result_text.insert(tk.END, f"  ∫cosφ×E\'\'dφ = {angle_integral_mu:.4e}\n")
                if use_fg:
                    self.integrand_result_text.insert(tk.END, f"  ε(q) = {strain_at_q:.4f}\n")
                self.integrand_result_text.insert(tk.END, "\n")

            self.ax_angle_integrand.legend(fontsize=12)
            self.ax_mu_integrand.legend(fontsize=12)

            self.integrand_progress_var.set(50)
            self.root.update_idletasks()

            # === G(q) integrand plot ===
            # Get PSD q range
            if hasattr(self.psd_model, 'q_data'):
                q_array = self.psd_model.q_data
            elif hasattr(self.psd_model, 'q'):
                q_array = self.psd_model.q
            else:
                q_array = np.logspace(2, 8, 200)

            # Calculate G(q) integrand for each q
            q_plot = np.logspace(np.log10(q_array.min()), np.log10(q_array.max()), 100)
            G_integrand_values = np.zeros_like(q_plot)

            for i, q in enumerate(q_plot):
                # Get PSD value
                C_q = self.psd_model(np.array([q]))[0]

                # Calculate angle integral
                omega = q * v * np.cos(phi)
                omega_eval = np.abs(omega)
                omega_eval[omega_eval < 1e-10] = 1e-10

                integrand = np.zeros_like(phi)
                for j, w in enumerate(omega_eval):
                    E_prime = self.material.get_storage_modulus(np.array([w]))[0]
                    E_loss = self.material.get_loss_modulus(np.array([w]))[0]

                    if use_fg and f_interp is not None and g_interp is not None:
                        strain_q = 0.01  # simplified
                        f_val = np.clip(f_interp(strain_q), 0.0, 1.0)
                        g_val = np.clip(g_interp(strain_q), 0.01, None)  # g can exceed 1.0
                        E_prime_eff = E_prime * f_val
                        E_loss_eff = E_loss * g_val
                    else:
                        E_prime_eff = E_prime
                        E_loss_eff = E_loss

                    E_star_sq = E_prime_eff**2 + E_loss_eff**2
                    integrand[j] = E_star_sq * prefactor**2

                angle_int = np.trapezoid(integrand, phi)
                G_integrand_values[i] = q**3 * C_q * angle_int

            # Plot G(q) integrand
            self.ax_q_integrand.plot(q_plot, G_integrand_values, 'b-', linewidth=1.5,
                                     label=f'v = {v:.2e} m/s')

            # Mark the selected q values
            for q in q_values:
                if q >= q_plot.min() and q <= q_plot.max():
                    idx = np.abs(q_plot - q).argmin()
                    self.ax_q_integrand.axvline(q, color='r', linestyle='--', alpha=0.5)
                    self.ax_q_integrand.plot(q, G_integrand_values[idx], 'ro', markersize=8)

            self.ax_q_integrand.legend(fontsize=12)

            self.integrand_progress_var.set(70)
            self.root.update_idletasks()

            # === Frequency range vs velocity plot ===
            v_range = np.logspace(-4, 1, 50)  # 0.0001 to 10 m/s
            q_ref = float(q_values[0])  # Use first q value as reference

            # ω_min = 0 (at cosφ = 0), ω_max = q*v (at cosφ = 1)
            omega_max = q_ref * v_range
            omega_min = 1e-10 * np.ones_like(v_range)  # small but not zero

            # Plot frequency range
            self.ax_freq_range.fill_between(v_range, omega_min, omega_max, alpha=0.3, label=f'q = {q_ref:.1e} 1/m')
            self.ax_freq_range.plot(v_range, omega_max, 'b-', linewidth=1.5, label='ω_max = qv')

            # Show DMA frequency range
            if hasattr(self, 'raw_dma_data') and self.raw_dma_data is not None:
                omega_dma_min = self.raw_dma_data['omega'].min()
                omega_dma_max = self.raw_dma_data['omega'].max()
                self.ax_freq_range.axhline(omega_dma_min, color='r', linestyle='--', alpha=0.7, label=f'DMA ω_min = {omega_dma_min:.1e}')
                self.ax_freq_range.axhline(omega_dma_max, color='r', linestyle='--', alpha=0.7, label=f'DMA ω_max = {omega_dma_max:.1e}')

            # Mark current velocity
            self.ax_freq_range.axvline(v, color='g', linestyle='-', linewidth=2, alpha=0.7, label=f'현재 v = {v:.2e}')

            self.ax_freq_range.legend(fontsize=12, loc='lower right')

            # Frequency range info text
            self.freq_range_text.insert(tk.END, f"선택 q = {q_ref:.2e} 1/m, v = {v:.2e} m/s\n")
            self.freq_range_text.insert(tk.END, f"ω 범위: 0 ~ {q_ref * v:.2e} rad/s\n\n")
            if hasattr(self, 'raw_dma_data') and self.raw_dma_data is not None:
                self.freq_range_text.insert(tk.END, f"DMA 데이터 범위:\n")
                self.freq_range_text.insert(tk.END, f"  {omega_dma_min:.2e} ~ {omega_dma_max:.2e} rad/s\n")

            self.integrand_progress_var.set(100)
            self.fig_integrand.subplots_adjust(left=0.12, right=0.95, top=0.94, bottom=0.10, hspace=0.42, wspace=0.35)
            self.canvas_integrand.draw()

            self.status_var.set("피적분함수 계산 완료")

        except Exception as e:
            import traceback
            messagebox.showerror("오류", f"계산 실패:\n{str(e)}\n\n{traceback.format_exc()}")
        finally:
            self.integrand_calc_btn.config(state='normal')

    def _create_variables_tab(self, parent):
        """Create variable relationship explanation tab - matching 수식 정리 tab style."""
        # Toolbar
        self._create_panel_toolbar(parent)

        # Single scrollable canvas (same structure as equations tab)
        canvas = tk.Canvas(parent, bg='white', highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='white')

        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        def _on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)
        canvas.bind('<Configure>', _on_canvas_configure)

        def _update_scrollregion(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))
        scrollable_frame.bind("<Configure>", _update_scrollregion)

        # Mouse wheel scroll (cross-platform)
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        def _on_mousewheel_linux_up(event):
            canvas.yview_scroll(-3, "units")
        def _on_mousewheel_linux_down(event):
            canvas.yview_scroll(3, "units")

        def _bind_mousewheel(event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
            canvas.bind_all("<Button-4>", _on_mousewheel_linux_up)
            canvas.bind_all("<Button-5>", _on_mousewheel_linux_down)
        def _unbind_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")

        canvas.bind('<Enter>', _bind_mousewheel)
        canvas.bind('<Leave>', _unbind_mousewheel)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # --- Helper functions (same as equations tab) ---
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        math_fontfamily = 'cm'  # Computer Modern (built-in, best LaTeX look)

        def add_section_title(title_text, bg_color='#1B2A4A', fg_color='white'):
            frame = tk.Frame(scrollable_frame, bg=bg_color, padx=15, pady=12)
            frame.pack(fill=tk.X, padx=10, pady=(18, 4))
            tk.Label(frame, text=title_text, bg=bg_color, fg=fg_color,
                     font=('Segoe UI', 24, 'bold')).pack(anchor=tk.W)

        def add_text(text, font_size=19, fg='#1E293B', bold=False, padx=20, pady=4):
            weight = 'bold' if bold else 'normal'
            lbl = tk.Label(scrollable_frame, text=text, bg='white', fg=fg,
                           font=('Segoe UI', font_size, weight),
                           justify=tk.LEFT, anchor='w', wraplength=1800)
            lbl.pack(fill=tk.X, padx=padx, pady=pady, anchor='w')

        def add_equation(latex_str, fig_height=1.2, font_size=24):
            fig = Figure(figsize=(14, fig_height), facecolor='white')
            ax = fig.add_subplot(111)
            ax.axis('off')
            ax.text(0.02, 0.5, latex_str, transform=ax.transAxes,
                    fontsize=font_size, verticalalignment='center',
                    horizontalalignment='left', usetex=False,
                    math_fontfamily=math_fontfamily)
            fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.12)
            eq_canvas = FigureCanvasTkAgg(fig, master=scrollable_frame)
            eq_canvas.draw()
            eq_canvas.get_tk_widget().configure(height=int(fig_height * 80))
            eq_canvas.get_tk_widget().pack(fill=tk.X, padx=20, pady=(8, 8))

        def add_separator():
            tk.Frame(scrollable_frame, bg='#CBD5E1', height=2).pack(fill=tk.X, padx=10, pady=10)

        def add_graph(plot_func, fig_height=3.5):
            """Add an illustrative matplotlib graph."""
            import numpy as np
            fig = Figure(figsize=(12, fig_height), facecolor='#FAFBFC')
            ax = fig.add_subplot(111)
            ax.set_facecolor('#FAFBFC')
            plot_func(ax, np)
            ax.tick_params(labelsize=14)
            for spine in ax.spines.values():
                spine.set_color('#CBD5E1')
            fig.tight_layout(pad=1.5)
            graph_canvas = FigureCanvasTkAgg(fig, master=scrollable_frame)
            graph_canvas.draw()
            graph_canvas.get_tk_widget().configure(height=int(fig_height * 72))
            graph_canvas.get_tk_widget().pack(fill=tk.X, padx=30, pady=(6, 14))

        # === Title ===
        title_frame = tk.Frame(scrollable_frame, bg='white', pady=10)
        title_frame.pack(fill=tk.X, padx=10)
        tk.Label(title_frame, text='Persson 마찰 이론 - 변수 관계도',
                 bg='white', fg='#1B2A4A',
                 font=('Segoe UI', 28, 'bold')).pack(anchor='w', padx=10)

        # ═══════════════════════════════════════════════════════
        # Section 1: 입력 데이터
        # ═══════════════════════════════════════════════════════
        add_section_title('1. 입력 데이터')

        add_text('DMA 데이터 (재료 물성) — 고무의 점탄성 특성:', bold=True, pady=(8, 0))
        add_equation(r"$E(\omega) = E'(\omega) + i\,E''(\omega)$", fig_height=0.9)
        add_text('  \u03c9 : 각진동수 [rad/s] — 고무에 가해지는 진동의 빠르기', font_size=17, fg='#64748B')
        add_text('  E\'(\u03c9) : 저장 탄성률 [Pa] — 탄성 에너지를 저장하는 능력 (스프링 성분)', font_size=17, fg='#64748B')
        add_text('  E\'\'(\u03c9) : 손실 탄성률 [Pa] — 에너지를 열로 소산하는 능력 (댐퍼 성분)', font_size=17, fg='#64748B')
        add_text('  tan(\u03b4) = E\'\'/E\' : 손실 탄젠트 — E\'에 대한 E\'\'의 비율, 에너지 소산 효율의 척도', font_size=17, fg='#64748B')

        def _plot_var_dma(ax, np):
            omega = np.logspace(-2, 10, 500)
            E_stor = 1e6 + (1e9 - 1e6) / (1 + (1e4 / omega)**0.6)
            E_loss = 0.35e9 * (omega / 1e4)**0.5 / (1 + (omega / 1e4)**0.85)
            ax.loglog(omega, E_stor, '-', linewidth=2.5, color='#2563EB', label="E' (저장)")
            ax.loglog(omega, E_loss, '--', linewidth=2.5, color='#DC2626', label="E'' (손실)")
            ax.set_xlabel(r'$\omega$ (rad/s)', fontsize=16)
            ax.set_ylabel('E (Pa)', fontsize=16)
            ax.legend(fontsize=14)
            ax.grid(True, alpha=0.3, which='both')
            ax.set_title('DMA 마스터 커브 — 대표적 형상', fontsize=16, pad=10)
        add_graph(_plot_var_dma)

        add_separator()
        add_text('PSD 데이터 (표면 거칠기) — 바닥면의 요철 특성:', bold=True, pady=(6, 0))
        add_text('  q : 파수 [1/m] — 거칠기의 공간 진동수 (q = 2\u03c0/\u03bb, \u03bb = 파장)', font_size=17, fg='#64748B')
        add_text('  C(q) : 파워 스펙트럼 밀도 [m\u2074] — 파수 q에서의 거칠기 진폭의 제곱', font_size=17, fg='#64748B')

        def _plot_var_psd(ax, np):
            q = np.logspace(2, 8, 500)
            C = 1e-10 * (q / 1e2)**(-2.2)
            ax.loglog(q, C, '-', linewidth=2.5, color='#059669')
            ax.set_xlabel('q (1/m)', fontsize=16)
            ax.set_ylabel(r'C(q) (m$^4$)', fontsize=16)
            ax.grid(True, alpha=0.3, which='both')
            ax.set_title('PSD — 파워 스펙트럼 밀도 (대표적 형상)', fontsize=16, pad=10)
        add_graph(_plot_var_psd)

        add_separator()
        add_text('Strain Sweep 데이터 (비선형 보정용, 선택):', bold=True, pady=(6, 0))
        add_text('  \u03b3 : strain [%] — 변형률 진폭', font_size=17, fg='#64748B')
        add_text('  f(\u03b3) = E\'(\u03b3)/E\'(0) : 저장 탄성률 감소율 (대변형 → f < 1)', font_size=17, fg='#64748B')
        add_text('  g(\u03b3) = E\'\'(\u03b3)/E\'\'(0) : 손실 탄성률 변화율 (Payne 효과)', font_size=17, fg='#64748B')

        def _plot_var_strain(ax, np):
            gamma = np.linspace(0, 50, 200)
            f_g = 1 / (1 + 0.8 * (gamma / 10)**0.9)
            g_g = (1 + 2.5 * (gamma / 10)) / (1 + 3 * (gamma / 10)**1.4)
            ax.plot(gamma, f_g, '-', linewidth=2.5, color='#2563EB', label=r"f($\gamma$) — E' 감소")
            ax.plot(gamma, g_g, '--', linewidth=2.5, color='#DC2626', label=r"g($\gamma$) — E'' 변화")
            ax.set_xlabel(r'$\gamma$ (%)', fontsize=16)
            ax.set_ylabel('보정 계수', fontsize=16)
            ax.legend(fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
            ax.set_title('Strain Sweep — Payne 효과', fontsize=16, pad=10)
        add_graph(_plot_var_strain)

        # ═══════════════════════════════════════════════════════
        # Section 2: 계산 파라미터
        # ═══════════════════════════════════════════════════════
        add_section_title('2. 계산 파라미터')

        # Parameters table
        param_frame = tk.Frame(scrollable_frame, bg='white', padx=20, pady=6)
        param_frame.pack(fill=tk.X, padx=10)
        param_data = [
            ('\u03c3\u2080', '공칭 접촉 압력 [Pa]', '하중/면적. 고무가 바닥을 누르는 평균 압력'),
            ('v', '슬라이딩 속도 [m/s]', '고무가 바닥 위를 미끄러지는 속도'),
            ('T', '온도 [\u00b0C]', 'DMA 마스터 커브의 시프트 인자에 영향'),
            ('\u03bd', '푸아송 비', '고무 \u2248 0.5 (거의 비압축성)'),
            ('\u03b3', '접촉 보정 인자', '\u2248 0.5, S(q) 계산에 사용'),
            ('q\u2080 ~ q\u2081', 'PSD 적분 범위 [1/m]', '고려하는 거칠기 파수의 최소~최대'),
        ]
        for col_idx, header in enumerate(['기호', '의미', '설명']):
            lbl = tk.Label(param_frame, text=header, bg='#1B2A4A', fg='white',
                           font=('Segoe UI', 17, 'bold'), padx=12, pady=7, anchor='center')
            lbl.grid(row=0, column=col_idx, sticky='nsew', padx=1, pady=1)
        for row_idx, (sym, meaning, desc) in enumerate(param_data, start=1):
            bg = '#F1F5F9' if row_idx % 2 == 0 else 'white'
            for col_idx, val in enumerate([sym, meaning, desc]):
                weight = 'bold' if col_idx == 0 else 'normal'
                lbl = tk.Label(param_frame, text=val, bg=bg, fg='#1E293B',
                               font=('Segoe UI', 17, weight), padx=12, pady=6, anchor='w')
                lbl.grid(row=row_idx, column=col_idx, sticky='nsew', padx=1, pady=1)
        param_frame.columnconfigure(0, weight=1)
        param_frame.columnconfigure(1, weight=2)
        param_frame.columnconfigure(2, weight=3)

        # ═══════════════════════════════════════════════════════
        # Section 3: 중간 계산 변수
        # ═══════════════════════════════════════════════════════
        add_section_title('3. 중간 계산 변수')

        add_text('G(q) 계산 과정:', bold=True, pady=(8, 0))
        add_equation(r"$E^*(\omega) = E'(\omega) + i\,E''(\omega), \qquad \omega = q \cdot v \cdot \cos\phi$", fig_height=0.9)
        add_equation(
            r'$G(q) = \frac{1}{8} \int_{q_0}^{q} dq^{\prime}\, (q^{\prime})^3\, C(q^{\prime})'
            r' \int_{0}^{2\pi} d\phi\, \left| \frac{E^*(q^{\prime}v\cos\phi)}{(1-\nu^2)\sigma_0} \right|^2$',
            fig_height=1.3)

        def _plot_var_G(ax, np):
            q = np.logspace(2, 8, 500)
            G = 0.01 * (q / 1e2)**1.2 / (1 + (q / 1e7)**0.3)
            ax.loglog(q, G, '-', linewidth=2.5, color='#2563EB')
            ax.set_xlabel('q (1/m)', fontsize=16)
            ax.set_ylabel('G(q)', fontsize=16)
            ax.grid(True, alpha=0.3, which='both')
            ax.set_title('G(q) — 탄성 에너지 누적 함수', fontsize=16, pad=10)
        add_graph(_plot_var_G)

        add_text('  [비선형 보정 시]', font_size=17, bold=True, fg='#64748B')
        add_equation(r"$E'_{eff} = E' \times f(\varepsilon), \qquad E''_{eff} = E'' \times g(\varepsilon)$", fig_height=0.9)

        add_separator()
        add_text('접촉 면적 P(q)와 보정 계수 S(q):', bold=True, pady=(6, 0))
        add_equation(r'$P(q) = \mathrm{erf}\!\left(\frac{1}{2\sqrt{G(q)}}\right)$', fig_height=1.0)
        add_text('  G \u2192 0 : P \u2192 1 (완전 접촉)  |  G \u2192 \u221e : P \u2192 0 (접촉 없음)', font_size=17, fg='#64748B')
        add_equation(r'$S(q) = \gamma + (1-\gamma) \cdot P^2(q)$', fig_height=0.9)

        def _plot_var_PS(ax, np):
            from scipy.special import erf
            G = np.linspace(0.01, 20, 500)
            P = erf(1 / (2 * np.sqrt(G)))
            ax.plot(G, P, '-', linewidth=2.5, color='#2563EB', label='P(q)')
            gamma = 0.5
            S = gamma + (1 - gamma) * P**2
            ax.plot(G, S, '--', linewidth=2.5, color='#059669', label='S(q)')
            ax.set_xlabel('G(q)', fontsize=16)
            ax.set_ylabel('값', fontsize=16)
            ax.legend(fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.set_title('P(q)와 S(q) — G에 따른 변화', fontsize=16, pad=10)
        add_graph(_plot_var_PS)

        add_separator()
        add_text('표면 거칠기 통계량:', bold=True, pady=(6, 0))
        add_equation(r"$\xi^2(q) = h_{rms}^{\prime\,2}(q) = 2\pi \int_{q_0}^{q} k^3\, C(k)\, dk$", fig_height=1.0)
        add_text('  \u03be(q) = h\'_rms(q) : 누적 RMS 기울기 (파수 q까지의 표면 경사)', font_size=17, fg='#64748B')
        add_equation(r'$\varepsilon(q) = \alpha \cdot \xi(q) \qquad (\alpha \approx 0.5)$', fig_height=0.9)
        add_text('  \u03b5(q) : 국소 변형률 — 거칠기에 의한 고무의 국소 변형 크기', font_size=17, fg='#64748B')

        def _plot_var_xi_eps(ax, np):
            q = np.logspace(2, 8, 500)
            xi = 0.001 * (q / 1e2)**0.8
            eps = 0.5 * xi
            ax.loglog(q, xi, '-', linewidth=2.5, color='#DC2626', label=r"$\xi(q) = h'_{rms}$")
            ax.loglog(q, eps, '--', linewidth=2.5, color='#7C3AED', label=r"$\varepsilon(q) = 0.5 \cdot \xi$")
            ax.set_xlabel('q (1/m)', fontsize=16)
            ax.set_ylabel('값', fontsize=16)
            ax.legend(fontsize=14)
            ax.grid(True, alpha=0.3, which='both')
            ax.set_title(r"$\xi(q)$와 $\varepsilon(q)$ — 파수에 따른 변화", fontsize=16, pad=10)
        add_graph(_plot_var_xi_eps)

        # ═══════════════════════════════════════════════════════
        # Section 4: 최종 출력
        # ═══════════════════════════════════════════════════════
        add_section_title('4. 최종 출력: \u03bc_visc')

        add_equation(
            r'$\mu_{visc} = \frac{1}{2} \int_{q_0}^{q_1} dq\, q^3 C(q)\, P(q)\, S(q)'
            r' \int_{0}^{2\pi} d\phi\, \cos\phi\, \frac{\mathrm{Im}[E(qv\cos\phi)]}{(1-\nu^2)\sigma_0}$',
            fig_height=1.3)
        add_text('  [선형] G, P, S 계산 → Im[E] = Im[E_linear]', font_size=17, fg='#64748B')
        add_text('  [비선형] E_eff 사용, G\u00b7P\u00b7S 재계산, Im[E_eff] = Im[E] \u00d7 g(\u03b5)', font_size=17, fg='#64748B')

        def _plot_var_mu(ax, np):
            v = np.logspace(-6, 2, 500)
            mu = 0.8 * np.exp(-0.5 * ((np.log10(v) + 2) / 2)**2) + 0.1
            ax.semilogx(v, mu, '-', linewidth=2.5, color='#DC2626')
            ax.set_xlabel('v (m/s)', fontsize=16)
            ax.set_ylabel(r'$\mu_{visc}$', fontsize=16)
            ax.grid(True, alpha=0.3)
            ax.set_title(r'$\mu_{visc}$ vs 슬라이딩 속도 (대표적 형상)', fontsize=16, pad=10)
            peak_idx = np.argmax(mu)
            ax.annotate('마찰 피크', xy=(v[peak_idx], mu[peak_idx]),
                        fontsize=14, fontweight='bold', color='#DC2626',
                        xytext=(v[peak_idx]*20, mu[peak_idx]*0.85),
                        arrowprops=dict(arrowstyle='->', color='#DC2626'))
        add_graph(_plot_var_mu)

        # ═══════════════════════════════════════════════════════
        # Section 5: 데이터 흐름도
        # ═══════════════════════════════════════════════════════
        add_section_title('5. 데이터 흐름도', bg_color='#7C3AED')

        add_text('DMA + PSD → Tab1(검증) → Tab2(설정) → Tab3(G, P 계산)', bold=True, pady=(8, 0))
        add_text('→ Tab4(h\'_rms, \u03b5 계산) → Tab5(\u03bc_visc 계산)', bold=True, pady=(0, 4))
        add_text('Strain Sweep → f(\u03b5), g(\u03b5) 함수 → 비선형 보정에 반영', bold=True, pady=(0, 8))

        # ═══════════════════════════════════════════════════════
        # Section 6: 단위 정리
        # ═══════════════════════════════════════════════════════
        add_section_title('6. 단위 정리')

        unit_frame = tk.Frame(scrollable_frame, bg='white', padx=20, pady=6)
        unit_frame.pack(fill=tk.X, padx=10)
        unit_data = [
            ('q', '[1/m]', '파수'),
            ('C(q)', '[m\u2074]', '파워 스펙트럼 밀도'),
            ("E', E''", '[Pa]', '저장/손실 탄성률'),
            ('\u03c3\u2080', '[Pa]', '공칭 접촉 압력'),
            ('v', '[m/s]', '슬라이딩 속도'),
            ('\u03c9', '[rad/s]', '각진동수'),
            ('G(q)', '[무차원]', '탄성 에너지 적분'),
            ('P(q)', '[0~1]', '실접촉 면적 비율'),
            ('S(q)', '[무차원]', '접촉 보정 계수'),
            ('h_rms', '[m]', 'RMS 높이'),
            ("h'_rms = \u03be", '[무차원]', 'RMS 기울기'),
            ('\u03b5(q)', '[0~1]', '국소 변형률'),
            ('\u03bc_visc', '[무차원]', '점탄성 마찰 계수'),
        ]
        for col_idx, header in enumerate(['기호', '단위', '의미']):
            lbl = tk.Label(unit_frame, text=header, bg='#1B2A4A', fg='white',
                           font=('Segoe UI', 17, 'bold'), padx=12, pady=7, anchor='center')
            lbl.grid(row=0, column=col_idx, sticky='nsew', padx=1, pady=1)
        for row_idx, (sym, unit, meaning) in enumerate(unit_data, start=1):
            bg = '#F1F5F9' if row_idx % 2 == 0 else 'white'
            for col_idx, val in enumerate([sym, unit, meaning]):
                weight = 'bold' if col_idx == 0 else 'normal'
                lbl = tk.Label(unit_frame, text=val, bg=bg, fg='#1E293B',
                               font=('Segoe UI', 17, weight), padx=12, pady=6, anchor='center')
                lbl.grid(row=row_idx, column=col_idx, sticky='nsew', padx=1, pady=1)
        for col_idx in range(3):
            unit_frame.columnconfigure(col_idx, weight=1)

        # Bottom padding for scroll
        tk.Frame(scrollable_frame, bg='white', height=80).pack(fill=tk.X)

        # Force scrollregion update after all widgets are added
        self.root.after(100, _update_scrollregion)

    def _create_debug_tab(self, parent):
        """Create debug log tab for monitoring calculation values."""
        # Toolbar
        self._create_panel_toolbar(parent, buttons=[
            ("진단 실행", self._run_debug_diagnostic, 'Accent.TButton'),
        ])

        # Instruction label
        instruction = ttk.LabelFrame(parent, text="디버그 로그 탭 설명", padding=10)
        instruction.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(instruction, text=
            "이 탭에서는 μ_visc 계산 과정의 모든 중간 변수 값을 확인할 수 있습니다.\n"
            "문제 진단 및 계산 검증에 사용하세요. '진단 실행' 버튼으로 상세 분석을 수행합니다.",
            font=('Segoe UI', 17)
        ).pack()

        # Control buttons
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(
            btn_frame,
            text="진단 실행",
            command=self._run_debug_diagnostic,
            width=20
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            btn_frame,
            text="로그 지우기",
            command=self._clear_debug_log,
            width=15
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            btn_frame,
            text="로그 저장",
            command=self._save_debug_log,
            width=15
        ).pack(side=tk.LEFT, padx=5)

        # Debug log text area with scrollbar
        log_frame = ttk.LabelFrame(parent, text="계산 로그", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Create text widget with scrollbar
        log_scroll = ttk.Scrollbar(log_frame)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.debug_log_text = tk.Text(
            log_frame,
            wrap=tk.WORD,
            font=('Courier New', 15),
            yscrollcommand=log_scroll.set
        )
        self.debug_log_text.pack(fill=tk.BOTH, expand=True)
        log_scroll.config(command=self.debug_log_text.yview)

        # Initialize with welcome message
        self.debug_log_text.insert(tk.END, "=" * 70 + "\n")
        self.debug_log_text.insert(tk.END, "  디버그 로그 탭 - μ_visc 계산 진단 도구\n")
        self.debug_log_text.insert(tk.END, "=" * 70 + "\n\n")
        self.debug_log_text.insert(tk.END, "  '진단 실행' 버튼을 클릭하여 계산 상세 분석을 시작하세요.\n")
        self.debug_log_text.insert(tk.END, "  이 탭에서는 다음 값들을 확인할 수 있습니다:\n\n")
        self.debug_log_text.insert(tk.END, "  • 마스터 커브 데이터 (E', E'', 주파수 범위)\n")
        self.debug_log_text.insert(tk.END, "  • PSD 데이터 (C(q), q 범위)\n")
        self.debug_log_text.insert(tk.END, "  • G(q) 계산 중간값\n")
        self.debug_log_text.insert(tk.END, "  • P(q), S(q) 값\n")
        self.debug_log_text.insert(tk.END, "  • 각도 적분값 (angle integral)\n")
        self.debug_log_text.insert(tk.END, "  • 피적분함수 (integrand)\n")
        self.debug_log_text.insert(tk.END, "  • μ_visc 최종값 및 진단 결과\n")

    def _log_debug(self, message: str):
        """Add a message to the debug log."""
        if hasattr(self, 'debug_log_text'):
            self.debug_log_text.insert(tk.END, message + "\n")
            self.debug_log_text.see(tk.END)
            self.root.update()

    def _clear_debug_log(self):
        """Clear the debug log."""
        if hasattr(self, 'debug_log_text'):
            self.debug_log_text.delete(1.0, tk.END)
            self._log_debug("로그가 지워졌습니다.\n")

    def _save_debug_log(self):
        """Save debug log to file."""
        if hasattr(self, 'debug_log_text'):
            from tkinter import filedialog
            filepath = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                title="디버그 로그 저장"
            )
            if filepath:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(self.debug_log_text.get(1.0, tk.END))
                self._show_status(f"로그가 저장되었습니다:\n{filepath}", 'success')

    def _run_debug_diagnostic(self):
        """Run comprehensive debug diagnostic for mu_visc calculation."""
        self._clear_debug_log()
        self._log_debug("=" * 70)
        self._log_debug("  μ_visc 계산 진단 시작")
        self._log_debug("=" * 70 + "\n")

        # 1. Check Material (DMA) data
        self._log_debug("[1] 재료 데이터 (DMA) 검사")
        self._log_debug("-" * 50)

        if self.material is None:
            self._log_debug("  ❌ 오류: 재료 데이터가 없습니다!")
            self._log_debug("     → Tab 0에서 마스터 커브를 생성하고 Tab 1에서 가져오세요.\n")
            return
        else:
            self._log_debug("  ✓ 재료 데이터 로드됨")

            # Check material attributes
            if hasattr(self.material, '_omega') and self.material._omega is not None:
                omega = self.material._omega
                self._log_debug(f"  • 주파수 범위: {omega[0]:.2e} ~ {omega[-1]:.2e} rad/s")
                self._log_debug(f"  • 데이터 점 수: {len(omega)}")

            # Sample E' and E'' at various frequencies
            test_freqs = [1e-4, 1e-2, 1, 1e2, 1e4, 1e6]
            self._log_debug("\n  주파수별 E', E'' 샘플:")
            self._log_debug("  " + "-" * 45)
            self._log_debug("  {:<12} {:<15} {:<15}".format("w (rad/s)", "E' (Pa)", "E'' (Pa)"))
            self._log_debug("  " + "-" * 45)

            temperature = float(self.temperature_var.get()) if hasattr(self, 'temperature_var') else 20.0
            any_zero_E = False
            any_nan_E = False

            for freq in test_freqs:
                try:
                    E_prime = self.material.get_storage_modulus(freq, temperature=temperature)
                    E_loss = self.material.get_loss_modulus(freq, temperature=temperature)

                    if not np.isfinite(E_prime) or not np.isfinite(E_loss):
                        any_nan_E = True
                        self._log_debug(f"  {freq:<12.2e} {'NaN!':<15} {'NaN!':<15} ⚠️")
                    elif E_prime < 1e3 or E_loss < 1e2:
                        any_zero_E = True
                        self._log_debug(f"  {freq:<12.2e} {E_prime:<15.2e} {E_loss:<15.2e} ⚠️ 너무 작음")
                    else:
                        self._log_debug(f"  {freq:<12.2e} {E_prime:<15.2e} {E_loss:<15.2e}")
                except Exception as e:
                    self._log_debug(f"  {freq:<12.2e} {'오류!':<15} {str(e)[:15]:<15}")

            if any_nan_E:
                self._log_debug("\n  ❌ 경고: 일부 주파수에서 E' 또는 E''가 NaN입니다!")
            if any_zero_E:
                self._log_debug("\n  ⚠️ 경고: 일부 주파수에서 E' 또는 E''가 너무 작습니다!")

            # Check master curve frequency range vs expected omega range
            self._log_debug("\n  마스터 커브 주파수 범위 분석:")
            if hasattr(self.material, '_frequencies') and self.material._frequencies is not None:
                omega_data_min = np.min(self.material._frequencies)
                omega_data_max = np.max(self.material._frequencies)
                self._log_debug(f"  • 마스터 커브 ω 범위: {omega_data_min:.2e} ~ {omega_data_max:.2e} rad/s")

                # Estimate typical omega range for mu_visc calculation
                q_typical = [1e2, 1e4, 1e6]  # typical q values
                v_typical = [0.001, 0.1, 10]  # typical velocities
                self._log_debug("\n  μ_visc 계산 시 예상 ω 범위 (ω = q × v):")
                for q in q_typical:
                    for v in v_typical:
                        omega_calc = q * v
                        in_range = omega_data_min <= omega_calc <= omega_data_max
                        status = "✓" if in_range else "⚠️ 범위 밖"
                        self._log_debug(f"    q={q:.0e}, v={v:.0e} → ω={omega_calc:.2e} {status}")

        # 2. Check PSD data
        self._log_debug("\n\n[2] PSD 데이터 검사")
        self._log_debug("-" * 50)

        if self.psd_model is None:
            self._log_debug("  ❌ 오류: PSD 데이터가 없습니다!")
            self._log_debug("     → Tab 1에서 PSD 데이터를 로드하세요.\n")
            return
        else:
            self._log_debug("  ✓ PSD 데이터 로드됨")

            # Sample C(q) at various wavenumbers
            test_qs = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7]
            self._log_debug("\n  파수별 C(q) 샘플:")
            self._log_debug("  " + "-" * 35)
            self._log_debug(f"  {'q (1/m)':<12} {'C(q) (m^4)':<15}")
            self._log_debug("  " + "-" * 35)

            for q in test_qs:
                try:
                    C_q = self.psd_model(np.array([q]))[0]
                    if np.isfinite(C_q) and C_q > 0:
                        self._log_debug(f"  {q:<12.2e} {C_q:<15.2e}")
                    else:
                        self._log_debug(f"  {q:<12.2e} {'무효!':<15} ⚠️")
                except Exception as e:
                    self._log_debug(f"  {q:<12.2e} {'오류!':<15}")

        # 3. Check G(q,v) results
        self._log_debug("\n\n[3] G(q,v) 계산 결과 검사")
        self._log_debug("-" * 50)

        if not hasattr(self, 'results') or self.results is None or '2d_results' not in self.results:
            self._log_debug("  ❌ 오류: G(q,v) 계산 결과가 없습니다!")
            self._log_debug("     → Tab 2에서 G(q,v) 계산을 먼저 실행하세요.\n")
            return
        else:
            results_2d = self.results['2d_results']
            q = results_2d['q']
            v = results_2d['v']
            G_matrix = results_2d['G_matrix']

            self._log_debug("  ✓ G(q,v) 계산 결과 있음")
            self._log_debug(f"  • q 범위: {q[0]:.2e} ~ {q[-1]:.2e} (1/m), {len(q)}점")
            self._log_debug(f"  • v 범위: {v[0]:.2e} ~ {v[-1]:.2e} (m/s), {len(v)}점")
            self._log_debug(f"  • G 행렬 크기: {G_matrix.shape}")

            # G statistics
            G_min = np.nanmin(G_matrix)
            G_max = np.nanmax(G_matrix)
            G_mean = np.nanmean(G_matrix)

            self._log_debug(f"\n  G(q,v) 통계:")
            self._log_debug(f"  • 최소값: {G_min:.4e}")
            self._log_debug(f"  • 최대값: {G_max:.4e}")
            self._log_debug(f"  • 평균값: {G_mean:.4e}")

            # Check for NaN values
            nan_count = np.sum(~np.isfinite(G_matrix))
            if nan_count > 0:
                self._log_debug(f"  ⚠️ 경고: G 행렬에 {nan_count}개의 NaN/Inf 값이 있습니다!")

            # Calculate P(q) from G
            self._log_debug("\n  G(q) → P(q) 변환 (중간 속도):")
            mid_v_idx = len(v) // 2
            G_mid = G_matrix[:, mid_v_idx]
            from scipy.special import erf

            # Safe P calculation
            P_mid = np.zeros_like(G_mid)
            valid_G = np.isfinite(G_mid) & (G_mid > 0)
            P_mid[~valid_G] = 1.0  # Full contact for invalid G
            if np.any(valid_G):
                sqrt_G = np.sqrt(G_mid[valid_G])
                arg = 1.0 / (2.0 * np.minimum(sqrt_G, 1e5))
                P_mid[valid_G] = erf(np.minimum(arg, 10.0))

            P_min = np.min(P_mid)
            P_max = np.max(P_mid)
            P_mean = np.mean(P_mid)

            self._log_debug(f"  • P(q) 최소값: {P_min:.6f}")
            self._log_debug(f"  • P(q) 최대값: {P_max:.6f}")
            self._log_debug(f"  • P(q) 평균값: {P_mean:.6f}")

            if P_max < 0.001:
                self._log_debug("  ❌ 심각: P(q)가 거의 0! G(q)가 너무 큽니다.")
                self._log_debug("     → sigma_0 (공칭 압력)를 높이거나 PSD를 확인하세요.")

        # 4. Check calculation parameters
        self._log_debug("\n\n[4] 계산 파라미터 검사")
        self._log_debug("-" * 50)

        sigma_0 = float(self.sigma_0_var.get()) * 1e6 if hasattr(self, 'sigma_0_var') else 0.3e6
        temperature = float(self.temperature_var.get()) if hasattr(self, 'temperature_var') else 20.0
        poisson = float(self.poisson_var.get()) if hasattr(self, 'poisson_var') else 0.5
        gamma = float(self.gamma_var.get()) if hasattr(self, 'gamma_var') else 0.6
        n_phi = int(self.n_phi_var.get()) if hasattr(self, 'n_phi_var') else 72

        self._log_debug(f"  • sigma_0 (공칭 압력): {sigma_0/1e6:.3f} MPa = {sigma_0:.2e} Pa")
        self._log_debug(f"  • T (온도): {temperature:.1f} °C")
        self._log_debug(f"  • ν (푸아송비): {poisson:.2f}")
        self._log_debug(f"  • γ (접촉 보정): {gamma:.2f}")
        self._log_debug(f"  • n_φ (각도 적분점): {n_phi}")

        prefactor = 1.0 / ((1 - poisson**2) * sigma_0)
        self._log_debug(f"  • prefactor = 1/((1-ν²)sigma_0) = {prefactor:.4e}")

        # Check if prefactor is reasonable
        # For E'' ~ 1e7 Pa and prefactor ~ 4e-6, angle_integral ~ 4 * (pi/2) * 1e7 * 4e-6 ~ 0.25
        # This should give non-zero mu_visc
        if prefactor > 1e-4:
            self._log_debug("  ⚠️ 경고: prefactor가 큼 (sigma_0가 너무 작음)")
        elif prefactor < 1e-8:
            self._log_debug("  ⚠️ 경고: prefactor가 작음 (sigma_0가 너무 큼)")

        # 5. Test single point calculation
        self._log_debug("\n\n[5] 단일점 계산 테스트")
        self._log_debug("-" * 50)

        # Pick a middle q and v
        mid_q_idx = len(q) // 2
        mid_v_idx = len(v) // 2
        test_q = q[mid_q_idx]
        test_v = v[mid_v_idx]

        self._log_debug(f"  테스트 지점:")
        self._log_debug(f"  • q = {test_q:.2e} (1/m)")
        self._log_debug(f"  • v = {test_v:.4f} (m/s)")

        # Calculate omega range for this q and v
        omega_max = test_q * test_v  # φ = 0
        omega_min = test_q * test_v * np.cos(np.pi/2 - 0.001)  # near φ = π/2

        self._log_debug(f"  • ω 범위: {omega_min:.2e} ~ {omega_max:.2e} rad/s")

        # Get E' and E'' at omega_max
        try:
            E_prime_test = self.material.get_storage_modulus(omega_max, temperature=temperature)
            E_loss_test = self.material.get_loss_modulus(omega_max, temperature=temperature)
            self._log_debug(f"\n  ω={omega_max:.2e} 에서:")
            self._log_debug(f"  • E' = {E_prime_test:.4e} Pa")
            self._log_debug(f"  • E'' = {E_loss_test:.4e} Pa")

            if E_loss_test < 1:
                self._log_debug("  ❌ 심각: E''가 거의 0! 마스터 커브를 확인하세요.")
            elif E_loss_test < 1e4:
                self._log_debug(f"  ⚠️ 경고: E''가 작음 ({E_loss_test:.2e})")
        except Exception as e:
            self._log_debug(f"  ❌ 오류: {e}")

        # Calculate angle integral contribution
        self._log_debug("\n  각도 적분 테스트:")
        phi_test = np.linspace(0, np.pi/2, n_phi)
        omega_test = test_q * test_v * np.cos(phi_test)
        cos_phi_test = np.cos(phi_test)

        integrand_test = np.zeros(n_phi)
        ImE_test = np.zeros(n_phi)

        for i, (w, c) in enumerate(zip(omega_test, cos_phi_test)):
            w = max(w, 1e-10)
            try:
                E_loss = self.material.get_loss_modulus(w, temperature=temperature)
                ImE_test[i] = E_loss
                integrand_test[i] = c * E_loss * prefactor
            except:
                ImE_test[i] = 0
                integrand_test[i] = 0

        # Show some sample values
        self._log_debug(f"\n  φ별 샘플 (5개):")
        self._log_debug("  {:<10} {:<10} {:<12} {:<12} {:<12}".format("phi (rad)", "cos(phi)", "w (rad/s)", "E'' (Pa)", "integrand"))
        for i in [0, n_phi//4, n_phi//2, 3*n_phi//4, n_phi-1]:
            self._log_debug(f"  {phi_test[i]:<10.4f} {cos_phi_test[i]:<10.4f} {omega_test[i]:<12.2e} {ImE_test[i]:<12.2e} {integrand_test[i]:<12.4e}")

        # Calculate angle integral
        from scipy.integrate import simpson
        angle_integral = 4.0 * simpson(integrand_test, x=phi_test)
        self._log_debug(f"\n  각도 적분 결과: {angle_integral:.6e}")

        if abs(angle_integral) < 1e-10:
            self._log_debug("  ❌ 심각: 각도 적분이 0! E''가 문제입니다.")

        # 6. Calculate full integrand for this q
        self._log_debug("\n\n[6] 전체 피적분함수 테스트")
        self._log_debug("-" * 50)

        C_q_test = self.psd_model(np.array([test_q]))[0]
        G_test = G_matrix[mid_q_idx, mid_v_idx]

        # Calculate P and S
        if G_test > 0 and np.isfinite(G_test):
            P_test = erf(1.0 / (2.0 * np.sqrt(G_test)))
        else:
            P_test = 1.0
        S_test = gamma + (1 - gamma) * P_test**2

        q3_test = test_q**3
        qCPS_test = q3_test * C_q_test * P_test * S_test
        full_integrand = qCPS_test * angle_integral

        self._log_debug(f"  q = {test_q:.2e}")
        self._log_debug(f"  C(q) = {C_q_test:.4e}")
        self._log_debug(f"  G(q) = {G_test:.4e}")
        self._log_debug(f"  P(q) = erf(1/(2√G)) = {P_test:.6f}")
        self._log_debug(f"  S(q) = γ + (1-γ)P² = {S_test:.6f}")
        self._log_debug(f"  q³ = {q3_test:.4e}")
        self._log_debug(f"  q³·C·P·S = {qCPS_test:.4e}")
        self._log_debug(f"  angle_integral = {angle_integral:.4e}")
        self._log_debug(f"  피적분함수 = q³·C·P·S × angle_int = {full_integrand:.6e}")

        if abs(full_integrand) < 1e-15:
            self._log_debug("\n  ❌ 피적분함수가 거의 0!")
            if P_test < 0.01:
                self._log_debug("     → P(q)가 너무 작음 (G가 너무 큼)")
            if abs(angle_integral) < 1e-10:
                self._log_debug("     → 각도 적분이 0 (E''가 문제)")

        # 7. Existing mu_visc results
        self._log_debug("\n\n[7] μ_visc 결과 확인")
        self._log_debug("-" * 50)

        if hasattr(self, 'mu_visc_results') and self.mu_visc_results is not None:
            mu_array = self.mu_visc_results.get('mu', [])
            v_array = self.mu_visc_results.get('v', [])

            if len(mu_array) > 0:
                mu_min = np.min(mu_array)
                mu_max = np.max(mu_array)
                mu_mean = np.mean(mu_array)

                self._log_debug(f"  ✓ μ_visc 결과 있음")
                self._log_debug(f"  • μ_visc 범위: {mu_min:.6f} ~ {mu_max:.6f}")
                self._log_debug(f"  • μ_visc 평균: {mu_mean:.6f}")

                if mu_max < 1e-6:
                    self._log_debug("\n  ❌ 심각: μ_visc가 거의 0!")
                    self._log_debug("     가능한 원인:")
                    self._log_debug("     1. E''(손실탄성률)가 너무 작음")
                    self._log_debug("     2. P(q)(접촉면적비)가 너무 작음 → G(q)가 너무 큼")
                    self._log_debug("     3. 각도 적분이 0")
                    self._log_debug("     4. 마스터 커브 주파수 범위가 맞지 않음")
        else:
            self._log_debug("  ℹ️ μ_visc 계산 결과가 아직 없습니다.")
            self._log_debug("     → Tab 5에서 μ_visc 계산을 실행하세요.")

        # 8. Summary and recommendations
        self._log_debug("\n\n" + "=" * 70)
        self._log_debug("  진단 요약 및 권장 사항")
        self._log_debug("=" * 70)

        issues_found = False

        # Check for common issues
        if P_max < 0.01:
            issues_found = True
            self._log_debug("\n  ⚠️ P(q)가 매우 작음 (< 0.01)")
            self._log_debug("     권장: sigma_0 (공칭 압력)를 높이거나 표면 거칠기 확인")

        if abs(angle_integral) < 1e-10:
            issues_found = True
            self._log_debug("\n  ⚠️ 각도 적분이 0에 가까움")
            self._log_debug("     권장: 마스터 커브 데이터 (특히 E'') 확인")
            self._log_debug("     → ω 범위가 qv 범위를 포함하는지 확인")

        if any_nan_E:
            issues_found = True
            self._log_debug("\n  ⚠️ E' 또는 E''에 NaN 값 존재")
            self._log_debug("     권장: 마스터 커브 생성 과정 확인")

        if not issues_found:
            self._log_debug("\n  ✓ 주요 문제가 발견되지 않았습니다.")
            self._log_debug("    계산이 정상적으로 진행되어야 합니다.")

        self._log_debug("\n\n진단 완료.\n")

    def _create_friction_factors_tab(self, parent):
        """Create friction factors analysis tab - explains how to increase/decrease μ_visc."""
        # Toolbar
        self._create_panel_toolbar(parent)

        # Main scrollable frame
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Mouse wheel scroll
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Title
        title_frame = ttk.LabelFrame(scrollable_frame, text="μ_visc 영향 인자 분석 - 0.1~10 m/s 범위에서 마찰계수 조절 가이드", padding=15)
        title_frame.pack(fill=tk.X, padx=10, pady=10)

        content = """
════════════════════════════════════════════════════════════════════════════════
                   μ_visc 마찰계수 영향 인자 분석
════════════════════════════════════════════════════════════════════════════════

【μ_visc 공식】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  μ_visc = (1/2) × ∫[q0→q1] q³ C(q) P(q) S(q) × [∫₀²π cosφ × Im[E(qv·cosφ)] dφ] / ((1-ν²)sigma_0) dq

  분해하면:
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ μ_visc ∝ q³ × C(q) × P(q) × S(q) × Im[E(ω)] / sigma_0                           │
  └─────────────────────────────────────────────────────────────────────────────┘


【영향 인자별 상세 분석】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─ 1. C(q) - PSD 표면 거칠기 ──────────────────────────────────────────────────┐
│                                                                              │
│  【μ 증가】 C(q) ↑                                                           │
│  ├─ 표면이 더 거칠면 → C(q) 증가 → μ 증가                                    │
│  ├─ C(q0) 값 증가 (PSD 설정에서)                                             │
│  └─ Hurst 지수 H 감소 → 고주파 성분 증가 → μ 증가                            │
│                                                                              │
│  【코드 위치】                                                                │
│  ├─ Tab 1: PSD 파일 로드                                                     │
│  └─ Tab 2: PSD 설정 (q0, C(q0), H)                                           │
│                                                                              │
│  ※ 영향도: ★★★★★ (가장 큰 영향)                                              │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

┌─ 2. P(q) - 접촉 면적 비율 ───────────────────────────────────────────────────┐
│                                                                              │
│  P(q) = erf(1 / (2√G(q)))                                                    │
│                                                                              │
│  【μ 증가】 P(q) ↑ ← G(q) ↓                                                   │
│  ├─ G(q)가 작으면 → P(q) ≈ 1 (완전 접촉)                                     │
│  ├─ sigma_0 (압력) ↑ → G(q) ↓ → P(q) ↑ → μ ↑                                      │
│  │   (단, 분모의 sigma_0도 증가하므로 순효과 확인 필요)                            │
│  └─ |E*| ↓ (더 부드러운 재료) → G(q) ↓ → P(q) ↑                              │
│                                                                              │
│  【P(q) = 1로 단순화하면】                                                    │
│  일부 구현에서는 P(q)=1, S(q)=1로 가정 → μ가 더 높게 계산됨!                 │
│  → 이것이 0.43 vs 0.6 차이의 주요 원인일 수 있음                             │
│                                                                              │
│  ※ 영향도: ★★★★☆                                                             │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

┌─ 3. S(q) - 접촉 보정 인자 ───────────────────────────────────────────────────┐
│                                                                              │
│  S(q) = γ + (1-γ) × P(q)²                                                    │
│                                                                              │
│  【μ 증가】 S(q) ↑                                                            │
│  ├─ γ = 1.0 설정 → S(q) = 1 (항상 최대)                                      │
│  ├─ γ = 0.5 (기본값) → S(q) = 0.5 + 0.5×P²                                   │
│  └─ γ = 0 설정 → S(q) = P² (최소)                                            │
│                                                                              │
│  【S(q) = 1로 단순화하면】                                                    │
│  γ=1 또는 S(q)=1 가정 → μ 증가                                               │
│                                                                              │
│  【코드 위치】                                                                │
│  └─ persson_model/core/friction.py: gamma 파라미터                           │
│                                                                              │
│  ※ 영향도: ★★★☆☆                                                             │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

┌─ 4. Im[E(ω)] - 손실 탄성률 ──────────────────────────────────────────────────┐
│                                                                              │
│  ω = q × v × cosφ  (주파수 = 파수 × 속도)                                    │
│                                                                              │
│  【μ 증가】 Im[E(ω)] = E''(ω) ↑                                               │
│  ├─ E'' 피크가 높은 재료 → tan(δ)_max 큰 재료                                │
│  ├─ E'' 피크 위치가 qv 범위 내에 있어야 함                                   │
│  │   - v = 1 m/s, q = 10⁴ → ω ≈ 10⁴ rad/s → f ≈ 1600 Hz                     │
│  │   - v = 1 m/s, q = 10⁶ → ω ≈ 10⁶ rad/s → f ≈ 160 kHz                     │
│  └─ 마스터 커브가 이 주파수 범위를 포함해야 함                               │
│                                                                              │
│  【0.1~10 m/s 범위에서 μ 증가】                                               │
│  ├─ 이 속도 범위의 ω 범위: ~10² ~ 10⁸ rad/s                                  │
│  ├─ E'' 피크가 이 범위에 있으면 μ 증가                                       │
│  └─ 온도 ↓ → WLF shift → E'' 피크 이동 (고주파 쪽)                           │
│                                                                              │
│  【코드 위치】                                                                │
│  ├─ Tab 0: 마스터 커브 생성 (DMA 데이터)                                     │
│  └─ Tab 2: 온도 설정 (WLF shift)                                             │
│                                                                              │
│  ※ 영향도: ★★★★★ (C(q)와 함께 가장 큰 영향)                                   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

┌─ 5. sigma_0 - 공칭 압력 ──────────────────────────────────────────────────────────┐
│                                                                              │
│  μ_visc ∝ 1/sigma_0  (분모에 있음)                                                │
│                                                                              │
│  【μ 증가】 sigma_0 ↓                                                              │
│  ├─ 압력 감소 → μ 증가 (직접적)                                              │
│  ├─ 단, G(q) ∝ 1/sigma_0² → sigma_0↓ 시 G↑ → P↓ (간접적으로 μ 감소)                   │
│  └─ 순효과는 상황에 따라 다름                                                │
│                                                                              │
│  【코드 위치】                                                                │
│  └─ Tab 2: 공칭 압력 (MPa) 설정                                              │
│                                                                              │
│  ※ 영향도: ★★★☆☆                                                             │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

┌─ 6. q 적분 범위 (q0 ~ q1) ───────────────────────────────────────────────────┐
│                                                                              │
│  【μ 증가】 적분 범위 확대                                                    │
│  ├─ q1 ↑ → 더 미세한 거칠기 포함 → μ 증가                                    │
│  ├─ q0 ↓ → 더 큰 스케일 거칠기 포함                                          │
│  └─ q1은 h'rms(ξ) 목표값으로 결정                                            │
│                                                                              │
│  【주의】                                                                     │
│  └─ q1이 너무 크면 PSD 데이터 범위 초과 → 외삽 필요                          │
│                                                                              │
│  【코드 위치】                                                                │
│  └─ Tab 2: q_min, q_max 설정 또는 h'rms(ξ)/q1 모드                           │
│                                                                              │
│  ※ 영향도: ★★★☆☆                                                             │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘


【구현 차이로 인한 μ 값 차이 원인】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  이 프로그램 μ ≈ 0.43  vs  다른 구현 μ ≈ 0.6 차이 원인:

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ 1. P(q), S(q) 포함 여부                                                     │
  │    ├─ 이 프로그램: P(q) = erf(...), S(q) = γ + (1-γ)P² 적용                │
  │    └─ 단순 구현: P(q) = 1, S(q) = 1로 가정 → μ 더 높음 (약 1.3~1.5배)      │
  │                                                                             │
  │ 2. γ 값 차이                                                                │
  │    ├─ 이 프로그램: γ = 0.5 (기본값)                                         │
  │    └─ 다른 구현: γ = 1.0 → S(q) = 1                                         │
  │                                                                             │
  │ 3. G(q) 계산 방식                                                           │
  │    ├─ 누적 적분 vs 단순 공식                                                │
  │    └─ |E*|² 계산 시 평균 방식 차이                                          │
  │                                                                             │
  │ 4. 각도 적분 처리                                                           │
  │    ├─ 이 프로그램: ∫cosφ × Im[E(qv·cosφ)] dφ 정확히 계산                   │
  │    └─ 단순화: 평균값 근사 사용                                              │
  └─────────────────────────────────────────────────────────────────────────────┘


【0.1~10 m/s에서 μ 높이는 체크리스트】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  □ 1. PSD C(q) 증가
     ├─ C(q0) 값 증가
     ├─ Hurst 지수 H 감소 (0.8 → 0.6)
     └─ 더 거친 표면 데이터 사용

  □ 2. 재료 E'' 최적화
     ├─ tan(δ)_max가 큰 재료 선택
     ├─ E'' 피크가 계산 주파수 범위에 있는지 확인
     └─ 온도 조정으로 E'' 피크 이동

  □ 3. 압력 sigma_0 조정
     └─ sigma_0 감소 시도 (단, P(q) 영향 고려)

  □ 4. q 범위 확대
     └─ q1 (목표 h'rms 또는 직접 입력) 증가

  □ 5. γ 값 조정 (고급)
     └─ friction.py에서 gamma=1.0 으로 설정 시 S(q)=1

  □ 6. P(q)=1, S(q)=1 단순화 (비교용)
     └─ 다른 구현과 비교 시 이 가정 사용


【결론】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  μ_visc = f(C(q), P(q), S(q), E''(ω), sigma_0, q범위)

  • C(q)와 E''(ω)가 가장 큰 영향 (재료 및 표면 특성)
  • P(q), S(q) 적용 여부가 구현 간 차이의 주요 원인
  • 단순 구현 (P=1, S=1)은 μ를 과대평가할 수 있음

════════════════════════════════════════════════════════════════════════════════
"""

        text_widget = tk.Text(title_frame, wrap=tk.WORD, font=('Courier New', 15), height=50, width=90)
        text_widget.insert(tk.END, content)
        text_widget.config(state='disabled')  # Read-only
        text_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    # ===== 내장 데이터 관리 메서드들 =====

    def _get_preset_data_dir(self, data_type):
        """Get the preset data directory for a given data type."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        preset_dir = os.path.join(base_dir, 'preset_data', data_type)
        os.makedirs(preset_dir, exist_ok=True)
        return preset_dir

    def _refresh_preset_psd_list(self):
        """Refresh the preset PSD list in the combobox."""
        try:
            preset_dir = self._get_preset_data_dir('psd')
            files = [f for f in os.listdir(preset_dir) if f.endswith(('.txt', '.csv'))]
            if files:
                self.preset_psd_combo['values'] = files
            else:
                self.preset_psd_combo['values'] = ['(데이터 없음)']
        except Exception as e:
            print(f"[내장 PSD] 목록 로드 오류: {e}")
            self.preset_psd_combo['values'] = ['(오류)']

    def _load_preset_psd(self):
        """Load a preset PSD file."""
        selected = self.preset_psd_var.get()
        if not selected or selected.startswith('('):
            self._show_status("내장 PSD 파일을 선택하세요.", 'warning')
            return

        try:
            preset_dir = self._get_preset_data_dir('psd')
            filepath = os.path.join(preset_dir, selected)

            # 다중 인코딩 시도
            psd_data = None
            for enc in ['utf-8', 'cp949', 'euc-kr', 'latin-1']:
                try:
                    psd_data = np.loadtxt(filepath, comments='#', encoding=enc)
                    break
                except UnicodeDecodeError:
                    continue
            if psd_data is None:
                raise ValueError("파일 인코딩을 인식할 수 없습니다.")

            q_data = psd_data[:, 0]
            C_data = psd_data[:, 1]

            # Store for later use - psd_direct_data에도 저장
            self.psd_direct_data = {
                'q': q_data,
                'C_q': C_data,
                'filename': selected,
                'source': f'내장: {selected}'
            }

            self.psd_direct_info_var.set(f"내장 PSD: {selected} ({len(q_data)}pts)")
            self.apply_psd_type_var.set("direct")
            self.status_var.set(f"내장 PSD 로드 완료: {selected}")

            # 그래프에 표시
            self._plot_psd_direct_on_tab0()

            self._show_status(f"내장 PSD 로드 완료:\n{selected}", 'success')

        except Exception as e:
            messagebox.showerror("오류", f"내장 PSD 로드 실패:\n{str(e)}")

    def _delete_preset_psd(self):
        """Delete selected preset PSD file."""
        selected = self.preset_psd_var.get()
        if not selected or selected.startswith('('):
            self._show_status("삭제할 내장 PSD 파일을 선택하세요.", 'warning')
            return

        if not messagebox.askyesno("확인", f"'{selected}' 파일을 삭제하시겠습니까?"):
            return

        try:
            preset_dir = self._get_preset_data_dir('psd')
            filepath = os.path.join(preset_dir, selected)
            os.remove(filepath)
            self._refresh_preset_psd_list()
            self.preset_psd_var.set("(선택...)")
            self._show_status(f"삭제 완료: {selected}", 'success')
        except Exception as e:
            messagebox.showerror("오류", f"삭제 실패:\n{str(e)}")

    def _add_preset_psd(self):
        """Add current PSD to preset list."""
        # 현재 로드된 PSD가 있는지 확인 (psd_direct_data 또는 profile_psd_analyzer)
        psd_data = None
        source_name = "unknown"

        if hasattr(self, 'psd_direct_data') and self.psd_direct_data is not None:
            psd_data = self.psd_direct_data
            source_name = psd_data.get('filename', 'direct load')
        elif hasattr(self, 'profile_psd_analyzer') and self.profile_psd_analyzer is not None:
            if hasattr(self.profile_psd_analyzer, 'q') and self.profile_psd_analyzer.q is not None:
                psd_data = {
                    'q': self.profile_psd_analyzer.q,
                    'C_q': self.profile_psd_analyzer.C_q
                }
                source_name = "profile analysis"

        if psd_data is None:
            self._show_status("먼저 PSD 파일을 로드하세요.\n(Tab 0에서 PSD 직접 로드 또는 프로파일 분석)", 'warning')
            return

        # 파일 이름 입력 받기
        from tkinter import simpledialog
        name = simpledialog.askstring("내장 PSD 추가", "저장할 이름을 입력하세요:")
        if not name:
            return

        # 확장자 추가
        if not name.endswith('.txt'):
            name += '.txt'

        try:
            preset_dir = self._get_preset_data_dir('psd')
            filepath = os.path.join(preset_dir, name)

            # 데이터 저장 (키 이름 호환성 처리)
            q = psd_data.get('q', psd_data.get('q'))
            C = psd_data.get('C_q', psd_data.get('C', psd_data.get('C_q')))
            header = f"# 내장 PSD 데이터\n# 원본: {source_name}\n# q (1/m)\tC(q) (m^4)"
            np.savetxt(filepath, np.column_stack([q, C]), header=header, comments='', delimiter='\t')

            self._refresh_preset_psd_list()
            self._show_status(f"내장 PSD 추가 완료:\n{name}", 'success')

        except Exception as e:
            messagebox.showerror("오류", f"내장 PSD 추가 실패:\n{str(e)}")

    def _refresh_preset_mastercurve_list(self):
        """Refresh the preset master curve list in the combobox."""
        try:
            preset_dir = self._get_preset_data_dir('mastercurve')
            files = [f for f in os.listdir(preset_dir) if f.endswith(('.txt', '.csv'))]
            if files:
                self.preset_mc_combo['values'] = files
            else:
                self.preset_mc_combo['values'] = ['(데이터 없음)']
        except Exception as e:
            print(f"[내장 마스터커브] 목록 로드 오류: {e}")
            self.preset_mc_combo['values'] = ['(오류)']

    def _load_preset_mastercurve(self):
        """Load a preset master curve file."""
        selected = self.preset_mc_var.get()
        if not selected or selected.startswith('('):
            self._show_status("내장 마스터 커브 파일을 선택하세요.", 'warning')
            return

        try:
            preset_dir = self._get_preset_data_dir('mastercurve')
            filepath = os.path.join(preset_dir, selected)

            # 다중 인코딩 시도
            data = None
            for enc in ['utf-8', 'cp949', 'euc-kr', 'latin-1']:
                try:
                    data = np.loadtxt(filepath, comments='#', encoding=enc)
                    break
                except UnicodeDecodeError:
                    continue
            if data is None:
                raise ValueError("파일 인코딩을 인식할 수 없습니다.")

            freq = data[:, 0]
            E_storage = data[:, 1]
            E_loss = data[:, 2]
            omega = 2 * np.pi * freq

            # 저장
            self.persson_master_curve = {
                'freq': freq,
                'f': freq,
                'omega': omega,
                'E_storage': E_storage,
                'E_loss': E_loss,
                'filename': selected,
                'source': f'내장: {selected}'
            }

            self.mc_data_info_var.set(f"내장: {selected} ({len(freq)}pts)")
            self.status_var.set(f"내장 마스터 커브 로드 완료: {selected}")

            # 그래프에 표시
            self._plot_persson_master_curve()

            self._show_status(f"내장 마스터 커브 로드 완료:\n{selected}", 'success')

        except Exception as e:
            messagebox.showerror("오류", f"내장 마스터 커브 로드 실패:\n{str(e)}")

    def _delete_preset_mastercurve(self):
        """Delete selected preset master curve file."""
        selected = self.preset_mc_var.get()
        if not selected or selected.startswith('('):
            self._show_status("삭제할 내장 마스터 커브 파일을 선택하세요.", 'warning')
            return

        if not messagebox.askyesno("확인", f"'{selected}' 파일을 삭제하시겠습니까?"):
            return

        try:
            preset_dir = self._get_preset_data_dir('mastercurve')
            filepath = os.path.join(preset_dir, selected)
            os.remove(filepath)
            self._refresh_preset_mastercurve_list()
            self.preset_mc_var.set("(선택...)")
            self._show_status(f"삭제 완료: {selected}", 'success')
        except Exception as e:
            messagebox.showerror("오류", f"삭제 실패:\n{str(e)}")

    def _add_preset_mastercurve(self):
        """Add current master curve to preset list."""
        if not hasattr(self, 'persson_master_curve') or self.persson_master_curve is None:
            self._show_status("먼저 마스터 커브를 로드하세요.\n(Tab 1에서 Persson 정품 마스터 커브 로드)", 'warning')
            return

        from tkinter import simpledialog
        name = simpledialog.askstring("내장 마스터 커브 추가", "저장할 이름을 입력하세요:")
        if not name:
            return

        if not name.endswith('.txt'):
            name += '.txt'

        try:
            preset_dir = self._get_preset_data_dir('mastercurve')
            filepath = os.path.join(preset_dir, name)

            # 키 이름 호환성 처리: 'f' 또는 'freq' 키 모두 지원
            freq = self.persson_master_curve.get('freq', self.persson_master_curve.get('f'))
            E_storage = self.persson_master_curve['E_storage']
            E_loss = self.persson_master_curve['E_loss']
            source_name = self.persson_master_curve.get('source', self.persson_master_curve.get('filename', 'unknown'))
            header = f"# 내장 마스터 커브 데이터\n# 원본: {source_name}\n# freq (Hz)\tE' (Pa)\tE'' (Pa)"
            np.savetxt(filepath, np.column_stack([freq, E_storage, E_loss]), header=header, comments='', delimiter='\t')

            self._refresh_preset_mastercurve_list()
            self._show_status(f"내장 마스터 커브 추가 완료:\n{name}", 'success')

        except Exception as e:
            messagebox.showerror("오류", f"내장 마스터 커브 추가 실패:\n{str(e)}")

    def _refresh_preset_aT_list(self):
        """Refresh the preset aT shift factor list in the combobox."""
        try:
            preset_dir = self._get_preset_data_dir('aT')
            files = [f for f in os.listdir(preset_dir) if f.endswith(('.txt', '.csv'))]
            if files:
                self.preset_aT_combo['values'] = files
            else:
                self.preset_aT_combo['values'] = ['(데이터 없음)']
        except Exception as e:
            print(f"[내장 aT] 목록 로드 오류: {e}")
            self.preset_aT_combo['values'] = ['(오류)']

    def _load_preset_aT(self):
        """Load a preset aT shift factor file."""
        selected = self.preset_aT_var.get()
        if not selected or selected.startswith('('):
            self._show_status("내장 aT 시프트 팩터 파일을 선택하세요.", 'warning')
            return

        try:
            preset_dir = self._get_preset_data_dir('aT')
            filepath = os.path.join(preset_dir, selected)

            # 다중 인코딩 시도
            data = None
            for enc in ['utf-8', 'cp949', 'euc-kr', 'latin-1']:
                try:
                    data = np.loadtxt(filepath, comments='#', encoding=enc)
                    break
                except UnicodeDecodeError:
                    continue
            if data is None:
                raise ValueError("파일 인코딩을 인식할 수 없습니다.")

            T = data[:, 0]
            aT = data[:, 1]

            # Check for bT column
            has_bT = data.shape[1] >= 3
            if has_bT:
                bT = data[:, 2]
            else:
                bT = np.ones_like(T)

            # Compute log_aT
            log_aT = np.log10(np.maximum(aT, 1e-20))

            # Sort by temperature
            sort_idx = np.argsort(T)
            T = T[sort_idx]
            aT = aT[sort_idx]
            log_aT = log_aT[sort_idx]
            bT = bT[sort_idx]

            # Find reference temperature (where aT ≈ 1)
            ref_idx = np.argmin(np.abs(aT - 1.0))
            T_ref = T[ref_idx]

            # 저장 - persson_aT_data에도 저장
            self.persson_aT_data = {
                'T': T.copy(),
                'aT': aT.copy(),
                'log_aT': log_aT.copy(),
                'bT': bT.copy(),
                'T_ref': T_ref,
                'has_bT': has_bT,
                'filename': selected,
                'source': f'내장: {selected}'
            }

            # Create interpolation functions
            from scipy.interpolate import interp1d
            self.persson_aT_interp = interp1d(
                T, log_aT,
                kind='linear',
                bounds_error=False,
                fill_value='extrapolate'
            )
            self.persson_bT_interp = interp1d(
                T, bT,
                kind='linear',
                bounds_error=False,
                fill_value='extrapolate'
            )

            # Update info display
            bT_info = f", bT={bT.min():.2f}~{bT.max():.2f}" if has_bT else ""
            self.mc_aT_info_var.set(
                f"내장 aT: {selected} ({len(T)}pts, Tref={T_ref:.0f}°C{bT_info})"
            )

            # Plot aT on the master curve tab
            self._plot_persson_aT()

            self.status_var.set(f"내장 aT 시프트 팩터 로드 완료: {selected}")
            self._show_status(f"내장 aT 시프트 팩터 로드 완료:\n{selected}\n\n"
                f"데이터: {len(T)}pts, Tref={T_ref:.0f}°C\n"
                f"aT 범위: {aT.min():.2e} ~ {aT.max():.2e}", 'success')

        except Exception as e:
            import traceback
            messagebox.showerror("오류", f"내장 aT 시프트 팩터 로드 실패:\n{str(e)}\n\n{traceback.format_exc()}")

    def _delete_preset_aT(self):
        """Delete selected preset aT shift factor file."""
        selected = self.preset_aT_var.get()
        if not selected or selected.startswith('('):
            self._show_status("삭제할 내장 aT 파일을 선택하세요.", 'warning')
            return

        if not messagebox.askyesno("확인", f"'{selected}' 파일을 삭제하시겠습니까?"):
            return

        try:
            preset_dir = self._get_preset_data_dir('aT')
            filepath = os.path.join(preset_dir, selected)
            os.remove(filepath)
            self._refresh_preset_aT_list()
            self.preset_aT_var.set("(선택...)")
            self._show_status(f"삭제 완료: {selected}", 'success')
        except Exception as e:
            messagebox.showerror("오류", f"삭제 실패:\n{str(e)}")

    def _add_preset_aT(self):
        """Add current aT shift factor to preset list."""
        # persson_aT_data 또는 persson_aT 중 하나 확인
        aT_data = None
        if hasattr(self, 'persson_aT_data') and self.persson_aT_data is not None:
            aT_data = self.persson_aT_data
        elif hasattr(self, 'persson_aT') and self.persson_aT is not None:
            aT_data = self.persson_aT

        if aT_data is None:
            self._show_status("먼저 aT 시프트 팩터를 로드하세요.\n(Tab 1에서 aT 시프트 팩터 로드)", 'warning')
            return

        from tkinter import simpledialog
        name = simpledialog.askstring("내장 aT 추가", "저장할 이름을 입력하세요:")
        if not name:
            return

        if not name.endswith('.txt'):
            name += '.txt'

        try:
            preset_dir = self._get_preset_data_dir('aT')
            filepath = os.path.join(preset_dir, name)

            T = aT_data['T']
            aT = aT_data['aT']
            source_name = aT_data.get('source', aT_data.get('filename', 'unknown'))
            header = f"# 내장 aT 시프트 팩터 데이터\n# 원본: {source_name}\n# T (°C)\taT"
            np.savetxt(filepath, np.column_stack([T, aT]), header=header, comments='', delimiter='\t')

            self._refresh_preset_aT_list()
            self._show_status(f"내장 aT 시프트 팩터 추가 완료:\n{name}", 'success')

        except Exception as e:
            messagebox.showerror("오류", f"내장 aT 시프트 팩터 추가 실패:\n{str(e)}")


    # ===== Strain Sweep 내장 데이터 관리 =====

    def _refresh_preset_strain_sweep_list(self):
        """Refresh the preset Strain Sweep list in the combobox."""
        try:
            preset_dir = self._get_preset_data_dir('strain_sweep')
            files = [f for f in os.listdir(preset_dir)
                     if f.endswith(('.txt', '.csv', '.xlsx', '.xls'))]
            if files:
                self.preset_ss_combo['values'] = sorted(files)
            else:
                self.preset_ss_combo['values'] = ['(데이터 없음)']
        except Exception as e:
            print(f"[내장 Strain Sweep] 목록 로드 오류: {e}")
            self.preset_ss_combo['values'] = ['(오류)']

    def _load_preset_strain_sweep(self):
        """Load a preset Strain Sweep file."""
        selected = self.preset_ss_var.get()
        if not selected or selected.startswith('('):
            self._show_status("내장 Strain Sweep 파일을 선택하세요.", 'warning')
            return

        try:
            preset_dir = self._get_preset_data_dir('strain_sweep')
            filepath = os.path.join(preset_dir, selected)

            self.strain_data = load_strain_sweep_file(filepath)

            # Fallback: 이전 형식 (Temperature 마커 + 4열) 파일 처리
            if not self.strain_data:
                self.strain_data = self._parse_legacy_strain_sweep(filepath)

            if not self.strain_data:
                messagebox.showerror("오류", "유효한 데이터를 찾을 수 없습니다.")
                return

            # Update label
            self.strain_file_label.config(
                text=f"내장: {selected} ({len(self.strain_data)}개 온도)"
            )

            # Populate temperature listboxes
            temps = sorted(self.strain_data.keys())
            for lb in [self.temp_listbox_A, self.temp_listbox_B, self.temp_listbox]:
                lb.delete(0, tk.END)
                for T in temps:
                    lb.insert(tk.END, f"{T:.2f} °C")
                    lb.selection_set(tk.END)

            self.status_var.set(f"내장 Strain Sweep 로드 완료: {selected}")
            self._show_status(f"내장 Strain Sweep 로드 완료:\n{selected}\n온도 블록: {len(self.strain_data)}개", 'success')

        except Exception as e:
            messagebox.showerror("오류", f"내장 Strain Sweep 로드 실패:\n{str(e)}")

    def _parse_legacy_strain_sweep(self, filepath):
        """Parse legacy format strain sweep file (Temperature marker + 4-column data)."""
        import re
        float_re = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
        data_by_T = {}
        current_T = None

        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith('#'):
                        continue

                    # Check for "Temperature <value>" marker line
                    if line.lower().startswith('temperature'):
                        nums = float_re.findall(line)
                        if nums:
                            current_T = float(nums[0])
                        continue

                    # Skip header-like lines
                    if any(h in line.lower() for h in ['strain', 'frequency', 'freq', "e'"]):
                        continue

                    # Parse numeric data rows (4 columns: strain, freq, E', E'')
                    nums = float_re.findall(line)
                    if len(nums) >= 4 and current_T is not None:
                        strain_val = float(nums[0])
                        freq = float(nums[1])
                        ReE = float(nums[2])
                        ImE = float(nums[3])
                        data_by_T.setdefault(current_T, []).append((freq, strain_val, ReE, ImE))

            # Sort each temperature block by strain
            for T in list(data_by_T.keys()):
                rows = data_by_T[T]
                rows.sort(key=lambda x: x[1])
                data_by_T[T] = rows

        except Exception:
            pass

        return data_by_T

    def _delete_preset_strain_sweep(self):
        """Delete selected preset Strain Sweep file."""
        selected = self.preset_ss_var.get()
        if not selected or selected.startswith('('):
            self._show_status("삭제할 내장 Strain Sweep 파일을 선택하세요.", 'warning')
            return

        if not messagebox.askyesno("확인", f"'{selected}' 파일을 삭제하시겠습니까?"):
            return

        try:
            preset_dir = self._get_preset_data_dir('strain_sweep')
            filepath = os.path.join(preset_dir, selected)
            os.remove(filepath)
            self._refresh_preset_strain_sweep_list()
            self.preset_ss_var.set("(선택...)")
            self._show_status(f"삭제 완료: {selected}", 'success')
        except Exception as e:
            messagebox.showerror("오류", f"삭제 실패:\n{str(e)}")

    def _add_preset_strain_sweep(self):
        """Add current Strain Sweep file to preset list (copy original file)."""
        if not hasattr(self, 'strain_data') or self.strain_data is None:
            self._show_status("먼저 Strain Sweep 파일을 로드하세요.", 'warning')
            return

        from tkinter import simpledialog
        name = simpledialog.askstring("내장 Strain Sweep 추가", "저장할 이름을 입력하세요:")
        if not name:
            return

        if not name.endswith(('.txt', '.csv', '.xlsx', '.xls')):
            name += '.txt'

        try:
            preset_dir = self._get_preset_data_dir('strain_sweep')
            filepath = os.path.join(preset_dir, name)

            # Strain sweep 데이터를 로더가 읽을 수 있는 5열 형식으로 저장
            # 형식: T  freq  strain  ReE  ImE
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("# Strain Sweep 내장 데이터\n")
                f.write("# T(C)\tFreq(Hz)\tStrain\tE'(Pa)\tE''(Pa)\n")
                for T in sorted(self.strain_data.keys()):
                    block = self.strain_data[T]
                    # block is list of (freq, strain, ReE, ImE) tuples
                    if isinstance(block, list):
                        for row in block:
                            freq, strain, ReE, ImE = row[0], row[1], row[2], row[3]
                            f.write(f"{T}\t{freq}\t{strain}\t{ReE}\t{ImE}\n")
                    elif isinstance(block, dict):
                        # Legacy dict format fallback
                        strains = block.get('strain', [])
                        freqs = block.get('freq', [])
                        E_stor = block.get('E_storage', block.get('E_prime', []))
                        E_loss = block.get('E_loss', block.get('E_double_prime', []))
                        for k in range(len(strains)):
                            s = strains[k] if k < len(strains) else 0
                            freq = freqs[k] if k < len(freqs) else 0
                            es = E_stor[k] if k < len(E_stor) else 0
                            el = E_loss[k] if k < len(E_loss) else 0
                            f.write(f"{T}\t{freq}\t{s}\t{es}\t{el}\n")

            self._refresh_preset_strain_sweep_list()
            self._show_status(f"내장 Strain Sweep 추가 완료:\n{name}", 'success')

        except Exception as e:
            messagebox.showerror("오류", f"내장 Strain Sweep 추가 실패:\n{str(e)}")

    # ===== f,g 곡선 내장 데이터 관리 =====

    def _refresh_preset_fg_list(self):
        """Refresh the preset f,g curve list in the combobox."""
        try:
            preset_dir = self._get_preset_data_dir('fg_curve')
            files = [f for f in os.listdir(preset_dir) if f.endswith(('.txt', '.csv', '.dat'))]
            if files:
                self.preset_fg_combo['values'] = sorted(files)
            else:
                self.preset_fg_combo['values'] = ['(데이터 없음)']
        except Exception as e:
            print(f"[내장 f,g] 목록 로드 오류: {e}")
            self.preset_fg_combo['values'] = ['(오류)']

    def _load_preset_fg(self):
        """Load a preset f,g curve file."""
        selected = self.preset_fg_var.get()
        if not selected or selected.startswith('('):
            self._show_status("내장 f,g 곡선 파일을 선택하세요.", 'warning')
            return

        try:
            preset_dir = self._get_preset_data_dir('fg_curve')
            filepath = os.path.join(preset_dir, selected)

            # 다중 인코딩 시도
            fg_raw = None
            for enc in ['utf-8', 'cp949', 'euc-kr', 'latin-1']:
                try:
                    fg_raw = np.loadtxt(filepath, comments='#', encoding=enc)
                    break
                except UnicodeDecodeError:
                    continue
            if fg_raw is None:
                raise ValueError("파일 인코딩을 인식할 수 없습니다.")

            strain = fg_raw[:, 0]
            f_vals = fg_raw[:, 1]
            g_vals = fg_raw[:, 2] if fg_raw.shape[1] >= 3 else f_vals.copy()

            # Create interpolators
            self.f_interpolator, self.g_interpolator = create_fg_interpolator(
                strain, f_vals, g_vals
            )

            # Store for plotting
            self.fg_averaged = {
                'strain': strain,
                'f_avg': f_vals,
                'g_avg': g_vals
            }

            self.fg_file_label.config(text=f"내장: {selected} ({len(strain)}pts)")
            self._update_fg_plot()
            self.status_var.set(f"내장 f,g 곡선 로드 완료: {selected}")
            self._show_status(f"내장 f,g 곡선 로드 완료:\n{selected}", 'success')

        except Exception as e:
            messagebox.showerror("오류", f"내장 f,g 곡선 로드 실패:\n{str(e)}")

    def _delete_preset_fg(self):
        """Delete selected preset f,g curve file."""
        selected = self.preset_fg_var.get()
        if not selected or selected.startswith('('):
            self._show_status("삭제할 내장 f,g 곡선 파일을 선택하세요.", 'warning')
            return

        if not messagebox.askyesno("확인", f"'{selected}' 파일을 삭제하시겠습니까?"):
            return

        try:
            preset_dir = self._get_preset_data_dir('fg_curve')
            filepath = os.path.join(preset_dir, selected)
            os.remove(filepath)
            self._refresh_preset_fg_list()
            self.preset_fg_var.set("(선택...)")
            self._show_status(f"삭제 완료: {selected}", 'success')
        except Exception as e:
            messagebox.showerror("오류", f"삭제 실패:\n{str(e)}")

    def _add_preset_fg(self):
        """Add current f,g curve to preset list."""
        if not hasattr(self, 'fg_averaged') or self.fg_averaged is None:
            self._show_status("먼저 f,g 곡선을 계산하거나 로드하세요.", 'warning')
            return

        from tkinter import simpledialog
        name = simpledialog.askstring("내장 f,g 추가", "저장할 이름을 입력하세요:")
        if not name:
            return

        if not name.endswith('.txt'):
            name += '.txt'

        try:
            preset_dir = self._get_preset_data_dir('fg_curve')
            filepath = os.path.join(preset_dir, name)

            strain = self.fg_averaged['strain']
            f_avg = self.fg_averaged['f_avg']
            g_avg = self.fg_averaged['g_avg']

            header = "# 내장 f,g 곡선 데이터\n# strain\tf(strain)\tg(strain)"
            np.savetxt(filepath, np.column_stack([strain, f_avg, g_avg]),
                       header=header, comments='', delimiter='\t')

            self._refresh_preset_fg_list()
            self._show_status(f"내장 f,g 곡선 추가 완료:\n{name}", 'success')

        except Exception as e:
            messagebox.showerror("오류", f"내장 f,g 곡선 추가 실패:\n{str(e)}")


    # ================================================================
    # ====  점탄성 설계 (Viscoelastic Design Advisor) Tab  ====
    # ================================================================

    def _create_ve_advisor_tab(self, parent):
        """Create Viscoelastic Design Advisor tab."""
        main_container = ttk.Frame(parent)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # ── Left panel (scrollable controls, 600px) ──
        left_frame = ttk.Frame(main_container, width=getattr(self, '_left_panel_width', 600))
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        left_frame.pack_propagate(False)

        # Toolbar (fixed at top, always accessible)
        self._create_panel_toolbar(left_frame, buttons=[
            ("주파수 감도 분석 실행", self._run_ve_advisor_analysis, 'Accent.TButton'),
        ])

        left_canvas = tk.Canvas(left_frame, highlightthickness=0)
        left_scroll = ttk.Scrollbar(left_frame, orient='vertical', command=left_canvas.yview)
        left_panel = ttk.Frame(left_canvas)

        left_panel.bind("<Configure>",
                        lambda e: left_canvas.configure(scrollregion=left_canvas.bbox("all")))
        left_canvas.create_window((0, 0), window=left_panel, anchor="nw", width=getattr(self, '_left_panel_width', 600) - 20)
        left_canvas.configure(yscrollcommand=left_scroll.set)

        left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        left_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Mousewheel scroll
        def _on_mw(event):
            left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        def _on_mw_up(event):
            left_canvas.yview_scroll(-1, "units")
        def _on_mw_dn(event):
            left_canvas.yview_scroll(1, "units")

        def _bind_mw(widget):
            widget.bind("<MouseWheel>", _on_mw)
            widget.bind("<Button-4>", _on_mw_up)
            widget.bind("<Button-5>", _on_mw_dn)
            for child in widget.winfo_children():
                _bind_mw(child)
        left_panel.bind("<Map>", lambda e: _bind_mw(left_panel))

        # ── 1) 분석 설정 ──
        settings_frame = ttk.LabelFrame(left_panel, text="1) 분석 설정", padding=5)
        settings_frame.pack(fill=tk.X, pady=3, padx=3)

        # 속도 범위
        row = ttk.Frame(settings_frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="속도 범위:", font=self.FONTS['body']).pack(side=tk.LEFT)
        self.ve_v_min_var = tk.StringVar(value="0.01")
        self.ve_v_max_var = tk.StringVar(value="10.0")
        ttk.Entry(row, textvariable=self.ve_v_min_var, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(row, text="~", font=self.FONTS['body']).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.ve_v_max_var, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(row, text="m/s", font=self.FONTS['body']).pack(side=tk.LEFT)

        # 속도 포인트 수
        row2 = ttk.Frame(settings_frame)
        row2.pack(fill=tk.X, pady=2)
        ttk.Label(row2, text="속도 포인트:", font=self.FONTS['body']).pack(side=tk.LEFT)
        self.ve_n_v_var = tk.StringVar(value="30")
        ttk.Entry(row2, textvariable=self.ve_n_v_var, width=6).pack(side=tk.LEFT, padx=2)

        # 온도
        row3 = ttk.Frame(settings_frame)
        row3.pack(fill=tk.X, pady=2)
        ttk.Label(row3, text="온도:", font=self.FONTS['body']).pack(side=tk.LEFT)
        self.ve_temp_var = tk.StringVar(value="20.0")
        ttk.Entry(row3, textvariable=self.ve_temp_var, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(row3, text="\u00b0C", font=self.FONTS['body']).pack(side=tk.LEFT)

        # 압력
        row4 = ttk.Frame(settings_frame)
        row4.pack(fill=tk.X, pady=2)
        ttk.Label(row4, text="공칭 압력:", font=self.FONTS['body']).pack(side=tk.LEFT)
        self.ve_sigma_var = tk.StringVar(value="0.3")
        ttk.Entry(row4, textvariable=self.ve_sigma_var, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(row4, text="MPa", font=self.FONTS['body']).pack(side=tk.LEFT)

        # ── 2) 최적화 목표 ──
        opt_frame = ttk.LabelFrame(left_panel, text="2) 최적화 목표", padding=5)
        opt_frame.pack(fill=tk.X, pady=3, padx=3)

        self.ve_goal_var = tk.StringVar(value="maximize")
        ttk.Radiobutton(opt_frame, text="마찰 극대화 (그립 향상)",
                        variable=self.ve_goal_var, value="maximize").pack(anchor=tk.W)
        ttk.Radiobutton(opt_frame, text="마찰 극소화 (저마찰)",
                        variable=self.ve_goal_var, value="minimize").pack(anchor=tk.W)

        # E'' 최대 증가율
        row5 = ttk.Frame(opt_frame)
        row5.pack(fill=tk.X, pady=2)
        ttk.Label(row5, text="E'' 최대 변경 배율:", font=self.FONTS['body']).pack(side=tk.LEFT)
        self.ve_max_boost_var = tk.StringVar(value="3.0")
        ttk.Entry(row5, textvariable=self.ve_max_boost_var, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(row5, text="x (기존 대비)", font=self.FONTS['body']).pack(side=tk.LEFT)

        # ── 3) 분석 실행 ──
        btn_frame = ttk.Frame(left_panel)
        btn_frame.pack(fill=tk.X, pady=5, padx=3)

        self.ve_run_btn = ttk.Button(
            btn_frame, text="주파수 감도 분석 실행",
            command=self._run_ve_advisor_analysis,
            style='Accent.TButton')
        self.ve_run_btn.pack(fill=tk.X, pady=2)

        self.ve_progress_var = tk.DoubleVar(value=0)
        ttk.Progressbar(btn_frame, variable=self.ve_progress_var,
                        maximum=100).pack(fill=tk.X, pady=2)

        # ── 4) 주파수 감도 스펙트럼 설명 ──
        info_frame = ttk.LabelFrame(left_panel, text="주파수 감도 W(f) 란?", padding=5)
        info_frame.pack(fill=tk.X, pady=3, padx=3)
        info_text = (
            "Persson 마찰 적분에서:\n"
            "  \u03bc = \u222b W(f) \u00b7 E''(f) df\n\n"
            "W(f)는 '주파수 f에서 E''가 1 Pa 변할 때\n"
            "\u03bc가 얼마나 변하는가'를 나타냅니다.\n\n"
            "W(f)가 큰 대역 = 마찰에 가장 영향력 있는\n"
            "주파수 영역. 이 대역에서 E''를 조절하면\n"
            "마찰 계수를 효과적으로 변경할 수 있습니다.\n\n"
            "W(f)는 노면 PSD C(q), 접촉면적 P(q),\n"
            "속도, 파수(q)에 의해 결정됩니다."
        )
        ttk.Label(info_frame, text=info_text, font=('Segoe UI', 14),
                  foreground='#334155', wraplength=540,
                  justify=tk.LEFT).pack(anchor=tk.W)

        # ── 5) 결과 / 제안 ──
        result_frame = ttk.LabelFrame(left_panel, text="3) 분석 결과 / 제안", padding=5)
        result_frame.pack(fill=tk.X, pady=3, padx=3)

        self.ve_result_text = tk.Text(result_frame, height=22, width=60,
                                      font=('Consolas', 15), wrap=tk.WORD,
                                      bg='#F8FAFC', relief='solid', bd=1)
        ve_scroll_r = ttk.Scrollbar(result_frame, orient='vertical',
                                    command=self.ve_result_text.yview)
        self.ve_result_text.configure(yscrollcommand=ve_scroll_r.set)
        self.ve_result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ve_scroll_r.pack(side=tk.RIGHT, fill=tk.Y)

        # ── Right panel (plots, 2 rows x 3 cols) ──
        plot_frame = ttk.Frame(main_container)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig_ve_advisor = Figure(figsize=(14, 8), dpi=100)
        gs = self.fig_ve_advisor.add_gridspec(2, 3, hspace=0.38, wspace=0.35)

        self.ax_ve_Ep = self.fig_ve_advisor.add_subplot(gs[0, 0])
        self.ax_ve_Epp = self.fig_ve_advisor.add_subplot(gs[0, 1])
        self.ax_ve_sensitivity = self.fig_ve_advisor.add_subplot(gs[0, 2])
        self.ax_ve_mu_compare = self.fig_ve_advisor.add_subplot(gs[1, 0])
        self.ax_ve_temp_Ep = self.fig_ve_advisor.add_subplot(gs[1, 1])
        self.ax_ve_temp_Epp = self.fig_ve_advisor.add_subplot(gs[1, 2])

        # Initial axis setup
        for ax, title in [
            (self.ax_ve_Ep, "E'(f) 저장 탄성률"),
            (self.ax_ve_Epp, "E''(f) 손실 탄성률"),
            (self.ax_ve_sensitivity, "주파수 감도 W(f)"),
            (self.ax_ve_mu_compare, "\u03bc_visc 비교"),
            (self.ax_ve_temp_Ep, "E'(T) @10Hz"),
            (self.ax_ve_temp_Epp, "E''(T) @10Hz"),
        ]:
            ax.set_title(title, fontweight='bold', fontsize=13)
            ax.grid(True, alpha=0.3)

        # GridSpec already handles spacing via hspace/wspace, so skip tight_layout
        # to avoid "Axes not compatible with tight_layout" warning
        self.fig_ve_advisor.subplots_adjust(left=0.08, right=0.97, top=0.95, bottom=0.08)

        self.canvas_ve_advisor = FigureCanvasTkAgg(self.fig_ve_advisor, plot_frame)
        self.canvas_ve_advisor.draw()
        self.canvas_ve_advisor.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas_ve_advisor, plot_frame)
        toolbar.update()

    def _ve_get_loss_modulus_at_temp(self, omega, temperature):
        """Get E''(omega) at a specific temperature using aT shift.

        Uses persson_aT_interp if available, otherwise falls back to
        the current material (no shift).
        """
        if (hasattr(self, 'persson_aT_interp') and self.persson_aT_interp is not None
                and hasattr(self, 'persson_master_curve') and self.persson_master_curve is not None):
            mc = self.persson_master_curve
            T_ref = self.persson_aT_data.get('T_ref', 20.0)
            log_aT = float(self.persson_aT_interp(temperature))
            aT = 10**log_aT

            # Shifted frequency: omega_shifted = omega * aT
            omega_shifted = omega * aT

            # Interpolate from reference master curve
            from scipy.interpolate import interp1d
            log_om_ref = np.log10(mc['omega'])
            log_Epp_ref = np.log10(np.maximum(mc['E_loss'], 1e-3))
            interp_fn = interp1d(log_om_ref, log_Epp_ref, kind='linear',
                                 bounds_error=False, fill_value='extrapolate')
            return 10**interp_fn(np.log10(max(omega_shifted, 1e-6)))
        else:
            return self.material.get_loss_modulus(omega, temperature=temperature)

    def _ve_get_storage_modulus_at_temp(self, omega, temperature):
        """Get E'(omega) at a specific temperature using aT shift."""
        if (hasattr(self, 'persson_aT_interp') and self.persson_aT_interp is not None
                and hasattr(self, 'persson_master_curve') and self.persson_master_curve is not None):
            mc = self.persson_master_curve
            T_ref = self.persson_aT_data.get('T_ref', 20.0)
            log_aT = float(self.persson_aT_interp(temperature))
            aT = 10**log_aT

            omega_shifted = omega * aT

            from scipy.interpolate import interp1d
            log_om_ref = np.log10(mc['omega'])
            log_Ep_ref = np.log10(np.maximum(mc['E_storage'], 1e-3))
            interp_fn = interp1d(log_om_ref, log_Ep_ref, kind='linear',
                                 bounds_error=False, fill_value='extrapolate')
            return 10**interp_fn(np.log10(max(omega_shifted, 1e-6)))
        else:
            return self.material.get_storage_modulus(omega, temperature=temperature)

    def _run_ve_advisor_analysis(self):
        """Run frequency sensitivity analysis and suggest optimal E'(f), E''(f)."""
        import numpy as np
        from scipy.special import erf
        from scipy.integrate import simpson

        # ── Validate prerequisites ──
        if self.psd_model is None:
            self._show_status("PSD 데이터가 없습니다. Tab 0에서 PSD를 확정하세요.", 'warning')
            return
        if self.material is None:
            self._show_status("마스터 커브가 없습니다. Tab 1에서 확정하세요.", 'warning')
            return
        if not self.results or '2d_results' not in self.results:
            self._show_status("G(q,v) 결과가 없습니다. 탭 3에서 계산을 먼저 실행하세요.", 'warning')
            return

        try:
            self.ve_run_btn.config(state='disabled')
            self.ve_progress_var.set(0)
            self.status_var.set("점탄성 설계 분석 중...")
            self.root.update()

            # ── Read parameters ──
            v_min = float(self.ve_v_min_var.get())
            v_max = float(self.ve_v_max_var.get())
            n_v = int(self.ve_n_v_var.get())
            temperature = float(self.ve_temp_var.get())
            sigma_0 = float(self.ve_sigma_var.get()) * 1e6  # MPa -> Pa
            max_boost = float(self.ve_max_boost_var.get())
            goal = self.ve_goal_var.get()
            poisson = float(self.poisson_var.get())
            gamma = float(self.gamma_var.get())

            velocities = np.logspace(np.log10(v_min), np.log10(v_max), n_v)

            # ── Get G(q,v) data ──
            res2d = self.results['2d_results']
            q_arr = res2d['q']
            G_matrix = res2d['G_matrix']
            v_orig = res2d['v']

            C_q = self.psd_model(q_arr)
            prefactor = 1.0 / ((1 - poisson**2) * sigma_0)

            from scipy.interpolate import interp1d
            n_q = len(q_arr)

            log_v_orig = np.log10(v_orig)
            G_interp_list = []
            for iq in range(n_q):
                gi = interp1d(log_v_orig, G_matrix[iq, :], kind='linear',
                              bounds_error=False, fill_value='extrapolate')
                G_interp_list.append(gi)

            # ── Frequency sensitivity analysis ──
            n_phi = 14
            phi_arr = np.linspace(0, np.pi / 2, n_phi)
            cos_phi = np.cos(phi_arr)

            # Frequency range for histogram (in Hz)
            f_min_hz = q_arr[0] * v_min * 0.01 / (2 * np.pi)
            f_max_hz = q_arr[-1] * v_max * 1.1 / (2 * np.pi)
            n_freq_bins = 200
            log_f_edges = np.linspace(np.log10(max(f_min_hz, 1e-3)),
                                      np.log10(min(f_max_hz, 1e14)), n_freq_bins + 1)
            f_centers = 10**((log_f_edges[:-1] + log_f_edges[1:]) / 2)
            W_freq = np.zeros(n_freq_bins)

            mu_current = np.zeros(n_v)
            total_steps = n_v * n_q
            step = 0

            for iv, v_val in enumerate(velocities):
                for iq in range(n_q):
                    q = q_arr[iq]
                    Cq = C_q[iq]
                    G_val = max(G_interp_list[iq](np.log10(v_val)), 1e-20)

                    P_val = erf(1.0 / (2.0 * np.sqrt(G_val)))
                    S_val = gamma + (1 - gamma) * P_val**2
                    qCPS = q**3 * Cq * P_val * S_val

                    for ip in range(n_phi):
                        c = cos_phi[ip]
                        if c < 1e-10:
                            continue
                        omega = q * v_val * c
                        freq_hz = omega / (2 * np.pi)
                        if freq_hz < 1e-3:
                            continue

                        weight = 0.5 * qCPS * c * prefactor

                        log_f = np.log10(freq_hz)
                        bin_idx = int((log_f - log_f_edges[0]) /
                                      (log_f_edges[-1] - log_f_edges[0]) * n_freq_bins)
                        if 0 <= bin_idx < n_freq_bins:
                            W_freq[bin_idx] += weight / n_v

                    step += 1
                    if step % max(1, total_steps // 20) == 0:
                        self.ve_progress_var.set(step / total_steps * 70)
                        self.root.update()

                mu_current[iv] = self._compute_mu_for_velocity(
                    v_val, q_arr, C_q, G_interp_list, temperature,
                    sigma_0, poisson, gamma, n_phi)

            self.ve_progress_var.set(75)
            self.root.update()

            # ── Get current E' and E'' over frequency range (Hz) ──
            f_plot = f_centers
            omega_plot = 2 * np.pi * f_plot
            E_prime_current = np.array([
                self._ve_get_storage_modulus_at_temp(w, temperature)
                for w in omega_plot])
            E_loss_current = np.array([
                self._ve_get_loss_modulus_at_temp(w, temperature)
                for w in omega_plot])

            # ── Suggest optimal E'' ──
            W_max = np.max(W_freq) if np.max(W_freq) > 0 else 1.0
            W_norm = W_freq / W_max

            if goal == "maximize":
                boost_factor = 1.0 + (max_boost - 1.0) * W_norm
                E_loss_suggested = E_loss_current * boost_factor
                E_prime_suggested = E_prime_current.copy()
            else:
                reduce_factor = 1.0 / (1.0 + (max_boost - 1.0) * W_norm)
                E_loss_suggested = E_loss_current * reduce_factor
                E_prime_suggested = E_prime_current.copy()

            # ── Compute mu with suggested E'' ──
            log_f_sugg = np.log10(f_plot)
            log_Epp_sugg = np.log10(np.maximum(E_loss_suggested, 1e-3))
            suggested_interp = interp1d(log_f_sugg, log_Epp_sugg,
                                        kind='linear', bounds_error=False,
                                        fill_value='extrapolate')

            def suggested_loss_func(omega_val, T):
                f_hz = omega_val / (2 * np.pi)
                return 10**suggested_interp(np.log10(max(f_hz, 1e-3)))

            mu_suggested = np.zeros(n_v)
            for iv, v_val in enumerate(velocities):
                mu_suggested[iv] = self._compute_mu_for_velocity(
                    v_val, q_arr, C_q, G_interp_list, temperature,
                    sigma_0, poisson, gamma, n_phi,
                    loss_func_override=suggested_loss_func)

            self.ve_progress_var.set(90)
            self.root.update()

            # ── Temperature sweep at 10 Hz (using aT) ──
            temp_sweep_available = (
                hasattr(self, 'persson_aT_interp') and self.persson_aT_interp is not None
                and hasattr(self, 'persson_aT_data') and self.persson_aT_data is not None)

            T_sweep = None
            Ep_vs_T = None
            Epp_vs_T = None
            Ep_sugg_vs_T = None
            Epp_sugg_vs_T = None

            if temp_sweep_available:
                T_data = self.persson_aT_data['T']
                T_min_d, T_max_d = float(np.min(T_data)), float(np.max(T_data))
                T_sweep = np.linspace(T_min_d, T_max_d, 80)
                omega_10hz = 2 * np.pi * 10.0  # 10 Hz

                mc = self.persson_master_curve
                log_om_ref = np.log10(mc['omega'])
                log_Ep_ref = np.log10(np.maximum(mc['E_storage'], 1e-3))
                log_Epp_ref = np.log10(np.maximum(mc['E_loss'], 1e-3))
                Ep_interp_ref = interp1d(log_om_ref, log_Ep_ref, kind='linear',
                                         bounds_error=False, fill_value='extrapolate')
                Epp_interp_ref = interp1d(log_om_ref, log_Epp_ref, kind='linear',
                                          bounds_error=False, fill_value='extrapolate')

                Ep_vs_T = np.zeros(len(T_sweep))
                Epp_vs_T = np.zeros(len(T_sweep))
                Ep_sugg_vs_T = np.zeros(len(T_sweep))
                Epp_sugg_vs_T = np.zeros(len(T_sweep))

                # Suggested E'' interpolator (in omega space)
                log_om_sugg = np.log10(omega_plot)
                log_Epp_sugg_om = np.log10(np.maximum(E_loss_suggested, 1e-3))
                Epp_sugg_interp_om = interp1d(log_om_sugg, log_Epp_sugg_om,
                                              kind='linear', bounds_error=False,
                                              fill_value='extrapolate')

                for it, T_val in enumerate(T_sweep):
                    log_aT = float(self.persson_aT_interp(T_val))
                    aT = 10**log_aT
                    omega_shifted = omega_10hz * aT

                    Ep_vs_T[it] = 10**Ep_interp_ref(np.log10(max(omega_shifted, 1e-6)))
                    Epp_vs_T[it] = 10**Epp_interp_ref(np.log10(max(omega_shifted, 1e-6)))

                    # Suggested: use shifted omega to look up in suggested curve
                    Ep_sugg_vs_T[it] = Ep_vs_T[it]  # E' unchanged
                    Epp_sugg_vs_T[it] = 10**Epp_sugg_interp_om(
                        np.log10(max(omega_shifted, 1e-6)))

            self.ve_progress_var.set(95)
            self.root.update()

            # ── Find peak sensitivity frequency band ──
            peak_idx = np.argmax(W_freq)
            peak_freq_hz = f_centers[peak_idx]

            threshold = 0.5 * W_max
            band_mask = W_freq > threshold
            if np.any(band_mask):
                band_indices = np.where(band_mask)[0]
                freq_band_low = f_centers[band_indices[0]]
                freq_band_high = f_centers[band_indices[-1]]
            else:
                freq_band_low = peak_freq_hz * 0.1
                freq_band_high = peak_freq_hz * 10

            # ── Convert frequency band to temperature range via aT ──
            T_band_lo = None
            T_band_hi = None
            if temp_sweep_available and T_sweep is not None:
                # f_eff(T) = 10 Hz × aT(T)  =>  aT = f / 10
                log_aT_for_flo = np.log10(freq_band_low / 10.0)
                log_aT_for_fhi = np.log10(freq_band_high / 10.0)

                log_aT_vs_T = np.array([
                    float(self.persson_aT_interp(T_val))
                    for T_val in T_sweep])
                # Invert aT(T): given log_aT, find T
                sort_idx = np.argsort(log_aT_vs_T)
                T_from_logaT = interp1d(
                    log_aT_vs_T[sort_idx], T_sweep[sort_idx],
                    kind='linear', bounds_error=False, fill_value='extrapolate')
                # high freq → high aT → low T, low freq → low aT → high T
                T_band_lo = float(np.clip(T_from_logaT(log_aT_for_fhi),
                                          T_sweep[0], T_sweep[-1]))
                T_band_hi = float(np.clip(T_from_logaT(log_aT_for_flo),
                                          T_sweep[0], T_sweep[-1]))
                if T_band_lo > T_band_hi:
                    T_band_lo, T_band_hi = T_band_hi, T_band_lo

            # ── Compute improvement ──
            mu_curr_avg = np.mean(mu_current[mu_current > 0]) if np.any(mu_current > 0) else 0
            mu_sugg_avg = np.mean(mu_suggested[mu_suggested > 0]) if np.any(mu_suggested > 0) else 0
            improvement = ((mu_sugg_avg - mu_curr_avg) / mu_curr_avg * 100) if mu_curr_avg > 0 else 0

            # ── Update plots ──
            self._update_ve_advisor_plots(
                f_plot, E_prime_current, E_prime_suggested,
                E_loss_current, E_loss_suggested,
                f_centers, W_freq,
                velocities, mu_current, mu_suggested, goal,
                T_sweep, Ep_vs_T, Epp_vs_T, Ep_sugg_vs_T, Epp_sugg_vs_T,
                temperature, T_band_lo, T_band_hi)

            # ── Generate result text ──
            self.ve_result_text.delete(1.0, tk.END)
            txt = self.ve_result_text
            txt.insert(tk.END, "=" * 44 + "\n")
            txt.insert(tk.END, "  점탄성 설계 제안 결과\n")
            txt.insert(tk.END, "=" * 44 + "\n\n")

            aT_info = ""
            if temp_sweep_available:
                log_aT_val = float(self.persson_aT_interp(temperature))
                aT_val = 10**log_aT_val
                T_ref = self.persson_aT_data.get('T_ref', '?')
                aT_info = f"  aT({temperature}\u00b0C) = {aT_val:.3e}  (T_ref={T_ref}\u00b0C)\n"

            txt.insert(tk.END, f"분석 조건:\n")
            txt.insert(tk.END, f"  속도 범위: {v_min} ~ {v_max} m/s\n")
            txt.insert(tk.END, f"  온도: {temperature}\u00b0C\n")
            if aT_info:
                txt.insert(tk.END, aT_info)
            txt.insert(tk.END, f"  압력: {sigma_0/1e6:.2f} MPa\n")
            txt.insert(tk.END, f"  목표: {'마찰 극대화' if goal == 'maximize' else '마찰 극소화'}\n\n")

            txt.insert(tk.END, "\u2501" * 44 + "\n")
            txt.insert(tk.END, "  핵심 주파수 영역 (E''가 가장 중요한 대역)\n")
            txt.insert(tk.END, "\u2501" * 44 + "\n")
            txt.insert(tk.END, f"  피크 주파수: {peak_freq_hz:.1f} Hz\n")
            txt.insert(tk.END, f"  중요 대역: {freq_band_low:.1f} ~ "
                       f"{freq_band_high:.1f} Hz\n")
            if T_band_lo is not None and T_band_hi is not None:
                txt.insert(tk.END,
                    f"  참고 온도 범위 (@10Hz): "
                    f"{T_band_lo:.1f} ~ {T_band_hi:.1f}\u00b0C\n")
            txt.insert(tk.END, "\n")

            txt.insert(tk.END, "\u2501" * 44 + "\n")
            txt.insert(tk.END, "  설계 제안\n")
            txt.insert(tk.END, "\u2501" * 44 + "\n")

            if goal == "maximize":
                txt.insert(tk.END,
                    f"  \u25b6 {freq_band_low:.0f}~{freq_band_high:.0f} Hz 대역에서\n"
                    f"    E''(손실 탄성률)를 최대 {max_boost:.1f}x 증가시키세요.\n\n"
                    f"  \u25b6 이 대역은 노면 거칠기와 속도 범위가\n"
                    f"    만들어내는 핵심 여기(excitation) 주파수입니다.\n\n"
                    f"  \u25b6 tan\u03b4 = E''/E' 를 이 대역에서 높이면\n"
                    f"    에너지 손실(히스테리시스)이 증가하여\n"
                    f"    마찰 그립이 향상됩니다.\n\n")
            else:
                txt.insert(tk.END,
                    f"  \u25b6 {freq_band_low:.0f}~{freq_band_high:.0f} Hz 대역에서\n"
                    f"    E''(손실 탄성률)를 줄이세요.\n\n"
                    f"  \u25b6 이 대역의 히스테리시스 손실이\n"
                    f"    마찰의 주 원인입니다.\n\n"
                    f"  \u25b6 E'(저장 탄성률)를 유지하면서\n"
                    f"    tan\u03b4를 낮추는 것이 핵심입니다.\n\n")

            # Temperature advice
            if temp_sweep_available and Epp_vs_T is not None:
                txt.insert(tk.END, "\u2501" * 44 + "\n")
                txt.insert(tk.END, "  온도 영역 분석 (@10 Hz)\n")
                txt.insert(tk.END, "\u2501" * 44 + "\n")
                peak_T_idx = np.argmax(Epp_vs_T)
                peak_T = T_sweep[peak_T_idx]
                txt.insert(tk.END,
                    f"  E'' 피크 온도: {peak_T:.1f}\u00b0C (@10 Hz)\n"
                    f"  현재 온도({temperature}\u00b0C)에서 E''(10Hz):\n"
                    f"    {self._ve_get_loss_modulus_at_temp(2*np.pi*10, temperature):.2e} Pa\n\n"
                    f"  \u25b6 E'' 피크를 현재 사용 온도({temperature}\u00b0C)\n"
                    f"    부근으로 이동시키면 마찰 향상에 유리합니다.\n"
                    f"    (Tg 조절, 배합 변경 등)\n\n")

            txt.insert(tk.END, "\u2501" * 44 + "\n")
            txt.insert(tk.END, "  예상 효과\n")
            txt.insert(tk.END, "\u2501" * 44 + "\n")
            txt.insert(tk.END, f"  현재 평균 \u03bc: {mu_curr_avg:.4f}\n")
            txt.insert(tk.END, f"  제안 평균 \u03bc: {mu_sugg_avg:.4f}\n")
            txt.insert(tk.END, f"  변화율: {improvement:+.1f}%\n\n")

            # Per-velocity breakdown
            txt.insert(tk.END, "\u2501" * 44 + "\n")
            txt.insert(tk.END, "  속도별 \u03bc 비교\n")
            txt.insert(tk.END, "\u2501" * 44 + "\n")
            mu_sym = "\u03bc"
            hdr = f"  {'v (m/s)':>10} {f'현재 {mu_sym}':>10} {f'제안 {mu_sym}':>10} {'변화%':>8}"
            txt.insert(tk.END, hdr + "\n")
            txt.insert(tk.END, "  " + "-" * 40 + "\n")
            for iv in range(0, n_v, max(1, n_v // 10)):
                v_val = velocities[iv]
                mc = mu_current[iv]
                ms = mu_suggested[iv]
                ch = ((ms - mc) / mc * 100) if mc > 0 else 0
                txt.insert(tk.END, f"  {v_val:10.4f} {mc:10.4f} {ms:10.4f} {ch:+7.1f}%\n")

            self.ve_progress_var.set(100)
            self.status_var.set("점탄성 설계 분석 완료")

        except Exception as e:
            import traceback
            messagebox.showerror("오류", f"분석 중 오류:\n{str(e)}\n\n{traceback.format_exc()}")
        finally:
            self.ve_run_btn.config(state='normal')

    def _compute_mu_for_velocity(self, v_val, q_arr, C_q, G_interp_list,
                                  temperature, sigma_0, poisson, gamma, n_phi,
                                  loss_func_override=None):
        """Compute mu_visc for a single velocity using simplified Persson integral."""
        import numpy as np
        from scipy.special import erf
        from scipy.integrate import simpson

        prefactor = 1.0 / ((1 - poisson**2) * sigma_0)
        phi_arr = np.linspace(0, np.pi / 2, n_phi)
        cos_phi = np.cos(phi_arr)
        n_q = len(q_arr)

        integrand_q = np.zeros(n_q)

        for iq in range(n_q):
            q = q_arr[iq]
            Cq = C_q[iq]
            G_val = max(G_interp_list[iq](np.log10(v_val)), 1e-20)

            P_val = erf(1.0 / (2.0 * np.sqrt(G_val)))
            S_val = gamma + (1 - gamma) * P_val**2
            qCPS = q**3 * Cq * P_val * S_val

            angle_integrand = np.zeros(n_phi)
            for ip in range(n_phi):
                c = cos_phi[ip]
                if c < 1e-10:
                    continue
                omega = q * v_val * c
                if omega < 1e-3:
                    continue

                if loss_func_override is not None:
                    E_loss = loss_func_override(omega, temperature)
                else:
                    E_loss = self._ve_get_loss_modulus_at_temp(omega, temperature)

                angle_integrand[ip] = c * E_loss * prefactor

            angle_result = 4.0 * simpson(angle_integrand, x=phi_arr) if n_phi > 1 else 0
            integrand_q[iq] = qCPS * angle_result

        mu = 0.5 * simpson(integrand_q, x=q_arr) if n_q > 1 else 0
        return max(mu, 0)

    def _update_ve_advisor_plots(self, f_hz, Ep_curr, Ep_sugg,
                                  Epp_curr, Epp_sugg,
                                  f_w, W_freq,
                                  velocities, mu_curr, mu_sugg, goal,
                                  T_sweep, Ep_vs_T, Epp_vs_T,
                                  Ep_sugg_vs_T, Epp_sugg_vs_T,
                                  temperature,
                                  T_band_lo=None, T_band_hi=None):
        """Update the 6 plots (2x3) in the advisor tab."""

        # ── Plot 1: E'(f) comparison ──
        ax = self.ax_ve_Ep
        ax.clear()
        ax.loglog(f_hz, Ep_curr, 'b-', linewidth=2, label="현재 E'", alpha=0.8)
        ax.loglog(f_hz, Ep_sugg, 'r--', linewidth=2, label="제안 E'", alpha=0.8)
        ax.set_xlabel('f (Hz)')
        ax.set_ylabel("E' (Pa)")
        ax.set_title("E'(f) 저장 탄성률", fontweight='bold', fontsize=13)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)

        # ── Plot 2: E''(f) comparison ──
        ax = self.ax_ve_Epp
        ax.clear()
        ax.loglog(f_hz, Epp_curr, 'b-', linewidth=2, label="현재 E''", alpha=0.8)
        ax.loglog(f_hz, Epp_sugg, 'r--', linewidth=2, label="제안 E''", alpha=0.8)

        W_max = np.max(W_freq) if np.max(W_freq) > 0 else 1
        band_mask = W_freq > 0.5 * W_max
        if np.any(band_mask):
            band_indices = np.where(band_mask)[0]
            lo = f_w[band_indices[0]]
            hi = f_w[band_indices[-1]]
            ax.axvspan(lo, hi, alpha=0.15, color='orange', label='핵심 대역')

        ax.set_xlabel('f (Hz)')
        ax.set_ylabel("E'' (Pa)")
        ax.set_title("E''(f) 손실 탄성률", fontweight='bold', fontsize=13)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)

        # ── Plot 3: Frequency sensitivity W(f) ──
        ax = self.ax_ve_sensitivity
        ax.clear()
        ax.semilogx(f_w, W_freq, 'g-', linewidth=2)
        ax.fill_between(f_w, 0, W_freq, alpha=0.2, color='green')

        peak_idx = np.argmax(W_freq)
        if W_freq[peak_idx] > 0:
            peak_f = f_w[peak_idx]
            ax.axvline(peak_f, color='red', linestyle='--', alpha=0.7,
                       label=f'피크: {peak_f:.0f} Hz')

        ax.set_xlabel('f (Hz)')
        ax.set_ylabel('W(f) 감도 가중치')
        ax.set_title("주파수 감도 스펙트럼 W(f)", fontweight='bold', fontsize=13)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)

        # ── Plot 4: mu_visc comparison ──
        ax = self.ax_ve_mu_compare
        ax.clear()
        valid_c = mu_curr > 0
        valid_s = mu_sugg > 0

        ax.semilogx(velocities[valid_c], mu_curr[valid_c], 'b-o',
                     linewidth=2, markersize=3, label='현재 \u03bc')
        ax.semilogx(velocities[valid_s], mu_sugg[valid_s], 'r--s',
                     linewidth=2, markersize=3, label='제안 \u03bc')

        if np.any(valid_c & valid_s):
            v_both = velocities[valid_c & valid_s]
            mc_v = mu_curr[valid_c & valid_s]
            ms_v = mu_sugg[valid_c & valid_s]
            if goal == "maximize":
                ax.fill_between(v_both, mc_v, ms_v, where=(ms_v > mc_v),
                                alpha=0.15, color='green', label='향상 영역')
            else:
                ax.fill_between(v_both, mc_v, ms_v, where=(ms_v < mc_v),
                                alpha=0.15, color='green', label='감소 영역')

        ax.set_xlabel('v (m/s)')
        ax.set_ylabel('\u03bc_visc')
        ax.set_title("\u03bc_visc 현재 vs 제안", fontweight='bold', fontsize=13)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)

        # ── Plot 5: E'(T) @10Hz ──
        ax = self.ax_ve_temp_Ep
        ax.clear()
        if T_sweep is not None and Ep_vs_T is not None:
            ax.semilogy(T_sweep, Ep_vs_T, 'b-', linewidth=2, label="현재 E'")
            if Ep_sugg_vs_T is not None:
                ax.semilogy(T_sweep, Ep_sugg_vs_T, 'r--', linewidth=2, label="제안 E'")
            ax.axvline(temperature, color='gray', linestyle=':', alpha=0.8,
                       label=f'현재 T={temperature}\u00b0C')
            if T_band_lo is not None and T_band_hi is not None:
                ax.axvspan(T_band_lo, T_band_hi, alpha=0.15, color='orange',
                           label=f'핵심 대역 ({T_band_lo:.0f}~{T_band_hi:.0f}\u00b0C)')
            ax.set_xlabel('Temperature (\u00b0C)')
            ax.set_ylabel("E' (Pa)")
            ax.set_title("E'(T) @10 Hz", fontweight='bold', fontsize=13)
            ax.legend(fontsize=10, loc='best')
        else:
            ax.text(0.5, 0.5, 'aT 데이터 필요\n(마스터커브 탭에서 로드)',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=12, color='gray')
            ax.set_title("E'(T) @10 Hz", fontweight='bold', fontsize=13)
        ax.grid(True, alpha=0.3)

        # ── Plot 6: E''(T) @10Hz ──
        ax = self.ax_ve_temp_Epp
        ax.clear()
        if T_sweep is not None and Epp_vs_T is not None:
            ax.semilogy(T_sweep, Epp_vs_T, 'b-', linewidth=2, label="현재 E''")
            if Epp_sugg_vs_T is not None:
                ax.semilogy(T_sweep, Epp_sugg_vs_T, 'r--', linewidth=2, label="제안 E''")
            ax.axvline(temperature, color='gray', linestyle=':', alpha=0.8,
                       label=f'현재 T={temperature}\u00b0C')

            # Mark E'' peak temperature
            peak_T_idx = np.argmax(Epp_vs_T)
            peak_T = T_sweep[peak_T_idx]
            ax.axvline(peak_T, color='orange', linestyle='--', alpha=0.7,
                       label=f"E'' 피크: {peak_T:.0f}\u00b0C")

            # Mark critical temperature band from frequency spectrum
            if T_band_lo is not None and T_band_hi is not None:
                ax.axvspan(T_band_lo, T_band_hi, alpha=0.15, color='orange',
                           label=f'핵심 대역 ({T_band_lo:.0f}~{T_band_hi:.0f}\u00b0C)')

            ax.set_xlabel('Temperature (\u00b0C)')
            ax.set_ylabel("E'' (Pa)")
            ax.set_title("E''(T) @10 Hz", fontweight='bold', fontsize=13)
            ax.legend(fontsize=10, loc='best')
        else:
            ax.text(0.5, 0.5, 'aT 데이터 필요\n(마스터커브 탭에서 로드)',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=12, color='gray')
            ax.set_title("E''(T) @10 Hz", fontweight='bold', fontsize=13)
        ax.grid(True, alpha=0.3)

        self.fig_ve_advisor.subplots_adjust(left=0.08, right=0.97, top=0.95, bottom=0.08)
        self.canvas_ve_advisor.draw()


def _enable_dpi_awareness():
    """Enable High-DPI awareness BEFORE any window is created (Windows 10+).

    SetProcessDpiAwareness must be called before tk.Tk() or any GUI element
    is instantiated.  Value 1 = System DPI Aware, 2 = Per-Monitor DPI Aware.
    We use Per-Monitor V2 first (most accurate), falling back to System-aware.
    """
    if sys.platform != 'win32':
        return
    try:
        from ctypes import windll
        # Try Per-Monitor V2 awareness (Windows 10 1703+)
        try:
            windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            # Fallback to System DPI Aware
            try:
                windll.shcore.SetProcessDpiAwareness(1)
            except Exception:
                # Legacy fallback (Windows Vista+)
                windll.user32.SetProcessDPIAware()
    except Exception:
        pass


def _get_system_dpi_scale():
    """Return the Windows display scaling factor (e.g. 1.0, 1.25, 1.5, 2.0).

    Must be called AFTER _enable_dpi_awareness() and AFTER tk.Tk().
    """
    scale = 1.0
    if sys.platform != 'win32':
        return scale
    try:
        from ctypes import windll
        # GetDpiForSystem requires Windows 10 1607+
        try:
            dpi = windll.user32.GetDpiForSystem()
        except Exception:
            # Fallback: use device caps
            hdc = windll.user32.GetDC(0)
            dpi = windll.gdi32.GetDeviceCaps(hdc, 88)  # LOGPIXELSX
            windll.user32.ReleaseDC(0, hdc)
        scale = dpi / 96.0
    except Exception:
        pass
    return scale


def main():
    """Run the enhanced application."""
    # ── DPI awareness MUST be set BEFORE creating any window ──
    _enable_dpi_awareness()

    root = tk.Tk()

    # ── Detect system DPI scaling and adjust Tk scaling factor ──
    dpi_scale = _get_system_dpi_scale()
    # Tk internally uses a scaling factor (default ~1.33 on 96 DPI).
    # When the OS scale > 1.0, we need to compensate so that hardcoded
    # font sizes do not get double-scaled by both Windows and Tk.
    if dpi_scale > 1.05:
        # Reset Tk scaling to neutralise the OS-level magnification.
        # Default Tk scaling at 96 DPI ≈ 1.333;  at 144 DPI (150%) the OS
        # already enlarges everything, so we keep Tk at the 96-DPI baseline.
        root.tk.call('tk', 'scaling', 96.0 / 72.0)   # = 1.333 (96 DPI base)

    app = PerssonModelGUI_V2(root)
    root.mainloop()


if __name__ == "__main__":
    main()
