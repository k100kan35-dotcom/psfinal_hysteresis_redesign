# -*- mode: python ; coding: utf-8 -*-
"""
NexenRubberFriction PyInstaller spec file
build_installer.py / build_exe.py 기준으로 생성.

사용법:
  pyinstaller NexenRubberFriction.spec
"""

from PyInstaller.utils.hooks import collect_data_files, collect_all
import os

# ===== matplotlib 데이터 (폰트, 스타일, matplotlibrc) =====
mpl_datas = collect_data_files('matplotlib')

# ===== jaraco.text (namespace package) =====
jaraco_datas, jaraco_binaries, jaraco_hiddenimports = collect_all('jaraco.text')

# ===== 프로젝트 데이터 =====
sep = ';' if os.name == 'nt' else ':'
project_datas = [
    ('persson_model', 'persson_model'),
    ('reference_data', 'reference_data'),
]
if os.path.isdir('assets'):
    project_datas.append(('assets', 'assets'))
if os.path.isdir('preset_data'):
    project_datas.append(('preset_data', 'preset_data'))
if os.path.isfile('strain.py'):
    project_datas.append(('strain.py', '.'))
if os.path.isfile('braking_simulation.py'):
    project_datas.append(('braking_simulation.py', '.'))

# ===== hidden imports (build_installer.py와 동일) =====
hidden_imports = [
    # matplotlib
    'matplotlib', 'matplotlib.pyplot',
    'matplotlib.backends.backend_tkagg',
    'matplotlib.figure', 'matplotlib.font_manager',
    'matplotlib.ft2font', 'matplotlib.mathtext',
    'matplotlib._mathtext', 'matplotlib.ticker',
    'matplotlib.colors', 'matplotlib.cm',
    'matplotlib.collections',
    # numpy / scipy
    'numpy', 'numpy.core',
    'scipy.integrate', 'scipy.interpolate',
    'scipy.optimize', 'scipy.signal', 'scipy.special',
    'scipy.stats', 'scipy.stats.qmc',   # Sobol init for differential_evolution
    'scipy._lib', 'scipy._lib.messagestream',
    # pandas (DMA/PSD 파일 로딩)
    'pandas', 'pandas.core',
    # tkinter
    'tkinter', 'tkinter.ttk', 'tkinter.filedialog',
    'tkinter.messagebox', 'tkinter.simpledialog',
    # stdlib
    'platform', 'tempfile', 'csv', 're',
    # importlib
    'importlib_resources',
    # pkg_resources / jaraco
    'pkg_resources',
    'jaraco', 'jaraco.text',
    'jaraco.functools', 'jaraco.context',
    # persson_model 전체 서브모듈
    'persson_model', 'persson_model.core',
    'persson_model.core.contact',
    'persson_model.core.friction',
    'persson_model.core.g_calculator',
    'persson_model.core.master_curve',
    'persson_model.core.psd_from_profile',
    'persson_model.core.psd_models',
    'persson_model.core.viscoelastic',
    'persson_model.core.flash_temperature',
    'braking_simulation',
    'persson_model.utils',
    'persson_model.utils.data_loader',
    'persson_model.utils.numerical',
    'persson_model.utils.output',
]

# ===== 불필요 패키지 제외 =====
excludes = [
    # 대형 ML/DL
    'torch', 'torchvision', 'torchaudio',
    'tensorflow', 'keras',
    'numba', 'llvmlite',
    'tensorboard', 'tensorboardX',
    'onnx', 'onnxruntime',
    'xgboost', 'lightgbm', 'catboost',
    # 불필요 외부
    'IPython', 'jupyter', 'notebook', 'pytest',
    # 불필요 matplotlib 백엔드 (TkAgg만 사용)
    'matplotlib.tests',
    'matplotlib.backends.backend_qt5agg',
    'matplotlib.backends.backend_qt5',
    'matplotlib.backends.backend_qt',
    'matplotlib.backends.backend_qtagg',
    'matplotlib.backends.backend_gtk3',
    'matplotlib.backends.backend_gtk3agg',
    'matplotlib.backends.backend_gtk4',
    'matplotlib.backends.backend_gtk4agg',
    'matplotlib.backends.backend_wx',
    'matplotlib.backends.backend_wxagg',
    'matplotlib.backends.backend_webagg',
    'matplotlib.backends.backend_nbagg',
    'matplotlib.backends.backend_cairo',
    'matplotlib.backends.backend_macosx',
]

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=jaraco_binaries,
    datas=project_datas + mpl_datas + jaraco_datas,
    hiddenimports=hidden_imports + jaraco_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='NexenRubberFriction',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/app_icon.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='NexenRubberFriction',
)
