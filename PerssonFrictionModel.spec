# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files, collect_submodules
import os

# matplotlib 데이터 (폰트, 스타일, matplotlibrc)
mpl_datas = collect_data_files('matplotlib')

# matplotlib 서브모듈 (font_manager, ft2font, mathtext 등)
mpl_hiddenimports = collect_submodules('matplotlib')

# 프로젝트 데이터 디렉토리
project_datas = [
    ('persson_model', 'persson_model'),
    ('reference_data', 'reference_data'),
]
if os.path.isdir('preset_data'):
    project_datas.append(('preset_data', 'preset_data'))
if os.path.isfile('strain.py'):
    project_datas.append(('strain.py', '.'))

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=project_datas + mpl_datas,
    hiddenimports=mpl_hiddenimports + [
        # numpy/scipy
        'numpy', 'numpy.core',
        'scipy.integrate', 'scipy.interpolate', 'scipy.optimize',
        'scipy.signal', 'scipy.special',
        # pandas (DMA/PSD 파일 로딩)
        'pandas', 'pandas.core',
        # tkinter
        'tkinter', 'tkinter.ttk', 'tkinter.filedialog', 'tkinter.messagebox',
        # 한글 폰트 탐색
        'platform',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # 불필요 matplotlib 백엔드 (TkAgg만 사용)
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
        # 불필요 패키지
        'IPython', 'jupyter', 'notebook',
        'pytest', 'setuptools', 'pip', 'wheel',
        'pdb', 'doctest', 'pydoc', 'unittest', 'test',
        'lib2to3', 'ensurepip', 'idlelib', 'distutils',
        'curses',
        # 대형 ML/DL
        'torch', 'torchvision', 'torchaudio',
        'tensorflow', 'keras',
        'numba', 'llvmlite',
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='PerssonFrictionModel',
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    manifest='assets/dpi_aware.manifest',
)
