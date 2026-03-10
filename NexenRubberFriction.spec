# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_all

datas = [('assets', 'assets'), ('persson_model', 'persson_model'), ('reference_data', 'reference_data'), ('preset_data', 'preset_data'), ('strain.py', '.'), ('braking_simulation.py', '.')]
binaries = []
hiddenimports = ['matplotlib', 'matplotlib.pyplot', 'matplotlib.backends.backend_tkagg', 'matplotlib.figure', 'matplotlib.font_manager', 'matplotlib.ft2font', 'matplotlib.mathtext', 'matplotlib._mathtext', 'matplotlib.ticker', 'matplotlib.colors', 'matplotlib.cm', 'matplotlib.collections', 'numpy', 'numpy.core', 'scipy.integrate', 'scipy.interpolate', 'scipy.optimize', 'scipy.signal', 'scipy.special', 'scipy.stats', 'scipy.stats.qmc', 'scipy._lib', 'scipy._lib.messagestream', 'pandas', 'pandas.core', 'tkinter', 'tkinter.ttk', 'tkinter.filedialog', 'tkinter.messagebox', 'tkinter.simpledialog', 'platform', 'tempfile', 'csv', 're', 'importlib_resources', 'pkg_resources', 'jaraco', 'jaraco.text', 'jaraco.functools', 'jaraco.context', 'persson_model', 'persson_model.core', 'persson_model.core.contact', 'persson_model.core.friction', 'persson_model.core.g_calculator', 'persson_model.core.master_curve', 'persson_model.core.psd_from_profile', 'persson_model.core.psd_models', 'persson_model.core.viscoelastic', 'persson_model.core.flash_temperature', 'braking_simulation', 'persson_model.utils', 'persson_model.utils.data_loader', 'persson_model.utils.numerical', 'persson_model.utils.output']
datas += collect_data_files('matplotlib')
tmp_ret = collect_all('jaraco.text')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch', 'torchvision', 'torchaudio', 'tensorflow', 'keras', 'numba', 'llvmlite', 'tensorboard', 'tensorboardX', 'onnx', 'onnxruntime', 'xgboost', 'lightgbm', 'catboost', 'IPython', 'jupyter', 'notebook', 'pytest', 'matplotlib.tests', 'matplotlib.backends.backend_qt5agg', 'matplotlib.backends.backend_qt5', 'matplotlib.backends.backend_qt', 'matplotlib.backends.backend_qtagg', 'matplotlib.backends.backend_gtk3', 'matplotlib.backends.backend_gtk3agg', 'matplotlib.backends.backend_gtk4', 'matplotlib.backends.backend_gtk4agg', 'matplotlib.backends.backend_wx', 'matplotlib.backends.backend_wxagg', 'matplotlib.backends.backend_webagg', 'matplotlib.backends.backend_nbagg', 'matplotlib.backends.backend_cairo', 'matplotlib.backends.backend_macosx'],
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
    icon=['assets\\app_icon.ico'],
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
