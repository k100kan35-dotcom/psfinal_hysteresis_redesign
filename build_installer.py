"""
Persson Friction Model - 설치 파일 빌드 스크립트
================================================
Windows 설치 프로그램(.exe installer)을 자동 생성합니다.

사용법:
    python build_installer.py

빌드 과정:
    1단계: PyInstaller로 앱을 dist/PerssonFrictionModel/ 폴더에 빌드 (onedir 모드)
    2단계: Inno Setup으로 설치 프로그램 생성

사전 설치:
    pip install pyinstaller numpy scipy matplotlib
    Inno Setup 6: https://jrsoftware.org/issetup.php
      (설치 시 "Install Inno Setup Preprocessor" 체크)
"""
import os
import sys
import shutil
import subprocess
import time
import glob

# =====================================================================
# 설정
# =====================================================================
APP_NAME = "NexenRubberFriction"
APP_VERSION = "1.1.0"
MAIN_SCRIPT = "main.py"
ISS_FILE = "installer.iss"
OUTPUT_DIR = "installer_output"

# PyInstaller에서 제외할 불필요 패키지
EXCLUDES = [
    'torch', 'torchvision', 'torchaudio',
    'tensorflow', 'keras',
    'numba', 'llvmlite',
    'tensorboard', 'tensorboardX',
    'onnx', 'onnxruntime',
    'xgboost', 'lightgbm', 'catboost',
    'IPython', 'jupyter', 'notebook',
    'pytest',
    # 불필요 matplotlib 백엔드
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

# hidden imports
HIDDEN_IMPORTS = [
    # matplotlib
    'matplotlib', 'matplotlib.pyplot',
    'matplotlib.backends.backend_tkagg',
    'matplotlib.figure', 'matplotlib.font_manager',
    'matplotlib.ft2font', 'matplotlib.mathtext',
    'matplotlib._mathtext', 'matplotlib.ticker',
    'matplotlib.colors', 'matplotlib.cm',
    'matplotlib.collections',
    # numpy/scipy
    'numpy', 'numpy.core',
    'scipy.integrate', 'scipy.interpolate',
    'scipy.optimize', 'scipy.signal', 'scipy.special',
    # pandas
    'pandas', 'pandas.core',
    # tkinter
    'tkinter', 'tkinter.ttk', 'tkinter.filedialog',
    'tkinter.messagebox', 'tkinter.simpledialog',
    # stdlib
    'platform', 'tempfile', 'csv', 're',
    # importlib
    'importlib_resources',
    # jaraco
    'pkg_resources', 'jaraco', 'jaraco.text',
    'jaraco.functools', 'jaraco.context',
    # persson_model
    'persson_model', 'persson_model.core',
    'persson_model.core.contact', 'persson_model.core.friction',
    'persson_model.core.g_calculator', 'persson_model.core.master_curve',
    'persson_model.core.psd_from_profile', 'persson_model.core.psd_models',
    'persson_model.core.viscoelastic',
    'persson_model.utils', 'persson_model.utils.data_loader',
    'persson_model.utils.numerical', 'persson_model.utils.output',
]


def find_inno_setup():
    """Inno Setup 컴파일러(iscc.exe) 경로를 찾습니다."""
    # 일반적인 설치 경로
    candidates = [
        r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
        r"C:\Program Files\Inno Setup 6\ISCC.exe",
        r"C:\Program Files (x86)\Inno Setup 5\ISCC.exe",
        r"C:\Program Files\Inno Setup 5\ISCC.exe",
    ]

    # PATH에서 찾기
    iscc = shutil.which("iscc") or shutil.which("ISCC")
    if iscc:
        return iscc

    for path in candidates:
        if os.path.isfile(path):
            return path

    return None


def kill_old_processes():
    """기존 실행 중인 프로세스 종료 (Windows)"""
    if sys.platform != 'win32':
        return

    exe_name = f"{APP_NAME}.exe"
    try:
        subprocess.run(
            ['taskkill', '/F', '/IM', exe_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(1)
    except FileNotFoundError:
        pass


def clean_build():
    """이전 빌드 산출물을 정리합니다."""
    for d in ['build', f'dist/{APP_NAME}']:
        if os.path.isdir(d):
            print(f"  [CLEAN] Removing {d}/")
            shutil.rmtree(d, ignore_errors=True)


def step1_pyinstaller():
    """1단계: PyInstaller로 앱 빌드 (onedir 모드)"""
    print()
    print("=" * 60)
    print("  Step 1: PyInstaller Build (onedir)")
    print("=" * 60)

    import PyInstaller.__main__

    sep = ';' if sys.platform == 'win32' else ':'

    args = [
        MAIN_SCRIPT,
        '--onedir',
        f'--name={APP_NAME}',
        '--clean',
        '--noconfirm',
        '--noconsole',
        '--log-level', 'WARN',
        '--icon=assets/app_icon.ico',

        # matplotlib 데이터 번들
        '--collect-data', 'matplotlib',
        '--collect-all', 'jaraco.text',
    ]

    # hidden imports
    for imp in HIDDEN_IMPORTS:
        args.extend(['--hidden-import', imp])

    # excludes
    for exc in EXCLUDES:
        args.extend(['--exclude-module', exc])

    # 데이터 디렉토리 포함
    if os.path.isdir('assets'):
        args.extend(['--add-data', f'assets{sep}assets'])

    args.extend(['--add-data', f'persson_model{sep}persson_model'])

    if os.path.isdir('reference_data'):
        args.extend(['--add-data', f'reference_data{sep}reference_data'])

    if os.path.isdir('preset_data'):
        args.extend(['--add-data', f'preset_data{sep}preset_data'])

    if os.path.isfile('strain.py'):
        args.extend(['--add-data', f'strain.py{sep}.'])

    # 빌드 실행
    PyInstaller.__main__.run(args)

    # 결과 확인
    dist_dir = os.path.join('dist', APP_NAME)
    exe_name = f'{APP_NAME}.exe' if sys.platform == 'win32' else APP_NAME
    exe_path = os.path.join(dist_dir, exe_name)

    if os.path.exists(exe_path):
        size_mb = os.path.getsize(exe_path) / (1024 * 1024)
        print(f"\n  [OK] PyInstaller build successful: {exe_path}")
        print(f"  [OK] EXE size: {size_mb:.1f} MB")

        # dist 폴더 전체 크기
        total_size = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, fns in os.walk(dist_dir)
            for f in fns
        )
        print(f"  [OK] Total dist size: {total_size / (1024 * 1024):.1f} MB")
        return True
    else:
        print(f"\n  [ERROR] EXE not found at {exe_path}")
        print("  Check the PyInstaller output above for errors.")
        return False


def step2_inno_setup():
    """2단계: Inno Setup으로 설치 프로그램 생성"""
    print()
    print("=" * 60)
    print("  Step 2: Inno Setup Installer")
    print("=" * 60)

    if sys.platform != 'win32':
        print()
        print("  [INFO] Inno Setup은 Windows에서만 실행 가능합니다.")
        print("  [INFO] Windows에서 다음 명령으로 설치 파일을 생성하세요:")
        print(f"         iscc {ISS_FILE}")
        print()
        print("  [INFO] 또는 Inno Setup GUI에서 installer.iss 파일을 열어 컴파일하세요.")
        return False

    iscc = find_inno_setup()
    if not iscc:
        print()
        print("  [WARN] Inno Setup이 설치되지 않았습니다.")
        print("  [WARN] 다운로드: https://jrsoftware.org/issetup.php")
        print()
        print("  Inno Setup 설치 후 다음 명령으로 설치 파일을 생성하세요:")
        print(f'    "{iscc or "iscc"}" {ISS_FILE}')
        print(f"  또는 Inno Setup GUI에서 {ISS_FILE} 파일을 열어 컴파일하세요.")
        return False

    # ISS 파일 존재 확인
    if not os.path.isfile(ISS_FILE):
        print(f"  [ERROR] {ISS_FILE} 파일을 찾을 수 없습니다.")
        return False

    print(f"  Using: {iscc}")
    print(f"  Script: {ISS_FILE}")
    print()

    # Inno Setup 컴파일 실행
    result = subprocess.run(
        [iscc, ISS_FILE],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        # 생성된 설치 파일 찾기
        setup_files = glob.glob(os.path.join(OUTPUT_DIR, "*.exe"))
        if setup_files:
            for sf in setup_files:
                size_mb = os.path.getsize(sf) / (1024 * 1024)
                print(f"\n  [OK] Installer created: {sf}")
                print(f"  [OK] Installer size: {size_mb:.1f} MB")
            return True
        else:
            print(f"\n  [OK] Inno Setup completed. Check {OUTPUT_DIR}/ folder.")
            return True
    else:
        print(f"\n  [ERROR] Inno Setup failed (exit code {result.returncode})")
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
        return False


def main():
    print("=" * 60)
    print(f"  {APP_NAME} v{APP_VERSION} - Installer Build")
    print("=" * 60)
    print(f"  Python: {sys.version}")
    print(f"  Platform: {sys.platform}")
    print(f"  Working Dir: {os.getcwd()}")

    # 프로젝트 루트 확인
    if not os.path.isfile(MAIN_SCRIPT):
        print(f"\n  [ERROR] {MAIN_SCRIPT} not found.")
        print("  This script must be run from the project root directory.")
        sys.exit(1)

    # 이전 프로세스 종료
    kill_old_processes()

    # 이전 빌드 정리
    clean_build()

    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1단계: PyInstaller
    if not step1_pyinstaller():
        print("\n  [ABORT] PyInstaller build failed. Cannot create installer.")
        sys.exit(1)

    # 2단계: Inno Setup
    step2_inno_setup()

    # 완료 요약
    print()
    print("=" * 60)
    print("  Build Summary")
    print("=" * 60)

    dist_dir = os.path.join('dist', APP_NAME)
    if os.path.isdir(dist_dir):
        print(f"  PyInstaller output : dist/{APP_NAME}/")

    setup_files = glob.glob(os.path.join(OUTPUT_DIR, "*.exe"))
    if setup_files:
        for sf in setup_files:
            print(f"  Installer          : {sf}")
    else:
        print(f"  Installer          : (Inno Setup 미실행 - Windows에서 생성 필요)")

    print()
    print("  설치 파일 생성 방법 (Windows):")
    print("    1. Inno Setup 6 설치: https://jrsoftware.org/issetup.php")
    print(f"    2. 명령 프롬프트에서: iscc {ISS_FILE}")
    print(f"    3. 또는 Inno Setup GUI에서 {ISS_FILE}을 열어 컴파일")
    print(f"    4. 결과물: {OUTPUT_DIR}/NexenRubberFriction_v{APP_VERSION}_Setup.exe")
    print("=" * 60)


if __name__ == '__main__':
    main()
