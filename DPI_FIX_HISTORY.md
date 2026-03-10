# DPI / 해상도 수정 이력

> 모든 시도는 **실패**했음. 다른 컴퓨터에서 빌드 후 테스트할 때마다 UI가 여전히 비정상적으로 커짐.

---

## 시도 1 — `c89f736` (2026-03-10 08:13)
**방법**: Tk scaling 보정 임계값 수정 (dpi > 1.05일 때 96/72로 고정)
- Tk 스케일링 보정이 dpi_scale >= 1.5 이상에서만 작동하던 것을 1.05 이상으로 변경
- 스플래시 화면 크기를 화면 너비의 42% 비례로 변경
- **결과: 실패** — UI 여전히 커짐

## 시도 2 — `7d017f1` (2026-03-10 09:02)
**방법**: "Chrome-like scaling" 전면 수정
- DPI manifest를 System DPI Aware로 설정
- `SetProcessDpiAwareness(1)` 호출
- Tk scaling을 96/72로 고정
- build_installer.py에 --manifest 옵션 추가
- font_settings.json 경로를 %APPDATA%로 변경
- **결과: 실패** — UI 여전히 커짐

## 시도 3 — `9094450` (2026-03-10 09:16)
**방법**: DPI 처리 완전 제거 — Windows DPI 가상화에 위임
- 시도 2의 접근이 다중 모니터에서 문제라고 판단
- `SetProcessDpiAwareness` 호출 제거, `_enable_dpi_awareness()` → no-op
- manifest를 `dpiAware=false`로 변경
- Tk scaling 강제 설정 제거
- **결과: 실패** — Windows가 앱 전체를 비트맵 업스케일링하여 모든 게 더 커짐

## 시도 4 — `8048c11` (2026-03-10 09:37)
**방법**: 다시 System DPI Aware로 전환
- manifest를 `dpiAware=true`로 되돌림
- `SetProcessDpiAwareness(1)` 다시 호출
- Tk scaling을 1.0으로 강제 (이전의 96/72가 아닌 1.0)
- **결과: 미확인** — 빌드 후 다른 컴퓨터에서 테스트 필요

---

## 핵심 문제
- 개발 환경에서 직접 테스트할 수 없음 (빌드 → 다른 PC 복사 → 설치 → 실행 필요)
- 시도 2와 시도 4는 거의 동일한 접근 (System DPI Aware + Tk scaling 고정)
- 시도 3은 정반대 접근 (DPI Unaware + Windows 가상화)
- 어느 쪽이든 실패

## 테스트 환경 정보
- 빌드: PyInstaller (`build_installer.py`)
- manifest 임베딩: `--manifest=assets/dpi_aware.manifest`
- 대상 OS: Windows (고DPI 모니터, 125% 이상 스케일링)
