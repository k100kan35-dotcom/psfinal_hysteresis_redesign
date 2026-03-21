# 데이터 입력 탭 + 결과 탭 구현 계획

## 파일 구조
- **새 파일**: `data_input_tab.py` (monte_carlo_tab.py 패턴)
- **main.py 수정**: 탭 리스트 맨 앞에 2개 탭 추가

## Tab 0: 데이터 입력 탭

### 레이아웃
```
┌─────────────────────────────────────────────────────────┐
│ [컴파운드 수: ___] [확인]   [선형/비선형: ○선형 ○비선형]  │
│                            [Flash Temp: □ 적용]          │
├─────────────────────────────────────────────────────────┤
│ ┌── 공통 입력 ─────────────────────────────────────────┐│
│ │ 노면 PSD:  [파일 선택] ✓ loaded                      ││
│ └──────────────────────────────────────────────────────┘│
│ ┌── 컴파운드 1 (R0) ──┬── 컴파운드 2 (R20) ──┬─ ... ──┐│
│ │ 마스터커브: [선택] ✓ │ 마스터커브: [선택] ✓ │        ││
│ │ Strain:    [선택] ✓ │ Strain:    [선택] ✓ │        ││
│ │ μ_dry:     [선택] ✓ │ μ_dry:     [선택] ✓ │        ││
│ └──────────────────────┴──────────────────────┴────────┘│
│                                                         │
│          [ ★ 전체 계산 실행 ★ ]                          │
│          상태: 대기 중                                   │
└─────────────────────────────────────────────────────────┘
```

### 구현 세부

1. **컴파운드 수 설정** (1~8)
   - Spinbox + 확인 버튼
   - 확인 시 컴파운드 열 동적 생성/삭제
   - 컴파운드 이름 편집 가능 (기본: Compound 1, 2, ...)

2. **공통 PSD 입력**
   - "PSD 파일 로드 (CSV/TXT)" 버튼 → q, C(q) 2열
   - "노면 프로파일 로드" 버튼 → PSDComputer로 변환
   - 로드 상태 표시 (파일명, 포인트 수, q 범위)

3. **컴파운드별 입력** (각 열)
   - 마스터커브 DMA 파일 → main.py의 _load_multi_temp_dma 로직 재사용
   - Strain sweep 파일 → main.py의 _load_strain_sweep_data 로직 재사용
   - μ_dry 실측 CSV → main.py의 _load_measured_mu_dry_csv 로직 재사용
   - 각각 로드 상태 아이콘 (✓/✗)

4. **모델 옵션**
   - 선형/비선형 라디오버튼
   - Flash Temperature 체크박스
   - σ₀ (접촉 압력) 입력

5. **전체 계산** 버튼
   - 각 컴파운드에 대해 순차적으로:
     a. 마스터커브 생성 (WLF fitting)
     b. PSD 설정
     c. G(q) 계산
     d. h'rms / strain 계산
     e. μ_hys 계산 (선형 or 비선형)
     f. μ_adh 계산
     g. μ_total = μ_hys + μ_adh
   - 스레드로 실행, 진행률 표시
   - 결과를 Tab 1 (결과 탭)에 플롯

## Tab 1: 결과 탭

### 레이아웃 (4개 서브탭)
```
┌─ μ_total ──┬─ 분리 플롯 ──┬─ 비교 ──┬─ 상세 ──┐
│            │              │         │         │
│ 모든 컴파  │ 서브플롯 3개 │ 컴파별  │ G(q),   │
│ 운드의     │ μ_hys        │ 오버레이│ C(q),   │
│ μ_total    │ μ_adh        │ + 실측  │ strain  │
│ vs v       │ μ_total      │ 데이터  │ 등      │
└────────────┴──────────────┴─────────┴─────────┘
```

## 통합 포인트

### main.py 수정
1. tabs 리스트에 맨 앞 2개 추가:
   ```python
   ('tab_data_input', '데이터 입력', self._create_data_input_tab),
   ('tab_results_summary', '계산 결과', self._create_results_summary_tab),
   ```
2. `data_input_tab.py`에서 bind 패턴 사용 (monte_carlo_tab.py와 동일)

### Monte Carlo 탭 연동
- `data_input_tab`에서 계산된 material 객체들을 MC탭의 `_compound_materials`에 전달
- PSD도 공유 (self.app.psd_model)

### 기존 탭 활용
- 기존 탭들은 "고급 설정"으로 유지
- data_input_tab에서 로드한 데이터는 기존 탭에도 반영
  (self.app.material, self.app.psd_model 등 공유 변수 사용)

## 핵심 클래스 설계

```python
# data_input_tab.py

class CompoundData:
    """단일 컴파운드의 모든 데이터를 담는 컨테이너"""
    name: str
    material: object  # ViscoelasticMaterial
    strain_data: dict
    mu_dry_data: list  # [(v, mu), ...]
    results: dict  # μ_hys, μ_adh, μ_total vs v

class DataInputTab:
    def __init__(self, app):
        self.app = app
        self.compounds = []  # list of CompoundData
        self.psd_data = None  # (q, C) tuple

    def _create_tab(self, parent):
        # Tab 0: 데이터 입력
        # Tab 1: 결과 (self.app.tab_results_summary에 빌드)

    def _build_input_tab(self, parent): ...
    def _build_results_tab(self, parent): ...
    def _update_compound_columns(self, n): ...
    def _load_psd(self): ...
    def _load_dma_for_compound(self, idx): ...
    def _load_strain_for_compound(self, idx): ...
    def _load_mu_dry_for_compound(self, idx): ...
    def _run_all_calculations(self): ...  # threaded
    def _plot_results(self): ...
```

## 재사용할 main.py 메서드/로직
| 기능 | main.py 메서드 | 줄 번호 |
|------|---------------|---------|
| DMA 로드 | _load_multi_temp_dma | 3050 |
| 마스터커브 생성 | _generate_master_curve | 3900 |
| PSD 로드 | psd 관련 Tab0 로직 | 2500-2650 |
| Strain 로드 | _load_strain_sweep_data | 11528 |
| f(g) 계산 | _compute_fg_curves | 11640 |
| μ_dry 로드 | _load_measured_mu_dry_csv | 19898 |
| μ_visc 계산 | μ_visc 탭 계산 로직 | ~13000 |
| μ_adh 계산 | μ_adh 탭 계산 로직 | ~20000 |
