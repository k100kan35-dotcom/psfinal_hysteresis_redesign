# 데이터 입력 탭 + 결과 탭 구현 계획

## 파일 구조
- **새 파일 1**: `data_input_tab.py` (monte_carlo_tab.py 패턴)
- **새 파일 2**: `results_overview_tab.py`
- **main.py 수정**: 탭 리스트 맨 앞에 2개 탭 추가 + 상태 변수 추가

## main.py 수정사항

### 1. import 추가 (line ~200)
```python
from data_input_tab import bind_data_input_tab
from results_overview_tab import bind_results_overview_tab
```

### 2. bind 호출 (line ~428, _create_main_layout 전)
```python
bind_data_input_tab(self)
bind_results_overview_tab(self)
```

### 3. tabs 리스트 맨 앞에 삽입 (line 945)
```python
('tab_data_input',       '데이터 입력',  self._create_data_input_tab),
('tab_results_overview', '계산 결과',    self._create_results_overview_tab),
```

### 4. __init__에 상태 변수 추가 (line ~357)
```python
self.compound_count = 1
self.compound_data = []          # list of CompoundData dicts
self.shared_psd_model = None
self.all_compound_results = []   # 계산 결과
self.data_input_finalized = False
```

---

## Tab 0: 데이터 입력 탭

### 레이아웃
```
┌─────────────────────────────────────────────────────────┐
│ [컴파운드 수: ___] [확인]   [○선형 ○비선형] [□Flash Temp]│
├─────────────────────────────────────────────────────────┤
│ ┌── 공통 PSD ─────────────────────────────────────────┐ │
│ │ [PSD 파일 로드] [노면 프로파일 로드]  ✓ loaded       │ │
│ └─────────────────────────────────────────────────────┘ │
│ ┌── Compound 1 ──┬── Compound 2 ──┬── Compound 3 ──┐  │
│ │ 이름: [R0    ] │ 이름: [R20   ] │ 이름: [R40   ] │  │
│ │ MC:  [선택] ✓  │ MC:  [선택] ✓  │ MC:  [선택] ✓  │  │
│ │ Str: [선택] ✓  │ Str: [선택] ✓  │ Str: [선택] ✓  │  │
│ │ μ_d: [선택] ✓  │ μ_d: [선택] ✓  │ μ_d: [선택] ✓  │  │
│ └────────────────┴────────────────┴────────────────┘  │
│                                                        │
│  σ₀: [0.3] MPa  T_ref: [20] °C  속도범위: [1e-4~10]   │
│         [ ★ 전체 계산 실행 ★ ]  진행률: 67%             │
└────────────────────────────────────────────────────────┘
```

### 주요 구현 포인트

1. **컴파운드 수 설정** (1~8, Spinbox + 확인)
   - 확인 시 _rebuild_compound_columns(n) 호출
   - 기존 열 삭제 후 n개 열 동적 생성

2. **공통 PSD** — 2가지 로드 방식
   - PSD CSV/TXT: `np.loadtxt` → `MeasuredPSD`
   - 노면 프로파일: `PSDComputer.load_profile()` → `compute_psd()`

3. **컴파운드별 입력**
   - 마스터커브(DMA): `persson_model.utils.data_loader.load_dma_from_file()`
     → `create_material_from_dma()` → `ViscoelasticMaterial`
   - Strain sweep: `load_strain_sweep_file()` → `compute_fg_from_strain_sweep()`
   - μ_dry CSV: (log10_v, mu_dry) 2열 → 보간 함수

4. **전체 계산** (threading.Thread)
   - 각 컴파운드에 대해:
     a. PSD + Material 설정
     b. GCalculator → G(q,v)
     c. FrictionCalculator → μ_hys(v)
     d. μ_adh(v) (μ_dry 있을 때만)
     e. μ_total = μ_hys + μ_adh
   - **핵심**: `GCalculator`와 `FrictionCalculator`를 직접 호출
     (main.py의 GUI 메서드 대신 코어 클래스 사용 → 부작용 없음)

### 주의: State 오염 방지
기존 코드는 `self.app.material` 하나를 공유하므로,
계산 시 직접 코어 클래스를 사용하고 app 상태를 변경하지 않음.
마지막 컴파운드의 material만 `self.app.material`에 설정.

---

## Tab 1: 결과 탭

### 4개 서브탭
```
┌─ μ_total ──┬─ 분리 플롯 ──┬─ 비교 ──┬─ 상세 ──┐
│ 모든 컴파  │ 3개 서브플롯 │ 컴파별  │ G(q),   │
│ 운드       │ μ_hys        │ 오버레이│ C(q),   │
│ μ_total    │ μ_adh        │ + 실측  │ h'rms   │
│ vs v       │ μ_total      │ 데이터  │ 등      │
└────────────┴──────────────┴─────────┴─────────┘
```

---

## 재사용할 핵심 모듈 (GUI 메서드가 아닌 코어 클래스)

| 기능 | 모듈/클래스 | 용도 |
|------|------------|------|
| PSD | `persson_model.core.psd_models.MeasuredPSD` | C(q) 보간 |
| 접촉역학 | `persson_model.core.g_calculator.GCalculator` | G(q,v) 계산 |
| 마찰 | `persson_model.core.friction.FrictionCalculator` | μ_hys 계산 |
| 재료 | `persson_model.core.material.ViscoelasticMaterial` | E*(ω) |
| 데이터 로드 | `persson_model.utils.data_loader` | DMA, strain, PSD 파일 |
| PSD 계산 | `monte_carlo_tab.PSDComputer` | 프로파일 → PSD |

## Monte Carlo 탭 연동
- 계산된 material 객체 → `mc_tab._compound_materials[i]`에 전달
- 공통 PSD → `mc_tab.psd_scans`에 추가
- `COMPOUND_NAMES`를 data_input_tab의 이름으로 업데이트
