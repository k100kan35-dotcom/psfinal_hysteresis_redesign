"""
Vehicle Test Matching Tab — 실차 평가 결과 vs 마찰 맵 상관성 분석

실차 평가 항목(제동거리, 소음, 마모 등)의 인덱스를 기반으로
각 컴파운드별 Friction Map에서 추출한 mu 인덱스와의 상관성을 분석하는 탭.

핵심 원리:
  - 각 샘플(컴파운드)마다 저장된 마찰맵을 연결
  - 각 샘플마다 실차 평가 결과를 입력 (제동거리, 소음 등 — 측정 마찰계수 불필요)
  - 지정된 조건(T, p0, v, branch)에서 마찰맵으로부터 mu 값을 추출
  - 평가 결과를 인덱스화 → 마찰맵 mu 인덱스와 상관성 계산
  - 최적 상관성 조건을 자동 탐색

All methods are designed to be bound to the main PerssonModelGUI_V2 class.
Usage in main.py:
    from vehicle_test_matching import bind_vehicle_test_matching
    bind_vehicle_test_matching(PerssonModelGUI_V2)
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import json
import os


# ================================================================
# ==== Vehicle Test Matching (실차 매칭) Tab ====
# ================================================================

def _create_vehicle_test_matching_tab(self, parent):
    """실차 평가 결과와 마찰 맵 상관성 분석 탭."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

    layout = self._create_tab_layout(parent, toolbar_buttons=[
        ("상관성 분석 실행", self._run_vehicle_test_matching, 'Accent.TButton'),
        ("평가항목 관리", self._open_vtm_item_manager, 'TButton'),
        ("CSV 내보내기", self._export_vtm_results_csv, 'TButton'),
    ])
    left_panel = layout['content']

    # ── Internal state ──
    self._vtm_items = []        # list of dicts: {name, unit, direction, values:{sample:val}}
    self._vtm_samples = []      # sample names e.g. ["A", "B", "C"]
    self._vtm_sample_maps = {}  # {sample: friction_map_name} — 샘플별 마찰맵 매핑
    self._vtm_results = None    # analysis results
    self._vtm_threshold = 0.80  # correlation threshold
    self._vtm_map_combos = {}   # {sample: combo_widget} — 샘플별 마찰맵 콤보박스

    # ── 1) 실차 평가 항목 정의 ──
    sec1 = self._create_section(left_panel, "1) 평가 항목 관리")

    ttk.Label(sec1, text="실차 평가 항목을 추가/편집합니다.\n"
              "인덱스 방향: '작을수록 좋음' → 역방향 인덱싱",
              font=self.FONTS['small'], foreground='#64748B').pack(anchor='w')

    # Quick-add row
    row_add = ttk.Frame(sec1); row_add.pack(fill=tk.X, pady=(4, 2))
    ttk.Label(row_add, text="항목명:", font=self.FONTS['body']).pack(side=tk.LEFT)
    self._vtm_new_item_name_var = tk.StringVar()
    ttk.Entry(row_add, textvariable=self._vtm_new_item_name_var, width=14).pack(side=tk.LEFT, padx=2)

    ttk.Label(row_add, text="단위:", font=self.FONTS['body']).pack(side=tk.LEFT, padx=(4, 0))
    self._vtm_new_item_unit_var = tk.StringVar()
    ttk.Entry(row_add, textvariable=self._vtm_new_item_unit_var, width=6).pack(side=tk.LEFT, padx=2)

    row_add2 = ttk.Frame(sec1); row_add2.pack(fill=tk.X, pady=1)
    ttk.Label(row_add2, text="방향:", font=self.FONTS['body']).pack(side=tk.LEFT)
    self._vtm_new_item_dir_var = tk.StringVar(value="higher_better")
    ttk.Radiobutton(row_add2, text="클수록 좋음", variable=self._vtm_new_item_dir_var,
                    value="higher_better").pack(side=tk.LEFT, padx=2)
    ttk.Radiobutton(row_add2, text="작을수록 좋음", variable=self._vtm_new_item_dir_var,
                    value="lower_better").pack(side=tk.LEFT, padx=2)

    row_add3 = ttk.Frame(sec1); row_add3.pack(fill=tk.X, pady=2)
    ttk.Button(row_add3, text="+ 항목 추가", command=self._vtm_add_item,
               style='Accent.TButton').pack(side=tk.LEFT, padx=2)
    ttk.Button(row_add3, text="선택 삭제", command=self._vtm_remove_selected_item).pack(side=tk.LEFT, padx=2)

    # Items listbox
    self._vtm_items_listbox = tk.Listbox(sec1, height=5, font=('NanumGothicCoding', 11),
                                          selectmode=tk.SINGLE)
    self._vtm_items_listbox.pack(fill=tk.X, pady=2)

    # Preset buttons
    row_preset = ttk.Frame(sec1); row_preset.pack(fill=tk.X, pady=2)
    ttk.Button(row_preset, text="기본 프리셋 로드",
               command=self._vtm_load_default_preset).pack(side=tk.LEFT, padx=2)
    ttk.Button(row_preset, text="파일에서 로드",
               command=self._vtm_load_items_from_file).pack(side=tk.LEFT, padx=2)
    ttk.Button(row_preset, text="파일로 저장",
               command=self._vtm_save_items_to_file).pack(side=tk.LEFT, padx=2)

    # ── 2) 샘플 & 마찰맵 매핑 ──
    sec2 = self._create_section(left_panel, "2) 샘플 및 마찰맵 연결")

    ttk.Label(sec2, text="샘플(컴파운드) 이름, 쉼표 구분, 2개 이상:",
              font=self.FONTS['body']).pack(anchor='w')
    self._vtm_samples_var = tk.StringVar(value="A, B, C")
    ttk.Entry(sec2, textvariable=self._vtm_samples_var, width=30).pack(fill=tk.X, pady=2)

    ttk.Button(sec2, text="샘플 & 마찰맵 매핑 생성",
               command=self._vtm_create_sample_map_table,
               style='Accent.TButton').pack(anchor='w', pady=4)

    ttk.Label(sec2, text="※ 각 샘플에 저장소의 마찰맵을 연결하세요.\n"
              "   Friction Map 탭에서 '마찰맵 저장' 후 여기서 선택.\n"
              "   측정 마찰계수는 선택사항 (Traction 평가 시에만).",
              font=self.FONTS['small'], foreground='#0369A1').pack(anchor='w')

    # Sample-map assignment area
    self._vtm_sample_map_frame = ttk.Frame(sec2)
    self._vtm_sample_map_frame.pack(fill=tk.X, pady=2)

    # ── 3) 실차 데이터 입력 ──
    sec2b = self._create_section(left_panel, "3) 실차 평가 데이터 입력")

    ttk.Button(sec2b, text="데이터 입력 테이블 생성",
               command=self._vtm_create_data_table,
               style='Accent.TButton').pack(anchor='w', pady=4)

    ttk.Label(sec2b, text="※ 실차 평가 항목별 결과값을 입력합니다.\n"
              "   측정 마찰계수(mu) 행은 선택사항입니다 (비워도 됨).",
              font=self.FONTS['small'], foreground='#64748B').pack(anchor='w')

    # Data table area (scrollable)
    self._vtm_table_frame = ttk.Frame(sec2b)
    self._vtm_table_frame.pack(fill=tk.BOTH, expand=True, pady=2)
    self._vtm_entry_vars = {}   # (row_idx, sample_idx) → StringVar
    self._vtm_mu_entry_vars = {}  # sample_idx → StringVar (선택사항)

    # ── 4) 분석 설정 ──
    sec3 = self._create_section(left_panel, "4) 분석 설정")

    row_thr = ttk.Frame(sec3); row_thr.pack(fill=tk.X, pady=2)
    ttk.Label(row_thr, text="상관성 임계값:", font=self.FONTS['body']).pack(side=tk.LEFT)
    self._vtm_threshold_var = tk.StringVar(value="0.80")
    ttk.Entry(row_thr, textvariable=self._vtm_threshold_var, width=6).pack(side=tk.LEFT, padx=2)
    ttk.Label(row_thr, text="(0~1, |r| ≥ 값이면 결과에 포함)",
              font=self.FONTS['small'], foreground='#64748B').pack(side=tk.LEFT, padx=4)

    row_branch = ttk.Frame(sec3); row_branch.pack(fill=tk.X, pady=2)
    ttk.Label(row_branch, text="분석 Branch:", font=self.FONTS['body']).pack(side=tk.LEFT)
    self._vtm_branch_var = tk.StringVar(value="both")
    ttk.Radiobutton(row_branch, text="Cold", variable=self._vtm_branch_var,
                    value="cold").pack(side=tk.LEFT, padx=4)
    ttk.Radiobutton(row_branch, text="Hot", variable=self._vtm_branch_var,
                    value="hot").pack(side=tk.LEFT, padx=4)
    ttk.Radiobutton(row_branch, text="Both", variable=self._vtm_branch_var,
                    value="both").pack(side=tk.LEFT, padx=4)

    row_metric = ttk.Frame(sec3); row_metric.pack(fill=tk.X, pady=2)
    ttk.Label(row_metric, text="마찰 데이터:", font=self.FONTS['body']).pack(side=tk.LEFT)
    self._vtm_mu_type_var = tk.StringVar(value="mu_total")
    for label, val in [("mu_total", "mu_total"), ("mu_hys", "mu_visc"), ("mu_adh", "mu_adh")]:
        ttk.Radiobutton(row_metric, text=label, variable=self._vtm_mu_type_var,
                        value=val).pack(side=tk.LEFT, padx=3)

    row_idx_base = ttk.Frame(sec3); row_idx_base.pack(fill=tk.X, pady=2)
    ttk.Label(row_idx_base, text="인덱스 기준:", font=self.FONTS['body']).pack(side=tk.LEFT)
    self._vtm_index_base_var = tk.StringVar(value="first")
    ttk.Label(row_idx_base, text="첫 번째 샘플(A) = 100",
              font=self.FONTS['body'], foreground='#0369A1').pack(side=tk.LEFT, padx=4)

    # ── 탐색 조건 설정 ──
    sec3b = self._create_section(left_panel, "4-1) 맵 탐색 조건 (고정값)")
    ttk.Label(sec3b, text="마찰맵에서 mu 추출 시 고정할 조건을 지정합니다.\n"
              "비워두면 자동으로 최적 조건을 탐색합니다.",
              font=self.FONTS['small'], foreground='#64748B').pack(anchor='w')

    row_fix_T = ttk.Frame(sec3b); row_fix_T.pack(fill=tk.X, pady=1)
    ttk.Label(row_fix_T, text="온도 T (°C):", font=self.FONTS['body']).pack(side=tk.LEFT)
    self._vtm_fix_T_var = tk.StringVar(value="")
    ttk.Entry(row_fix_T, textvariable=self._vtm_fix_T_var, width=8).pack(side=tk.LEFT, padx=2)
    ttk.Label(row_fix_T, text="(비우면 자동탐색)", font=self.FONTS['small'],
              foreground='#64748B').pack(side=tk.LEFT)

    row_fix_p = ttk.Frame(sec3b); row_fix_p.pack(fill=tk.X, pady=1)
    ttk.Label(row_fix_p, text="압력 p₀ (MPa):", font=self.FONTS['body']).pack(side=tk.LEFT)
    self._vtm_fix_p0_var = tk.StringVar(value="")
    ttk.Entry(row_fix_p, textvariable=self._vtm_fix_p0_var, width=8).pack(side=tk.LEFT, padx=2)
    ttk.Label(row_fix_p, text="(비우면 자동탐색)", font=self.FONTS['small'],
              foreground='#64748B').pack(side=tk.LEFT)

    row_fix_v = ttk.Frame(sec3b); row_fix_v.pack(fill=tk.X, pady=1)
    ttk.Label(row_fix_v, text="속도 v (m/s):", font=self.FONTS['body']).pack(side=tk.LEFT)
    self._vtm_fix_v_var = tk.StringVar(value="")
    ttk.Entry(row_fix_v, textvariable=self._vtm_fix_v_var, width=8).pack(side=tk.LEFT, padx=2)
    ttk.Label(row_fix_v, text="(비우면 자동탐색)", font=self.FONTS['small'],
              foreground='#64748B').pack(side=tk.LEFT)

    # ── 5) 결과 요약 ──
    sec4 = self._create_section(left_panel, "5) 결과 요약")
    self._vtm_result_text = tk.Text(sec4, height=15, font=('NanumGothicCoding', 11),
                                     wrap=tk.WORD, state=tk.DISABLED)
    self._vtm_result_text.pack(fill=tk.BOTH, expand=True, pady=2)

    # ── Right panel: plots ──
    right_panel = layout.get('right')
    self._vtm_right_notebook = ttk.Notebook(right_panel)
    self._vtm_right_notebook.pack(fill=tk.BOTH, expand=True)

    # Tab 1: Correlation Heatmap
    tab_heatmap = ttk.Frame(self._vtm_right_notebook)
    self._vtm_right_notebook.add(tab_heatmap, text='  상관성 히트맵  ')

    fig_hm = Figure(figsize=(10, 7), dpi=100)
    self._vtm_fig_heatmap = fig_hm
    canvas_hm = FigureCanvasTkAgg(fig_hm, master=tab_heatmap)
    canvas_hm.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    self._vtm_canvas_heatmap = canvas_hm
    toolbar_hm = NavigationToolbar2Tk(canvas_hm, tab_heatmap)
    toolbar_hm.update()

    # Tab 2: Index comparison chart
    tab_idx = ttk.Frame(self._vtm_right_notebook)
    self._vtm_right_notebook.add(tab_idx, text='  인덱스 비교  ')

    fig_idx = Figure(figsize=(10, 7), dpi=100)
    self._vtm_fig_index = fig_idx
    canvas_idx = FigureCanvasTkAgg(fig_idx, master=tab_idx)
    canvas_idx.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    self._vtm_canvas_index = canvas_idx
    toolbar_idx = NavigationToolbar2Tk(canvas_idx, tab_idx)
    toolbar_idx.update()

    # Tab 3: Best-match detail
    tab_detail = ttk.Frame(self._vtm_right_notebook)
    self._vtm_right_notebook.add(tab_detail, text='  매칭 상세  ')

    fig_detail = Figure(figsize=(10, 7), dpi=100)
    self._vtm_fig_detail = fig_detail
    canvas_detail = FigureCanvasTkAgg(fig_detail, master=tab_detail)
    canvas_detail.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    self._vtm_canvas_detail = canvas_detail
    toolbar_detail = NavigationToolbar2Tk(canvas_detail, tab_detail)
    toolbar_detail.update()

    # Tab 4: Friction Map 조건 탐색 — 3D 시각화 (스크롤 지원, 대형 플롯)
    tab_map = ttk.Frame(self._vtm_right_notebook)
    self._vtm_right_notebook.add(tab_map, text='  맵 조건 탐색  ')

    # 옵션 바: 슬라이스 축 선택
    opt_bar = ttk.Frame(tab_map)
    opt_bar.pack(fill=tk.X, padx=4, pady=2)
    ttk.Label(opt_bar, text="슬라이스 축:").pack(side=tk.LEFT)
    self._vtm_map_slice_var = tk.StringVar(value="pressure")
    for txt, val in [("압력(p₀)별", "pressure"), ("온도(T)별", "temperature"), ("속도(v)별", "velocity")]:
        ttk.Radiobutton(opt_bar, text=txt, variable=self._vtm_map_slice_var,
                        value=val).pack(side=tk.LEFT, padx=3)
    ttk.Label(opt_bar, text=" | Branch:").pack(side=tk.LEFT, padx=(8, 2))
    self._vtm_map_branch_var = tk.StringVar(value="cold")
    for txt, val in [("Cold", "cold"), ("Hot", "hot")]:
        ttk.Radiobutton(opt_bar, text=txt, variable=self._vtm_map_branch_var,
                        value=val).pack(side=tk.LEFT, padx=2)
    ttk.Button(opt_bar, text="새로고침",
               command=lambda: self._vtm_plot_map_conditions(self._vtm_results) if self._vtm_results else None
               ).pack(side=tk.LEFT, padx=8)

    # 동적 플롯 영역 (스크롤 지원) — 플롯은 _vtm_plot_map_conditions에서 생성
    self._vtm_map_plot_frame = ttk.Frame(tab_map)
    self._vtm_map_plot_frame.pack(fill=tk.BOTH, expand=True)


# ── Sample-Map assignment ──

def _vtm_create_sample_map_table(self):
    """Create sample ↔ friction map assignment table."""
    samples_str = self._vtm_samples_var.get().strip()
    samples = [s.strip() for s in samples_str.split(',') if s.strip()]
    if len(samples) < 2:
        messagebox.showwarning("샘플 설정", "샘플을 2개 이상 입력하세요 (쉼표 구분).")
        return

    self._vtm_samples = samples

    # Clear previous
    for w in self._vtm_sample_map_frame.winfo_children():
        w.destroy()
    self._vtm_map_combos = {}
    self._vtm_sample_map_vars = {}

    # Get available map names
    map_names = [entry['name'] for entry in self._friction_map_store]

    if not map_names:
        ttk.Label(self._vtm_sample_map_frame,
                  text="⚠ 저장된 마찰맵이 없습니다.\nFriction Map 탭에서 '마찰맵 저장' 버튼으로 먼저 저장하세요.",
                  font=self.FONTS['small'], foreground='#DC2626').pack(anchor='w', pady=4)
        return

    # Header
    ttk.Label(self._vtm_sample_map_frame, text="샘플 ↔ 마찰맵 연결:",
              font=('', 10, 'bold')).pack(anchor='w', pady=(2, 4))

    for sample in samples:
        row = ttk.Frame(self._vtm_sample_map_frame)
        row.pack(fill=tk.X, pady=1)

        ttk.Label(row, text=f"{sample}:", font=self.FONTS['body'], width=8).pack(side=tk.LEFT)

        var = tk.StringVar(value=self._vtm_sample_maps.get(sample, ''))
        self._vtm_sample_map_vars[sample] = var
        combo = ttk.Combobox(row, textvariable=var, width=24,
                             values=map_names, state='readonly')
        combo.pack(side=tk.LEFT, padx=2)
        self._vtm_map_combos[sample] = combo

        # Auto-select if only one or if previously set
        if len(map_names) >= 1 and not var.get():
            # Try to match by sample name
            matched = False
            for mn in map_names:
                if sample.lower() in mn.lower():
                    var.set(mn)
                    matched = True
                    break
            if not matched and len(map_names) == 1:
                var.set(map_names[0])

    self._show_status(f"{len(samples)}개 샘플의 마찰맵 연결 테이블 생성", 'success')


# ── Item management ──

def _vtm_add_item(self):
    """Add a new evaluation item."""
    name = self._vtm_new_item_name_var.get().strip()
    unit = self._vtm_new_item_unit_var.get().strip()
    direction = self._vtm_new_item_dir_var.get()
    if not name:
        messagebox.showwarning("항목 추가", "항목명을 입력하세요.")
        return
    for item in self._vtm_items:
        if item['name'] == name:
            messagebox.showwarning("항목 추가", f"'{name}' 항목이 이미 존재합니다.")
            return
    self._vtm_items.append({
        'name': name,
        'unit': unit or '-',
        'direction': direction,
        'values': {},
    })
    self._vtm_refresh_items_listbox()
    self._vtm_new_item_name_var.set("")
    self._vtm_new_item_unit_var.set("")


def _vtm_remove_selected_item(self):
    """Remove the selected evaluation item."""
    sel = self._vtm_items_listbox.curselection()
    if not sel:
        messagebox.showwarning("항목 삭제", "삭제할 항목을 선택하세요.")
        return
    idx = sel[0]
    name = self._vtm_items[idx]['name']
    if messagebox.askyesno("항목 삭제", f"'{name}' 항목을 삭제하시겠습니까?"):
        del self._vtm_items[idx]
        self._vtm_refresh_items_listbox()


def _vtm_refresh_items_listbox(self):
    """Refresh the items listbox display."""
    self._vtm_items_listbox.delete(0, tk.END)
    for item in self._vtm_items:
        dir_str = "↑클수록좋음" if item['direction'] == 'higher_better' else "↓작을수록좋음"
        self._vtm_items_listbox.insert(tk.END,
            f"{item['name']} [{item['unit']}] ({dir_str})")


def _vtm_load_default_preset(self):
    """Load default evaluation items."""
    defaults = [
        {'name': '건조 제동거리', 'unit': 'm', 'direction': 'lower_better', 'values': {}},
        {'name': '습윤 제동거리', 'unit': 'm', 'direction': 'lower_better', 'values': {}},
        {'name': '고속 제동거리', 'unit': 'm', 'direction': 'lower_better', 'values': {}},
        {'name': '소음 레벨', 'unit': 'dB', 'direction': 'lower_better', 'values': {}},
        {'name': '마모율', 'unit': 'mm/1000km', 'direction': 'lower_better', 'values': {}},
        {'name': '페이드 성능', 'unit': '%', 'direction': 'higher_better', 'values': {}},
        {'name': '고온 마찰력', 'unit': '-', 'direction': 'higher_better', 'values': {}},
        {'name': '저온 마찰력', 'unit': '-', 'direction': 'higher_better', 'values': {}},
    ]
    self._vtm_items = defaults
    self._vtm_refresh_items_listbox()
    self._show_status("기본 프리셋 8개 항목 로드 완료", 'success')


def _vtm_save_items_to_file(self):
    """Save evaluation items and data to JSON file."""
    path = filedialog.asksaveasfilename(
        defaultextension=".json",
        filetypes=[("JSON", "*.json"), ("All", "*.*")],
        title="평가항목 저장")
    if not path:
        return
    self._vtm_collect_table_data()
    data = {
        'items': self._vtm_items,
        'samples': self._vtm_samples,
        'sample_maps': self._vtm_sample_maps,
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    self._show_status(f"평가항목 저장: {os.path.basename(path)}", 'success')


def _vtm_load_items_from_file(self):
    """Load evaluation items from JSON file."""
    path = filedialog.askopenfilename(
        filetypes=[("JSON", "*.json"), ("All", "*.*")],
        title="평가항목 로드")
    if not path:
        return
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self._vtm_items = data.get('items', [])
        loaded_samples = data.get('samples', [])
        self._vtm_sample_maps = data.get('sample_maps', {})
        if loaded_samples:
            self._vtm_samples = loaded_samples
            self._vtm_samples_var.set(", ".join(loaded_samples))
        self._vtm_refresh_items_listbox()
        if self._vtm_items and self._vtm_samples:
            self._vtm_create_sample_map_table()
            self._vtm_create_data_table()
        self._show_status(f"평가항목 로드: {os.path.basename(path)}", 'success')
    except Exception as e:
        messagebox.showerror("로드 오류", str(e))


def _open_vtm_item_manager(self):
    """Open a popup window for managing evaluation items."""
    self._vtm_items_listbox.focus_set()
    if not self._vtm_items:
        self._vtm_load_default_preset()


# ── Data table ──

def _vtm_create_data_table(self):
    """Create data input table with optional mu row + evaluation item rows."""
    samples_str = self._vtm_samples_var.get().strip()
    samples = [s.strip() for s in samples_str.split(',') if s.strip()]
    if len(samples) < 2:
        messagebox.showwarning("데이터 입력", "샘플을 2개 이상 입력하세요 (쉼표 구분).")
        return
    if not self._vtm_items:
        messagebox.showwarning("데이터 입력", "평가 항목을 먼저 추가하세요.")
        return

    self._vtm_samples = samples

    # Clear previous table
    for w in self._vtm_table_frame.winfo_children():
        w.destroy()
    self._vtm_entry_vars = {}
    self._vtm_mu_entry_vars = {}

    # Create scrollable table
    canvas = tk.Canvas(self._vtm_table_frame, highlightthickness=0, height=220)
    scrollbar = ttk.Scrollbar(self._vtm_table_frame, orient='vertical', command=canvas.yview)
    table_inner = ttk.Frame(canvas)

    table_inner.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
    canvas.create_window((0, 0), window=table_inner, anchor='nw')
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Header row
    ttk.Label(table_inner, text="항목 \\ 샘플", font=('', 10, 'bold'),
              width=18).grid(row=0, column=0, padx=2, pady=1, sticky='w')
    for j, sample in enumerate(samples):
        ttk.Label(table_inner, text=sample, font=('', 10, 'bold'),
                  width=10).grid(row=0, column=j+1, padx=2, pady=1)

    # Row 1: Optional measured mu (마찰계수) — 선택사항
    lbl = ttk.Label(table_inner, text="측정 마찰계수 (선택)",
                    font=('', 9), foreground='#64748B', width=22)
    lbl.grid(row=1, column=0, padx=2, pady=1, sticky='w')
    for j, sample in enumerate(samples):
        var = tk.StringVar(value='')
        self._vtm_mu_entry_vars[j] = var
        e = ttk.Entry(table_inner, textvariable=var, width=10)
        e.grid(row=1, column=j+1, padx=2, pady=1)

    # Separator
    ttk.Separator(table_inner, orient=tk.HORIZONTAL).grid(
        row=2, column=0, columnspan=len(samples)+1, sticky='ew', pady=2)

    # Evaluation item rows
    for i, item in enumerate(self._vtm_items):
        dir_mark = "↓" if item['direction'] == 'lower_better' else "↑"
        ttk.Label(table_inner, text=f"{dir_mark} {item['name']} [{item['unit']}]",
                  font=('', 9), width=22).grid(row=i+3, column=0, padx=2, pady=1, sticky='w')
        for j, sample in enumerate(samples):
            var = tk.StringVar(value=str(item.get('values', {}).get(sample, '')))
            self._vtm_entry_vars[(i, j)] = var
            ttk.Entry(table_inner, textvariable=var, width=10).grid(
                row=i+3, column=j+1, padx=2, pady=1)

    # Mouse wheel scrolling
    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    canvas.bind('<MouseWheel>', _on_mousewheel)


def _vtm_collect_table_data(self):
    """Collect data from the table entries into internal state."""
    # Collect optional mu values
    self._vtm_mu_values = {}
    for j, sample in enumerate(self._vtm_samples):
        if j in self._vtm_mu_entry_vars:
            try:
                val = float(self._vtm_mu_entry_vars[j].get())
                self._vtm_mu_values[sample] = val
            except (ValueError, TypeError):
                pass

    # Collect sample-map assignments
    self._vtm_sample_maps = {}
    for sample in self._vtm_samples:
        var = getattr(self, '_vtm_sample_map_vars', {}).get(sample)
        if var and var.get():
            self._vtm_sample_maps[sample] = var.get()

    # Collect item values
    for i, item in enumerate(self._vtm_items):
        item['values'] = {}
        for j, sample in enumerate(self._vtm_samples):
            key = (i, j)
            if key in self._vtm_entry_vars:
                try:
                    val = float(self._vtm_entry_vars[key].get())
                    item['values'][sample] = val
                except (ValueError, TypeError):
                    pass


# ── Indexing ──

def _vtm_compute_indices(self, raw_values, direction, base_mode='first'):
    """Compute index values from raw data.
    base_mode='first': 첫 번째 샘플(A) = 100 기준
    """
    if not raw_values:
        return {}

    vals = np.array(list(raw_values.values()))
    names = list(raw_values.keys())

    if base_mode == 'first':
        # 첫 번째 샘플 기준 = 100
        # _vtm_samples 순서대로, 해당 순서의 첫 샘플 값 사용
        first_sample = None
        if hasattr(self, '_vtm_samples') and self._vtm_samples:
            for s in self._vtm_samples:
                if s in raw_values:
                    first_sample = s
                    break
        if first_sample is not None:
            base = raw_values[first_sample]
        else:
            base = vals[0]
    elif base_mode == 'min':
        base = np.min(vals)
    elif base_mode == 'max':
        base = np.max(vals)
    else:  # mean
        base = np.mean(vals)

    if base == 0:
        base = 1e-10

    indices = {}
    for name, val in zip(names, vals):
        if val == 0:
            val = 1e-10
        if direction == 'lower_better':
            indices[name] = (base / val) * 100.0
        else:
            indices[name] = (val / base) * 100.0

    return indices


# ── Extract mu from friction map ──

def _vtm_extract_mu_from_map(self, fm_results, T_fix=None, p0_fix=None, v_fix=None,
                              branch='both', mu_type='mu_total'):
    """Extract mu value from a friction map at specified (or all) conditions.

    Returns dict with best conditions and mu values.
    If T/p0/v are fixed, returns mu at that condition.
    If any are None, returns the full grid for sweeping.
    """
    if fm_results is None:
        return None

    T_arr = fm_results['T_array']
    p0_arr = fm_results['p0_array_MPa']
    v_arr = fm_results['v_array']

    lut_key_map = {
        'mu_total': ('LUT_cold', 'LUT_hot'),
        'mu_visc': ('LUT_mu_visc_cold', 'LUT_mu_visc_hot'),
        'mu_adh': ('LUT_mu_adh_cold', 'LUT_mu_adh_hot'),
    }
    cold_key, hot_key = lut_key_map.get(mu_type, ('LUT_cold', 'LUT_hot'))

    # Determine indices
    iT = np.argmin(np.abs(T_arr - T_fix)) if T_fix is not None else None
    ip = np.argmin(np.abs(p0_arr - p0_fix)) if p0_fix is not None else None
    iv = np.argmin(np.abs(v_arr - v_fix)) if v_fix is not None else None

    # Select branch
    branches = []
    if branch in ('cold', 'both'):
        branches.append(('cold', cold_key))
    if branch in ('hot', 'both') and fm_results.get('use_flash', False):
        branches.append(('hot', hot_key))
    if not branches:
        branches.append(('cold', cold_key))

    best_mu = None
    best_cond = None

    for branch_name, lut_key in branches:
        LUT = fm_results[lut_key]

        if iT is not None and ip is not None and iv is not None:
            mu_val = float(LUT[iT, ip, iv])
            if best_mu is None or mu_val > best_mu:
                best_mu = mu_val
                best_cond = {
                    'T_C': float(T_arr[iT]),
                    'p0_MPa': float(p0_arr[ip]),
                    'v_ms': float(v_arr[iv]),
                    'branch': branch_name,
                    'mu': mu_val,
                }
        else:
            # Find the mu value (use mean over unspecified dims)
            if iT is not None:
                LUT = LUT[iT:iT+1, :, :]
            if ip is not None:
                LUT = LUT[:, ip:ip+1, :]
            if iv is not None:
                LUT = LUT[:, :, iv:iv+1]
            mu_val = float(np.mean(LUT))
            if best_mu is None or mu_val > best_mu:
                best_mu = mu_val
                t_idx = iT if iT is not None else LUT.shape[0] // 2
                p_idx = ip if ip is not None else LUT.shape[1] // 2
                v_idx = iv if iv is not None else LUT.shape[2] // 2
                best_cond = {
                    'T_C': float(T_arr[min(t_idx, len(T_arr)-1)]),
                    'p0_MPa': float(p0_arr[min(p_idx, len(p0_arr)-1)]),
                    'v_ms': float(v_arr[min(v_idx, len(v_arr)-1)]),
                    'branch': branch_name,
                    'mu': mu_val,
                }

    return best_cond


def _vtm_find_all_correlation_conditions(self, fm_results_dict, item_indices,
                                           base_mode='min', branch='both', mu_type='mu_total',
                                           threshold=0.80):
    """마찰맵의 모든 (T, p0, v, branch) 조건을 전수탐색하여
    |r| >= threshold인 조건을 모두 수집하고, 최적 조건과 범위를 리턴.

    fm_results_dict: {sample_name: friction_map_results}
    item_indices: {sample_name: index_value}
    threshold: 상관성 임계값 (기본 0.80)

    Returns dict:
        'best_corr': float,
        'best_condition': {T_C, p0_MPa, v_ms, branch, corr, mu_indices},
        'passing_conditions': list of {T_C, p0_MPa, v_ms, branch, corr, mu_indices},
        'passing_count': int,
        'total_searched': int,
        'T_range': (min, max) or None,
        'p0_range': (min, max) or None,
        'v_range': (min, max) or None,
        'branch_list': list of branch names that passed,
    """
    result = {
        'best_corr': 0.0,
        'best_condition': None,
        'passing_conditions': [],
        'passing_count': 0,
        'total_searched': 0,
        'T_range': None,
        'p0_range': None,
        'v_range': None,
        'branch_list': [],
    }

    # We need at least one FM with LUT data
    any_fm = None
    for fm in fm_results_dict.values():
        if fm is not None:
            any_fm = fm
            break
    if any_fm is None:
        return result

    T_arr = any_fm['T_array']
    p0_arr = any_fm['p0_array_MPa']
    v_arr = any_fm['v_array']

    lut_key_map = {
        'mu_total': ('LUT_cold', 'LUT_hot'),
        'mu_visc': ('LUT_mu_visc_cold', 'LUT_mu_visc_hot'),
        'mu_adh': ('LUT_mu_adh_cold', 'LUT_mu_adh_hot'),
    }
    cold_key, hot_key = lut_key_map.get(mu_type, ('LUT_cold', 'LUT_hot'))

    common_samples = [s for s in item_indices.keys()
                      if s in fm_results_dict and fm_results_dict[s] is not None]
    if len(common_samples) < 2:
        return result

    test_arr = np.array([item_indices[s] for s in common_samples])

    branches = []
    if branch in ('cold', 'both'):
        branches.append(('cold', cold_key))
    if branch in ('hot', 'both') and any_fm.get('use_flash', False):
        branches.append(('hot', hot_key))
    if not branches:
        branches.append(('cold', cold_key))

    # 전수 탐색: 모든 (T, p0, v, branch) 조합
    all_passing = []
    total_count = 0

    for branch_name, lut_key in branches:
        for iT, T in enumerate(T_arr):
            for ip, p0 in enumerate(p0_arr):
                for iv, v in enumerate(v_arr):
                    total_count += 1

                    # Extract mu for each sample at this condition
                    mu_vals = {}
                    for s in common_samples:
                        fm = fm_results_dict[s]
                        if fm is None:
                            continue
                        try:
                            LUT = fm[lut_key]
                            s_iT = np.argmin(np.abs(fm['T_array'] - T))
                            s_ip = np.argmin(np.abs(fm['p0_array_MPa'] - p0))
                            s_iv = np.argmin(np.abs(fm['v_array'] - v))
                            mu_vals[s] = float(LUT[s_iT, s_ip, s_iv])
                        except (KeyError, IndexError):
                            pass

                    if len(mu_vals) < 2:
                        continue

                    # Compute mu indices
                    mu_idx = self._vtm_compute_indices(mu_vals, 'higher_better', base_mode)
                    mu_arr = np.array([mu_idx.get(s, 0) for s in common_samples])

                    # Pearson correlation
                    if np.std(test_arr) < 1e-15 or np.std(mu_arr) < 1e-15:
                        corr = 0.0
                    else:
                        corr = float(np.corrcoef(test_arr, mu_arr)[0, 1])

                    cond_entry = {
                        'T_C': float(T),
                        'p0_MPa': float(p0),
                        'v_ms': float(v),
                        'branch': branch_name,
                        'corr': corr,
                        'mu_indices': mu_idx,
                    }

                    # 임계값 이상이면 수집
                    if abs(corr) >= threshold:
                        all_passing.append(cond_entry)

                    # 최적 조건 추적
                    if abs(corr) > abs(result['best_corr']):
                        result['best_corr'] = corr
                        result['best_condition'] = cond_entry

    result['total_searched'] = total_count
    result['passing_conditions'] = all_passing
    result['passing_count'] = len(all_passing)

    # 임계값 이상 조건들의 범위 계산
    if all_passing:
        Ts = [c['T_C'] for c in all_passing]
        ps = [c['p0_MPa'] for c in all_passing]
        vs = [c['v_ms'] for c in all_passing]
        result['T_range'] = (min(Ts), max(Ts))
        result['p0_range'] = (min(ps), max(ps))
        result['v_range'] = (min(vs), max(vs))
        result['branch_list'] = list(set(c['branch'] for c in all_passing))

    return result


# ── Main analysis ──

def _run_vehicle_test_matching(self):
    """Run correlation analysis between vehicle test indices and friction map mu."""
    # Validate
    if not self._vtm_items:
        messagebox.showwarning("실차 매칭", "평가 항목을 추가하세요.")
        return
    if len(self._vtm_samples) < 2:
        messagebox.showwarning("실차 매칭", "샘플을 2개 이상 입력하세요.")
        return

    # Collect table data
    self._vtm_collect_table_data()

    # Build friction map dict per sample
    fm_dict = {}
    samples_with_map = 0
    for sample in self._vtm_samples:
        map_name = self._vtm_sample_maps.get(sample)
        if map_name:
            fm, _ = self._get_friction_map_by_name(map_name)
            fm_dict[sample] = fm
            if fm is not None:
                samples_with_map += 1
        else:
            fm_dict[sample] = None

    # Check if we have enough friction maps
    if samples_with_map < 2:
        # Check if optional measured mu is available
        mu_vals = getattr(self, '_vtm_mu_values', {})
        if len(mu_vals) < 2:
            messagebox.showwarning("실차 매칭",
                "마찰맵이 2개 이상의 샘플에 연결되어야 합니다.\n\n"
                "1) Friction Map 탭에서 각 컴파운드의 마찰맵을 생성하고 '마찰맵 저장'\n"
                "2) 이 탭의 '샘플 & 마찰맵 매핑 생성'에서 각 샘플에 마찰맵을 연결\n\n"
                "또는 측정 마찰계수(mu)를 직접 입력하세요 (Traction 평가 시).")
            return

    # Verify items have data
    for item in self._vtm_items:
        if len(item['values']) < 2:
            messagebox.showwarning("실차 매칭",
                f"'{item['name']}' 항목에 데이터가 부족합니다 (2개 이상 필요).")
            return

    try:
        threshold = float(self._vtm_threshold_var.get())
    except ValueError:
        threshold = 0.80
    self._vtm_threshold = threshold

    base_mode = self._vtm_index_base_var.get()
    branch_mode = self._vtm_branch_var.get()
    mu_type = self._vtm_mu_type_var.get()

    # Parse fixed conditions
    T_fix = None
    p0_fix = None
    v_fix = None
    try:
        t_str = self._vtm_fix_T_var.get().strip()
        if t_str:
            T_fix = float(t_str)
    except ValueError:
        pass
    try:
        p_str = self._vtm_fix_p0_var.get().strip()
        if p_str:
            p0_fix = float(p_str)
    except ValueError:
        pass
    try:
        v_str = self._vtm_fix_v_var.get().strip()
        if v_str:
            v_fix = float(v_str)
    except ValueError:
        pass

    # ── Determine mu source for each sample ──
    # Priority: 1) Friction map mu at condition, 2) Measured mu (optional)
    mu_from_map = {}
    mu_conditions = {}

    if samples_with_map >= 2:
        if T_fix is not None and p0_fix is not None and v_fix is not None:
            # Fixed condition: extract mu directly
            for sample in self._vtm_samples:
                fm = fm_dict.get(sample)
                if fm is not None:
                    cond = self._vtm_extract_mu_from_map(
                        fm, T_fix=T_fix, p0_fix=p0_fix, v_fix=v_fix,
                        branch=branch_mode, mu_type=mu_type)
                    if cond:
                        mu_from_map[sample] = cond['mu']
                        mu_conditions[sample] = cond
        else:
            # Extract mu at center of each FM's grid as initial values
            for sample in self._vtm_samples:
                fm = fm_dict.get(sample)
                if fm is not None:
                    T_mid = float(np.median(fm['T_array']))
                    p0_mid = float(np.median(fm['p0_array_MPa']))
                    v_mid = float(np.median(fm['v_array']))
                    cond = self._vtm_extract_mu_from_map(
                        fm, T_fix=T_fix or T_mid, p0_fix=p0_fix or p0_mid,
                        v_fix=v_fix or v_mid, branch=branch_mode, mu_type=mu_type)
                    if cond:
                        mu_from_map[sample] = cond['mu']
                        mu_conditions[sample] = cond

    # Fallback: use measured mu if available
    mu_vals = getattr(self, '_vtm_mu_values', {})
    effective_mu = {}
    for sample in self._vtm_samples:
        if sample in mu_from_map:
            effective_mu[sample] = mu_from_map[sample]
        elif sample in mu_vals:
            effective_mu[sample] = mu_vals[sample]

    # Compute mu indices
    mu_index = self._vtm_compute_indices(effective_mu, 'higher_better', base_mode) if len(effective_mu) >= 2 else {}

    # Step: Correlate each item's index with mu index, and find best conditions
    all_results = []

    for item in self._vtm_items:
        item_idx = self._vtm_compute_indices(item['values'], item['direction'], base_mode)

        # Build aligned arrays
        common_samples = [s for s in self._vtm_samples
                          if s in item_idx and s in mu_index]

        if len(common_samples) < 2:
            all_results.append({
                'item_name': item['name'],
                'item_unit': item['unit'],
                'item_direction': item['direction'],
                'test_indices': item_idx,
                'mu_indices': mu_index,
                'mu_source': 'map' if samples_with_map >= 2 else 'measured',
                'correlation': None,
                'common_samples': [],
                'best_condition': "",
                'map_matches': [],
            })
            continue

        test_arr = np.array([item_idx[s] for s in common_samples])
        mu_arr = np.array([mu_index[s] for s in common_samples])

        # Pearson correlation at current/fixed condition
        if np.std(test_arr) < 1e-15 or np.std(mu_arr) < 1e-15:
            corr = 0.0
        else:
            corr = float(np.corrcoef(test_arr, mu_arr)[0, 1])

        # Auto-search for best condition & all passing conditions
        best_condition = ""
        search_result = None
        if samples_with_map >= 2:
            if T_fix is not None and p0_fix is not None and v_fix is not None:
                best_condition = f"T={T_fix:.0f}°C, p₀={p0_fix:.3g} MPa, v={v_fix:.4g} m/s (고정)"
            else:
                # 전수 탐색: 모든 조건에서 상관성 계산
                search_result = self._vtm_find_all_correlation_conditions(
                    fm_dict, item_idx, base_mode, branch_mode, mu_type, threshold)

                if search_result['best_condition'] is not None:
                    bc = search_result['best_condition']
                    if abs(bc['corr']) > abs(corr):
                        corr = bc['corr']
                        best_condition = (
                            f"T={bc['T_C']:.0f}°C, p₀={bc['p0_MPa']:.3g} MPa, "
                            f"v={bc['v_ms']:.4g} m/s [{bc['branch']}]")
                        # Update mu_arr with best condition values
                        mu_arr = np.array([bc['mu_indices'].get(s, 0) for s in common_samples])

        # Map matches per sample
        map_matches = []
        for sample in common_samples:
            cond = mu_conditions.get(sample)
            if cond:
                map_matches.append({
                    'sample': sample,
                    'target_mu': effective_mu.get(sample, 0),
                    'matched_mu': cond['mu'],
                    'T_C': cond['T_C'],
                    'p0_MPa': cond['p0_MPa'],
                    'v_ms': cond['v_ms'],
                    'branch': cond['branch'],
                    'error': 0.0,
                    'error_pct': 0.0,
                })

        all_results.append({
            'item_name': item['name'],
            'item_unit': item['unit'],
            'item_direction': item['direction'],
            'test_indices': item_idx,
            'mu_indices': mu_index,
            'mu_source': 'map' if samples_with_map >= 2 else 'measured',
            'correlation': corr,
            'common_samples': common_samples,
            'best_condition': best_condition,
            'search_result': search_result,  # 전수 탐색 결과 (범위 정보 포함)
            'map_matches': map_matches,
        })

    self._vtm_results = all_results

    # Display
    self._vtm_display_results(all_results)
    self._vtm_plot_heatmap(all_results)
    self._vtm_plot_index_comparison(all_results)
    self._vtm_plot_match_detail(all_results)
    self._vtm_plot_map_conditions(all_results)

    self._show_status(
        f"실차 매칭 완료: {len(self._vtm_items)}개 항목 분석", 'success')


# ── Display results ──

def _vtm_display_results(self, all_results):
    """Display correlation results in the text widget."""
    self._vtm_result_text.config(state=tk.NORMAL)
    self._vtm_result_text.delete('1.0', tk.END)

    base_mode = self._vtm_index_base_var.get()
    base_label = {'first': '첫 번째 샘플(A)', 'min': '최솟값', 'max': '최댓값', 'mean': '평균'}.get(base_mode, '첫 번째 샘플(A)')
    first_s = self._vtm_samples[0] if self._vtm_samples else 'A'

    self._vtm_result_text.insert(tk.END,
        f"═══ 실차 평가 ↔ 마찰맵 상관성 분석 ═══\n"
        f"샘플: {', '.join(self._vtm_samples)}\n"
        f"인덱스 기준: {first_s} = 100\n"
        f"{'─'*50}\n\n")

    # Show sample ↔ friction map mapping
    self._vtm_result_text.insert(tk.END, "▶ 샘플별 마찰맵 연결\n")
    for s in self._vtm_samples:
        map_name = self._vtm_sample_maps.get(s, '(미연결)')
        self._vtm_result_text.insert(tk.END, f"  {s}: {map_name}\n")
    self._vtm_result_text.insert(tk.END, "\n")

    # Show mu values and indices
    mu_idx = all_results[0]['mu_indices'] if all_results else {}
    mu_source = all_results[0].get('mu_source', 'unknown') if all_results else 'unknown'
    effective_mu = getattr(self, '_vtm_mu_values', {})
    # Merge with map-extracted mu
    for s in self._vtm_samples:
        if s not in effective_mu:
            if s in mu_idx:
                # reverse-compute from index is complex, just note it
                pass

    self._vtm_result_text.insert(tk.END,
        f"▶ 마찰계수 (소스: {'마찰맵' if mu_source == 'map' else '측정값'})\n")
    for s in self._vtm_samples:
        idx_val = mu_idx.get(s, '?')
        idx_str = f"{idx_val:.1f}" if isinstance(idx_val, float) else str(idx_val)
        self._vtm_result_text.insert(tk.END, f"  {s}: (인덱스: {idx_str})\n")
    self._vtm_result_text.insert(tk.END, "\n")

    # Per-item results
    for res in all_results:
        name = res['item_name']
        unit = res['item_unit']
        direction = res['item_direction']
        dir_str = "↑클수록좋음" if direction == 'higher_better' else "↓작을수록좋음"
        corr = res['correlation']

        self._vtm_result_text.insert(tk.END,
            f"■ {name} [{unit}] ({dir_str})\n")

        # Test indices
        test_idx = res['test_indices']
        idx_strs = [f"  {s}: {test_idx.get(s, '?'):.1f}" for s in self._vtm_samples
                     if s in test_idx]
        self._vtm_result_text.insert(tk.END,
            f"  인덱스: {' | '.join(idx_strs)}\n")

        if corr is not None:
            corr_str = f"{corr:+.4f}"
            strength = "매우 강함" if abs(corr) >= 0.9 else \
                       "강함" if abs(corr) >= 0.7 else \
                       "보통" if abs(corr) >= 0.5 else "약함"
            sign_str = "양(+)" if corr > 0 else "음(-)"
            self._vtm_result_text.insert(tk.END,
                f"  ★ 최대 상관계수: r = {corr_str} [{strength}, {sign_str} 상관]\n")
        else:
            self._vtm_result_text.insert(tk.END,
                f"  ★ 상관계수 계산 불가 (데이터 부족)\n")

        # Best condition
        best_cond = res.get('best_condition', '')
        if best_cond:
            self._vtm_result_text.insert(tk.END,
                f"  최적 조건 (r 최대): {best_cond}\n")

        # 전수 탐색 결과 — 범위 표시
        sr = res.get('search_result')
        if sr and sr['total_searched'] > 0:
            thr = self._vtm_threshold
            self._vtm_result_text.insert(tk.END,
                f"\n  ── 전수 탐색 결과 (|r| ≥ {thr:.2f}) ──\n"
                f"  총 탐색: {sr['total_searched']}개 조건\n"
                f"  통과: {sr['passing_count']}개 조건 "
                f"({sr['passing_count']/max(sr['total_searched'],1)*100:.1f}%)\n")

            if sr['passing_count'] > 0:
                T_r = sr['T_range']
                p_r = sr['p0_range']
                v_r = sr['v_range']
                self._vtm_result_text.insert(tk.END,
                    f"  ── 상관성 충족 조건 범위 ──\n"
                    f"    온도 T: {T_r[0]:.0f} ~ {T_r[1]:.0f} °C\n"
                    f"    압력 p₀: {p_r[0]:.3g} ~ {p_r[1]:.3g} MPa\n"
                    f"    속도 v: {v_r[0]:.4g} ~ {v_r[1]:.4g} m/s\n"
                    f"    Branch: {', '.join(sr['branch_list'])}\n")

                # 상위 5개 조건 상세
                top_n = sorted(sr['passing_conditions'],
                               key=lambda c: abs(c['corr']), reverse=True)[:5]
                self._vtm_result_text.insert(tk.END,
                    f"  ── 상위 {min(5, len(top_n))}개 조건 ──\n")
                for rank, c in enumerate(top_n, 1):
                    self._vtm_result_text.insert(tk.END,
                        f"    #{rank} r={c['corr']:+.4f}  "
                        f"T={c['T_C']:.0f}°C, p₀={c['p0_MPa']:.3g} MPa, "
                        f"v={c['v_ms']:.4g} m/s [{c['branch']}]\n")
            else:
                self._vtm_result_text.insert(tk.END,
                    f"  ⚠ |r| ≥ {thr:.2f} 조건이 없습니다.\n")

        # Map matches per sample
        matches = res.get('map_matches', [])
        if matches:
            self._vtm_result_text.insert(tk.END, "\n  ── 마찰 맵 mu 값 (샘플별) ──\n")
            for m in matches:
                self._vtm_result_text.insert(tk.END,
                    f"    [{m['sample']}] mu={m['matched_mu']:.4f}  "
                    f"T={m['T_C']:.0f}°C, p₀={m['p0_MPa']:.3g} MPa, "
                    f"v={m['v_ms']:.4g} m/s [{m['branch']}]\n")

        self._vtm_result_text.insert(tk.END, "\n")

    self._vtm_result_text.config(state=tk.DISABLED)


def _vtm_plot_heatmap(self, all_results):
    """Plot correlation summary: bar chart of correlation per item."""
    import matplotlib.pyplot as plt

    fig = self._vtm_fig_heatmap
    fig.clear()

    items_with_corr = [(res['item_name'], res['correlation'])
                        for res in all_results if res['correlation'] is not None]
    if not items_with_corr:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "표시할 데이터가 없습니다.",
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_axis_off()
        self._vtm_canvas_heatmap.draw()
        return

    names = [x[0] for x in items_with_corr]
    corrs = [x[1] for x in items_with_corr]

    ax = fig.add_subplot(111)
    x = np.arange(len(names))
    colors = ['#EF4444' if c < -0.5 else '#3B82F6' if c > 0.5 else '#94A3B8'
              for c in corrs]

    bars = ax.barh(x, corrs, color=colors, edgecolor='gray', linewidth=0.5, height=0.6)

    # Add value labels
    for bar_obj, corr_val in zip(bars, corrs):
        xpos = bar_obj.get_width()
        offset = 0.02 if corr_val >= 0 else -0.02
        ha = 'left' if corr_val >= 0 else 'right'
        ax.text(xpos + offset, bar_obj.get_y() + bar_obj.get_height()/2,
                f'{corr_val:+.3f}', ha=ha, va='center', fontsize=10, fontweight='bold')

    ax.set_yticks(x)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel('Pearson r (실차 인덱스 vs 마찰맵 mu 인덱스)', fontsize=11)
    ax.set_title("항목별 상관계수 (실차 평가 ↔ 마찰맵)", fontsize=13, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.axvline(x=self._vtm_threshold, color='green', linewidth=1, linestyle='--', alpha=0.6,
               label=f'임계값 +{self._vtm_threshold}')
    ax.axvline(x=-self._vtm_threshold, color='green', linewidth=1, linestyle='--', alpha=0.6,
               label=f'임계값 -{self._vtm_threshold}')
    ax.set_xlim(-1.1, 1.1)
    ax.legend(fontsize=9)
    ax.invert_yaxis()

    fig.tight_layout()
    self._vtm_canvas_heatmap.draw()


def _vtm_plot_index_comparison(self, all_results):
    """Plot index comparison bar chart for all items and samples."""
    import matplotlib.pyplot as plt

    fig = self._vtm_fig_index
    fig.clear()

    items_with_data = [res for res in all_results if res['test_indices']]
    if not items_with_data:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "표시할 데이터가 없습니다.",
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_axis_off()
        self._vtm_canvas_index.draw()
        return

    # Include mu index as first item
    n_items = len(items_with_data) + 1  # +1 for mu
    n_samples = len(self._vtm_samples)

    n_cols = min(4, n_items)
    n_rows = int(np.ceil(n_items / n_cols))

    axes = fig.subplots(n_rows, n_cols, squeeze=False)
    colors = plt.cm.Set2(np.linspace(0, 1, max(n_samples, 3)))

    # First subplot: mu index
    ax = axes[0][0]
    mu_idx = all_results[0]['mu_indices'] if all_results else {}
    x = np.arange(n_samples)
    vals = [mu_idx.get(s, 0) for s in self._vtm_samples]
    bars = ax.bar(x, vals, color=colors[:n_samples], edgecolor='gray', linewidth=0.5)
    for bar_obj, val in zip(bars, vals):
        ax.text(bar_obj.get_x() + bar_obj.get_width()/2, bar_obj.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(self._vtm_samples, fontsize=9)
    ax.set_ylabel('인덱스', fontsize=9)
    mu_src = all_results[0].get('mu_source', 'map') if all_results else 'map'
    ax.set_title(f"마찰맵 mu ({'맵' if mu_src == 'map' else '측정'})",
                 fontsize=10, fontweight='bold', color='#0369A1')
    ax.axhline(y=100, color='red', linestyle='--', alpha=0.5, linewidth=0.8)

    # Remaining subplots: evaluation items
    for idx, res in enumerate(items_with_data):
        plot_idx = idx + 1
        row = plot_idx // n_cols
        col = plot_idx % n_cols
        ax = axes[row][col]

        test_idx = res['test_indices']
        vals = [test_idx.get(s, 0) for s in self._vtm_samples]
        bars = ax.bar(x, vals, color=colors[:n_samples], edgecolor='gray', linewidth=0.5)
        for bar_obj, val in zip(bars, vals):
            ax.text(bar_obj.get_x() + bar_obj.get_width()/2, bar_obj.get_height() + 0.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(self._vtm_samples, fontsize=9)
        ax.set_ylabel('인덱스', fontsize=9)

        corr = res['correlation']
        corr_str = f"r={corr:+.3f}" if corr is not None else ""
        ax.set_title(f"{res['item_name']} {corr_str}", fontsize=10, fontweight='bold')
        ax.axhline(y=100, color='red', linestyle='--', alpha=0.5, linewidth=0.8)

    # Hide unused subplots
    for idx in range(n_items, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row][col].set_visible(False)

    fig.suptitle("실차 평가 인덱스 비교", fontsize=13, fontweight='bold')
    fig.tight_layout()
    self._vtm_canvas_index.draw()


def _vtm_plot_match_detail(self, all_results):
    """Plot scatter plot: test index vs mu index for each item.
    - 샘플 순서 보존 (self._vtm_samples 순서대로)
    - 상관성 표시
    - 더 많은 정보 표현
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig = self._vtm_fig_detail
    fig.clear()

    items_with_corr = [res for res in all_results
                        if res['correlation'] is not None and len(res['common_samples']) >= 2]
    if not items_with_corr:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "매칭 결과가 없습니다.",
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_axis_off()
        self._vtm_canvas_detail.draw()
        return

    n_plots = min(len(items_with_corr), 6)
    n_cols = min(3, n_plots)
    n_rows = int(np.ceil(n_plots / n_cols))

    # Use subplots for proper layout
    axes = fig.subplots(n_rows, n_cols, squeeze=False)

    # Color per sample — consistent across all subplots
    sample_colors = {}
    cmap = plt.cm.Set1(np.linspace(0, 1, max(len(self._vtm_samples), 3)))
    for i, s in enumerate(self._vtm_samples):
        sample_colors[s] = cmap[i]

    for idx in range(n_plots):
        res = items_with_corr[idx]
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row][col]

        # 샘플 순서 보존: self._vtm_samples 순서대로 정렬
        common = [s for s in self._vtm_samples if s in res['common_samples']]
        test_vals = np.array([res['test_indices'][s] for s in common])
        mu_vals = np.array([res['mu_indices'][s] for s in common])
        colors = [sample_colors[s] for s in common]

        # Scatter with sample labels — 순서대로
        for i, (s, mx, ty) in enumerate(zip(common, mu_vals, test_vals)):
            ax.scatter(mx, ty, s=100, c=[colors[i]], edgecolors='black',
                       linewidth=0.8, zorder=5)
            ax.annotate(s, (mx, ty), textcoords="offset points",
                        xytext=(8, 8), fontsize=10, fontweight='bold',
                        color=colors[i])

        # Trend line
        corr = res['correlation']
        if len(common) >= 2:
            z = np.polyfit(mu_vals, test_vals, 1)
            p = np.poly1d(z)
            x_line = np.linspace(mu_vals.min() - 5, mu_vals.max() + 5, 50)
            ax.plot(x_line, p(x_line), 'r--', alpha=0.7, linewidth=1.5)

        ax.set_xlabel('mu 인덱스 (마찰맵)', fontsize=9)
        ax.set_ylabel(f'{res["item_name"]} 인덱스', fontsize=9)

        # 상관성 정보 상세 표시
        strength = "매우 강함" if abs(corr) >= 0.9 else \
                   "강함" if abs(corr) >= 0.7 else \
                   "보통" if abs(corr) >= 0.5 else "약함"
        sign_str = "(+)" if corr > 0 else "(-)"
        ax.set_title(f"{res['item_name']}\nr = {corr:+.4f}  [{strength} {sign_str}]",
                     fontsize=10, fontweight='bold')

        # 기준선
        ax.axhline(y=100, color='gray', linestyle=':', alpha=0.4)
        ax.axvline(x=100, color='gray', linestyle=':', alpha=0.4)

        # 최적 조건 정보 텍스트 박스
        best_cond = res.get('best_condition', '')
        sr = res.get('search_result')
        info_lines = []
        if best_cond:
            info_lines.append(f"최적: {best_cond}")
        if sr and sr['passing_count'] > 0:
            info_lines.append(f"통과: {sr['passing_count']}/{sr['total_searched']}")
        if info_lines:
            info_text = '\n'.join(info_lines)
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                    fontsize=7, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                              alpha=0.8, edgecolor='gray'))

        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_plots, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row][col].set_visible(False)

    fig.suptitle("항목별 상관성 상세 (마찰맵 mu 인덱스 vs 실차 인덱스)",
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    self._vtm_canvas_detail.draw()


def _vtm_plot_map_conditions(self, all_results):
    """3D 마찰맵 표면 위에 상관성 통과 영역을 색상으로 표시.

    Friction Map 탭과 유사한 대형 스크롤 가능 3D 플롯.
    상관성 높은 영역 = 빨간색 음영, 비통과 영역 = 기본 컬러맵.
    Best Fit = 별표 마커.
    """
    import tkinter as tk
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.colors import Normalize
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    # ── 기존 플롯 영역 클리어 ──
    plot_frame = self._vtm_map_plot_frame
    for w in plot_frame.winfo_children():
        w.destroy()

    # ── 데이터 검증 ──
    if all_results is None:
        lbl = ttk.Label(plot_frame, text="상관성 분석을 먼저 실행하세요.",
                        font=self.FONTS['body'])
        lbl.pack(expand=True)
        return

    # Collect search results from first item with search_result
    sr = None
    sr_item_name = None
    for res in all_results:
        if res.get('search_result') and res['search_result']['total_searched'] > 0:
            sr = res['search_result']
            sr_item_name = res['item_name']
            break

    if sr is None:
        lbl = ttk.Label(plot_frame,
                        text="전수 탐색 결과가 없습니다.\n조건을 비우고 분석을 실행하세요.",
                        font=self.FONTS['body'])
        lbl.pack(expand=True)
        return

    # Get friction map from first sample
    any_fm = None
    for s in self._vtm_samples:
        map_name = self._vtm_sample_maps.get(s)
        if map_name:
            fm, _ = self._get_friction_map_by_name(map_name)
            if fm is not None:
                any_fm = fm
                break

    if any_fm is None:
        lbl = ttk.Label(plot_frame, text="마찰맵 데이터를 찾을 수 없습니다.",
                        font=self.FONTS['body'])
        lbl.pack(expand=True)
        return

    T_arr = any_fm['T_array']
    p0_arr = any_fm['p0_array_MPa']
    v_arr = any_fm['v_array']
    log_v = np.log10(v_arr)
    thr = getattr(self, '_vtm_threshold', 0.80)

    # Determine LUT key
    mu_type = self._vtm_mu_type_var.get()
    branch_sel = getattr(self, '_vtm_map_branch_var', None)
    display_branch = branch_sel.get() if branch_sel else 'cold'
    lut_map = {
        'mu_total': ('LUT_cold', 'LUT_hot'),
        'mu_visc': ('LUT_mu_visc_cold', 'LUT_mu_visc_hot'),
        'mu_adh': ('LUT_mu_adh_cold', 'LUT_mu_adh_hot'),
    }
    cold_key, hot_key = lut_map.get(mu_type, ('LUT_cold', 'LUT_hot'))
    lut_key = hot_key if display_branch == 'hot' else cold_key
    LUT = any_fm.get(lut_key)
    if LUT is None:
        LUT = any_fm.get(cold_key)
    if LUT is None:
        lbl = ttk.Label(plot_frame, text="LUT 데이터를 찾을 수 없습니다.",
                        font=self.FONTS['body'])
        lbl.pack(expand=True)
        return

    # ── Build correlation map: 3D array [n_T, n_p, n_v] with |r| values ──
    # Initialize with NaN (= not computed)
    corr_map = np.full((len(T_arr), len(p0_arr), len(v_arr)), np.nan)
    best_cond = sr['best_condition']

    for c in sr['passing_conditions']:
        iT = np.argmin(np.abs(T_arr - c['T_C']))
        ip = np.argmin(np.abs(p0_arr - c['p0_MPa']))
        iv = np.argmin(np.abs(v_arr - c['v_ms']))
        corr_map[iT, ip, iv] = abs(c['corr'])

    # ── Slice axis ──
    slice_axis = getattr(self, '_vtm_map_slice_var', None)
    slice_mode = slice_axis.get() if slice_axis else 'pressure'

    if slice_mode == 'pressure':
        slice_arr = p0_arr
        slice_label = 'p₀'
        slice_unit = 'MPa'
        x_arr, y_arr = T_arr, log_v
        x_label, y_label = r'$T_0$ ($^\circ$C)', r'$\log_{10}(v)$'
        def get_Z(i): return LUT[:, i, :]
        def get_corr(i): return corr_map[:, i, :]
    elif slice_mode == 'temperature':
        slice_arr = T_arr
        slice_label = 'T'
        slice_unit = '°C'
        x_arr, y_arr = p0_arr, log_v
        x_label, y_label = r'$p_0$ (MPa)', r'$\log_{10}(v)$'
        def get_Z(i): return LUT[i, :, :]
        def get_corr(i): return corr_map[i, :, :]
    else:  # velocity
        slice_arr = v_arr
        slice_label = 'v'
        slice_unit = 'm/s'
        x_arr, y_arr = T_arr, p0_arr
        x_label, y_label = r'$T_0$ ($^\circ$C)', r'$p_0$ (MPa)'
        def get_Z(i): return LUT[:, :, i]
        def get_corr(i): return corr_map[:, :, i]

    n_slices = len(slice_arr)

    # z-label
    z_labels_map = {'mu_total': r'$\mu_{total}$', 'mu_visc': r'$\mu_{hys}$',
                    'mu_adh': r'$\mu_{adh}$'}
    z_label = z_labels_map.get(mu_type, r'$\mu$')

    # Global z-limits
    z_min = float(LUT.min())
    z_max = float(LUT.max())

    X_mesh, Y_mesh = np.meshgrid(x_arr, y_arr, indexing='ij')

    # Fine interpolation
    use_fine = len(x_arr) >= 2 and len(y_arr) >= 2
    if use_fine:
        try:
            from scipy.interpolate import RectBivariateSpline
            x_fine = np.linspace(x_arr[0], x_arr[-1], 30)
            y_fine = np.linspace(y_arr[0], y_arr[-1], 40)
            X_fine, Y_fine = np.meshgrid(x_fine, y_fine, indexing='ij')
        except ImportError:
            use_fine = False

    # ── Figure sizing (Friction Map 탭과 유사) ──
    MAX_COLS = 3
    n_cols = min(n_slices, MAX_COLS)
    n_rows = int(np.ceil(n_slices / n_cols))

    plot_frame.update_idletasks()
    avail_w = plot_frame.winfo_width()
    if avail_w < 200:
        avail_w = 900
    fig_dpi = 100
    fig_w = max(avail_w - 20, 600) / fig_dpi
    row_h = 5.5  # inches per row — large plots
    fig_h = max(row_h * n_rows, 6)

    fig = Figure(figsize=(fig_w, fig_h), dpi=fig_dpi, facecolor='white')
    fig.subplots_adjust(left=0.04, right=0.96, top=0.92, bottom=0.04,
                        wspace=0.18, hspace=0.30)

    # ── Custom colormap: viridis base + red overlay for passing regions ──
    base_cmap = plt.cm.viridis
    red_color = np.array([1.0, 0.15, 0.15, 1.0])  # bright red

    def _make_facecolors(Z_slice, corr_slice):
        """Create RGBA facecolors: viridis for non-passing, red-tinted for passing."""
        norm = Normalize(vmin=z_min, vmax=z_max)
        # Base colors from viridis
        rgba = base_cmap(norm(Z_slice))  # shape (nx, ny, 4)
        # Overlay red where correlation passes threshold
        # For plot_surface, facecolors shape = (nx-1, ny-1, 4) for cell centers
        nx, ny = Z_slice.shape
        fc = np.zeros((nx - 1, ny - 1, 4))
        for i in range(nx - 1):
            for j in range(ny - 1):
                # Average of 4 corner values
                z_avg = (Z_slice[i, j] + Z_slice[i+1, j] +
                         Z_slice[i, j+1] + Z_slice[i+1, j+1]) / 4
                base_rgba = base_cmap(norm(z_avg))

                # Check if any corner is a passing condition
                corners = [corr_slice[i, j], corr_slice[i+1, j],
                           corr_slice[i, j+1], corr_slice[i+1, j+1]]
                passing_corners = [c for c in corners if not np.isnan(c)]

                if passing_corners:
                    # Blend with red — intensity proportional to |r|
                    max_corr = max(passing_corners)
                    blend = min((max_corr - thr) / (1.0 - thr + 1e-10), 1.0) * 0.7 + 0.3
                    fc[i, j] = (1 - blend) * np.array(base_rgba) + blend * red_color
                else:
                    fc[i, j] = base_rgba
                fc[i, j, 3] = 0.85  # alpha
        return fc

    for idx in range(n_slices):
        row_i = idx // n_cols
        col_i = idx % n_cols
        subplot_idx = row_i * n_cols + col_i + 1
        ax = fig.add_subplot(n_rows, n_cols, subplot_idx, projection='3d')

        Z_slice = get_Z(idx)
        corr_slice = get_corr(idx)
        s_val = slice_arr[idx]

        # Count passing cells in this slice
        n_pass_slice = int(np.sum(~np.isnan(corr_slice)))

        # ── Surface with facecolors ──
        if use_fine:
            try:
                sp_z = RectBivariateSpline(x_arr, y_arr, Z_slice)
                Z_f = sp_z(x_fine, y_fine)
                # Also interpolate correlation mask (nearest for boolean-like data)
                from scipy.interpolate import RegularGridInterpolator
                corr_filled = np.where(np.isnan(corr_slice), -1.0, corr_slice)
                rgi = RegularGridInterpolator(
                    (x_arr, y_arr), corr_filled, method='nearest',
                    bounds_error=False, fill_value=-1.0)
                pts = np.stack(np.meshgrid(x_fine, y_fine, indexing='ij'), axis=-1)
                corr_f = rgi(pts.reshape(-1, 2)).reshape(len(x_fine), len(y_fine))
                corr_f[corr_f < 0] = np.nan

                fc = _make_facecolors(Z_f, corr_f)
                ax.plot_surface(X_fine, Y_fine, Z_f,
                                facecolors=fc,
                                rstride=1, cstride=1,
                                linewidth=0, antialiased=False,
                                shade=False)
            except Exception:
                fc = _make_facecolors(Z_slice, corr_slice)
                ax.plot_surface(X_mesh, Y_mesh, Z_slice,
                                facecolors=fc,
                                shade=False)
        else:
            fc = _make_facecolors(Z_slice, corr_slice)
            ax.plot_surface(X_mesh, Y_mesh, Z_slice,
                            facecolors=fc,
                            shade=False)

        # ── Best Fit marker ──
        if best_cond is not None:
            if slice_mode == 'pressure':
                bc_slice_val = best_cond['p0_MPa']
                bc_x = best_cond['T_C']
                bc_y = np.log10(best_cond['v_ms'])
            elif slice_mode == 'temperature':
                bc_slice_val = best_cond['T_C']
                bc_x = best_cond['p0_MPa']
                bc_y = np.log10(best_cond['v_ms'])
            else:
                bc_slice_val = best_cond['v_ms']
                bc_x = best_cond['T_C']
                bc_y = best_cond['p0_MPa']

            bc_slice_idx = np.argmin(np.abs(slice_arr - bc_slice_val))
            if bc_slice_idx == idx:
                bc_ix = np.argmin(np.abs(x_arr - bc_x))
                bc_iy = np.argmin(np.abs(y_arr - bc_y))
                bc_z = float(Z_slice[bc_ix, bc_iy])
                ax.scatter([bc_x], [bc_y], [bc_z],
                           c='yellow', s=300, marker='*',
                           edgecolors='black', linewidth=1.5,
                           zorder=20)
                ax.text(bc_x, bc_y, bc_z * 1.03,
                        'Best Fit', fontsize=9, fontweight='bold',
                        color='black', ha='center', zorder=21)

        ax.set_xlabel(x_label, fontsize=11, labelpad=6)
        ax.set_ylabel(y_label, fontsize=11, labelpad=6)
        ax.set_zlabel(z_label, fontsize=11, labelpad=6)
        ax.set_zlim(z_min, z_max)
        ax.view_init(elev=25, azim=225)
        ax.tick_params(labelsize=9)

        # Title
        if slice_mode == 'velocity':
            title_val = f"{s_val:.4g}"
        elif slice_mode == 'pressure':
            title_val = f"{s_val:.3g}"
        else:
            title_val = f"{s_val:.0f}"
        branch_disp = display_branch.capitalize()
        pass_info = f"({n_pass_slice}개 통과)" if n_pass_slice > 0 else "(통과 없음)"
        ax.set_title(f'{branch_disp}  {slice_label}={title_val} {slice_unit}\n'
                     f'{z_label}: {Z_slice.min():.4f} ~ {Z_slice.max():.4f}  {pass_info}',
                     fontsize=12, fontweight='bold',
                     color='#1565C0' if display_branch == 'cold' else '#C62828')

    # ── Text box with range info ──
    info_lines = []
    info_lines.append(f"[{sr_item_name}] 전수 탐색 결과")
    info_lines.append(f"총 탐색: {sr['total_searched']:,}개 조건")
    info_lines.append(f"|r| ≥ {thr:.2f} 통과: {sr['passing_count']:,}개 "
                      f"({sr['passing_count']/max(sr['total_searched'],1)*100:.1f}%)")

    if sr['passing_count'] > 0:
        T_r = sr['T_range']
        p_r = sr['p0_range']
        v_r = sr['v_range']
        info_lines.append("")
        info_lines.append("■ 상관성 충족 범위:")
        info_lines.append(f"  온도: {T_r[0]:.0f} ~ {T_r[1]:.0f} °C")
        info_lines.append(f"  하중: {p_r[0]:.3g} ~ {p_r[1]:.3g} MPa")
        info_lines.append(f"  속도: {v_r[0]:.4g} ~ {v_r[1]:.4g} m/s")
        info_lines.append(f"  Branch: {', '.join(sr['branch_list'])}")

    if best_cond is not None:
        info_lines.append("")
        info_lines.append(f"★ Best Fit (r={best_cond['corr']:+.4f}):")
        info_lines.append(f"  T={best_cond['T_C']:.0f}°C, "
                          f"p₀={best_cond['p0_MPa']:.3g} MPa, "
                          f"v={best_cond['v_ms']:.4g} m/s [{best_cond['branch']}]")

    info_text = '\n'.join(info_lines)
    fig.text(0.99, 0.01, info_text,
             fontsize=10, family='monospace',
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='lightyellow',
                       alpha=0.95, edgecolor='#444'))

    fig.suptitle(f"마찰맵 3D 표면 — 상관성 통과 영역 표시 ({slice_label}별, {display_branch.capitalize()})",
                 fontsize=14, fontweight='bold')

    # ── Scrollable canvas (Friction Map 탭과 동일 패턴) ──
    scroll_frame = tk.Frame(plot_frame, bg='white')
    scroll_frame.pack(fill=tk.BOTH, expand=True)

    v_scrollbar = ttk.Scrollbar(scroll_frame, orient=tk.VERTICAL)
    v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    plot_canvas = tk.Canvas(scroll_frame, yscrollcommand=v_scrollbar.set,
                            bg='white', highlightthickness=0)
    v_scrollbar.config(command=plot_canvas.yview)
    plot_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    inner_frame = tk.Frame(plot_canvas, bg='white')
    mpl_canvas = FigureCanvasTkAgg(fig, master=inner_frame)
    mpl_canvas.draw()
    canvas_widget = mpl_canvas.get_tk_widget()
    fig_w_px = int(fig_w * fig_dpi)
    fig_h_px = int(fig_h * fig_dpi)
    canvas_widget.config(width=fig_w_px, height=fig_h_px)
    canvas_widget.pack(fill=tk.X)

    win_id = plot_canvas.create_window((0, 0), window=inner_frame, anchor='nw',
                                       width=fig_w_px)

    def _on_configure(event):
        plot_canvas.configure(scrollregion=plot_canvas.bbox('all'))
        new_w = event.width
        if new_w > 100:
            plot_canvas.itemconfig(win_id, width=new_w)
    plot_canvas.bind('<Configure>', _on_configure)

    def _on_mousewheel(event):
        if event.delta:
            plot_canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')
        elif event.num == 4:
            plot_canvas.yview_scroll(-3, 'units')
        elif event.num == 5:
            plot_canvas.yview_scroll(3, 'units')

    plot_canvas.bind('<MouseWheel>', _on_mousewheel)
    plot_canvas.bind('<Button-4>', _on_mousewheel)
    plot_canvas.bind('<Button-5>', _on_mousewheel)


# ── Export ──

def _export_vtm_results_csv(self):
    """Export matching results to CSV."""
    if not self._vtm_results:
        messagebox.showwarning("CSV 내보내기", "먼저 상관성 분석을 실행하세요.")
        return

    path = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV", "*.csv"), ("All", "*.*")],
        title="실차 매칭 결과 CSV 내보내기")
    if not path:
        return

    try:
        import csv
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)

            writer.writerow(["실차 평가 ↔ 마찰맵 상관성 분석 결과"])
            writer.writerow([f"샘플: {', '.join(self._vtm_samples)}"])
            writer.writerow([])

            # Sample-map mapping
            writer.writerow(["샘플별 마찰맵"])
            writer.writerow(["샘플", "마찰맵"])
            for s in self._vtm_samples:
                writer.writerow([s, self._vtm_sample_maps.get(s, '(미연결)')])
            writer.writerow([])

            # Mu indices
            writer.writerow(["마찰맵 mu 인덱스"])
            writer.writerow(["샘플", "mu 인덱스"])
            mu_idx = self._vtm_results[0]['mu_indices'] if self._vtm_results else {}
            for s in self._vtm_samples:
                writer.writerow([s,
                                 f"{mu_idx.get(s, ''):.1f}"
                                 if s in mu_idx else ''])
            writer.writerow([])

            for res in self._vtm_results:
                writer.writerow([f"항목: {res['item_name']} [{res['item_unit']}]"])
                corr = res['correlation']
                writer.writerow([f"상관계수 r: {corr:+.4f}" if corr is not None else "상관계수: N/A"])
                best = res.get('best_condition', '')
                if best:
                    writer.writerow([f"최적 조건: {best}"])

                writer.writerow(["샘플", "실차 결과", "인덱스"])
                for s in self._vtm_samples:
                    item_data = next((it for it in self._vtm_items if it['name'] == res['item_name']), None)
                    raw = item_data['values'].get(s, '') if item_data else ''
                    idx = res['test_indices'].get(s, '')
                    writer.writerow([s,
                                     f"{raw:.4f}" if isinstance(raw, float) else str(raw),
                                     f"{idx:.1f}" if isinstance(idx, float) else str(idx)])

                # Map matches
                matches = res.get('map_matches', [])
                if matches:
                    writer.writerow([])
                    writer.writerow(["샘플", "맵 mu", "온도(°C)", "압력(MPa)",
                                     "속도(m/s)", "Branch"])
                    for m in matches:
                        writer.writerow([
                            m['sample'], f"{m['matched_mu']:.4f}",
                            f"{m['T_C']:.0f}", f"{m['p0_MPa']:.3g}",
                            f"{m['v_ms']:.4g}", m['branch']])

                writer.writerow([])

        self._show_status(f"CSV 내보내기 완료: {os.path.basename(path)}", 'success')
    except Exception as e:
        messagebox.showerror("CSV 오류", str(e))


# ── Binding function ──
def bind_vehicle_test_matching(cls):
    """Bind all vehicle test matching methods to the main GUI class."""
    cls._create_vehicle_test_matching_tab = _create_vehicle_test_matching_tab
    cls._vtm_create_sample_map_table = _vtm_create_sample_map_table
    cls._vtm_add_item = _vtm_add_item
    cls._vtm_remove_selected_item = _vtm_remove_selected_item
    cls._vtm_refresh_items_listbox = _vtm_refresh_items_listbox
    cls._vtm_load_default_preset = _vtm_load_default_preset
    cls._vtm_save_items_to_file = _vtm_save_items_to_file
    cls._vtm_load_items_from_file = _vtm_load_items_from_file
    cls._open_vtm_item_manager = _open_vtm_item_manager
    cls._vtm_create_data_table = _vtm_create_data_table
    cls._vtm_collect_table_data = _vtm_collect_table_data
    cls._vtm_compute_indices = _vtm_compute_indices
    cls._vtm_extract_mu_from_map = _vtm_extract_mu_from_map
    cls._vtm_find_all_correlation_conditions = _vtm_find_all_correlation_conditions
    cls._run_vehicle_test_matching = _run_vehicle_test_matching
    cls._vtm_display_results = _vtm_display_results
    cls._vtm_plot_heatmap = _vtm_plot_heatmap
    cls._vtm_plot_index_comparison = _vtm_plot_index_comparison
    cls._vtm_plot_match_detail = _vtm_plot_match_detail
    cls._vtm_plot_map_conditions = _vtm_plot_map_conditions
    cls._export_vtm_results_csv = _export_vtm_results_csv
