"""
Braking Simulation Tab — Straight-line braking with longitudinal slip.

All methods are designed to be bound to the main PerssonModelGUI_V2 class.
Usage in main.py:
    from braking_simulation import bind_braking_simulation
    bind_braking_simulation(PerssonModelGUI_V2)
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np


# ================================================================
# ====  Braking Simulation (직선 제동 시뮬레이션) Tab  ====
# ================================================================

def _create_braking_simulation_tab(self, parent):
    """Braking Simulation tab — straight-line braking.

    2x2 grid layout:
      Left-top:  Road animation + Speed/Decel graphs
      Right-top: Brake pedal gauge + Tire animation
      Left-bottom: 5 footprint contour plots
      Right-bottom: Fx vs SR + Fx vs Distance
    """
    from matplotlib.figure import Figure as _Fig
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg as _FCA
    from matplotlib.gridspec import GridSpec

    # ── State ──
    self._bk_sim_data = None
    self._bk_playing = False
    self._bk_frame_idx = 0
    self._bk_play_after_id = None
    self._bk_frame_accum = 0.0
    self._bk_road_phase = 0.0
    self._bk_tire_rot_deg = 0.0

    D = self.DIMS
    C = self.COLORS

    main = ttk.Frame(parent)
    main.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

    # ── Left: fixed-width scrollable controls ──
    left = ttk.Frame(main, width=D['panel_width'])
    left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 4))
    left.pack_propagate(False)
    self._add_logo_to_panel(left)
    self._create_panel_toolbar(left, buttons=[
        ("제동 시뮬레이션 실행", self._run_braking_simulation, 'Accent.TButton'),
    ])

    scroll_canvas = tk.Canvas(left, highlightthickness=0, bg=C['bg'])
    scrollbar = ttk.Scrollbar(left, orient='vertical',
                              command=scroll_canvas.yview)
    left_panel = ttk.Frame(scroll_canvas)
    left_panel.bind('<Configure>',
                    lambda e: scroll_canvas.configure(
                        scrollregion=scroll_canvas.bbox('all')))
    _cw_id = scroll_canvas.create_window(
        (0, 0), window=left_panel, anchor='nw',
        width=D['panel_width'] - 18)
    scroll_canvas.configure(yscrollcommand=scrollbar.set)
    scroll_canvas.bind('<Configure>',
                       lambda e, _c=scroll_canvas, _id=_cw_id:
                           _c.itemconfigure(_id, width=e.width))
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

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

    # ── Right: 2x2 visualization ──
    viz = ttk.Frame(main)
    viz.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    viz_vpane = ttk.PanedWindow(viz, orient=tk.VERTICAL)
    viz_vpane.pack(fill=tk.BOTH, expand=True)
    self._bk_viz_vpane = viz_vpane

    # ── Top row ──
    top_row = ttk.Frame(viz_vpane)
    viz_vpane.add(top_row, weight=55)
    top_hpane = ttk.PanedWindow(top_row, orient=tk.HORIZONTAL)
    top_hpane.pack(fill=tk.BOTH, expand=True)
    self._bk_top_hpane = top_hpane

    # Top-Left: Road animation + graphs (3 subplots stacked)
    road_frame = ttk.Frame(top_hpane)
    top_hpane.add(road_frame, weight=55)

    self._bk_road_fig = _Fig(figsize=(5, 4.2), dpi=100, facecolor='#1A1A2E')
    gs_road = GridSpec(3, 1, figure=self._bk_road_fig,
                       hspace=0.45,
                       left=0.12, right=0.95, top=0.95, bottom=0.08,
                       height_ratios=[1.2, 1, 1])

    # Road animation subplot
    self._bk_ax_road = self._bk_road_fig.add_subplot(gs_road[0, 0])
    self._bk_ax_road.set_facecolor('#4A4A4A')
    self._bk_ax_road.set_ylim(-1.5, 1.5)
    self._bk_ax_road.set_yticks([])
    self._bk_ax_road.set_xlabel('Distance [m]', fontsize=7, color='#CCC')
    self._bk_ax_road.set_title('Braking Road View', fontsize=9,
                                fontweight='bold', color='#EEE')
    self._bk_ax_road.tick_params(labelsize=6, colors='#AAA')

    # Speed vs Distance
    self._bk_ax_speed = self._bk_road_fig.add_subplot(gs_road[1, 0])
    self._bk_ax_speed.set_title('Speed vs Distance', fontsize=9, fontweight='bold')
    self._bk_ax_speed.set_xlabel('Distance [m]', fontsize=7)
    self._bk_ax_speed.set_ylabel('Speed [km/h]', fontsize=7)
    self._bk_ax_speed.tick_params(labelsize=6)
    self._bk_ax_speed.grid(True, alpha=0.3)

    # Deceleration vs Distance
    self._bk_ax_decel = self._bk_road_fig.add_subplot(gs_road[2, 0])
    self._bk_ax_decel.set_title('Deceleration vs Distance', fontsize=9, fontweight='bold')
    self._bk_ax_decel.set_xlabel('Distance [m]', fontsize=7)
    self._bk_ax_decel.set_ylabel('Decel [g]', fontsize=7)
    self._bk_ax_decel.tick_params(labelsize=6)
    self._bk_ax_decel.grid(True, alpha=0.3)

    self._bk_road_canvas = _FCA(self._bk_road_fig, road_frame)
    self._bk_road_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Top-Right: Brake pedal + Tire animation
    pedal_tire_frame = ttk.Frame(top_hpane)
    top_hpane.add(pedal_tire_frame, weight=45)
    pedal_inner = ttk.Frame(pedal_tire_frame)
    pedal_inner.pack(fill=tk.BOTH, expand=True)

    # Brake pedal gauge
    self._bk_pedal_fig = _Fig(figsize=(2.2, 2.2), dpi=80, facecolor='#F8FAFC')
    self._bk_pedal_ax = self._bk_pedal_fig.add_axes([0.05, 0.05, 0.9, 0.9])
    self._bk_pedal_ax.set_xlim(-1.5, 1.5)
    self._bk_pedal_ax.set_ylim(-1.5, 1.5)
    self._bk_pedal_ax.set_aspect('equal')
    self._bk_pedal_ax.axis('off')
    self._bk_pedal_canvas = _FCA(self._bk_pedal_fig, pedal_inner)
    self._bk_pedal_canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)

    # ABS Logic Graph (replaces tire side-view)
    self._bk_abs_fig = _Fig(figsize=(2.8, 2.8), dpi=80, facecolor='#F8FAFC')
    gs_abs = GridSpec(2, 1, figure=self._bk_abs_fig,
                      hspace=0.45,
                      left=0.18, right=0.95, top=0.92, bottom=0.10)
    self._bk_abs_ax_sr = self._bk_abs_fig.add_subplot(gs_abs[0, 0])
    self._bk_abs_ax_sr.set_title('Slip Ratio & ABS', fontsize=9, fontweight='bold')
    self._bk_abs_ax_sr.set_ylabel('|SR| [%]', fontsize=7)
    self._bk_abs_ax_sr.tick_params(labelsize=6)
    self._bk_abs_ax_sr.grid(True, alpha=0.3)

    self._bk_abs_ax_bp = self._bk_abs_fig.add_subplot(gs_abs[1, 0])
    self._bk_abs_ax_bp.set_title('Brake Pressure', fontsize=9, fontweight='bold')
    self._bk_abs_ax_bp.set_xlabel('Time [s]', fontsize=7)
    self._bk_abs_ax_bp.set_ylabel('Brake', fontsize=7)
    self._bk_abs_ax_bp.tick_params(labelsize=6)
    self._bk_abs_ax_bp.set_yticks([0, 1])
    self._bk_abs_ax_bp.set_yticklabels(['OFF', 'ON'], fontsize=6)
    self._bk_abs_ax_bp.grid(True, alpha=0.3)

    self._bk_abs_canvas = _FCA(self._bk_abs_fig, pedal_inner)
    self._bk_abs_canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)

    self._bk_hud_var = tk.StringVar(value="SR: 0.0%  |  Fx: 0 N")
    ttk.Label(pedal_tire_frame, textvariable=self._bk_hud_var,
              font=('Consolas', 9), foreground='#0369A1').pack(pady=1)

    # ── Bottom row ──
    bot_row = ttk.Frame(viz_vpane)
    viz_vpane.add(bot_row, weight=45)
    bot_hpane = ttk.PanedWindow(bot_row, orient=tk.HORIZONTAL)
    bot_hpane.pack(fill=tk.BOTH, expand=True)
    self._bk_bot_hpane = bot_hpane

    # Bottom-Left: 5 contour plots
    contour_frame = ttk.Frame(bot_hpane)
    bot_hpane.add(contour_frame, weight=55)

    self._bk_contour_fig = _Fig(figsize=(10, 3.2), dpi=100)
    gs_contour = GridSpec(1, 5, figure=self._bk_contour_fig,
                          wspace=0.50,
                          left=0.04, right=0.98, top=0.86, bottom=0.18)
    self._bk_ax_stick = self._bk_contour_fig.add_subplot(gs_contour[0, 0])
    self._bk_ax_speed_c = self._bk_contour_fig.add_subplot(gs_contour[0, 1])
    self._bk_ax_press = self._bk_contour_fig.add_subplot(gs_contour[0, 2])
    self._bk_ax_temp = self._bk_contour_fig.add_subplot(gs_contour[0, 3])
    self._bk_ax_fric = self._bk_contour_fig.add_subplot(gs_contour[0, 4])

    for ax, title in [(self._bk_ax_stick, 'Adhesion / Sliding'),
                       (self._bk_ax_speed_c, 'Sliding Speed'),
                       (self._bk_ax_press, 'Contact Pressure [bar]'),
                       (self._bk_ax_temp, 'Temperature [°C]'),
                       (self._bk_ax_fric, 'Friction Force [N/node]')]:
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.set_xlabel('x [m]', fontsize=7)
        ax.set_ylabel('y [m]', fontsize=7)
        ax.tick_params(labelsize=6)

    self._bk_contour_canvas = _FCA(self._bk_contour_fig, contour_frame)
    self._bk_contour_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Bottom-Right: Fx vs SR + Fx vs Distance
    fx_frame = ttk.Frame(bot_hpane)
    bot_hpane.add(fx_frame, weight=45)

    self._bk_fx_fig = _Fig(figsize=(4.5, 4.8), dpi=100)
    gs_fx = GridSpec(3, 1, figure=self._bk_fx_fig,
                     hspace=0.55,
                     left=0.15, right=0.95, top=0.94, bottom=0.08)
    self._bk_ax_fx_sr = self._bk_fx_fig.add_subplot(gs_fx[0, 0])
    self._bk_ax_fx_sr.set_title('Fx vs Slip Ratio', fontsize=9, fontweight='bold')
    self._bk_ax_fx_sr.set_xlabel('SR [%]', fontsize=8)
    self._bk_ax_fx_sr.set_ylabel('Fx [N]', fontsize=8)
    self._bk_ax_fx_sr.grid(True, alpha=0.3)

    self._bk_ax_ke_dist = self._bk_fx_fig.add_subplot(gs_fx[1, 0])
    self._bk_ax_ke_dist.set_title('Kinetic Energy vs Distance', fontsize=9, fontweight='bold')
    self._bk_ax_ke_dist.set_xlabel('Distance [m]', fontsize=8)
    self._bk_ax_ke_dist.set_ylabel('Energy [kJ]', fontsize=8)
    self._bk_ax_ke_dist.grid(True, alpha=0.3)

    self._bk_ax_fx_dist = self._bk_fx_fig.add_subplot(gs_fx[2, 0])
    self._bk_ax_fx_dist.set_title('Fx vs Distance', fontsize=9, fontweight='bold')
    self._bk_ax_fx_dist.set_xlabel('Distance [m]', fontsize=8)
    self._bk_ax_fx_dist.set_ylabel('Fx [N]', fontsize=8)
    self._bk_ax_fx_dist.grid(True, alpha=0.3)

    self._bk_fx_canvas = _FCA(self._bk_fx_fig, fx_frame)
    self._bk_fx_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ── Deferred layout ──
    self._bk_layout_initialized = False

    def _bk_force_layout(event=None):
        if self._bk_layout_initialized:
            return
        viz.update_idletasks()
        vh = viz_vpane.winfo_height()
        vw = viz_vpane.winfo_width()
        if vh < 10 or vw < 10:
            viz.after(100, _bk_force_layout)
            return
        self._bk_layout_initialized = True
        try:
            viz_vpane.sashpos(0, int(vh * 0.55))
        except Exception:
            pass
        tw = top_hpane.winfo_width()
        bw = bot_hpane.winfo_width()
        if tw > 10:
            try: top_hpane.sashpos(0, int(tw * 0.55))
            except Exception: pass
        if bw > 10:
            try: bot_hpane.sashpos(0, int(bw * 0.55))
            except Exception: pass
        viz.after(50, _bk_redraw_all)

    def _bk_redraw_all():
        for cv in (self._bk_road_canvas, self._bk_pedal_canvas,
                   self._bk_abs_canvas, self._bk_contour_canvas,
                   self._bk_fx_canvas):
            try:
                w = cv.get_tk_widget()
                ww, wh = w.winfo_width(), w.winfo_height()
                if ww > 1 and wh > 1:
                    cv.figure.set_size_inches(ww / cv.figure.dpi, wh / cv.figure.dpi,
                                              forward=False)
                cv.draw_idle()
            except Exception:
                pass

    viz.bind('<Map>', _bk_force_layout, add='+')

    # ===================== Left Panel Controls =====================
    sec1 = self._create_section(left_panel, "1) 제동 조건")
    for label, var_name, default, unit in [
        ("초기 속도:", 'bk_v0_var', "200", "km/h"),
        ("대기온도:", 'bk_T_amb_var', "25.0", "°C"),
        ("차량 질량 m:", 'bk_mass_var', "2000", "kg"),
        ("Cd·A (공기저항):", 'bk_cda_var', "1.2", "m²"),
        ("Cl·A (다운포스):", 'bk_cla_var', "3.5", "m²"),
    ]:
        row = ttk.Frame(sec1); row.pack(fill=tk.X, pady=1)
        ttk.Label(row, text=label, font=self.FONTS['body']).pack(side=tk.LEFT)
        var = tk.StringVar(value=default)
        setattr(self, var_name, var)
        ttk.Entry(row, textvariable=var, width=8).pack(side=tk.LEFT, padx=2)
        if unit:
            ttk.Label(row, text=unit, font=self.FONTS['small'],
                      foreground='#64748B').pack(side=tk.LEFT)

    sec2 = self._create_section(left_panel, "2) 브레이크 시스템")
    for label, var_name, default, unit in [
        ("최대 감속:", 'bk_brake_g_var', "5.0", "g"),
    ]:
        row = ttk.Frame(sec2); row.pack(fill=tk.X, pady=1)
        ttk.Label(row, text=label, font=self.FONTS['body']).pack(side=tk.LEFT)
        var = tk.StringVar(value=default)
        setattr(self, var_name, var)
        ttk.Entry(row, textvariable=var, width=8).pack(side=tk.LEFT, padx=2)
        if unit:
            ttk.Label(row, text=unit, font=self.FONTS['small'],
                      foreground='#64748B').pack(side=tk.LEFT)

    # ABS toggle
    abs_row = ttk.Frame(sec2); abs_row.pack(fill=tk.X, pady=2)
    ttk.Label(abs_row, text="ABS 모드:", font=self.FONTS['body']).pack(side=tk.LEFT)
    self.bk_abs_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(abs_row, text="ON", variable=self.bk_abs_var).pack(side=tk.LEFT, padx=4)

    for label, var_name, default, unit in [
        ("SR 상한 (release):", 'bk_sr_upper_var', "15", "%"),
        ("SR 하한 (apply):", 'bk_sr_lower_var', "8", "%"),
        ("ABS 주기:", 'bk_abs_freq_var', "15", "Hz"),
    ]:
        row = ttk.Frame(sec2); row.pack(fill=tk.X, pady=1)
        ttk.Label(row, text=label, font=self.FONTS['body']).pack(side=tk.LEFT)
        var = tk.StringVar(value=default)
        setattr(self, var_name, var)
        ttk.Entry(row, textvariable=var, width=8).pack(side=tk.LEFT, padx=2)
        if unit:
            ttk.Label(row, text=unit, font=self.FONTS['small'],
                      foreground='#64748B').pack(side=tk.LEFT)

    # 2D Brush sync
    sec_brush = self._create_section(left_panel, "2-1) 2D Brush 모델 연동")
    self.bk_brush_info_var = tk.StringVar(
        value="※ 2D Brush 탭의 Nx, Ny, 풋프린트,\n"
              "   마찰감도 계수 등을 자동 동기화합니다.")
    ttk.Label(sec_brush, textvariable=self.bk_brush_info_var,
              font=self.FONTS['small'], foreground='#1565C0',
              justify='left').pack(anchor='w', padx=4, pady=2)
    ttk.Button(sec_brush, text="Brush 설정 동기화 확인",
               command=self._show_braking_brush_sync_info, width=22).pack(anchor='w', padx=4, pady=2)

    # ── Pressure preset selector ──
    sec_press = self._create_section(left_panel, "2-2) 접촉압력 프리셋")
    pr_row1 = ttk.Frame(sec_press); pr_row1.pack(fill=tk.X, pady=1)
    ttk.Label(pr_row1, text="압력 분포:", font=self.FONTS['body']).pack(side=tk.LEFT)
    self.bk_pressure_type_var = tk.StringVar(value='braking_front')
    _pressure_presets = [
        ('parabolic', '포물선 (기본)'),
        ('elliptic', '타원 (헤르츠)'),
        ('uniform', '균일'),
        ('dual_peak', '이중 피크'),
        ('braking_front', '제동 전방편중'),
    ]
    _pr_combo = ttk.Combobox(pr_row1, textvariable=self.bk_pressure_type_var,
                              values=[k for k, _ in _pressure_presets],
                              state='readonly', width=14)
    _pr_combo.pack(side=tk.LEFT, padx=2)

    # Preset description label
    self._bk_press_desc_var = tk.StringVar(value='제동 시 전방(리딩엣지) 압력 편중')
    ttk.Label(sec_press, textvariable=self._bk_press_desc_var,
              font=self.FONTS['small'], foreground='#64748B',
              wraplength=200, justify='left').pack(anchor='w', padx=4, pady=1)

    _desc_map = dict(_pressure_presets)
    def _on_press_preset(event=None):
        k = self.bk_pressure_type_var.get()
        descs = {
            'parabolic': '대칭 포물선 압력 분포 (정적 상태)',
            'elliptic': '헤르츠 접촉이론 기반 타원형 분포',
            'uniform': '균일한 압력 분포',
            'dual_peak': '양측 피크 분포 (숄더 타이어)',
            'braking_front': '제동 시 전방(리딩엣지) 압력 편중',
        }
        self._bk_press_desc_var.set(descs.get(k, k))
    _pr_combo.bind('<<ComboboxSelected>>', _on_press_preset)

    # Braking bias intensity
    pr_row2 = ttk.Frame(sec_press); pr_row2.pack(fill=tk.X, pady=1)
    ttk.Label(pr_row2, text="전방 편중 강도:", font=self.FONTS['body']).pack(side=tk.LEFT)
    self.bk_braking_bias_var = tk.StringVar(value="0.5")
    ttk.Entry(pr_row2, textvariable=self.bk_braking_bias_var, width=6).pack(side=tk.LEFT, padx=2)
    ttk.Label(pr_row2, text="(0~1)", font=self.FONTS['small'],
              foreground='#64748B').pack(side=tk.LEFT)

    # Execution & playback
    sec3 = self._create_section(left_panel, "3) 실행 & 재생")
    calc_row = ttk.Frame(sec3); calc_row.pack(fill=tk.X, pady=2)
    self.bk_run_btn = ttk.Button(calc_row, text="▶ 제동 시뮬레이션 실행",
                                  command=self._run_braking_simulation, width=22,
                                  style='Accent.TButton')
    self.bk_run_btn.pack(side=tk.LEFT, padx=2)
    self.bk_progress_var = tk.IntVar()
    self.bk_progress_bar = ttk.Progressbar(calc_row, variable=self.bk_progress_var,
                                            maximum=100, length=120)
    self.bk_progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

    play_row = ttk.Frame(sec3); play_row.pack(fill=tk.X, pady=4)
    for text, cmd in [("▶", self._braking_play), ("⏸", self._braking_pause),
                      ("⏮", self._braking_reset)]:
        ttk.Button(play_row, text=text, width=3, command=cmd).pack(side=tk.LEFT, padx=1)

    self.bk_frame_label_var = tk.StringVar(value="t = 0.00 s  |  0 km/h")
    ttk.Label(play_row, textvariable=self.bk_frame_label_var,
              font=self.FONTS['small'], foreground='#0369A1').pack(side=tk.LEFT, padx=6)

    speed_row = ttk.Frame(sec3); speed_row.pack(fill=tk.X, pady=1)
    self._bk_speed_label_var = tk.StringVar(value="재생 속도: 2.0x")
    ttk.Label(speed_row, textvariable=self._bk_speed_label_var,
              font=self.FONTS['small']).pack(side=tk.LEFT, padx=4)
    self._bk_speed_mult = tk.DoubleVar(value=2.0)
    ttk.Scale(speed_row, from_=0.5, to=10.0, orient='horizontal',
              variable=self._bk_speed_mult,
              command=self._on_braking_speed_slider).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

    slider_row = ttk.Frame(sec3); slider_row.pack(fill=tk.X, pady=2)
    self.bk_time_slider = ttk.Scale(slider_row, from_=0, to=100,
                                     orient='horizontal',
                                     command=self._on_braking_slider_change)
    self.bk_time_slider.pack(fill=tk.X, padx=4)

    sec4 = self._create_section(left_panel, "4) 제동 결과")
    self.bk_result_text = tk.Text(sec4, height=14, font=('Consolas', 9),
                                   bg='#F8FAFC', relief='flat', wrap='word',
                                   state='disabled')
    self.bk_result_text.pack(fill=tk.X, padx=4, pady=2)

    # Initialize visualizations
    self._init_braking_pedal_gauge()
    self._init_braking_abs_graph()
    self._draw_braking_road_static()


# ── Simulation engine ──
def _run_braking_simulation(self):
    """Run straight-line braking simulation with optional ABS."""

    # Validate Cold & Hot friction map
    if self.cold_hot_results is None:
        messagebox.showerror(
            "데이터 없음",
            "Cold & Hot Branch 마찰맵 데이터가 없습니다.\n\n"
            "Braking Simulation을 실행하려면 먼저:\n"
            "  1) μ_visc 계산 탭에서 μ_visc 계산\n"
            "  2) μ_adh 계산 탭에서 μ_adh 계산\n"
            "  3) Cold & Hot Branch 탭에서 마찰맵 계산\n"
            "  4) 2D Brush Model 탭에서 시뮬레이션 실행\n\n"
            "위 단계를 완료한 후 다시 실행하세요.")
        return

    try:
        v0_kmh = float(self.bk_v0_var.get())
        T_amb = float(self.bk_T_amb_var.get())
        mass = float(self.bk_mass_var.get())
        cda = float(self.bk_cda_var.get())
        cla = float(self.bk_cla_var.get())
        brake_g_max = float(self.bk_brake_g_var.get())
    except ValueError:
        messagebox.showerror("오류", "파라미터를 확인하세요.")
        return

    abs_on = self.bk_abs_var.get()
    try:
        sr_upper = float(self.bk_sr_upper_var.get()) / 100.0
        sr_lower = float(self.bk_sr_lower_var.get()) / 100.0
        abs_freq = float(self.bk_abs_freq_var.get())
    except ValueError:
        sr_upper, sr_lower, abs_freq = 0.15, 0.08, 15.0

    g_acc = 9.81
    rho = 1.225
    v0 = v0_kmh / 3.6
    dt = 0.001  # 1 ms time step

    # Build friction LUT
    lut_cold, lut_hot = self._build_brush_lut()
    chr_ = self.cold_hot_results

    try:
        D_mm = float(self.br_D_macro_var.get())
        _s0 = 0.2 * D_mm * 1e-3
    except (AttributeError, ValueError):
        _s0 = chr_.get('s0', 0.001)
    try:
        _L_fp = float(self.br_L_var.get())
        _fscale = float(self.br_friction_scale_var.get())
    except (AttributeError, ValueError):
        _L_fp = 0.15
        _fscale = 0.5

    def mu_eff_at_v(v_q):
        v_q = np.atleast_1d(np.asarray(v_q, dtype=float))
        mu_c = lut_cold(v_q)
        mu_h = lut_hot(v_q)
        v_slide_rep = 0.1 * v_q
        t_max = _L_fp / np.clip(v_q, 0.5, None)
        s_max = v_slide_rep * t_max
        ratio = s_max / max(_s0, 1e-10)
        avg_blend = np.where(ratio > 1e-6,
                             (1.0 - np.exp(-ratio)) / ratio, 1.0)
        # NOTE: friction_scale is for brush-element tuning, NOT vehicle dynamics.
        # Vehicle-level mu uses the raw Cold/Hot friction map values.
        return mu_c * avg_blend + mu_h * (1.0 - avg_blend)

    # ── Time-stepping simulation ──
    max_steps = int(30.0 / dt)  # max 30 seconds
    t_arr = []
    v_arr = []
    dist_arr = []
    sr_arr = []
    fx_arr = []
    decel_g_arr = []
    brake_pressure_arr = []  # 0~1 normalized
    abs_events = 0

    v_car = v0
    v_wheel = v0  # wheel peripheral speed = omega * R_tire
    dist = 0.0
    t = 0.0
    R_tire = 0.33  # tire radius [m]
    I_wheel = 1.5  # wheel inertia [kg·m²] (per wheel)
    brake_on = True
    abs_cycle_timer = 0.0
    abs_cycle_period = 1.0 / abs_freq if abs_freq > 0 else 0.1

    for step in range(max_steps):
        if v_car <= 0.1:
            v_car = 0.0
            break

        # Fz per tire (weight + downforce)
        Fz_tire = (mass * g_acc + 0.5 * rho * cla * v_car**2) / 4.0

        # Slip ratio: SR = (v_wheel - v_car) / v_car
        # During braking: v_wheel < v_car → SR is negative
        if v_car > 0.5:
            sr = (v_wheel - v_car) / v_car
        else:
            sr = -1.0 if v_wheel < v_car else 0.0
        sr = np.clip(sr, -1.0, 0.0)

        # Effective friction at slip speed
        v_slip = abs(sr) * v_car
        mu = float(mu_eff_at_v(max(v_slip, 0.01)))

        # Braking force per tire (limited by friction)
        F_brake_max = mu * Fz_tire  # max friction force per tire

        # ABS logic
        if abs_on:
            abs_cycle_timer += dt
            if abs(sr) > sr_upper and brake_on:
                brake_on = False
                abs_events += 1
            elif abs(sr) < sr_lower and not brake_on:
                brake_on = True
            # ABS modulation with frequency limit
            if abs_cycle_timer < abs_cycle_period * 0.3 and not brake_on:
                brake_on = False  # hold release state
        else:
            brake_on = True

        # Applied brake torque → wheel deceleration
        if brake_on:
            brake_torque = brake_g_max * mass * g_acc * R_tire / 4.0
            bp = 1.0
        else:
            brake_torque = 0.0
            bp = 0.0

        # Wheel angular dynamics: I * alpha = -brake_torque + F_friction * R
        # F_friction opposes slip → if wheel slower, friction pulls wheel forward
        F_friction_on_wheel = -F_brake_max * np.sign(sr) if abs(sr) > 0.001 else 0.0
        omega = v_wheel / R_tire
        alpha = (-brake_torque + F_friction_on_wheel * R_tire) / (I_wheel + 1e-10)
        omega_new = omega + alpha * dt
        omega_new = max(omega_new, 0.0)  # wheel can't spin backward
        v_wheel = omega_new * R_tire

        # Vehicle dynamics: F = ma
        F_brake_total = 4 * F_brake_max * min(abs(sr) / 0.01, 1.0)  # friction force
        F_aero = 0.5 * rho * cda * v_car**2
        a_decel = (F_brake_total + F_aero) / mass
        v_car_new = v_car - a_decel * dt
        v_car_new = max(v_car_new, 0.0)

        # Update distance
        dist += 0.5 * (v_car + v_car_new) * dt

        # Store (downsample later)
        t_arr.append(t)
        v_arr.append(v_car * 3.6)
        dist_arr.append(dist)
        sr_arr.append(sr * 100)  # in %
        fx_arr.append(F_brake_total)
        decel_g_arr.append(a_decel / g_acc)
        brake_pressure_arr.append(bp)

        v_car = v_car_new
        t += dt
        abs_cycle_timer = abs_cycle_timer % abs_cycle_period if abs_on else 0

    self.bk_progress_var.set(40)
    self.root.update_idletasks()

    # Convert to numpy
    t_arr = np.array(t_arr)
    v_arr = np.array(v_arr)
    dist_arr = np.array(dist_arr)
    sr_arr = np.array(sr_arr)
    fx_arr = np.array(fx_arr)
    decel_g_arr = np.array(decel_g_arr)
    brake_pressure_arr = np.array(brake_pressure_arr)

    # Downsample to ~500 frames for display
    n_total = len(t_arr)
    n_frames = min(n_total, 500)
    idx_sel = np.linspace(0, n_total - 1, n_frames, dtype=int)

    t_ds = t_arr[idx_sel]
    v_ds = v_arr[idx_sel]
    dist_ds = dist_arr[idx_sel]
    sr_ds = sr_arr[idx_sel]
    fx_ds = fx_arr[idx_sel]
    decel_ds = decel_g_arr[idx_sel]
    bp_ds = brake_pressure_arr[idx_sel]

    self.bk_progress_var.set(60)
    self.root.update_idletasks()

    # Compute brush data for each frame (longitudinal slip)
    Fz_ds = (mass * g_acc + 0.5 * rho * cla * (v_ds / 3.6)**2) / 4.0
    brush_data = self._compute_braking_brush_data(sr_ds, v_ds / 3.6, Fz_ds, T_amb)

    self.bk_progress_var.set(90)
    self.root.update_idletasks()

    # ── Also run ABS OFF comparison if ABS was ON ──
    stop_dist_abs_off = None
    stop_time_abs_off = None
    if abs_on:
        v_c2 = v0; dist2 = 0.0; t2 = 0.0; v_w2 = v0
        for _ in range(max_steps):
            if v_c2 <= 0.1: v_c2 = 0; break
            Fz2 = (mass * g_acc + 0.5 * rho * cla * v_c2**2) / 4.0
            sr2 = (v_w2 - v_c2) / max(v_c2, 0.5)
            sr2 = np.clip(sr2, -1.0, 0.0)
            mu2 = float(mu_eff_at_v(max(abs(sr2) * v_c2, 0.01)))
            bt2 = brake_g_max * mass * g_acc * R_tire / 4.0
            F_fw2 = -mu2 * Fz2 * np.sign(sr2) if abs(sr2) > 0.001 else 0.0
            om2 = v_w2 / R_tire
            al2 = (-bt2 + F_fw2 * R_tire) / (I_wheel + 1e-10)
            om2 = max(om2 + al2 * dt, 0.0)
            v_w2 = om2 * R_tire
            Fb2 = 4 * mu2 * Fz2 * min(abs(sr2) / 0.01, 1.0)
            Fa2 = 0.5 * rho * cda * v_c2**2
            ad2 = (Fb2 + Fa2) / mass
            v_c2_new = max(v_c2 - ad2 * dt, 0.0)
            dist2 += 0.5 * (v_c2 + v_c2_new) * dt
            t2 += dt; v_c2 = v_c2_new
        stop_dist_abs_off = dist2
        stop_time_abs_off = t2

    self._bk_sim_data = {
        'time': t_ds, 'v_kmh': v_ds, 'dist': dist_ds,
        'sr_pct': sr_ds, 'Fx': fx_ds, 'decel_g': decel_ds,
        'brake_pressure': bp_ds,
        'n_frames': n_frames,
        'stop_dist': dist_arr[-1] if len(dist_arr) > 0 else 0,
        'stop_time': t_arr[-1] if len(t_arr) > 0 else 0,
        'max_decel_g': float(np.max(decel_g_arr)) if len(decel_g_arr) > 0 else 0,
        'avg_decel_g': float(np.mean(decel_g_arr)) if len(decel_g_arr) > 0 else 0,
        'max_sr_pct': float(np.min(sr_arr)) if len(sr_arr) > 0 else 0,
        'abs_events': abs_events,
        'abs_on': abs_on,
        'v0_kmh': v0_kmh,
        'T_amb': T_amb,
        'stop_dist_abs_off': stop_dist_abs_off,
        'stop_time_abs_off': stop_time_abs_off,
        'Fz': Fz_ds,
        'brush_data': brush_data,
    }

    self.bk_progress_var.set(100)
    self._update_braking_results_text()
    self._draw_braking_road_static()
    self._init_braking_abs_graph()
    self._init_braking_contour_artists()
    self._bk_frame_idx = 0
    self._update_braking_frame(0)
    self.bk_time_slider.configure(to=n_frames - 1)
    self._braking_play()


def _compute_braking_brush_data(self, sr_arr_pct, v_arr, Fz_arr, T_amb):
    """Brush model for braking — longitudinal slip only.

    sr_arr_pct: slip ratio in % (negative = braking)
    v_arr: vehicle speed in m/s
    Fz_arr: normal force per tire in N
    T_amb: ambient temperature (initial rubber temperature)
    """
    from scipy.interpolate import interp1d
    from scipy.ndimage import binary_dilation

    lut_cold, lut_hot = self._build_brush_lut()
    chr_ = self.cold_hot_results
    T0_base = T_amb  # Use user-specified ambient temperature

    try:
        D_mm = float(self.br_D_macro_var.get())
        s0 = 0.2 * D_mm * 1e-3
    except (AttributeError, ValueError):
        s0 = chr_.get('s0', 0.001)

    dT_interp = interp1d(chr_['v'], chr_['delta_T'], kind='linear',
                         bounds_error=False,
                         fill_value=(chr_['delta_T'][0], chr_['delta_T'][-1]))

    # Read 2D Brush params
    try:
        nx = int(self.br_Nx_var.get())
        ny = int(self.br_Ny_var.get())
        L = float(self.br_L_var.get())
        W = float(self.br_W_var.get())
        kx = float(self.br_kx_var.get())
        friction_scale = float(self.br_friction_scale_var.get())
        vc_brush = float(self.br_vc_var.get())
    except (AttributeError, ValueError):
        nx, ny = 64, 64
        L, W = 0.15, 0.12
        kx = 8e5
        friction_scale = 0.5
        vc_brush = 16.67

    # Read pressure preset from braking tab (overrides 2D brush tab)
    try:
        ptype = self.bk_pressure_type_var.get()
    except (AttributeError, ValueError):
        try:
            ptype = self.br_pressure_type_var.get()
        except (AttributeError, ValueError):
            ptype = 'parabolic'

    try:
        braking_bias = float(self.bk_braking_bias_var.get())
        braking_bias = max(0.0, min(1.0, braking_bias))
    except (AttributeError, ValueError):
        braking_bias = 0.0

    n = len(sr_arr_pct)
    a, b = L / 2.0, W / 2.0

    se_n = getattr(self, '_ELLIPSE_POWER', 2.5)
    try:
        se_n = float(self._br_ellipse_power_var.get())
    except (AttributeError, ValueError):
        pass

    x_arr = np.linspace(-a, a, nx)
    y_arr = np.linspace(-b, b, ny)
    dx_g = L / max(nx - 1, 1)
    dy_g = W / max(ny - 1, 1)
    dA = dx_g * dy_g
    xx, yy = np.meshgrid(x_arr, y_arr, indexing='ij')

    r_se = np.abs(xx / a)**se_n + np.abs(yy / b)**se_n
    mask = r_se <= 1.0
    mask_fill = binary_dilation(mask, iterations=2)

    # Base pressure (no SA-dependent lateral shift for braking)
    if ptype == 'uniform':
        p_base = np.ones((nx, ny))
    elif ptype == 'elliptic':
        p_base = np.sqrt(np.clip(1 - r_se, 0, None))
    elif ptype == 'dual_peak':
        lon_env = np.clip(1 - np.abs(2 * xx / L)**se_n, 0, None)
        y_norm = 2 * yy / W
        pk_pos, pk_w = 0.40, 0.45
        p_base = (np.exp(-((y_norm - pk_pos) / pk_w)**2) +
                  np.exp(-((y_norm + pk_pos) / pk_w)**2)) * lon_env
        p_base *= mask
    elif ptype == 'braking_front':
        # Parabolic base with forward (leading-edge) bias
        p_base = (np.clip(1 - np.abs(2 * xx / L)**se_n, 0, None) *
                  np.clip(1 - np.abs(2 * yy / W)**se_n, 0, None))
        # Leading edge is at -x (rolling direction ← , tire contacts road at -x first)
        # Shift pressure toward leading edge: multiply by (1 + bias * (-x/a))
        x_norm = xx / a  # -1 to +1
        bias_mult = 1.0 + braking_bias * (-x_norm)
        p_base *= np.clip(bias_mult, 0, None)
    else:  # parabolic
        p_base = (np.clip(1 - np.abs(2 * xx / L)**se_n, 0, None) *
                  np.clip(1 - np.abs(2 * yy / W)**se_n, 0, None))

    # Apply forward bias to any non-braking_front presets if bias > 0
    if ptype != 'braking_front' and braking_bias > 0:
        x_norm = xx / a
        bias_mult = 1.0 + braking_bias * (-x_norm)
        p_base *= np.clip(bias_mult, 0, None)

    # Contact time (rolling)
    t_contact = np.clip((xx + L / 2.0) / max(vc_brush, 0.01), 0, None)

    n_in_mask = max(np.sum(mask), 1)
    kx_node = kx / n_in_mask

    # Allocate
    is_sliding_all = np.full((n, nx, ny), np.nan)
    v_slide_all = np.full((n, nx, ny), np.nan)
    pressure_bar_all = np.full((n, nx, ny), np.nan)
    temperature_all = np.full((n, nx, ny), np.nan)
    friction_all = np.full((n, nx, ny), np.nan)
    Fx_brush = np.zeros(n)

    for fi in range(n):
        sr = sr_arr_pct[fi] / 100.0  # convert % to ratio
        vc = max(v_arr[fi], 0.5)
        Fz_i = Fz_arr[fi]

        # Longitudinal slip velocity: v_rim_x = v_car * |SR|
        v_rim_x = vc * abs(sr)
        v_rim_mag = v_rim_x + 1e-15

        # Pressure map (symmetric for braking, no lateral shift)
        p_map = p_base.copy() * mask
        p_total = np.sum(p_map) * dA
        if p_total > 0:
            p_map *= Fz_i / p_total
        Fz_ij = p_map * dA

        # Cold→Hot blend
        s_dist = v_rim_mag * t_contact
        blend = np.exp(-s_dist / max(s0, 1e-10))
        mu_cold_vals = lut_cold(np.full_like(t_contact, v_rim_mag))
        mu_hot_vals = lut_hot(np.full_like(t_contact, v_rim_mag))
        mu_eff = mu_cold_vals * blend + mu_hot_vals * (1.0 - blend)

        # Friction capacity
        F_fric_max = mu_eff * Fz_ij * friction_scale

        # Stick deformation (longitudinal)
        ux_stick = v_rim_x * t_contact  # longitudinal deformation
        Fspring_x = kx_node * ux_stick
        F_spring_mag = np.abs(Fspring_x)

        # Stick/slip
        is_sliding = (F_spring_mag > F_fric_max) & mask

        # Actual force per node (longitudinal direction)
        deform_dir_x = np.sign(ux_stick + 1e-20)
        Fnode_x = np.where(is_sliding,
                           F_fric_max * deform_dir_x,
                           Fspring_x)
        Fnode_x *= mask

        Fx_brush[fi] = np.sum(Fnode_x)

        # Local slip velocity
        _safe_Fsp = np.where(F_spring_mag > 1e-15, F_spring_mag, 1.0)
        scale = np.where(F_spring_mag > 1e-15,
                         np.clip(F_fric_max / _safe_Fsp, 0, 1), 1.0)
        v_slip = v_rim_mag * (1.0 - scale)
        v_slip = np.where(is_sliding, v_slip, 0.0) * mask

        # Temperature
        dT_local = dT_interp(np.clip(v_slip.ravel(), 0, None)).reshape(nx, ny)
        p_mean = np.mean(p_map[mask]) if np.any(mask) else 1.0
        p_ratio = p_map / max(p_mean, 1e-10)
        t_max_c = np.max(t_contact) if np.max(t_contact) > 0 else 1.0
        t_ratio = np.sqrt(t_contact / t_max_c)
        T_contact = T0_base + dT_local * p_ratio * t_ratio
        T_contact = T_contact * mask + T0_base * (~mask)

        # Store
        is_sliding_all[fi] = np.where(mask_fill, is_sliding.astype(float), np.nan)
        v_slide_all[fi] = np.where(mask_fill, v_slip, np.nan)
        pressure_bar_all[fi] = np.where(mask_fill, p_map * 1e-5, np.nan)
        temperature_all[fi] = np.where(mask_fill, T_contact, np.nan)
        F_node_mag = np.abs(Fnode_x) * mask
        friction_all[fi] = np.where(mask_fill, F_node_mag, np.nan)

    dx_e = x_arr[1] - x_arr[0]; dy_e = y_arr[1] - y_arr[0]
    x_edges = np.concatenate([x_arr - dx_e / 2, [x_arr[-1] + dx_e / 2]])
    y_edges = np.concatenate([y_arr - dy_e / 2, [y_arr[-1] + dy_e / 2]])

    return {
        'is_sliding': is_sliding_all, 'v_slide': v_slide_all,
        'pressure': pressure_bar_all, 'temperature': temperature_all,
        'friction': friction_all, 'Fx_brush': Fx_brush,
        'x_edges': x_edges, 'y_edges': y_edges,
        'mask': mask, 'mask_fill': mask_fill,
        'nx': nx, 'ny': ny, 'half_L': a, 'half_W': b,
        'pressure_type': ptype,
    }


def _update_braking_results_text(self):
    d = self._bk_sim_data
    if d is None:
        return
    abs_str = "ON" if d['abs_on'] else "OFF"
    text = (
        f"═══ 제동 시뮬레이션 결과 ═══\n\n"
        f"  초기 속도:    {d['v0_kmh']:.1f} km/h\n"
        f"  대기온도:     {d['T_amb']:.1f} °C\n"
        f"  ABS:         {abs_str}\n\n"
        f"  ── 제동 성능 ──\n"
        f"  제동 거리:    {d['stop_dist']:.1f} m\n"
        f"  제동 시간:    {d['stop_time']:.3f} s\n"
        f"  평균 감속:    {d['avg_decel_g']:.2f} g\n"
        f"  최대 감속:    {d['max_decel_g']:.2f} g\n"
        f"  최대 SR:     {d['max_sr_pct']:.1f} %\n"
    )
    if d['abs_on']:
        text += f"  ABS 개입:    {d['abs_events']} 회\n"
    if d.get('stop_dist_abs_off') is not None:
        diff_pct = (d['stop_dist_abs_off'] - d['stop_dist']) / max(d['stop_dist'], 1) * 100
        text += (
            f"\n  ── ABS OFF 비교 ──\n"
            f"  제동 거리:    {d['stop_dist_abs_off']:.1f} m"
            f"  ({diff_pct:+.1f}%)\n"
            f"  제동 시간:    {d['stop_time_abs_off']:.3f} s\n"
        )

    # ── ABS 제어 로직 테이블 ──
    try:
        sr_up = float(self.bk_sr_upper_var.get())
        sr_lo = float(self.bk_sr_lower_var.get())
        abs_hz = float(self.bk_abs_freq_var.get())
    except (AttributeError, ValueError):
        sr_up, sr_lo, abs_hz = 15.0, 8.0, 15.0
    abs_period = 1.0 / abs_hz if abs_hz > 0 else 0.1

    text += (
        f"\n  ═══ ABS 제어 로직 ═══\n"
        f"  ┌──────────────────────────────────┐\n"
        f"  │ 파라미터       │ 값               │\n"
        f"  ├──────────────────────────────────┤\n"
        f"  │ SR 상한 (해제) │ {sr_up:>6.1f} %        │\n"
        f"  │ SR 하한 (인가) │ {sr_lo:>6.1f} %        │\n"
        f"  │ ABS 주파수     │ {abs_hz:>6.1f} Hz       │\n"
        f"  │ ABS 주기       │ {abs_period*1000:>6.1f} ms       │\n"
        f"  │ 해제 유지비    │ {0.3*100:>5.0f} %         │\n"
        f"  └──────────────────────────────────┘\n"
        f"\n"
        f"  ── ABS 상태 천이 ──\n"
        f"  ┌─────────────┬────────────┬───────────────┐\n"
        f"  │ 현재 상태   │ 조건       │ 다음 상태     │\n"
        f"  ├─────────────┼────────────┼───────────────┤\n"
        f"  │ Brake ON    │|SR|>{sr_up:g}% │ Brake OFF     │\n"
        f"  │ Brake OFF   │|SR|<{sr_lo:g}%  │ Brake ON      │\n"
        f"  │ Brake OFF   │ t<{abs_period*0.3*1000:.0f}ms   │ Hold OFF      │\n"
        f"  │ ABS OFF     │ (항상)     │ Brake ON      │\n"
        f"  └─────────────┴────────────┴───────────────┘\n"
    )

    bd = d.get('brush_data', {})
    text += (
        f"\n  ── 2D Brush 연동 ──\n"
        f"  격자: {bd.get('nx','?')}×{bd.get('ny','?')}\n"
        f"  풋프린트: {bd.get('half_L',0)*2:.3f}×{bd.get('half_W',0)*2:.3f} m\n"
        f"  압력 분포: {bd.get('pressure_type','?')}\n"
    )
    self.bk_result_text.configure(state='normal')
    self.bk_result_text.delete('1.0', tk.END)
    self.bk_result_text.insert('1.0', text)
    self.bk_result_text.configure(state='disabled')


# ── Road visualization ──
def _draw_braking_road_static(self):
    """Draw static road and graphs."""
    ax = self._bk_ax_road
    ax.clear()
    ax.set_facecolor('#4A4A4A')
    ax.set_ylim(-1.5, 1.5)
    ax.set_yticks([])
    ax.set_title('Braking Road View', fontsize=9, fontweight='bold', color='#EEE')
    ax.tick_params(labelsize=6, colors='#AAA')

    d = self._bk_sim_data
    if d is not None:
        max_dist = d['dist'][-1] if len(d['dist']) > 0 else 200
    else:
        max_dist = 200
    ax.set_xlim(-10, max_dist + 20)
    ax.set_xlabel('Distance [m]', fontsize=7, color='#CCC')

    # Road lanes
    ax.axhline(y=-1.2, color='white', lw=2, zorder=1)
    ax.axhline(y=1.2, color='white', lw=2, zorder=1)
    # Dashed center line
    for x0 in np.arange(0, max_dist + 20, 8):
        ax.plot([x0, x0 + 4], [0, 0], '-', color='#FFD54F', lw=1.5, alpha=0.7, zorder=1)

    if d is not None:
        # Stop marker
        sd = d['stop_dist']
        ax.axvline(x=sd, color='#F44336', lw=2, ls='--', alpha=0.8, zorder=2)
        ax.text(sd, 1.35, f'STOP\n{sd:.1f}m', fontsize=7, ha='center',
                color='#F44336', fontweight='bold', zorder=3)

        # Skid marks (based on SR)
        sr_abs = np.abs(d['sr_pct'])
        for i in range(len(d['dist']) - 1):
            if sr_abs[i] > 15:  # significant slip
                alpha_sk = min(sr_abs[i] / 100, 0.8)
                ax.plot([d['dist'][i], d['dist'][i+1]], [-0.4, -0.4],
                        '-', color='#1A1A1A', lw=3, alpha=alpha_sk, zorder=1)
                ax.plot([d['dist'][i], d['dist'][i+1]], [0.4, 0.4],
                        '-', color='#1A1A1A', lw=3, alpha=alpha_sk, zorder=1)

    # Car marker (dynamic, will be updated per frame)
    self._bk_car_marker, = ax.plot([], [], 's', color='#1565C0', markersize=12, zorder=5)
    self._bk_car_trail, = ax.plot([], [], '-', color='#F44336', lw=2, alpha=0.5, zorder=4)

    # Speed & decel graphs
    ax_s = self._bk_ax_speed
    ax_d = self._bk_ax_decel
    ax_s.clear(); ax_d.clear()
    ax_s.set_title('Speed vs Distance', fontsize=9, fontweight='bold')
    ax_s.set_xlabel('Distance [m]', fontsize=7)
    ax_s.set_ylabel('Speed [km/h]', fontsize=7)
    ax_s.tick_params(labelsize=6); ax_s.grid(True, alpha=0.3)
    ax_d.set_title('Deceleration vs Distance', fontsize=9, fontweight='bold')
    ax_d.set_xlabel('Distance [m]', fontsize=7)
    ax_d.set_ylabel('Decel [g]', fontsize=7)
    ax_d.tick_params(labelsize=6); ax_d.grid(True, alpha=0.3)

    if d is not None:
        ax_s.plot(d['dist'], d['v_kmh'], '-', color='#1565C0', lw=1.5, label='Speed')
        ax_d.plot(d['dist'], d['decel_g'], '-', color='#F44336', lw=1.5, label='Decel')
        ax_s.legend(fontsize=7, loc='upper right')
        ax_d.legend(fontsize=7, loc='upper right')
        # Cursor lines
        self._bk_speed_cursor = ax_s.axvline(x=0, color='#FF6600', lw=1, ls='--', alpha=0.7)
        self._bk_decel_cursor = ax_d.axvline(x=0, color='#FF6600', lw=1, ls='--', alpha=0.7)

    self._bk_road_canvas.draw_idle()


# ── Contour plot artists ──
def _init_braking_contour_artists(self):
    """Create contour pcolormesh + Fx/SR/KE plot artists.

    Matches Track Simulator style: egg-shaped contact patch outline,
    clip paths, discrete jet colormaps, inset colorbars, blit-based
    rendering for 120 Hz playback.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Polygon as MplPolygon, Patch
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    d = self._bk_sim_data
    bd = d['brush_data']
    xe, ye = bd['x_edges'], bd['y_edges']
    half_L = bd.get('half_L', 0.075)
    half_W = bd.get('half_W', 0.06)
    L_mm = half_L * 2 * 1000
    W_mm = half_W * 2 * 1000
    xe_mm = xe * 1000
    ye_mm = ye * 1000

    n_levels = 8  # discrete levels (matching Track Simulator / 2D Brush tab)
    _AX_MARGIN = 1.60
    _AX_X_HALF = L_mm / 2 * _AX_MARGIN
    _AX_Y_HALF = W_mm / 2 * _AX_MARGIN

    _axes = [self._bk_ax_stick, self._bk_ax_speed_c,
             self._bk_ax_press, self._bk_ax_temp, self._bk_ax_fric]
    for ax in _axes:
        ax.clear()

    self._bk_outline_patches = []
    self._bk_cbar_axes = {}

    def _add_contact_outline(ax):
        """Add egg-shaped contact patch outline (matching Track Simulator)."""
        if hasattr(self, '_egg_outline'):
            verts = self._egg_outline(0, L_mm / 2, W_mm / 2)
        else:
            theta = np.linspace(0, 2 * np.pi, 80)
            verts = np.column_stack([L_mm / 2 * np.cos(theta),
                                     W_mm / 2 * np.sin(theta)])
        poly = MplPolygon(verts, closed=True, fill=False,
                          edgecolor='k', linewidth=1.5, zorder=5)
        ax.add_patch(poly)
        self._bk_outline_patches.append(poly)
        return poly

    def _setup_contour_ax(ax, title):
        ax.clear()
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.set_xlabel('length [mm]', fontsize=7)
        ax.set_ylabel('width [mm]', fontsize=7)
        ax.set_xlim(-_AX_X_HALF, _AX_X_HALF)
        ax.set_ylim(-_AX_Y_HALF, _AX_Y_HALF)
        ax.tick_params(labelsize=6)

    def _make_inset_cb(mappable, ax, key, label='', **extra_kw):
        """Create inset colorbar (horizontal, bottom-right) matching Track Simulator."""
        cax = inset_axes(ax, width="45%", height="4%",
                         loc='lower right', borderpad=1.2)
        cb = self._bk_contour_fig.colorbar(mappable, cax=cax,
                                            orientation='horizontal', **extra_kw)
        cb.ax.tick_params(labelsize=4.5, length=2, pad=1)
        if label:
            cb.set_label(label, fontsize=5.5)
        self._bk_cbar_axes[key] = cax
        return cb

    # ── (1) Adhesion / Sliding (blue=stick, red=slip) ──
    _setup_contour_ax(self._bk_ax_stick, 'sliding vs adhesion')
    cmap_sa = ListedColormap(['#2196F3', '#F44336'])
    bounds_sa = [0, 0.5, 1.0]
    norm_sa = BoundaryNorm(bounds_sa, cmap_sa.N)
    init = bd['is_sliding'][0].T
    self._bk_pm_stick = self._bk_ax_stick.pcolormesh(
        xe_mm, ye_mm, init, cmap=cmap_sa, norm=norm_sa, shading='flat', zorder=1)
    outline1 = _add_contact_outline(self._bk_ax_stick)
    self._bk_pm_stick.set_clip_path(outline1)
    # Rolling direction arrow (green)
    _roll_y = -W_mm * 0.55
    self._bk_ax_stick.annotate('', xy=(-L_mm * 0.35, _roll_y),
                 xytext=(L_mm * 0.35, _roll_y),
                 arrowprops=dict(arrowstyle='->', color='#00C853', lw=2.5,
                                 mutation_scale=16), zorder=7)
    self._bk_ax_stick.text(0, _roll_y - W_mm * 0.08, '\u2190 Rolling Dir.',
             fontsize=7, ha='center', va='top', color='#00C853',
             fontweight='bold', zorder=7)
    # Braking direction arrow (white, longitudinal)
    self._bk_stick_arrow_line, = self._bk_ax_stick.plot(
        [], [], '-', color='white', lw=3, zorder=6)
    self._bk_stick_arrow_head, = self._bk_ax_stick.plot(
        [], [], marker=(3, 0, 0), color='white', markersize=14,
        linestyle='', zorder=6)
    # Legend
    legend_patches = [Patch(facecolor='#2196F3', edgecolor='k', linewidth=0.5, label='Stick (\ubd80\ucc29)'),
                      Patch(facecolor='#F44336', edgecolor='k', linewidth=0.5, label='Slip (\ubbf8\ub044\ub7ec)')]
    self._bk_ax_stick.legend(handles=legend_patches, loc='upper right', fontsize=7,
               framealpha=0.9, edgecolor='#999',
               bbox_to_anchor=(0.98, 0.98), borderaxespad=0,
               handlelength=2.5, handletextpad=0.8, columnspacing=1.5)

    # ── (2) Sliding Speed (pcolormesh with discrete jet levels) ──
    _setup_contour_ax(self._bk_ax_speed_c, 'sliding speed')
    sp_max = max(np.nanmax(bd['v_slide']), 0.01)
    self._bk_global_sp_max = sp_max
    sp_boundaries = np.linspace(0, sp_max, n_levels + 1)
    sp_cmap = plt.colormaps['jet'].resampled(n_levels)
    sp_norm = BoundaryNorm(sp_boundaries, sp_cmap.N)
    self._bk_pm_speed = self._bk_ax_speed_c.pcolormesh(
        xe_mm, ye_mm, bd['v_slide'][0].T, cmap=sp_cmap, norm=sp_norm,
        shading='flat', zorder=1)
    outline2 = _add_contact_outline(self._bk_ax_speed_c)
    self._bk_pm_speed.set_clip_path(outline2)
    cb_sp = _make_inset_cb(self._bk_pm_speed, self._bk_ax_speed_c, 'speed', label='m/s')
    sp_centers = 0.5 * (sp_boundaries[:-1] + sp_boundaries[1:])
    cb_sp.set_ticks(sp_centers[::2])
    cb_sp.set_ticklabels([f'{v:.1f}' for v in sp_centers[::2]])

    # ── (3) Contact Pressure (pcolormesh with discrete jet levels) ──
    _setup_contour_ax(self._bk_ax_press, 'contact pressure')
    pr_max = max(np.nanmax(bd['pressure']), 0.01)
    pr_boundaries = np.linspace(0, pr_max, n_levels + 1)
    pr_cmap = plt.colormaps['jet'].resampled(n_levels)
    pr_norm = BoundaryNorm(pr_boundaries, pr_cmap.N)
    self._bk_pm_press = self._bk_ax_press.pcolormesh(
        xe_mm, ye_mm, bd['pressure'][0].T, cmap=pr_cmap, norm=pr_norm,
        shading='flat', zorder=1)
    outline3 = _add_contact_outline(self._bk_ax_press)
    self._bk_pm_press.set_clip_path(outline3)
    cb_pr = _make_inset_cb(self._bk_pm_press, self._bk_ax_press, 'pressure', label='bar')
    pr_centers = 0.5 * (pr_boundaries[:-1] + pr_boundaries[1:])
    cb_pr.set_ticks(pr_centers[::2])
    cb_pr.set_ticklabels([f'{v:.1f}' for v in pr_centers[::2]])
    # Rolling direction arrow + LE/TE labels
    _roll_y3 = -W_mm * 0.55
    self._bk_ax_press.annotate('', xy=(-L_mm * 0.35, _roll_y3),
                 xytext=(L_mm * 0.35, _roll_y3),
                 arrowprops=dict(arrowstyle='->', color='#00C853', lw=2,
                                 mutation_scale=14), zorder=7)
    self._bk_ax_press.text(0, _roll_y3 - W_mm * 0.08, '\u2190 Rolling Dir.',
             fontsize=6, ha='center', va='top', color='#00C853',
             fontweight='bold', zorder=7)
    self._bk_ax_press.text(-L_mm * 0.42, 0, 'LE', fontsize=7, ha='right', va='center',
             color='#333', fontweight='bold', fontstyle='italic', zorder=7)
    self._bk_ax_press.text(L_mm * 0.42, 0, 'TE', fontsize=7, ha='left', va='center',
             color='#333', fontweight='bold', fontstyle='italic', zorder=7)

    # ── (4) Temperature (pcolormesh with discrete jet levels) ──
    _setup_contour_ax(self._bk_ax_temp, 'temperature')
    t_min = np.nanmin(bd['temperature'])
    t_max = max(np.nanmax(bd['temperature']), t_min + 1)
    t_boundaries = np.linspace(t_min, t_max, n_levels + 1)
    t_cmap = plt.colormaps['jet'].resampled(n_levels)
    t_norm = BoundaryNorm(t_boundaries, t_cmap.N)
    self._bk_pm_temp = self._bk_ax_temp.pcolormesh(
        xe_mm, ye_mm, bd['temperature'][0].T, cmap=t_cmap, norm=t_norm,
        shading='flat', zorder=1)
    outline4 = _add_contact_outline(self._bk_ax_temp)
    self._bk_pm_temp.set_clip_path(outline4)
    cb_t = _make_inset_cb(self._bk_pm_temp, self._bk_ax_temp, 'temperature', label='\u00b0C')
    t_centers = 0.5 * (t_boundaries[:-1] + t_boundaries[1:])
    cb_t.set_ticks(t_centers[::2])
    cb_t.set_ticklabels([f'{v:.1f}' for v in t_centers[::2]])

    # ── (5) Friction Force (pcolormesh with discrete jet levels) ──
    _setup_contour_ax(self._bk_ax_fric, 'friction force')
    f_max = max(np.nanmax(bd['friction']), 0.01)
    fric_boundaries = np.linspace(0, f_max, n_levels + 1)
    fric_cmap = plt.colormaps['jet'].resampled(n_levels)
    fric_norm = BoundaryNorm(fric_boundaries, fric_cmap.N)
    self._bk_pm_fric = self._bk_ax_fric.pcolormesh(
        xe_mm, ye_mm, bd['friction'][0].T, cmap=fric_cmap, norm=fric_norm,
        shading='flat', zorder=1)
    outline5 = _add_contact_outline(self._bk_ax_fric)
    self._bk_pm_fric.set_clip_path(outline5)
    cb_f = _make_inset_cb(self._bk_pm_fric, self._bk_ax_fric, 'friction', label='N/node')
    fric_centers = 0.5 * (fric_boundaries[:-1] + fric_boundaries[1:])
    cb_f.set_ticks(fric_centers[::2])
    cb_f.set_ticklabels([f'{v:.1e}' for v in fric_centers[::2]])

    # ── Blit setup: list dynamic artists, then cache CLEAN background ──
    self._bk_contour_dynamic = [
        self._bk_pm_stick, self._bk_pm_speed, self._bk_pm_press,
        self._bk_pm_temp, self._bk_pm_fric,
        self._bk_stick_arrow_line, self._bk_stick_arrow_head,
    ]
    self._bk_contour_dynamic.extend(self._bk_outline_patches)

    _vis_states = []
    for a in self._bk_contour_dynamic:
        _vis_states.append(a.get_visible())
        a.set_visible(False)
    self._bk_contour_canvas.draw()
    self._bk_contour_blit_bg = self._bk_contour_canvas.copy_from_bbox(
        self._bk_contour_fig.bbox)
    for a, vis in zip(self._bk_contour_dynamic, _vis_states):
        a.set_visible(vis)
    self._bk_use_blit = True

    # ── Fx vs SR plot ──
    ax_sr = self._bk_ax_fx_sr
    ax_sr.clear()
    ax_sr.set_title('Fx vs Slip Ratio', fontsize=9, fontweight='bold')
    ax_sr.set_xlabel('SR [%]', fontsize=8)
    ax_sr.set_ylabel('Fx [N]', fontsize=8)
    ax_sr.grid(True, alpha=0.3)
    ax_sr.plot(d['sr_pct'], d['Fx'], '-', color='#1565C0', lw=1.2, label='Fx')
    ax_sr.legend(fontsize=7)
    self._bk_cursor_fx_sr = ax_sr.axvline(x=0, color='#FF6600', lw=1, ls='--')
    self._bk_marker_fx_sr, = ax_sr.plot([], [], 'o', color='#FF6600', markersize=6, zorder=5)

    # ── Kinetic Energy vs Distance plot ──
    ax_ke = self._bk_ax_ke_dist
    ax_ke.clear()
    v0_ms = d['v_kmh'][0] / 3.6
    mass = float(self.bk_mass_var.get()) if hasattr(self, 'bk_mass_var') else 2000.0
    KE_initial = 0.5 * mass * v0_ms**2
    KE_arr = 0.5 * mass * (d['v_kmh'] / 3.6)**2
    KE_dissipated = KE_initial - KE_arr

    ax_ke.set_title('Kinetic Energy vs Distance', fontsize=9, fontweight='bold')
    ax_ke.set_xlabel('Distance [m]', fontsize=8)
    ax_ke.set_ylabel('Energy [kJ]', fontsize=8)
    ax_ke.grid(True, alpha=0.3)
    ax_ke.plot(d['dist'], KE_arr / 1000, '-', color='#1565C0', lw=1.5, label='KE remaining')
    ax_ke.fill_between(d['dist'], KE_arr / 1000, KE_initial / 1000,
                       alpha=0.3, color='#F44336', label='Dissipated')
    ax_ke.plot(d['dist'], KE_dissipated / 1000, '-', color='#F44336', lw=1.2,
               alpha=0.7, label='Cumul. dissipated')
    ax_ke.axhline(y=KE_initial / 1000, color='#999', lw=0.8, ls=':', alpha=0.6)
    ax_ke.text(d['dist'][-1] * 0.02, KE_initial / 1000 * 1.02,
               f'KE\u2080 = {KE_initial/1000:.1f} kJ', fontsize=6, color='#666')
    ax_ke.legend(fontsize=6, loc='center right')
    ax_ke.set_ylim(0, KE_initial / 1000 * 1.15)
    self._bk_cursor_ke_dist = ax_ke.axvline(x=0, color='#FF6600', lw=1, ls='--')

    # Fx vs Distance
    ax_fd = self._bk_ax_fx_dist
    ax_fd.clear()
    ax_fd.set_title('Fx vs Distance', fontsize=9, fontweight='bold')
    ax_fd.set_xlabel('Distance [m]', fontsize=8)
    ax_fd.set_ylabel('Fx [N]', fontsize=8)
    ax_fd.grid(True, alpha=0.3)
    ax_fd.plot(d['dist'], d['Fx'], '-', color='#F44336', lw=1.2, label='Fx')
    ax_fd.legend(fontsize=7)
    self._bk_cursor_fx_dist = ax_fd.axvline(x=0, color='#FF6600', lw=1, ls='--')

    self._bk_fx_canvas.draw_idle()


# ── Brake pedal gauge ──
def _init_braking_pedal_gauge(self):
    ax = self._bk_pedal_ax
    ax.clear()
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor('#F8FAFC')

    # Pedal outline
    from matplotlib.patches import FancyBboxPatch, Rectangle
    ax.add_patch(Rectangle((-0.4, -1.2), 0.8, 2.0,
                            facecolor='#E0E0E0', edgecolor='#999', lw=2, zorder=1))
    ax.text(0, 1.1, 'BRAKE', fontsize=11, ha='center', va='top',
            fontweight='bold', color='#333', zorder=3)

    # Pressure bar (dynamic)
    self._bk_pedal_bar = ax.bar([0], [0], width=0.6, bottom=-1.1,
                                 color='#F44336', alpha=0.8, zorder=2)[0]
    self._bk_pedal_pct_label = ax.text(0, -1.3, '0%', fontsize=10,
                                        ha='center', va='top',
                                        fontweight='bold', color='#F44336', zorder=3)
    # SR display
    self._bk_pedal_sr_label = ax.text(0, -0.0, 'SR: 0%', fontsize=10,
                                       ha='center', va='center',
                                       fontweight='bold', color='#1565C0', zorder=3)
    self._bk_pedal_canvas.draw()


# ── ABS Logic Graph (replaces tire side-view) ──
def _init_braking_abs_graph(self):
    """Initialize or redraw the ABS logic graph with SR and brake state."""
    d = self._bk_sim_data

    ax_sr = self._bk_abs_ax_sr
    ax_bp = self._bk_abs_ax_bp
    ax_sr.clear()
    ax_bp.clear()

    ax_sr.set_title('Slip Ratio & ABS', fontsize=9, fontweight='bold')
    ax_sr.set_ylabel('|SR| [%]', fontsize=7)
    ax_sr.tick_params(labelsize=6)
    ax_sr.grid(True, alpha=0.3)

    ax_bp.set_title('Brake Pressure', fontsize=9, fontweight='bold')
    ax_bp.set_xlabel('Time [s]', fontsize=7)
    ax_bp.set_ylabel('Brake', fontsize=7)
    ax_bp.tick_params(labelsize=6)
    ax_bp.set_yticks([0, 1])
    ax_bp.set_yticklabels(['OFF', 'ON'], fontsize=6)
    ax_bp.grid(True, alpha=0.3)

    if d is not None:
        t = d['time']
        sr_abs = np.abs(d['sr_pct'])
        bp = d['brake_pressure']

        # SR plot
        ax_sr.plot(t, sr_abs, '-', color='#1565C0', lw=1.2, label='|SR|')
        ax_sr.set_xlim(t[0], t[-1])
        ax_sr.set_ylim(0, max(np.max(sr_abs) * 1.1, 20))

        # ABS threshold lines
        try:
            sr_up = float(self.bk_sr_upper_var.get())
            sr_lo = float(self.bk_sr_lower_var.get())
        except (AttributeError, ValueError):
            sr_up, sr_lo = 15.0, 8.0
        ax_sr.axhline(y=sr_up, color='#F44336', lw=1.2, ls='--', alpha=0.8,
                       label=f'SR upper ({sr_up}%)')
        ax_sr.axhline(y=sr_lo, color='#4CAF50', lw=1.2, ls='--', alpha=0.8,
                       label=f'SR lower ({sr_lo}%)')
        # Shade optimal zone
        ax_sr.axhspan(sr_lo, sr_up, alpha=0.08, color='#FF9800',
                       label='ABS target zone')
        ax_sr.legend(fontsize=5.5, loc='upper right')

        # Brake pressure ON/OFF plot (step function)
        ax_bp.fill_between(t, bp, step='post', alpha=0.4, color='#F44336')
        ax_bp.step(t, bp, where='post', color='#F44336', lw=1.2)
        ax_bp.set_xlim(t[0], t[-1])
        ax_bp.set_ylim(-0.1, 1.3)

        # Cursor lines (dynamic)
        self._bk_abs_cursor_sr = ax_sr.axvline(x=0, color='#FF6600', lw=1.2,
                                                 ls='--', alpha=0.8)
        self._bk_abs_cursor_bp = ax_bp.axvline(x=0, color='#FF6600', lw=1.2,
                                                 ls='--', alpha=0.8)
        # SR marker dot
        self._bk_abs_sr_marker, = ax_sr.plot([], [], 'o', color='#FF6600',
                                               markersize=5, zorder=5)

    self._bk_abs_canvas.draw_idle()


# ── Frame update ──
def _update_braking_frame(self, idx):
    d = self._bk_sim_data
    if d is None:
        return
    n = d['n_frames']
    idx = int(idx) % n

    t_cur = d['time'][idx]
    v_cur = d['v_kmh'][idx]
    sr_cur = d['sr_pct'][idx]
    fx_cur = d['Fx'][idx]
    dist_cur = d['dist'][idx]
    bp_cur = d['brake_pressure'][idx]
    decel_cur = d['decel_g'][idx]

    # ── Road view: car position ──
    self._bk_car_marker.set_data([dist_cur], [0])
    # Trail
    trail_start = max(0, idx - 30)
    self._bk_car_trail.set_data(d['dist'][trail_start:idx+1],
                                 [0] * (idx - trail_start + 1))

    # Graph cursors
    if hasattr(self, '_bk_speed_cursor'):
        self._bk_speed_cursor.set_xdata([dist_cur, dist_cur])
        self._bk_decel_cursor.set_xdata([dist_cur, dist_cur])

    if idx % 4 == 0:
        self._bk_road_canvas.draw_idle()

    # ── Brake pedal ──
    bar_h = bp_cur * 1.8
    self._bk_pedal_bar.set_height(bar_h)
    color = '#F44336' if bp_cur > 0.5 else '#FF9800' if bp_cur > 0 else '#4CAF50'
    self._bk_pedal_bar.set_color(color)
    self._bk_pedal_pct_label.set_text(f'{bp_cur*100:.0f}%')
    self._bk_pedal_sr_label.set_text(f'SR: {sr_cur:.1f}%')
    if idx % 4 == 0:
        self._bk_pedal_canvas.draw_idle()

    # ── ABS Logic Graph cursor update ──
    if hasattr(self, '_bk_abs_cursor_sr'):
        self._bk_abs_cursor_sr.set_xdata([t_cur, t_cur])
        self._bk_abs_cursor_bp.set_xdata([t_cur, t_cur])
        self._bk_abs_sr_marker.set_data([t_cur], [abs(sr_cur)])
        if idx % 4 == 0:
            self._bk_abs_canvas.draw_idle()

    # ── Contour update ──
    bd = d['brush_data']
    _playing = getattr(self, '_bk_playing', False)

    if hasattr(self, '_bk_pm_stick'):
        self._bk_pm_stick.set_array(bd['is_sliding'][idx].T.ravel())
        self._bk_pm_speed.set_array(bd['v_slide'][idx].T.ravel())
        self._bk_pm_press.set_array(bd['pressure'][idx].T.ravel())
        self._bk_pm_temp.set_array(bd['temperature'][idx].T.ravel())
        self._bk_pm_fric.set_array(bd['friction'][idx].T.ravel())

        # Braking direction arrow on stick/slip contour
        if hasattr(self, '_bk_stick_arrow_line'):
            half_L = bd.get('half_L', 0.075)
            L_mm_a = half_L * 2 * 1000
            if abs(sr_cur) > 1.0:
                arrow_scale = L_mm_a * 0.3
                dx_a = -arrow_scale  # braking → arrow points backward (rolling dir)
                self._bk_stick_arrow_line.set_data([0, dx_a], [0, 0])
                self._bk_stick_arrow_line.set_visible(True)
                self._bk_stick_arrow_head.set_marker((3, 0, 180))
                self._bk_stick_arrow_head.set_data([dx_a], [0])
                self._bk_stick_arrow_head.set_visible(True)
            else:
                self._bk_stick_arrow_line.set_visible(False)
                self._bk_stick_arrow_head.set_visible(False)

        # Blit-based render for 120 Hz (matching Track Simulator)
        if getattr(self, '_bk_use_blit', False) and self._bk_contour_blit_bg is not None:
            c_canvas = self._bk_contour_canvas
            c_canvas.restore_region(self._bk_contour_blit_bg)
            for artist in self._bk_contour_dynamic:
                try:
                    artist.axes.draw_artist(artist)
                except Exception:
                    pass
            c_canvas.blit(self._bk_contour_fig.bbox)
            if idx % 4 == 0:
                c_canvas.flush_events()
        else:
            if idx % 4 == 0:
                self._bk_contour_canvas.draw_idle()

    # ── Fx / KE graphs cursor ──
    if hasattr(self, '_bk_cursor_fx_sr'):
        self._bk_cursor_fx_sr.set_xdata([sr_cur, sr_cur])
        self._bk_marker_fx_sr.set_data([sr_cur], [fx_cur])
        self._bk_cursor_fx_dist.set_xdata([dist_cur, dist_cur])
        if hasattr(self, '_bk_cursor_ke_dist'):
            self._bk_cursor_ke_dist.set_xdata([dist_cur, dist_cur])
        if not _playing or idx % 4 == 0:
            self._bk_fx_canvas.draw_idle()

    # ── HUD ──
    self.bk_frame_label_var.set(
        f"t={t_cur:.2f}s | {v_cur:.0f}km/h | {dist_cur:.0f}m")
    self._bk_hud_var.set(
        f"SR: {sr_cur:.1f}%  |  Fx: {fx_cur:.0f} N  |  {decel_cur:.1f}g")


# ── Playback ──
def _braking_play(self):
    if self._bk_sim_data is None:
        messagebox.showinfo("알림", "먼저 제동 시뮬레이션을 실행하세요.")
        return
    self._bk_playing = True
    self._bk_frame_accum = 0.0
    self._braking_animate_step()


def _braking_animate_step(self):
    if not self._bk_playing or self._bk_sim_data is None:
        return
    speed = self._bk_speed_mult.get()
    acc = self._bk_frame_accum + speed
    frame_skip = int(acc)
    self._bk_frame_accum = acc - frame_skip

    if frame_skip < 1:
        self._bk_play_after_id = self.root.after(1, self._braking_animate_step)
        return

    n = self._bk_sim_data['n_frames']
    self._bk_frame_idx = (self._bk_frame_idx + frame_skip)
    if self._bk_frame_idx >= n:
        self._bk_frame_idx = n - 1
        self._bk_playing = False
        self._update_braking_frame(self._bk_frame_idx)
        return

    if self._bk_frame_idx % 8 == 0:
        self.bk_time_slider.set(self._bk_frame_idx)

    self._update_braking_frame(self._bk_frame_idx)
    self._bk_play_after_id = self.root.after(1, self._braking_animate_step)


def _braking_pause(self):
    self._bk_playing = False
    if self._bk_play_after_id is not None:
        try:
            self.root.after_cancel(self._bk_play_after_id)
        except Exception:
            pass
        self._bk_play_after_id = None


def _braking_reset(self):
    self._braking_pause()
    self._bk_frame_idx = 0
    if self._bk_sim_data is not None:
        self._update_braking_frame(0)
        self.bk_time_slider.set(0)


def _on_braking_speed_slider(self, val):
    self._bk_speed_label_var.set(f"재생 속도: {self._bk_speed_mult.get():.1f}x")


def _on_braking_slider_change(self, val):
    if self._bk_sim_data is None:
        return
    self._bk_frame_idx = int(float(val))
    self._update_braking_frame(self._bk_frame_idx)


def _show_braking_brush_sync_info(self):
    try:
        nx = self.br_Nx_var.get()
        ny = self.br_Ny_var.get()
        L = self.br_L_var.get()
        W = self.br_W_var.get()
        fs = self.br_friction_scale_var.get()
        ep = getattr(self, '_br_ellipse_power_var', None)
        ep_val = f"{ep.get():.1f}" if ep else "2.0"
        self.bk_brush_info_var.set(
            f"Nx={nx}, Ny={ny}\n"
            f"L={L} m, W={W} m\n"
            f"마찰 감도: {fs}\n"
            f"타원 형상 (n): {ep_val}")
    except Exception:
        self.bk_brush_info_var.set("2D Brush 탭 파라미터 읽기 실패")


# ── Binding function ──
def bind_braking_simulation(cls):
    """Bind all braking simulation methods to the main GUI class."""
    cls._create_braking_simulation_tab = _create_braking_simulation_tab
    cls._run_braking_simulation = _run_braking_simulation
    cls._compute_braking_brush_data = _compute_braking_brush_data
    cls._update_braking_results_text = _update_braking_results_text
    cls._draw_braking_road_static = _draw_braking_road_static
    cls._init_braking_contour_artists = _init_braking_contour_artists
    cls._init_braking_pedal_gauge = _init_braking_pedal_gauge
    cls._init_braking_abs_graph = _init_braking_abs_graph
    cls._update_braking_frame = _update_braking_frame
    cls._braking_play = _braking_play
    cls._braking_animate_step = _braking_animate_step
    cls._braking_pause = _braking_pause
    cls._braking_reset = _braking_reset
    cls._on_braking_speed_slider = _on_braking_speed_slider
    cls._on_braking_slider_change = _on_braking_slider_change
    cls._show_braking_brush_sync_info = _show_braking_brush_sync_info
