"""
Results Overview Tab for Persson Friction Model
================================================
다중 컴파운드 계산 결과 종합 비교 탭

Usage in main.py:
    from results_overview_tab import bind_results_overview_tab
    # __init__ 내에서, _create_main_layout() 호출 전에:
    bind_results_overview_tab(self)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import os
import csv

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


def bind_results_overview_tab(app):
    """Results Overview 탭을 app에 바인딩한다."""
    rot = ResultsOverviewTab(app)
    app._results_overview_tab = rot


class ResultsOverviewTab:
    """계산 결과 종합 탭: 다중 컴파운드 비교 플롯 + 마찰 맵 + 비교 코멘트."""

    # 1=검정, 2=빨강, 3=파랑, 이후 구분 가능한 색상
    COMPOUND_COLORS = [
        '#000000', '#DC2626', '#2563EB', '#059669',
        '#D97706', '#7C3AED', '#DB2777', '#0891B2',
    ]
    PLOT_FONT_SIZE = 12
    PLOT_LINE_WIDTH = 2.5
    PLOT_LINE_WIDTH_SUB = 1.8

    def __init__(self, app):
        self.app = app
        app._create_results_overview_tab = self._create_tab

        self._plot_tabs = {}  # key -> (fig, canvas)
        self._compound_visible_vars = []
        self._comment_text = None

    def _create_tab(self, parent):
        """Build the Results Overview tab."""
        C = self.app.COLORS
        F = self.app.FONTS

        main = ttk.Frame(parent)
        main.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        # ── Left controls ──
        left = ttk.Frame(main, width=260)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 4))
        left.pack_propagate(False)

        # Compound visibility
        sec1 = ttk.LabelFrame(left, text="컴파운드 표시", padding=8)
        sec1.pack(fill=tk.X, pady=4, padx=2)
        self._visibility_frame = sec1

        # Export button
        sec2 = ttk.LabelFrame(left, text="내보내기", padding=8)
        sec2.pack(fill=tk.X, pady=4, padx=2)
        ttk.Button(sec2, text="CSV 내보내기", command=self._export_csv).pack(fill=tk.X, pady=2)
        ttk.Button(sec2, text="그래프 이미지 저장", command=self._save_figure).pack(fill=tk.X, pady=2)

        # Refresh button
        ttk.Button(left, text="새로고침", command=self.refresh).pack(fill=tk.X, pady=8, padx=4)

        # Status
        self._status_var = tk.StringVar(value="데이터 입력 탭에서 계산을 실행하세요")
        ttk.Label(left, textvariable=self._status_var, font=F['small'],
                  foreground=C['text_secondary'], wraplength=240).pack(fill=tk.X, pady=4, padx=4)

        # ── Right: sub-notebook ──
        right = ttk.Frame(main)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._plot_notebook = ttk.Notebook(right)
        self._plot_notebook.pack(fill=tk.BOTH, expand=True)

        # Create sub-tabs
        _tab_defs = [
            ('total_overlay', 'μ_total 오버레이'),
            ('cold_hot',      'Cold & Hot Branch'),
            ('separated',     '분리 플롯'),
            ('friction_map',  '마찰 맵 (컴파운드별)'),
            ('comparison',    '비교 분석'),
        ]
        for key, label in _tab_defs:
            tab_frame = ttk.Frame(self._plot_notebook)
            self._plot_notebook.add(tab_frame, text=f'  {label}  ')

            if key == 'comparison':
                # Comparison tab uses text widget + small plot
                pane = ttk.PanedWindow(tab_frame, orient=tk.VERTICAL)
                pane.pack(fill=tk.BOTH, expand=True)

                # Top: small comparison chart
                chart_frame = ttk.Frame(pane)
                pane.add(chart_frame, weight=1)
                fig = Figure(figsize=(10, 4), dpi=100, facecolor='white')
                canvas = FigureCanvasTkAgg(fig, chart_frame)
                toolbar = NavigationToolbar2Tk(canvas, chart_frame)
                toolbar.update()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                self._plot_tabs[key] = (fig, canvas)

                # Bottom: text comment area
                comment_frame = ttk.LabelFrame(pane, text="비교 분석 코멘트", padding=6)
                pane.add(comment_frame, weight=1)
                self._comment_text = tk.Text(comment_frame, wrap=tk.WORD, height=12,
                                             font=F.get('small', ('TkDefaultFont', 10)),
                                             bg='#FAFAFA', relief='flat', padx=8, pady=6)
                scrollbar = ttk.Scrollbar(comment_frame, orient=tk.VERTICAL,
                                          command=self._comment_text.yview)
                self._comment_text.configure(yscrollcommand=scrollbar.set)
                scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                self._comment_text.pack(fill=tk.BOTH, expand=True)
            else:
                fig = Figure(figsize=(10, 7), dpi=100, facecolor='white')
                fig.subplots_adjust(hspace=0.4, wspace=0.3)
                canvas = FigureCanvasTkAgg(fig, tab_frame)
                toolbar = NavigationToolbar2Tk(canvas, tab_frame)
                toolbar.update()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                self._plot_tabs[key] = (fig, canvas)

        # Legacy reference for _save_figure
        self._fig, self._canvas = self._plot_tabs['total_overlay']

    # ================================================================
    #  Plot styling helper
    # ================================================================
    @staticmethod
    def _style_ax(ax, title='', xlabel='', ylabel='', xscale=None, yscale=None):
        """Apply consistent styling: 12pt fonts, grid."""
        FS = ResultsOverviewTab.PLOT_FONT_SIZE
        if title:
            ax.set_title(title, fontsize=FS + 1, fontweight='bold')
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=FS)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=FS)
        if xscale:
            ax.set_xscale(xscale)
        if yscale:
            ax.set_yscale(yscale)
        ax.tick_params(axis='both', labelsize=FS - 1)
        ax.grid(True, alpha=0.3)

    # ================================================================
    #  Refresh
    # ================================================================
    def refresh(self):
        """Refresh all sub-tab plots."""
        compounds = getattr(self.app, 'compound_data', [])
        results = getattr(self.app, 'all_compound_results', [])

        if not compounds or not results:
            for key, (fig, canvas) in self._plot_tabs.items():
                fig.clear()
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, '데이터 입력 탭에서 계산을 실행하세요',
                        ha='center', va='center', fontsize=14,
                        transform=ax.transAxes, color='gray')
                ax.set_axis_off()
                canvas.draw_idle()
            self._status_var.set("결과 없음")
            return

        # Update visibility checkboxes
        self._update_visibility_checkboxes(compounds)

        # Get visible compounds
        visible = []
        for i, cpd in enumerate(compounds):
            if i < len(self._compound_visible_vars):
                if self._compound_visible_vars[i].get():
                    visible.append((i, cpd))
            else:
                visible.append((i, cpd))

        self._plot_total_overlay(visible)
        self._plot_cold_hot(visible)
        self._plot_separated(visible)
        self._plot_friction_maps(visible)
        self._plot_comparison(visible)
        self._status_var.set(f"{len(compounds)}개 컴파운드 결과 표시 중")

    def _update_visibility_checkboxes(self, compounds):
        """Update compound visibility checkboxes."""
        if len(self._compound_visible_vars) == len(compounds):
            return

        for w in self._visibility_frame.winfo_children():
            w.destroy()
        self._compound_visible_vars.clear()

        for i, cpd in enumerate(compounds):
            var = tk.BooleanVar(value=True)
            self._compound_visible_vars.append(var)
            cb = ttk.Checkbutton(self._visibility_frame, text=cpd.name,
                                 variable=var, command=self.refresh)
            cb.pack(anchor=tk.W, pady=1)

    # ================================================================
    #  Plot: μ_total overlay
    # ================================================================
    def _plot_total_overlay(self, visible_compounds):
        """All compounds' mu_total on one axis."""
        FS = self.PLOT_FONT_SIZE
        LW = self.PLOT_LINE_WIDTH
        LWs = self.PLOT_LINE_WIDTH_SUB
        fig, canvas = self._plot_tabs['total_overlay']
        fig.clear()

        ax = fig.add_subplot(111)
        self._style_ax(ax, 'μ_total vs Velocity (All Compounds)',
                        'Velocity (m/s)', 'μ_total', 'log')

        for i, cpd in visible_compounds:
            if cpd.results is None:
                continue
            r = cpd.results
            color = self.COMPOUND_COLORS[i % len(self.COMPOUND_COLORS)]
            ax.plot(r['v'], r['mu_total'], '-', color=color, linewidth=LW, label=cpd.name)
            ax.plot(r['v'], r['mu_visc'], '--', color=color, linewidth=LWs, alpha=0.5,
                    label=f'{cpd.name} (hys)')
            if r.get('mu_adh') is not None:
                ax.plot(r['v'], r['mu_adh'], ':', color=color, linewidth=LWs, alpha=0.5,
                        label=f'{cpd.name} (adh)')

        if visible_compounds:
            ax.legend(fontsize=FS - 1, loc='best')
        fig.tight_layout()
        canvas.draw_idle()

    # ================================================================
    #  Plot: Cold & Hot Branch (like Tab 10)
    # ================================================================
    def _plot_cold_hot(self, visible_compounds):
        """Cold & Hot Branch 비교 (10번 탭과 동일 스타일).

        2×2: μ_hys Cold/Hot, μ_adh Cold/Hot, A/A0 Cold/Hot, μ_total Cold/Hot
        """
        FS = self.PLOT_FONT_SIZE
        LW = self.PLOT_LINE_WIDTH
        LWs = self.PLOT_LINE_WIDTH_SUB
        fig, canvas = self._plot_tabs['cold_hot']
        fig.clear()

        # Check if any compound has hot data
        has_any_hot = any(
            cpd.results is not None and cpd.results.get('mu_hot_hys') is not None
            for _, cpd in visible_compounds
        )

        if not has_any_hot:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5,
                    'Flash Temperature를 활성화하고 계산하세요\n'
                    '(데이터 입력 탭에서 Flash Temperature 체크)',
                    ha='center', va='center', fontsize=FS, color='gray',
                    transform=ax.transAxes)
            ax.set_axis_off()
            canvas.draw_idle()
            return

        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)

        self._style_ax(ax1, 'μ_hys: Cold vs Hot', 'Velocity (m/s)', 'μ_hys', 'log')
        self._style_ax(ax2, 'μ_adh: Cold vs Hot', 'Velocity (m/s)', 'μ_adh', 'log')
        self._style_ax(ax3, 'A/A₀: Cold vs Hot', 'Velocity (m/s)', 'A/A₀', 'log')
        self._style_ax(ax4, 'μ_total: Cold vs Hot', 'Velocity (m/s)', 'μ_total', 'log')

        for i, cpd in visible_compounds:
            if cpd.results is None:
                continue
            r = cpd.results
            v = r['v']
            color = self.COMPOUND_COLORS[i % len(self.COMPOUND_COLORS)]

            # Cold branch
            mu_cold_hys = r.get('mu_cold_hys')
            mu_cold_adh = r.get('mu_cold_adh')
            mu_cold_total = r.get('mu_cold_total')
            A_A0_cold = r.get('A_A0_cold')

            # Hot branch
            mu_hot_hys = r.get('mu_hot_hys')
            mu_hot_adh = r.get('mu_hot_adh')
            mu_hot_total = r.get('mu_hot_total')
            A_A0_hot = r.get('A_A0_hot')

            lbl = cpd.name

            # μ_hys
            if mu_cold_hys is not None:
                ax1.plot(v, mu_cold_hys, '-', color=color, linewidth=LW,
                         label=f'{lbl} Cold')
            if mu_hot_hys is not None:
                ax1.plot(v, mu_hot_hys, '--', color=color, linewidth=LWs,
                         label=f'{lbl} Hot')

            # μ_adh
            if mu_cold_adh is not None:
                ax2.plot(v, mu_cold_adh, '-', color=color, linewidth=LW,
                         label=f'{lbl} Cold')
            if mu_hot_adh is not None:
                ax2.plot(v, mu_hot_adh, '--', color=color, linewidth=LWs,
                         label=f'{lbl} Hot')

            # A/A0
            if A_A0_cold is not None:
                ax3.plot(v, A_A0_cold, '-', color=color, linewidth=LW,
                         label=f'{lbl} Cold')
            if A_A0_hot is not None:
                ax3.plot(v, A_A0_hot, '--', color=color, linewidth=LWs,
                         label=f'{lbl} Hot')

            # μ_total
            if mu_cold_total is not None:
                ax4.plot(v, mu_cold_total, '-', color=color, linewidth=LW,
                         label=f'{lbl} Cold')
            if mu_hot_total is not None:
                ax4.plot(v, mu_hot_total, '--', color=color, linewidth=LWs,
                         label=f'{lbl} Hot')

        for ax in [ax1, ax2, ax3, ax4]:
            ax.legend(fontsize=FS - 3, loc='best')

        fig.tight_layout()
        canvas.draw_idle()

    # ================================================================
    #  Plot: separated (visc / adh / total / bar)
    # ================================================================
    def _plot_separated(self, visible_compounds):
        """4-panel: mu_visc, mu_adh, mu_total, bar chart."""
        FS = self.PLOT_FONT_SIZE
        LW = self.PLOT_LINE_WIDTH
        fig, canvas = self._plot_tabs['separated']
        fig.clear()

        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)

        for ax, title, key in [
            (ax1, 'μ_visc (Hysteresis)', 'mu_visc'),
            (ax2, 'μ_adh (Adhesion)', 'mu_adh'),
            (ax3, 'μ_total', 'mu_total'),
        ]:
            ylabel = title.split('(')[0].strip()
            self._style_ax(ax, title, 'Velocity (m/s)', ylabel, 'log')

            for i, cpd in visible_compounds:
                if cpd.results is None:
                    continue
                r = cpd.results
                color = self.COMPOUND_COLORS[i % len(self.COMPOUND_COLORS)]
                y = r.get(key)
                if y is not None:
                    ax.plot(r['v'], y, '-', color=color, linewidth=LW, label=cpd.name)

            ax.legend(fontsize=FS - 2, loc='best')

        # 4th panel: bar chart at key speeds
        self._style_ax(ax4, '속도별 μ_total 비교', '', 'μ_total')

        key_speeds = [0.001, 0.01, 0.1, 1.0]
        valid_compounds = [(i, cpd) for i, cpd in visible_compounds if cpd.results is not None]
        if valid_compounds:
            n_cpd = len(valid_compounds)
            n_spd = len(key_speeds)
            width = 0.8 / n_cpd
            x = np.arange(n_spd)

            for ci_local, (i, cpd) in enumerate(valid_compounds):
                r = cpd.results
                color = self.COMPOUND_COLORS[i % len(self.COMPOUND_COLORS)]
                vals = []
                for spd in key_speeds:
                    idx = np.argmin(np.abs(r['v'] - spd))
                    vals.append(r['mu_total'][idx])
                ax4.bar(x + ci_local * width, vals, width, label=cpd.name, color=color, alpha=0.85)

            ax4.set_xticks(x + width * (n_cpd - 1) / 2)
            ax4.set_xticklabels([f'{s} m/s' for s in key_speeds], fontsize=FS - 1)
            ax4.legend(fontsize=FS - 2, loc='best')

        fig.tight_layout()
        canvas.draw_idle()

    # ================================================================
    #  Plot: 3D friction map per compound (like Tab 11)
    # ================================================================
    def _plot_friction_maps(self, visible_compounds):
        """3D surface friction map per compound: log10(v) × component → μ.

        Similar to Tab 11 Friction Map style using plot_surface().
        X = log10(velocity), Y = component (total/visc/adh), Z = μ
        One 3D subplot per compound.
        """
        FS = self.PLOT_FONT_SIZE
        LW = self.PLOT_LINE_WIDTH
        fig, canvas = self._plot_tabs['friction_map']
        fig.clear()

        valid = [(i, cpd) for i, cpd in visible_compounds if cpd.results is not None]
        if not valid:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, '결과 없음', ha='center', va='center',
                    fontsize=14, color='gray', transform=ax.transAxes)
            ax.set_axis_off()
            canvas.draw_idle()
            return

        n = len(valid)
        cols = min(n, 2)
        rows = (n + cols - 1) // cols

        for plot_idx, (i, cpd) in enumerate(valid):
            ax = fig.add_subplot(rows, cols, plot_idx + 1, projection='3d')
            color = self.COMPOUND_COLORS[i % len(self.COMPOUND_COLORS)]
            r = cpd.results
            v = r['v']
            log_v = np.log10(np.maximum(v, 1e-20))
            mu_visc = r['mu_visc']
            mu_adh = r.get('mu_adh')
            mu_total = r['mu_total']

            n_v = len(v)

            # Build surface: Y=0 → mu_visc, Y=1 → mu_adh, Y=2 → mu_total
            has_adh = mu_adh is not None and np.any(mu_adh > 0)
            if has_adh:
                # 3 bands: visc, adh, total
                Y_labels = ['μ_hys', 'μ_adh', 'μ_total']
                n_bands = 3
                n_interp = 6  # interpolation subdivisions per band
                Y_pos = np.linspace(0, 2, n_bands * n_interp)
                Z_surface = np.zeros((n_bands * n_interp, n_v))
                for j in range(n_bands * n_interp):
                    band = j // n_interp
                    frac = (j % n_interp) / n_interp
                    if band == 0:
                        Z_surface[j, :] = mu_visc
                    elif band == 1:
                        # Transition from visc to adh
                        Z_surface[j, :] = mu_visc * (1 - frac) + mu_adh * frac
                    else:
                        # Transition from adh to total
                        Z_surface[j, :] = mu_adh * (1 - frac) + mu_total * frac
            else:
                # 2 bands: visc → total
                Y_labels = ['μ_hys', 'μ_total']
                n_bands = 2
                n_interp = 8
                Y_pos = np.linspace(0, 1, n_bands * n_interp)
                Z_surface = np.zeros((n_bands * n_interp, n_v))
                for j in range(n_bands * n_interp):
                    band = j // n_interp
                    frac = (j % n_interp) / n_interp
                    if band == 0:
                        Z_surface[j, :] = mu_visc
                    else:
                        Z_surface[j, :] = mu_visc * (1 - frac) + mu_total * frac

            X_mesh, Y_mesh = np.meshgrid(log_v, Y_pos)

            ax.plot_surface(X_mesh, Y_mesh, Z_surface,
                            cmap='viridis', alpha=0.85,
                            rstride=1, cstride=1,
                            linewidth=0, antialiased=True,
                            rasterized=True)

            # Overlay edge lines for each component
            ax.plot(log_v, np.zeros(n_v), mu_visc,
                    color='#3B82F6', linewidth=LW, label='μ_hys', zorder=5)
            if has_adh:
                ax.plot(log_v, np.ones(n_v), mu_adh,
                        color='#EF4444', linewidth=LW, label='μ_adh', zorder=5)
                ax.plot(log_v, np.full(n_v, 2.0), mu_total,
                        color=color, linewidth=LW + 0.5, label='μ_total', zorder=5)
            else:
                ax.plot(log_v, np.ones(n_v), mu_total,
                        color=color, linewidth=LW + 0.5, label='μ_total', zorder=5)

            # Mark peak
            peak_idx = np.argmax(mu_total)
            peak_v = log_v[peak_idx]
            peak_mu = mu_total[peak_idx]
            y_peak = 2.0 if has_adh else 1.0
            ax.scatter([peak_v], [y_peak], [peak_mu], color='red', s=50, zorder=10)
            ax.text(peak_v, y_peak, peak_mu * 1.05,
                    f'peak={peak_mu:.3f}\n@{v[peak_idx]:.1e}',
                    fontsize=FS - 4, ha='center', color='#333')

            # Labels and styling
            ax.set_xlabel('log₁₀(v)', fontsize=FS - 1, labelpad=8)
            ax.set_ylabel('Component', fontsize=FS - 1, labelpad=8)
            ax.set_zlabel('μ', fontsize=FS - 1, labelpad=6)
            ax.set_title(cpd.name, fontsize=FS, fontweight='bold', pad=10)

            # Y-axis tick labels
            if has_adh:
                ax.set_yticks([0, 1, 2])
                ax.set_yticklabels(Y_labels, fontsize=FS - 3)
            else:
                ax.set_yticks([0, 1])
                ax.set_yticklabels(Y_labels, fontsize=FS - 3)

            ax.tick_params(axis='x', labelsize=FS - 2)
            ax.tick_params(axis='z', labelsize=FS - 2)

            # Condition info
            sigma0 = r.get('sigma_0', 0)
            temp = r.get('temperature', 0)
            flash = 'Flash ON' if r.get('use_flash', False) else ''
            info = f"σ₀={sigma0/1e6:.1f}MPa  T={temp:.0f}°C  {flash}".strip()
            ax.text2D(0.02, 0.98, info, transform=ax.transAxes,
                      fontsize=FS - 3, va='top', color='#666')

            ax.view_init(elev=25, azim=225)
            ax.legend(fontsize=FS - 3, loc='upper left')

        fig.tight_layout()
        canvas.draw_idle()

    # ================================================================
    #  Plot: comparison analysis + comment
    # ================================================================
    def _plot_comparison(self, visible_compounds):
        """Comparison bar chart + auto-generated analysis comment."""
        FS = self.PLOT_FONT_SIZE
        fig, canvas = self._plot_tabs['comparison']
        fig.clear()

        valid = [(i, cpd) for i, cpd in visible_compounds if cpd.results is not None]
        if not valid:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, '결과 없음', ha='center', va='center',
                    fontsize=14, color='gray', transform=ax.transAxes)
            ax.set_axis_off()
            canvas.draw_idle()
            if self._comment_text:
                self._comment_text.delete('1.0', tk.END)
            return

        # Bar chart: peak μ, μ at low/mid/high speed
        ax = fig.add_subplot(111)
        speed_labels = ['Peak μ', 'v=0.001', 'v=0.01', 'v=0.1', 'v=1.0']
        n_cpd = len(valid)
        n_cat = len(speed_labels)
        width = 0.8 / n_cpd
        x = np.arange(n_cat)

        comparison_data = {}  # cpd_name -> dict of values

        for ci, (i, cpd) in enumerate(valid):
            r = cpd.results
            color = self.COMPOUND_COLORS[i % len(self.COMPOUND_COLORS)]

            peak_mu = np.max(r['mu_total'])
            peak_v = r['v'][np.argmax(r['mu_total'])]

            vals = [peak_mu]
            speed_vals = {}
            for spd in [0.001, 0.01, 0.1, 1.0]:
                idx = np.argmin(np.abs(r['v'] - spd))
                vals.append(r['mu_total'][idx])
                speed_vals[spd] = r['mu_total'][idx]

            ax.bar(x + ci * width, vals, width, label=cpd.name, color=color, alpha=0.85)

            comparison_data[cpd.name] = {
                'peak_mu': peak_mu,
                'peak_v': peak_v,
                'mu_visc_peak': np.max(r['mu_visc']),
                'mu_adh_peak': np.max(r['mu_adh']) if r.get('mu_adh') is not None else 0,
                'sigma_0': r.get('sigma_0', 0),
                'temperature': r.get('temperature', 0),
                'speed_vals': speed_vals,
            }

        ax.set_xticks(x + width * (n_cpd - 1) / 2)
        ax.set_xticklabels(speed_labels, fontsize=FS - 1)
        self._style_ax(ax, '컴파운드별 마찰 비교', '', 'μ_total')
        ax.legend(fontsize=FS - 2, loc='best')
        fig.tight_layout()
        canvas.draw_idle()

        # Generate comparison comment
        self._generate_comparison_comment(comparison_data)

    def _generate_comparison_comment(self, data):
        """Auto-generate comparison analysis text."""
        if self._comment_text is None or not data:
            return

        self._comment_text.delete('1.0', tk.END)
        lines = []
        names = list(data.keys())
        n = len(names)

        lines.append("=" * 60)
        lines.append(f"  컴파운드 비교 분석 ({n}개)")
        lines.append("=" * 60)
        lines.append("")

        # 1. Peak friction comparison
        lines.append("■ 최대 마찰계수 (Peak μ_total)")
        lines.append("-" * 40)
        ranked = sorted(data.items(), key=lambda x: x[1]['peak_mu'], reverse=True)
        for rank, (name, d) in enumerate(ranked, 1):
            lines.append(f"  {rank}위: {name}  →  μ_peak = {d['peak_mu']:.4f}  "
                         f"(@ v = {d['peak_v']:.2e} m/s)")
        if n >= 2:
            best_name, best = ranked[0]
            worst_name, worst = ranked[-1]
            diff_pct = (best['peak_mu'] - worst['peak_mu']) / worst['peak_mu'] * 100
            lines.append(f"  ▶ {best_name}이(가) {worst_name} 대비 {diff_pct:+.1f}% 높음")
        lines.append("")

        # 2. Hysteresis vs adhesion contribution
        lines.append("■ Hysteresis / Adhesion 기여도")
        lines.append("-" * 40)
        for name, d in data.items():
            total = d['mu_visc_peak'] + d['mu_adh_peak']
            if total > 0:
                hys_pct = d['mu_visc_peak'] / total * 100
                adh_pct = d['mu_adh_peak'] / total * 100
            else:
                hys_pct = adh_pct = 0
            lines.append(f"  {name}:  hys={d['mu_visc_peak']:.4f} ({hys_pct:.0f}%)  "
                         f"adh={d['mu_adh_peak']:.4f} ({adh_pct:.0f}%)")
        lines.append("")

        # 3. Speed-dependent comparison
        lines.append("■ 속도별 비교")
        lines.append("-" * 40)
        for spd in [0.001, 0.01, 0.1, 1.0]:
            vals = {name: d['speed_vals'].get(spd, 0) for name, d in data.items()}
            best_name = max(vals, key=vals.get)
            lines.append(f"  v = {spd} m/s:")
            for name, v in vals.items():
                marker = " ★" if name == best_name else ""
                lines.append(f"    {name}: μ = {v:.4f}{marker}")
        lines.append("")

        # 4. Conditions
        lines.append("■ 계산 조건")
        lines.append("-" * 40)
        for name, d in data.items():
            lines.append(f"  {name}:  σ₀ = {d['sigma_0']/1e6:.1f} MPa,  "
                         f"T = {d['temperature']:.0f}°C")
        lines.append("")
        lines.append("=" * 60)

        self._comment_text.insert('1.0', '\n'.join(lines))

    # ================================================================
    #  Export
    # ================================================================
    def _export_csv(self):
        """Export all compound results to CSV."""
        compounds = getattr(self.app, 'compound_data', [])
        if not compounds or not any(c.results for c in compounds):
            messagebox.showinfo("알림", "내보낼 결과가 없습니다.")
            return

        fp = filedialog.asksaveasfilename(
            title="CSV 내보내기",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")])
        if not fp:
            return

        try:
            v_ref = None
            for c in compounds:
                if c.results is not None:
                    v_ref = c.results['v']
                    break

            headers = ['velocity_m_s']
            data_cols = [v_ref]
            for c in compounds:
                name = c.name.replace(',', '_')
                if c.results is not None:
                    headers.extend([f'{name}_mu_visc', f'{name}_mu_adh', f'{name}_mu_total'])
                    data_cols.append(c.results['mu_visc'])
                    data_cols.append(c.results.get('mu_adh', np.zeros_like(v_ref)))
                    data_cols.append(c.results['mu_total'])

            with open(fp, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                for row_idx in range(len(v_ref)):
                    row = [col[row_idx] if col is not None else '' for col in data_cols]
                    writer.writerow(row)

            messagebox.showinfo("완료", f"CSV 저장 완료: {os.path.basename(fp)}")
        except Exception as e:
            messagebox.showerror("저장 실패", str(e))

    def _save_figure(self):
        """Save current sub-tab's figure as image."""
        # Get current sub-tab's figure
        try:
            current_idx = self._plot_notebook.index(self._plot_notebook.select())
            key = list(self._plot_tabs.keys())[current_idx]
            fig = self._plot_tabs[key][0]
        except Exception:
            fig = self._fig

        if fig is None:
            return
        fp = filedialog.asksaveasfilename(
            title="그래프 저장",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")])
        if not fp:
            return
        try:
            fig.savefig(fp, dpi=200, bbox_inches='tight')
            messagebox.showinfo("완료", f"저장 완료: {os.path.basename(fp)}")
        except Exception as e:
            messagebox.showerror("저장 실패", str(e))
