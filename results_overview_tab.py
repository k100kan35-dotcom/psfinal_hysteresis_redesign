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
    """계산 결과 종합 탭 (Tab 1): 다중 컴파운드 비교 플롯."""

    COMPOUND_COLORS = [
        '#2563EB', '#059669', '#D97706', '#DC2626',
        '#7C3AED', '#DB2777', '#0891B2', '#65A30D',
    ]

    def __init__(self, app):
        self.app = app
        app._create_results_overview_tab = self._create_tab

        self._fig = None
        self._canvas = None
        self._plot_mode_var = None
        self._compound_visible_vars = []

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

        # Plot mode
        sec1 = ttk.LabelFrame(left, text="플롯 모드", padding=8)
        sec1.pack(fill=tk.X, pady=4, padx=2)

        self._plot_mode_var = tk.StringVar(value='total_overlay')
        modes = [
            ('total_overlay', 'μ_total 오버레이'),
            ('separated', '분리 플롯 (3패널)'),
            ('per_compound', '컴파운드별'),
        ]
        for val, label in modes:
            ttk.Radiobutton(sec1, text=label, variable=self._plot_mode_var,
                            value=val, command=self.refresh).pack(anchor=tk.W, pady=1)

        # Compound visibility
        sec2 = ttk.LabelFrame(left, text="컴파운드 표시", padding=8)
        sec2.pack(fill=tk.X, pady=4, padx=2)
        self._visibility_frame = sec2

        # Export button
        sec3 = ttk.LabelFrame(left, text="내보내기", padding=8)
        sec3.pack(fill=tk.X, pady=4, padx=2)
        ttk.Button(sec3, text="CSV 내보내기", command=self._export_csv).pack(fill=tk.X, pady=2)
        ttk.Button(sec3, text="그래프 이미지 저장", command=self._save_figure).pack(fill=tk.X, pady=2)

        # Refresh button
        ttk.Button(left, text="새로고침", command=self.refresh).pack(fill=tk.X, pady=8, padx=4)

        # Status
        self._status_var = tk.StringVar(value="데이터 입력 탭에서 계산을 실행하세요")
        ttk.Label(left, textvariable=self._status_var, font=F['small'],
                  foreground=C['text_secondary'], wraplength=240).pack(fill=tk.X, pady=4, padx=4)

        # ── Right plot area ──
        right = ttk.Frame(main)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._fig = Figure(figsize=(10, 7), dpi=100, facecolor='white')
        self._fig.subplots_adjust(hspace=0.4, wspace=0.3)
        self._canvas = FigureCanvasTkAgg(self._fig, right)
        toolbar = NavigationToolbar2Tk(self._canvas, right)
        toolbar.update()
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ================================================================
    #  Refresh
    # ================================================================
    def refresh(self):
        """Refresh plots from app.all_compound_results."""
        compounds = getattr(self.app, 'compound_data', [])
        results = getattr(self.app, 'all_compound_results', [])

        if not compounds or not results:
            self._fig.clear()
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5, '데이터 입력 탭에서 계산을 실행하세요',
                    ha='center', va='center', fontsize=14,
                    transform=ax.transAxes, color='gray')
            ax.set_axis_off()
            self._canvas.draw_idle()
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

        mode = self._plot_mode_var.get() if self._plot_mode_var else 'total_overlay'
        self._fig.clear()

        if mode == 'total_overlay':
            self._plot_total_overlay(visible)
        elif mode == 'separated':
            self._plot_separated(visible)
        elif mode == 'per_compound':
            self._plot_per_compound(visible)

        self._fig.tight_layout()
        self._canvas.draw_idle()
        self._status_var.set(f"{len(compounds)}개 컴파운드 결과 표시 중")

    def _update_visibility_checkboxes(self, compounds):
        """Update compound visibility checkboxes."""
        # Only rebuild if count changed
        if len(self._compound_visible_vars) == len(compounds):
            return

        for w in self._visibility_frame.winfo_children():
            w.destroy()
        self._compound_visible_vars.clear()

        for i, cpd in enumerate(compounds):
            var = tk.BooleanVar(value=True)
            self._compound_visible_vars.append(var)
            color = self.COMPOUND_COLORS[i % len(self.COMPOUND_COLORS)]
            cb = ttk.Checkbutton(self._visibility_frame, text=cpd.name,
                                 variable=var, command=self.refresh)
            cb.pack(anchor=tk.W, pady=1)

    # ================================================================
    #  Plot modes
    # ================================================================
    def _plot_total_overlay(self, visible_compounds):
        """All compounds' mu_total on one axis."""
        ax = self._fig.add_subplot(111)
        ax.set_xlabel('Velocity (m/s)')
        ax.set_ylabel('μ_total')
        ax.set_title('μ_total vs Velocity')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)

        for i, cpd in visible_compounds:
            if cpd.results is None:
                continue
            r = cpd.results
            color = self.COMPOUND_COLORS[i % len(self.COMPOUND_COLORS)]
            ax.plot(r['v'], r['mu_total'], '-', color=color, linewidth=2, label=cpd.name)

        if visible_compounds:
            ax.legend(fontsize=10, loc='best')

    def _plot_separated(self, visible_compounds):
        """3-panel: mu_visc, mu_adh, mu_total."""
        ax1 = self._fig.add_subplot(2, 2, 1)
        ax2 = self._fig.add_subplot(2, 2, 2)
        ax3 = self._fig.add_subplot(2, 2, 3)
        ax4 = self._fig.add_subplot(2, 2, 4)

        for ax, title, key in [
            (ax1, 'μ_visc (Hysteresis)', 'mu_visc'),
            (ax2, 'μ_adh (Adhesion)', 'mu_adh'),
            (ax3, 'μ_total', 'mu_total'),
        ]:
            ax.set_xlabel('Velocity (m/s)')
            ax.set_ylabel(title.split('(')[0].strip())
            ax.set_title(title)
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)

            for i, cpd in visible_compounds:
                if cpd.results is None:
                    continue
                r = cpd.results
                color = self.COMPOUND_COLORS[i % len(self.COMPOUND_COLORS)]
                y = r.get(key)
                if y is not None:
                    ax.plot(r['v'], y, '-', color=color, linewidth=2, label=cpd.name)

            ax.legend(fontsize=8, loc='best')

        # 4th panel: bar chart at key speeds
        ax4.set_title('속도별 비교')
        ax4.set_ylabel('μ_total')
        ax4.grid(True, alpha=0.3, axis='y')

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
                ax4.bar(x + ci_local * width, vals, width, label=cpd.name, color=color, alpha=0.8)

            ax4.set_xticks(x + width * (n_cpd - 1) / 2)
            ax4.set_xticklabels([f'{s} m/s' for s in key_speeds], fontsize=9)
            ax4.legend(fontsize=8, loc='best')

    def _plot_per_compound(self, visible_compounds):
        """One subplot per compound."""
        n = len(visible_compounds)
        if n == 0:
            return
        cols = min(n, 3)
        rows = (n + cols - 1) // cols

        for plot_idx, (i, cpd) in enumerate(visible_compounds):
            ax = self._fig.add_subplot(rows, cols, plot_idx + 1)
            ax.set_title(cpd.name, fontsize=11)
            ax.set_xlabel('v (m/s)', fontsize=9)
            ax.set_ylabel('μ', fontsize=9)
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)

            if cpd.results is None:
                continue
            r = cpd.results
            color = self.COMPOUND_COLORS[i % len(self.COMPOUND_COLORS)]

            ax.plot(r['v'], r['mu_total'], '-', color=color, linewidth=2, label='μ_total')
            ax.plot(r['v'], r['mu_visc'], '--', color=color, linewidth=1, alpha=0.7, label='μ_visc')
            if r.get('mu_adh') is not None:
                ax.plot(r['v'], r['mu_adh'], ':', color=color, linewidth=1, alpha=0.7, label='μ_adh')
            ax.legend(fontsize=7, loc='best')

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
            # Use the first compound's velocity array as reference
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
        """Save current figure as image."""
        if self._fig is None:
            return
        fp = filedialog.asksaveasfilename(
            title="그래프 저장",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")])
        if not fp:
            return
        try:
            self._fig.savefig(fp, dpi=200, bbox_inches='tight')
            messagebox.showinfo("완료", f"저장 완료: {os.path.basename(fp)}")
        except Exception as e:
            messagebox.showerror("저장 실패", str(e))
