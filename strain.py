# strain_gui_piecewise.py
# Persson nonlinear f,g builder from strain sweep (T, f, strain, ReE, ImE)
# + Piecewise temperature-set averaging by strain ranges (Group A / Group B with split strain)
# + Target data tab (load target curve and overlay on plot)
# Python 3.9 compatible (no "|" union types)

import os
import re
import math
import csv
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


# -----------------------------
# Parsing
# -----------------------------
def parse_strain_sweep_file(path):
    """
    Parse rows containing at least 5 numbers:
      T  f  strain  ReE  ImE
    Return: dict[T] -> list of (freq, strain, ReE, ImE)
    """
    data_by_T = {}
    float_re = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

    def is_numeric_row(line):
        nums = float_re.findall(line)
        return len(nums) >= 5

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.upper() in ("AAA", "BBB"):
                continue
            if not is_numeric_row(line):
                continue

            nums = float_re.findall(line)
            T = float(nums[0])
            freq = float(nums[1])
            strain = float(nums[2])
            ReE = float(nums[3])
            ImE = float(nums[4])
            data_by_T.setdefault(T, []).append((freq, strain, ReE, ImE))

    for T in list(data_by_T.keys()):
        rows = data_by_T[T]
        rows.sort(key=lambda x: x[1])
        data_by_T[T] = rows

    return data_by_T


def parse_target_curve_file(path, strain_is_percent=False):
    """
    Target curve:
      - whitespace/CSV/TSV ok
      - columns: strain, f   (optional: g)
    IMPORTANT:
      - Your target strain is already a FRACTION (e.g., 1.49E-04), NOT percent.
      - So default strain_is_percent=False (no /100).
    """
    float_re = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
    strains, fvals, gvals = [], [], []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            nums = float_re.findall(line)
            if len(nums) < 2:
                continue
            s = float(nums[0])
            fv = float(nums[1])
            strains.append(s)
            fvals.append(fv)
            if len(nums) >= 3:
                gvals.append(float(nums[2]))

    if len(strains) < 2:
        return None

    s = np.array(strains, dtype=float)
    if strain_is_percent:
        s = s / 100.0  # ONLY if user says so

    f = np.array(fvals, dtype=float)
    g = np.array(gvals, dtype=float) if len(gvals) == len(strains) else None

    idx = np.argsort(s)
    s = s[idx]
    f = f[idx]
    if g is not None:
        g = g[idx]

    return {"strain": s, "f": f, "g": g}


# -----------------------------
# Interpolation helpers
# -----------------------------
def _safe_log(x):
    return np.log(np.clip(x, 1e-300, None))


def interp_curve(x, y, xq, kind="loglog_linear", extrap_mode="hold"):
    """
    Interpolate y(x) to y(xq).
    kind: "linear" | "loglog_linear" | "loglog_cubic"
    extrap_mode: "hold" | "linear" | "none"
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xq = np.asarray(xq, dtype=float)

    m = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x = x[m]
    y = y[m]
    if x.size < 2:
        return np.full_like(xq, np.nan, dtype=float)

    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]
    xmin, xmax = x[0], x[-1]

    yq = np.full_like(xq, np.nan, dtype=float)

    inside = (xq >= xmin) & (xq <= xmax)
    xqi = xq[inside]
    if xqi.size > 0:
        if kind == "linear":
            yq[inside] = np.interp(xqi, x, y)

        elif kind == "loglog_linear":
            if np.any(y <= 0):
                yq[inside] = np.interp(xqi, x, y)
            else:
                lx = _safe_log(x)
                ly = _safe_log(y)
                yq[inside] = np.exp(np.interp(_safe_log(xqi), lx, ly))

        elif kind == "loglog_cubic":
            if np.any(y <= 0):
                yq[inside] = np.interp(xqi, x, y)
            else:
                lx = _safe_log(x)
                ly = _safe_log(y)
                for k, xv in zip(np.where(inside)[0], xqi):
                    t = float(_safe_log(xv))
                    j = int(np.searchsorted(lx, t))
                    j0 = max(0, j - 2)
                    j1 = min(lx.size, j0 + 4)
                    j0 = max(0, j1 - 4)
                    if (j1 - j0) < 4:
                        yq[k] = np.exp(np.interp(t, lx, ly))
                    else:
                        xs = lx[j0:j1]
                        ys = ly[j0:j1]
                        coef = np.polyfit(xs, ys, 3)
                        yq[k] = math.exp(np.polyval(coef, t))
        else:
            raise ValueError("Unknown interp kind")

    left = xq < xmin
    right = xq > xmax

    if extrap_mode == "none":
        pass
    elif extrap_mode == "hold":
        if np.any(left):
            yq[left] = y[0]
        if np.any(right):
            yq[right] = y[-1]
    elif extrap_mode == "linear":
        if kind == "linear" or np.any(y <= 0):
            x0, x1 = x[0], x[1]
            y0, y1 = y[0], y[1]
            xl0, xl1 = x[-2], x[-1]
            yl0, yl1 = y[-2], y[-1]
            mL = (y1 - y0) / (x1 - x0)
            mR = (yl1 - yl0) / (xl1 - xl0)
            if np.any(left):
                yq[left] = y0 + mL * (xq[left] - x0)
            if np.any(right):
                yq[right] = yl1 + mR * (xq[right] - xl1)
        else:
            lx = _safe_log(x)
            ly = _safe_log(y)
            mL = (ly[1] - ly[0]) / (lx[1] - lx[0])
            mR = (ly[-1] - ly[-2]) / (lx[-1] - lx[-2])
            if np.any(left):
                t = _safe_log(xq[left])
                yq[left] = np.exp(ly[0] + mL * (t - lx[0]))
            if np.any(right):
                t = _safe_log(xq[right])
                yq[right] = np.exp(ly[-1] + mR * (t - lx[-1]))
    else:
        raise ValueError("Unknown extrap mode")

    return yq


def persson_strain_grid(max_strain_fraction):
    """
    Persson-like discrete grid:
      starts at 1.49e-4 and multiplies by 1.5
    """
    s0 = 1.49e-4
    r = 1.5
    out = []
    s = s0
    while s <= max_strain_fraction * 1.000001:
        out.append(s)
        s *= r
    # Always include max_strain_fraction as the final point
    # so the grid covers the full data range
    if out and out[-1] < max_strain_fraction * 0.999:
        out.append(max_strain_fraction)
    return np.array(out, dtype=float)


def make_grid(n_points, max_strain_fraction, use_persson_grid=True):
    if use_persson_grid:
        g = persson_strain_grid(max_strain_fraction)
        if g.size >= 2:
            return g
    smin = 1e-4
    smax = max(1.1 * smin, max_strain_fraction)
    return np.logspace(np.log10(smin), np.log10(smax), int(max(5, n_points)))


# -----------------------------
# Compute f,g per temperature
# -----------------------------
def compute_fg_by_T(
    data_by_T,
    target_freq=1.0,
    tol=0.01,
    freq_mode="nearest",
    strain_is_percent=True,
    e0_n_points=5,
    clip_leq_1=True
):
    """
    For each temperature:
      - take data at freq=1 Hz (nearest or tolerance)
      - convert strain to fraction if strain_is_percent True
      - E0 = mean of first e0_n_points in low strain (Re and Im separately)
      - f = ReE/E0_re, g = ImE/E0_im
    """
    fg_by_T = {}
    for T, rows in data_by_T.items():
        if not rows:
            continue

        arr = np.array(rows, dtype=float)
        freqs = arr[:, 0]
        strains = arr[:, 1]
        ReE = arr[:, 2]
        ImE = arr[:, 3]

        if freq_mode == "nearest":
            uniq = np.unique(freqs)
            if uniq.size == 0:
                continue
            fsel = float(uniq[np.argmin(np.abs(uniq - target_freq))])
            m = np.isclose(freqs, fsel)
        else:
            m = np.abs(freqs - target_freq) <= tol

        if not np.any(m):
            continue

        s = strains[m].astype(float)
        reE = ReE[m].astype(float)
        imE = ImE[m].astype(float)

        if strain_is_percent:
            s = s / 100.0  # convert % -> fraction

        good = np.isfinite(s) & np.isfinite(reE) & np.isfinite(imE) & (s > 0)
        s, reE, imE = s[good], reE[good], imE[good]

        if s.size < max(3, e0_n_points):
            continue

        idx = np.argsort(s)
        s, reE, imE = s[idx], reE[idx], imE[idx]

        n0 = int(max(1, min(e0_n_points, s.size)))
        E0_re = float(np.mean(reE[:n0]))
        E0_im = float(np.mean(imE[:n0]))

        f = reE / E0_re if E0_re != 0 else np.full_like(reE, np.nan)
        g = imE / E0_im if E0_im != 0 else np.full_like(imE, np.nan)

        if clip_leq_1:
            f = np.minimum(f, 1.0)
            # g is NOT clipped: g(ε) can exceed 1.0 (loss modulus overshoot)

        fg_by_T[T] = {"strain": s, "f": f, "g": g, "E0_re": E0_re, "E0_im": E0_im}

    return fg_by_T


def average_fg_on_grid(
    fg_by_T,
    selected_Ts,
    grid_strain,
    interp_kind="loglog_linear",
    extrap_mode="hold",
    missing_policy="ignore",
    avg_mode="mean",
    n_min=1,
    clip_leq_1=True
):
    """
    Build averaged f,g on common grid using selected temperatures.
    missing_policy:
      - "ignore": outside each T range => NaN, then avg ignores NaN.
      - "hold": outside each T range => hold boundary values.
      - "extrap": use extrap_mode (hold/linear/none).
    Tail-hold:
      If n_eff < n_min at some grid point, keep last valid value onward.
    """
    Ts = [T for T in selected_Ts if T in fg_by_T]
    if len(Ts) == 0:
        return None

    F, G = [], []
    for T in Ts:
        s = fg_by_T[T]["strain"]
        f = fg_by_T[T]["f"]
        g = fg_by_T[T]["g"]
        if s.size < 2:
            continue

        if missing_policy == "ignore":
            fq = interp_curve(s, f, grid_strain, kind=interp_kind, extrap_mode="none")
            gq = interp_curve(s, g, grid_strain, kind=interp_kind, extrap_mode="none")
        elif missing_policy == "hold":
            fq = interp_curve(s, f, grid_strain, kind=interp_kind, extrap_mode="hold")
            gq = interp_curve(s, g, grid_strain, kind=interp_kind, extrap_mode="hold")
        elif missing_policy == "extrap":
            fq = interp_curve(s, f, grid_strain, kind=interp_kind, extrap_mode=extrap_mode)
            gq = interp_curve(s, g, grid_strain, kind=interp_kind, extrap_mode=extrap_mode)
        else:
            raise ValueError("Unknown missing_policy")

        F.append(fq)
        G.append(gq)

    if len(F) == 0:
        return None

    F = np.vstack(F)
    G = np.vstack(G)

    def reduce_mat(M):
        if avg_mode == "mean":
            return np.nanmean(M, axis=0)
        if avg_mode == "median":
            return np.nanmedian(M, axis=0)
        if avg_mode == "max":
            return np.nanmax(M, axis=0)
        raise ValueError("Unknown avg_mode")

    f_avg = reduce_mat(F)
    g_avg = reduce_mat(G)

    n_eff = np.sum(np.isfinite(F), axis=0)

    # Tail hold when not enough participants
    n_min = int(max(1, n_min))
    hold_started = False
    last_f, last_g = None, None
    for i in range(grid_strain.size):
        if not hold_started:
            if n_eff[i] < n_min or (not np.isfinite(f_avg[i])) or (not np.isfinite(g_avg[i])):
                if i == 0:
                    last_f, last_g = 1.0, 1.0
                hold_started = True
            else:
                last_f, last_g = float(f_avg[i]), float(g_avg[i])
        if hold_started:
            f_avg[i] = last_f
            g_avg[i] = last_g

    if clip_leq_1:
        f_avg = np.minimum(f_avg, 1.0)
        # g_avg is NOT clipped: g(ε) can exceed 1.0 (loss modulus overshoot)

    return {"strain": grid_strain.copy(), "f_avg": f_avg, "g_avg": g_avg, "Ts_used": Ts, "n_eff": n_eff}


def stitch_two_ranges(resA, resB, split_strain_fraction):
    """
    Piecewise stitch:
      strain <= split -> resA
      strain > split  -> resB
    Assumes same strain grid.
    """
    if resA is None or resB is None:
        return None
    s = resA["strain"]
    if resB["strain"].shape != s.shape or np.any(resB["strain"] != s):
        return None

    split = float(split_strain_fraction)
    maskA = s <= split
    maskB = ~maskA

    f = np.empty_like(resA["f_avg"])
    g = np.empty_like(resA["g_avg"])
    f[maskA] = resA["f_avg"][maskA]
    g[maskA] = resA["g_avg"][maskA]
    f[maskB] = resB["f_avg"][maskB]
    g[maskB] = resB["g_avg"][maskB]

    n_eff = np.empty_like(resA["n_eff"])
    n_eff[maskA] = resA["n_eff"][maskA]
    n_eff[maskB] = resB["n_eff"][maskB]

    return {
        "strain": s.copy(),
        "f_avg": f,
        "g_avg": g,
        "n_eff": n_eff,
        "split": split,
        "Ts_used_A": list(resA["Ts_used"]),
        "Ts_used_B": list(resB["Ts_used"]),
        "resA": resA,
        "resB": resB
    }


def stitch_n_ranges(results, split_strain_fractions):
    """
    Piecewise stitch N regions:
      results: list of N averaging results
      split_strain_fractions: sorted list of N-1 split strain fractions

    Region 0: strain <= splits[0]
    Region i: splits[i-1] < strain <= splits[i]
    Region N-1: strain > splits[-1]
    """
    n = len(results)
    if n == 0:
        return None
    if any(r is None for r in results):
        return None

    s = results[0]["strain"]
    for r in results[1:]:
        if r["strain"].shape != s.shape or np.any(r["strain"] != s):
            return None

    splits = sorted(float(sp) for sp in split_strain_fractions)

    f = np.empty_like(results[0]["f_avg"])
    g = np.empty_like(results[0]["g_avg"])
    n_eff = np.empty_like(results[0]["n_eff"])

    for i in range(n):
        if i == 0:
            mask = s <= splits[0]
        elif i < n - 1:
            mask = (s > splits[i - 1]) & (s <= splits[i])
        else:
            mask = s > splits[-1]

        f[mask] = results[i]["f_avg"][mask]
        g[mask] = results[i]["g_avg"][mask]
        n_eff[mask] = results[i]["n_eff"][mask]

    result = {
        "strain": s.copy(),
        "f_avg": f,
        "g_avg": g,
        "n_eff": n_eff,
        "splits": list(splits),
    }
    for i in range(n):
        result["Ts_used_%d" % i] = list(results[i]["Ts_used"])
        result["res_%d" % i] = results[i]

    return result


# -----------------------------
# GUI helpers
# -----------------------------
class ScrollableFrame(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        vscroll = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)

        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=vscroll.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        vscroll.pack(side="right", fill="y")

        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        self.canvas.bind_all("<MouseWheel>", _on_mousewheel)


# -----------------------------
# App
# -----------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Persson Nonlinear f,g Builder (piecewise temperature averaging)")
        self.geometry("1300x820")

        self.data_by_T = {}
        self.fg_by_T = {}

        self.N_REGIONS = 5
        self.temp_vars_list = [{} for _ in range(self.N_REGIONS)]

        self.region_results = [None] * self.N_REGIONS
        self.stitched = None
        self.snapshots = []

        # target overlay
        self.target = None
        self.target_label = "(no target)"

        root = ttk.Frame(self)
        root.pack(fill="both", expand=True)

        self.left = ScrollableFrame(root)
        self.left.pack(side="left", fill="y")

        self.right = ttk.Frame(root)
        self.right.pack(side="right", fill="both", expand=True)

        self.nb = ttk.Notebook(self.left.inner)
        self.nb.pack(fill="both", expand=False, padx=8, pady=8)

        self.tab_builder = ttk.Frame(self.nb)
        self.tab_target = ttk.Frame(self.nb)
        self.nb.add(self.tab_builder, text="Builder")
        self.nb.add(self.tab_target, text="Target data")

        self._build_builder_controls(self.tab_builder)
        self._build_target_controls(self.tab_target)
        self._build_plot(self.right)

    def _build_builder_controls(self, parent):
        pad = {"padx": 10, "pady": 6}

        lf = ttk.LabelFrame(parent, text="1) Strain sweep file (T, f, strain, ReE, ImE)")
        lf.pack(fill="x", **pad)
        ttk.Button(lf, text="Load file", command=self.on_load_file).pack(side="left", padx=8, pady=8)
        self.file_label = ttk.Label(lf, text="(no file)")
        self.file_label.pack(side="left", padx=8)

        fr = ttk.LabelFrame(parent, text="2) Frequency selection")
        fr.pack(fill="x", **pad)

        self.target_freq = tk.StringVar(value="1.0")
        self.tol = tk.StringVar(value="0.01")
        self.freq_mode = tk.StringVar(value="nearest")

        row = ttk.Frame(fr); row.pack(fill="x", padx=8, pady=6)
        ttk.Label(row, text="Target freq (Hz):").pack(side="left")
        ttk.Entry(row, textvariable=self.target_freq, width=10).pack(side="left", padx=6)
        ttk.Label(row, text="Tol (Hz):").pack(side="left", padx=(16, 0))
        ttk.Entry(row, textvariable=self.tol, width=10).pack(side="left", padx=6)

        row2 = ttk.Frame(fr); row2.pack(fill="x", padx=8, pady=6)
        ttk.Label(row2, text="Mode:").pack(side="left")
        ttk.Radiobutton(row2, text="Nearest", variable=self.freq_mode, value="nearest").pack(side="left", padx=8)
        ttk.Radiobutton(row2, text="Tolerance", variable=self.freq_mode, value="tolerance").pack(side="left", padx=8)

        nr = ttk.LabelFrame(parent, text="3) Normalization")
        nr.pack(fill="x", **pad)

        self.strain_is_percent = tk.BooleanVar(value=True)
        self.e0_points = tk.StringVar(value="5")
        self.clip_leq_1 = tk.BooleanVar(value=True)

        ttk.Checkbutton(nr, text="File strain is percent (%)", variable=self.strain_is_percent).pack(anchor="w", padx=8, pady=4)
        row3 = ttk.Frame(nr); row3.pack(fill="x", padx=8, pady=4)
        ttk.Label(row3, text="E0 points (lowest strain avg):").pack(side="left")
        ttk.Entry(row3, textvariable=self.e0_points, width=8).pack(side="left", padx=6)
        ttk.Checkbutton(nr, text="Clip f,g ≤ 1 (Persson-like)", variable=self.clip_leq_1).pack(anchor="w", padx=8, pady=4)

        ia = ttk.LabelFrame(parent, text="4) Interp / Missing / N_min / Average")
        ia.pack(fill="x", **pad)

        self.interp_kind = tk.StringVar(value="loglog_linear")
        self.extrap_mode = tk.StringVar(value="hold")
        self.avg_mode = tk.StringVar(value="mean")
        self.missing_policy = tk.StringVar(value="ignore")
        self.n_min = tk.StringVar(value="3")

        row = ttk.Frame(ia); row.pack(fill="x", padx=8, pady=4)
        ttk.Label(row, text="Interp:").pack(side="left")
        ttk.Radiobutton(row, text="linear", variable=self.interp_kind, value="linear").pack(side="left", padx=6)
        ttk.Radiobutton(row, text="log-log Linear", variable=self.interp_kind, value="loglog_linear").pack(side="left", padx=6)
        ttk.Radiobutton(row, text="log-log Cubic", variable=self.interp_kind, value="loglog_cubic").pack(side="left", padx=6)

        row = ttk.Frame(ia); row.pack(fill="x", padx=8, pady=4)
        ttk.Label(row, text="Missing policy:").pack(side="left")
        ttk.Radiobutton(row, text="Ignore", variable=self.missing_policy, value="ignore").pack(side="left", padx=6)
        ttk.Radiobutton(row, text="Hold", variable=self.missing_policy, value="hold").pack(side="left", padx=6)
        ttk.Radiobutton(row, text="Extrap", variable=self.missing_policy, value="extrap").pack(side="left", padx=6)

        row = ttk.Frame(ia); row.pack(fill="x", padx=8, pady=4)
        ttk.Label(row, text="Extrap mode (if Extrap):").pack(side="left")
        ttk.Radiobutton(row, text="Hold", variable=self.extrap_mode, value="hold").pack(side="left", padx=6)
        ttk.Radiobutton(row, text="Linear", variable=self.extrap_mode, value="linear").pack(side="left", padx=6)
        ttk.Radiobutton(row, text="None", variable=self.extrap_mode, value="none").pack(side="left", padx=6)

        row = ttk.Frame(ia); row.pack(fill="x", padx=8, pady=4)
        ttk.Label(row, text="Average:").pack(side="left")
        ttk.Radiobutton(row, text="Mean", variable=self.avg_mode, value="mean").pack(side="left", padx=6)
        ttk.Radiobutton(row, text="Median", variable=self.avg_mode, value="median").pack(side="left", padx=6)
        ttk.Radiobutton(row, text="Max", variable=self.avg_mode, value="max").pack(side="left", padx=6)

        row = ttk.Frame(ia); row.pack(fill="x", padx=8, pady=4)
        ttk.Label(row, text="N_min (min temps, else tail-hold):").pack(side="left")
        ttk.Entry(row, textvariable=self.n_min, width=6).pack(side="left", padx=6)

        gg = ttk.LabelFrame(parent, text="5) Grid / Extend")
        gg.pack(fill="x", **pad)

        self.grid_points = tk.StringVar(value="20")
        self.use_persson_grid = tk.BooleanVar(value=True)
        self.extend_to = tk.StringVar(value="40")

        row = ttk.Frame(gg); row.pack(fill="x", padx=8, pady=4)
        ttk.Label(row, text="Grid points:").pack(side="left")
        ttk.Entry(row, textvariable=self.grid_points, width=8).pack(side="left", padx=6)
        ttk.Checkbutton(row, text="Persson grid", variable=self.use_persson_grid).pack(side="left", padx=12)

        row2 = ttk.Frame(gg); row2.pack(fill="x", padx=8, pady=4)
        ttk.Label(row2, text="Extend to strain (%):").pack(side="left")
        ttk.Entry(row2, textvariable=self.extend_to, width=10).pack(side="left", padx=6)

        pw = ttk.LabelFrame(parent, text="6) Piecewise temperature selection by strain range (5-region)")
        pw.pack(fill="x", **pad)

        # 4 split points → 5 regions
        self.split_strains = [
            tk.StringVar(value="5.0"),
            tk.StringVar(value="10.0"),
            tk.StringVar(value="15.0"),
            tk.StringVar(value="20.0"),
        ]
        row = ttk.Frame(pw); row.pack(fill="x", padx=8, pady=4)
        ttk.Label(row, text="Split strains (%):").pack(side="left")
        for i, sv in enumerate(self.split_strains):
            ttk.Entry(row, textvariable=sv, width=6).pack(side="left", padx=3)
            if i < len(self.split_strains) - 1:
                ttk.Label(row, text=",").pack(side="left")

        # 5 region tabs for temperature selection
        self.region_nb = ttk.Notebook(pw)
        self.region_nb.pack(fill="x", padx=8, pady=6)

        region_labels = [
            "R1 (0-5%)", "R2 (5-10%)", "R3 (10-15%)",
            "R4 (15-20%)", "R5 (>20%)"
        ]
        self.temp_frames = []
        self.temp_inners = []
        for i in range(self.N_REGIONS):
            tab = ttk.Frame(self.region_nb)
            self.region_nb.add(tab, text=region_labels[i])
            inner = ttk.Frame(tab)
            inner.pack(fill="x", padx=6, pady=6)
            self.temp_frames.append(tab)
            self.temp_inners.append(inner)

        pc = ttk.LabelFrame(parent, text="7) Plot / Compare / Export")
        pc.pack(fill="x", **pad)

        row = ttk.Frame(pc); row.pack(fill="x", padx=8, pady=6)
        ttk.Button(row, text="Compute / Refresh", command=self.on_compute).pack(side="left", padx=6)
        ttk.Button(row, text="Add (snapshot)", command=self.on_add_snapshot).pack(side="left", padx=6)

        self.snapshot_list = tk.Listbox(pc, height=4)
        self.snapshot_list.pack(fill="x", padx=8, pady=(0, 6))
        row2 = ttk.Frame(pc); row2.pack(fill="x", padx=8, pady=(0, 6))
        ttk.Button(row2, text="Delete", command=self.on_delete_snapshot).pack(side="left", padx=6)

        row3 = ttk.Frame(pc); row3.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Button(row3, text="Export stitched avg CSV", command=self.on_export_stitched).pack(side="left", padx=6)
        ttk.Button(row3, text="Export selected snapshot CSV", command=self.on_export_snapshot).pack(side="left", padx=6)

        self.status = ttk.Label(parent, text="")
        self.status.pack(fill="x", padx=10, pady=(4, 10))

    def _build_target_controls(self, parent):
        pad = {"padx": 10, "pady": 6}

        lf = ttk.LabelFrame(parent, text="Target curve overlay (strain, f [, g])")
        lf.pack(fill="x", **pad)

        ttk.Button(lf, text="Load target curve", command=self.on_load_target).pack(side="left", padx=8, pady=8)

        # IMPORTANT default: False (your target strain is NOT %)
        self.target_strain_is_percent = tk.BooleanVar(value=False)
        ttk.Checkbutton(lf, text="Target strain is percent (%)", variable=self.target_strain_is_percent).pack(side="left", padx=10)

        self.target_lbl = ttk.Label(lf, text="(no target)")
        self.target_lbl.pack(side="left", padx=8)

        row = ttk.Frame(parent)
        row.pack(fill="x", padx=10, pady=6)
        ttk.Button(row, text="Clear target", command=self.on_clear_target).pack(side="left", padx=6)
        ttk.Label(row, text="(If your target strain is like 1.49E-04, keep the % box OFF.)").pack(side="left", padx=10)

    def _build_plot(self, parent):
        self.fig = Figure(figsize=(7.7, 6.3), dpi=110)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("f(ε), g(ε) vs strain (linear axes)", fontsize=15, fontweight='bold')
        self.ax.set_xlabel("strain ε (fraction)", fontsize=13)
        self.ax.set_ylabel("factor", fontsize=13)
        self.ax.tick_params(labelsize=12)
        self.ax.grid(True, alpha=0.4)

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, parent)
        self.toolbar.update()

    def _read_float(self, var, default):
        try:
            return float(var.get())
        except Exception:
            return float(default)

    def _read_int(self, var, default):
        try:
            return int(float(var.get()))
        except Exception:
            return int(default)

    def _selected_temps(self, vars_dict):
        return [T for T, v in vars_dict.items() if v.get()]

    # -----------------------------
    # Callbacks
    # -----------------------------
    def on_load_file(self):
        path = filedialog.askopenfilename(
            title="Select strain sweep file",
            filetypes=[("Text files", "*.txt *.dat *.csv *.*"), ("All files", "*.*")]
        )
        if not path:
            return

        try:
            data_by_T = parse_strain_sweep_file(path)
        except Exception as e:
            messagebox.showerror("Load failed", str(e))
            return

        if not data_by_T:
            messagebox.showerror("Load failed", "No valid numeric rows found.")
            return

        self.data_by_T = data_by_T
        self.file_label.config(text=os.path.basename(path))
        self.status.config(text="Loaded %d temperature blocks." % len(data_by_T))

        # rebuild temp checkboxes for all regions
        for region_idx in range(self.N_REGIONS):
            for w in self.temp_inners[region_idx].winfo_children():
                w.destroy()
            self.temp_vars_list[region_idx] = {}

        temps = sorted(self.data_by_T.keys())
        for region_idx in range(self.N_REGIONS):
            for i, T in enumerate(temps):
                var = tk.BooleanVar(value=True)
                self.temp_vars_list[region_idx][T] = var
                cb = ttk.Checkbutton(
                    self.temp_inners[region_idx],
                    text="%.3f °C" % T, variable=var,
                    command=self.on_compute
                )
                r = i // 3
                c = i % 3
                cb.grid(row=r, column=c, sticky="w", padx=4, pady=2)

        self.on_compute()

    def on_load_target(self):
        path = filedialog.askopenfilename(
            title="Select target curve file",
            filetypes=[("Text/CSV files", "*.txt *.csv *.dat *.*"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            t = parse_target_curve_file(path, strain_is_percent=self.target_strain_is_percent.get())
        except Exception as e:
            messagebox.showerror("Target load failed", str(e))
            return
        if t is None:
            messagebox.showerror("Target load failed", "Could not parse target data (need at least 2 numeric columns).")
            return

        self.target = t
        self.target_label = os.path.basename(path)
        self.target_lbl.config(text=self.target_label)
        self._redraw_plot()

    def on_clear_target(self):
        self.target = None
        self.target_label = "(no target)"
        self.target_lbl.config(text=self.target_label)
        self._redraw_plot()

    def on_compute(self):
        if not self.data_by_T:
            return

        target = self._read_float(self.target_freq, 1.0)
        tol = self._read_float(self.tol, 0.01)
        freq_mode = self.freq_mode.get()

        strain_is_percent = bool(self.strain_is_percent.get())
        e0_n = self._read_int(self.e0_points, 5)
        clip = bool(self.clip_leq_1.get())

        interp_kind = self.interp_kind.get()
        extrap_mode = self.extrap_mode.get()
        avg_mode = self.avg_mode.get()
        missing_policy = self.missing_policy.get()
        n_min = self._read_int(self.n_min, 1)

        n_grid = self._read_int(self.grid_points, 20)
        extend_percent = self._read_float(self.extend_to, 40.0)
        max_strain = extend_percent / 100.0
        use_persson = bool(self.use_persson_grid.get())

        # Read 4 split points
        split_defaults = [5.0, 10.0, 15.0, 20.0]
        splits_percent = [self._read_float(sv, split_defaults[i])
                          for i, sv in enumerate(self.split_strains)]
        splits_fraction = [sp / 100.0 for sp in sorted(splits_percent)]

        self.fg_by_T = compute_fg_by_T(
            self.data_by_T,
            target_freq=target,
            tol=tol,
            freq_mode=("nearest" if freq_mode == "nearest" else "tolerance"),
            strain_is_percent=strain_is_percent,
            e0_n_points=e0_n,
            clip_leq_1=clip
        )

        if not self.fg_by_T:
            messagebox.showerror("Compute failed", "No temperature blocks had enough data after freq filtering.")
            return

        grid = make_grid(n_grid, max_strain, use_persson_grid=use_persson)

        # Compute averages for each of the 5 regions
        region_temps = []
        for ri in range(self.N_REGIONS):
            ts = self._selected_temps(self.temp_vars_list[ri])
            if len(ts) == 0:
                messagebox.showwarning(
                    "Selection",
                    "Select at least one temperature in Region %d." % (ri + 1)
                )
                return
            region_temps.append(ts)

        self.region_results = []
        for ri in range(self.N_REGIONS):
            res = average_fg_on_grid(
                self.fg_by_T, region_temps[ri], grid,
                interp_kind=interp_kind,
                extrap_mode=extrap_mode,
                missing_policy=missing_policy,
                avg_mode=avg_mode,
                n_min=n_min,
                clip_leq_1=clip
            )
            if res is None:
                messagebox.showerror(
                    "Compute failed",
                    "Averaging failed for Region %d (insufficient data)." % (ri + 1)
                )
                return
            self.region_results.append(res)

        self.stitched = stitch_n_ranges(self.region_results, splits_fraction)
        if self.stitched is None:
            messagebox.showerror("Compute failed", "Stitch failed (grid mismatch).")
            return

        self._redraw_plot()
        splits_str = ", ".join("%.1f%%" % sp for sp in splits_percent)
        temps_str = ", ".join("R%d=%d" % (i + 1, len(self.region_results[i]["Ts_used"]))
                              for i in range(self.N_REGIONS))
        self.status.config(
            text="Computed. Splits=[%s] | %s | Missing=%s | N_min=%d | GridN=%d"
                 % (splits_str, temps_str, missing_policy, n_min, grid.size)
        )

    def _redraw_plot(self):
        self.ax.clear()
        self.ax.set_title("f(ε), g(ε) vs strain (linear axes)", fontsize=15, fontweight='bold')
        self.ax.set_xlabel("strain ε (fraction)", fontsize=13)
        self.ax.set_ylabel("factor", fontsize=13)
        self.ax.tick_params(labelsize=12)
        self.ax.grid(True, alpha=0.4)

        # show individual T curves (thin), union of all region-selected temps
        if self.fg_by_T:
            Ts_union = set()
            for ri in range(self.N_REGIONS):
                Ts_union |= set(self._selected_temps(self.temp_vars_list[ri]))
            for T, d in self.fg_by_T.items():
                if T not in Ts_union:
                    continue
                s = d["strain"]
                f = d["f"]
                g = d["g"]
                self.ax.plot(s, f, linewidth=1.0, alpha=0.18)
                self.ax.plot(s, g, linewidth=1.0, alpha=0.18)

        # region averages (thin dashed)
        region_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        for ri in range(self.N_REGIONS):
            if ri < len(self.region_results) and self.region_results[ri] is not None:
                s = self.region_results[ri]["strain"]
                color = region_colors[ri % len(region_colors)]
                self.ax.plot(s, self.region_results[ri]["f_avg"],
                             linewidth=1.8, linestyle="--", color=color, alpha=0.5,
                             label="R%d f" % (ri + 1))
                self.ax.plot(s, self.region_results[ri]["g_avg"],
                             linewidth=1.8, linestyle=":", color=color, alpha=0.5,
                             label="R%d g" % (ri + 1))

        # stitched final
        if self.stitched is not None:
            s = self.stitched["strain"]
            f = self.stitched["f_avg"]
            g = self.stitched["g_avg"]
            self.ax.plot(s, f, linewidth=4.0, color='black', label="STITCHED f(ε)")
            self.ax.plot(s, g, linewidth=4.0, color='red', label="STITCHED g(ε)")
            # draw split lines
            splits = self.stitched.get("splits", [])
            for sp in splits:
                self.ax.axvline(sp, linewidth=1.2, alpha=0.5, color='gray', linestyle='--')

        # snapshots
        for snap in self.snapshots:
            s = snap["strain"]
            self.ax.plot(s, snap["f_avg"], linewidth=2.2, linestyle=":", label="%s f" % snap["label"])
            self.ax.plot(s, snap["g_avg"], linewidth=2.2, linestyle=":", label="%s g" % snap["label"])

        # target overlay (your target is: strain (fraction) + f)
        if self.target is not None:
            ts = self.target["strain"]
            tf = self.target["f"]
            self.ax.plot(ts, tf, linewidth=3.0, label="TARGET f (%s)" % self.target_label)
            if self.target["g"] is not None:
                tg = self.target["g"]
                self.ax.plot(ts, tg, linewidth=3.0, linestyle="--", label="TARGET g (%s)" % self.target_label)

        # linear axes only
        self.ax.set_xlim(left=0.0)
        self.ax.set_ylim(bottom=0.0)
        self.ax.legend(loc="upper right", frameon=True, fontsize=12)
        self.canvas.draw_idle()

    def on_add_snapshot(self):
        if self.stitched is None:
            return
        idx = len(self.snapshots) + 1
        splits = self.stitched.get("splits", [])
        splits_str = ",".join("%.1f%%" % (sp * 100.0) for sp in splits)
        label = "stitch_%d (splits=[%s])" % (idx, splits_str)
        snap = {
            "label": label,
            "strain": self.stitched["strain"].copy(),
            "f_avg": self.stitched["f_avg"].copy(),
            "g_avg": self.stitched["g_avg"].copy(),
            "n_eff": self.stitched["n_eff"].copy(),
            "splits": list(splits),
        }
        for i in range(self.N_REGIONS):
            key = "Ts_used_%d" % i
            if key in self.stitched:
                snap[key] = list(self.stitched[key])
        self.snapshots.append(snap)
        self.snapshot_list.insert("end", label)
        self._redraw_plot()

    def on_delete_snapshot(self):
        sel = self.snapshot_list.curselection()
        if not sel:
            return
        i = int(sel[0])
        self.snapshot_list.delete(i)
        del self.snapshots[i]
        self._redraw_plot()

    def _export_result_to_csv(self, res, default_name="fg.csv"):
        path = filedialog.asksaveasfilename(
            title="Save CSV",
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["strain_fraction", "f_value", "g_value", "n_eff"])
                s = res["strain"]
                fval = res["f_avg"]
                gval = res["g_avg"]
                n_eff = res.get("n_eff", np.full_like(s, np.nan))
                for i in range(len(s)):
                    w.writerow([float(s[i]), float(fval[i]), float(gval[i]), float(n_eff[i])])
        except Exception as e:
            messagebox.showerror("Export failed", str(e))
            return
        messagebox.showinfo("Export", "Saved: %s" % path)

    def on_export_stitched(self):
        if self.stitched is None:
            messagebox.showinfo("Export", "No stitched result to export.")
            return
        splits = self.stitched.get("splits", [])
        splits_tag = "_".join("%.0f" % (sp * 100.0) for sp in splits)
        self._export_result_to_csv(self.stitched, default_name="stitched_fg_splits_%spct.csv" % splits_tag)

    def on_export_snapshot(self):
        sel = self.snapshot_list.curselection()
        if not sel:
            messagebox.showinfo("Export", "Select a snapshot to export.")
            return
        i = int(sel[0])
        snap = self.snapshots[i]
        safe = snap["label"].replace(" ", "_").replace(":", "_")
        self._export_result_to_csv(snap, default_name="%s.csv" % safe)


if __name__ == "__main__":
    import sys

    # DPI: let Windows DPI virtualisation handle scaling (no SetProcessDpiAwareness).
    app = App()

    app.mainloop()
