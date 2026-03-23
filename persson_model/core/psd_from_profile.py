"""
PSD calculation from surface profile data.

This module calculates Power Spectral Density (PSD) from 1D surface height profiles
for use in Persson's contact mechanics theory.

Pipeline: detrend → window/FFT → 1D PSD → sinc² correction → 1D→2D conversion → log-bin

Reference:
    J. Chem. Phys. 162, 074704 (2025) - Top PSD calculation method
    PSD_SPECIFICATION.md — canonical parameter defaults
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.optimize import curve_fit
from scipy.special import gamma as gamma_func
from typing import Tuple, Optional, Dict, Any
import warnings


def logarithmic_binning(
    q: np.ndarray,
    C: np.ndarray,
    n_bins: int = 88
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Persson-style log-binning with geometric mean averaging.

    At low q where FFT frequency spacing exceeds the desired log-bin width,
    each FFT frequency gets its own bin to avoid information loss.

    Parameters
    ----------
    q : np.ndarray
        Wavenumber array (1/m)
    C : np.ndarray
        PSD values
    n_bins : int
        Total target number of log-bins (default: 88, Persson convention)

    Returns
    -------
    q_binned : np.ndarray
        Binned wavenumber array (bin-centre in log space)
    C_binned : np.ndarray
        Binned PSD values (geometric mean within bin)
    """
    # Filter valid data
    valid = (q > 0) & (C > 0) & np.isfinite(q) & np.isfinite(C)
    q = q[valid]
    C = C[valid]

    if len(q) < 2:
        return q, C

    # Sort by q
    idx = np.argsort(q)
    q_s = q[idx]
    C_s = C[idx]
    lq_s = np.log10(q_s)

    lq_min, lq_max = lq_s[0], lq_s[-1]
    target_dlogq = (lq_max - lq_min) / n_bins

    # Phase 1: Individual FFT points at low-q where spacing > target bin width
    bin_q = []
    bin_C = []
    i_transition = 0
    for i in range(len(q_s) - 1):
        spacing = lq_s[i + 1] - lq_s[i]
        if spacing > target_dlogq * 0.7:
            if C_s[i] > 0:
                bin_q.append(q_s[i])
                bin_C.append(C_s[i])
            i_transition = i + 1
        else:
            break

    # Phase 2: Log-uniform bins with geometric mean for the dense region
    if i_transition < len(q_s):
        lq_start = lq_s[i_transition]
        n_remaining = max(1, n_bins - len(bin_q))
        edges = np.linspace(lq_start, lq_max, n_remaining + 1)
        lq_tail = lq_s[i_transition:]
        C_tail = C_s[i_transition:]
        lC_tail = np.log10(np.maximum(C_tail, 1e-50))
        for j in range(n_remaining):
            if j == n_remaining - 1:
                mask = (lq_tail >= edges[j]) & (lq_tail <= edges[j + 1])
            else:
                mask = (lq_tail >= edges[j]) & (lq_tail < edges[j + 1])
            if np.any(mask):
                geo_C = 10.0 ** np.mean(lC_tail[mask])
                if geo_C > 0:
                    bin_q.append(10.0 ** ((edges[j] + edges[j + 1]) / 2.0))
                    bin_C.append(geo_C)

    return np.array(bin_q), np.array(bin_C)


def load_profile_data(
    filepath: str,
    x_col: int = 0,
    h_col: int = 1,
    delimiter: Optional[str] = None,
    skip_header: int = 0,
    x_unit: str = 'm',
    h_unit: str = 'm'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load surface profile data from file.

    Parameters
    ----------
    filepath : str
        Path to data file (.txt or .csv)
    x_col : int
        Column index for x position (0-based)
    h_col : int
        Column index for height (0-based)
    delimiter : str, optional
        Column delimiter. If None, auto-detect (whitespace or comma)
    skip_header : int
        Number of header lines to skip
    x_unit : str
        Unit of x data ('m', 'mm', 'um', 'nm')
    h_unit : str
        Unit of height data ('m', 'mm', 'um', 'nm')

    Returns
    -------
    x : np.ndarray
        Position array in meters
    h : np.ndarray
        Height array in meters
    """
    # Unit conversion factors to meters
    unit_factors = {
        'm': 1.0,
        'mm': 1e-3,
        'um': 1e-6,
        'μm': 1e-6,
        'micrometer': 1e-6,
        'nm': 1e-9
    }

    x_factor = unit_factors.get(x_unit.lower(), 1.0)
    h_factor = unit_factors.get(h_unit.lower(), 1.0)

    # Try to detect delimiter
    if delimiter is None:
        with open(filepath, 'r') as f:
            # Skip header lines
            for _ in range(skip_header):
                f.readline()
            first_line = f.readline()
            if ',' in first_line:
                delimiter = ','
            elif '\t' in first_line:
                delimiter = '\t'
            else:
                delimiter = None  # whitespace

    # Load data
    try:
        data = np.loadtxt(filepath, delimiter=delimiter, skiprows=skip_header)
    except Exception as e:
        raise ValueError(f"Failed to load data: {e}")

    if data.ndim == 1:
        raise ValueError("Data appears to be 1D. Need at least 2 columns (x, height).")

    x = data[:, x_col] * x_factor
    h = data[:, h_col] * h_factor

    return x, h


def detrend_profile(h: np.ndarray, method: str = 'mean') -> np.ndarray:
    """
    Remove trend from height profile.

    Parameters
    ----------
    h : np.ndarray
        Height array
    method : str
        'mean': subtract mean (default)
        'linear': remove linear trend
        'quadratic': remove quadratic trend

    Returns
    -------
    h_detrended : np.ndarray
        Detrended height array
    """
    if method == 'mean':
        return h - np.mean(h)
    elif method == 'linear':
        return signal.detrend(h, type='linear')
    elif method == 'quadratic':
        x = np.arange(len(h))
        coeffs = np.polyfit(x, h, 2)
        trend = np.polyval(coeffs, x)
        return h - trend
    else:
        raise ValueError(f"Unknown detrend method: {method}")


def _sinc2_correction(q: np.ndarray, C1D: np.ndarray) -> np.ndarray:
    """Compensate sinc²(f·dx) attenuation from rectangular sampling.

    Near the Nyquist frequency the DFT of discretely-sampled data is
    attenuated by sinc²(f·dx).  This correction recovers the true PSD.
    """
    q_max = q.max()  # ≈ π/dx (Nyquist)
    x = q / (2.0 * q_max)
    sinc_val = np.where(x < 1e-10, 1.0, np.sin(np.pi * x) / (np.pi * x))
    return C1D / (sinc_val ** 2)


def _compute_1d_psd_single_window(
    h: np.ndarray,
    dx: float,
    window: str = 'none',
    sinc2_correct: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute 1D PSD with a single window function.

    C1D(q) = (dx / (2π·N)) × |FFT(h·w)|²  (window-energy-corrected)
    """
    N = len(h)
    if window in ('hann', 'hanning'):
        w = np.hanning(N)
    elif window == 'hamming':
        w = np.hamming(N)
    elif window == 'blackman':
        w = np.blackman(N)
    else:
        w = np.ones(N)

    energy_corr = np.sqrt(N / np.sum(w ** 2))
    H_fft = fft(h * w * energy_corr)
    freqs = fftfreq(N, d=dx)
    q = 2.0 * np.pi * freqs
    C1D = (dx / (2.0 * np.pi * N)) * np.abs(H_fft) ** 2

    mask = q > 0
    q_pos = q[mask]
    C1D_pos = C1D[mask]
    if sinc2_correct:
        C1D_pos = _sinc2_correction(q_pos, C1D_pos)
    return q_pos, C1D_pos


def _compute_1d_psd_multitaper(
    h: np.ndarray,
    dx: float,
    NW: float = 4.0,
    sinc2_correct: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Multitaper PSD using DPSS (Discrete Prolate Spheroidal Sequences).

    NW = 4, K = 2*NW - 1 = 7 tapers.
    Each taper → FFT → PSD, then weighted average.
    """
    N = len(h)
    K = int(2 * NW - 1)
    norm = dx / (2.0 * np.pi * N)
    freqs = fftfreq(N, d=dx)
    q = 2.0 * np.pi * freqs

    try:
        # scipy.signal.windows.dpss returns (K, N) array of tapers
        # and concentration eigenvalues
        tapers, eigenvalues = signal.windows.dpss(N, NW, Kmax=K, return_ratios=True)
    except Exception:
        # Fallback: average rect/hann/hamm/blackman
        return _compute_1d_psd_multitaper_fallback(h, dx, sinc2_correct)

    C1D_sum = np.zeros(N)
    weight_sum = 0.0
    for k in range(K):
        w = tapers[k]
        energy_corr = np.sqrt(N / np.sum(w ** 2))
        H_fft = fft(h * w * energy_corr)
        wt = eigenvalues[k]
        C1D_sum += wt * norm * np.abs(H_fft) ** 2
        weight_sum += wt
    C1D_avg = C1D_sum / weight_sum

    mask = q > 0
    q_pos = q[mask]
    C1D_pos = C1D_avg[mask]
    if sinc2_correct:
        C1D_pos = _sinc2_correction(q_pos, C1D_pos)
    return q_pos, C1D_pos


def _compute_1d_psd_multitaper_fallback(
    h: np.ndarray,
    dx: float,
    sinc2_correct: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback multitaper: average rect / hann / hamm / blackman windows."""
    N = len(h)
    norm = dx / (2.0 * np.pi * N)
    freqs = fftfreq(N, d=dx)
    q = 2.0 * np.pi * freqs

    windows = [np.ones(N), np.hanning(N), np.hamming(N), np.blackman(N)]
    C1D_sum = np.zeros(N)
    for w in windows:
        energy_corr = np.sqrt(N / np.sum(w ** 2))
        H_fft = fft(h * w * energy_corr)
        C1D_sum += norm * np.abs(H_fft) ** 2
    C1D_avg = C1D_sum / len(windows)

    mask = q > 0
    q_pos = q[mask]
    C1D_pos = C1D_avg[mask]
    if sinc2_correct:
        C1D_pos = _sinc2_correction(q_pos, C1D_pos)
    return q_pos, C1D_pos


def _compute_1d_psd_welch(
    h: np.ndarray,
    dx: float,
    nperseg: Optional[int] = None,
    overlap: float = 0.5,
    sinc2_correct: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Welch method: overlapping Hanning-windowed segments, averaged."""
    N = len(h)
    if nperseg is None:
        nperseg = max(64, N // 8)
    nperseg = min(nperseg, N)

    step = max(1, int(nperseg * (1 - overlap)))
    w = np.hanning(nperseg)
    S1 = np.sum(w ** 2)

    starts = list(range(0, N - nperseg + 1, step))
    if not starts:
        starts = [0]

    C1D_accum = None
    freqs = fftfreq(nperseg, d=dx)
    q = 2.0 * np.pi * freqs
    for s in starts:
        seg = h[s:s + nperseg] * w
        H_fft = fft(seg)
        C1D_seg = (dx / (2.0 * np.pi * S1)) * np.abs(H_fft) ** 2
        if C1D_accum is None:
            C1D_accum = C1D_seg.copy()
        else:
            C1D_accum += C1D_seg
    C1D_accum /= len(starts)

    mask = q > 0
    q_pos = q[mask]
    C1D_pos = C1D_accum[mask]
    if sinc2_correct:
        C1D_pos = _sinc2_correction(q_pos, C1D_pos)
    return q_pos, C1D_pos


def calculate_1d_psd(
    x: np.ndarray,
    h: np.ndarray,
    window: str = 'multitaper',
    detrend_method: str = 'mean',
    sinc2_correct: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate 1D Power Spectral Density from height profile.

    C1D(q) = (dx / (2π·N)) × |FFT(h)|²

    Parameters
    ----------
    x : np.ndarray
        Position array (m)
    h : np.ndarray
        Height array (m)
    window : str
        Window function ('multitaper', 'welch', 'hann', 'hamming', 'blackman', 'none')
    detrend_method : str
        Detrending method ('mean', 'linear', 'quadratic')
    sinc2_correct : bool
        Apply sinc² correction near Nyquist (default: True)

    Returns
    -------
    q : np.ndarray
        Wavenumber array (1/m)
    C1d : np.ndarray
        1D PSD (m³)
    """
    dx = np.abs(x[1] - x[0])
    h_detrended = detrend_profile(h, method=detrend_method)

    if window == 'multitaper':
        return _compute_1d_psd_multitaper(h_detrended, dx, sinc2_correct=sinc2_correct)
    elif window == 'welch':
        return _compute_1d_psd_welch(h_detrended, dx, sinc2_correct=sinc2_correct)
    else:
        return _compute_1d_psd_single_window(h_detrended, dx, window=window,
                                              sinc2_correct=sinc2_correct)


def calculate_top_psd(
    x: np.ndarray,
    h: np.ndarray,
    window: str = 'multitaper',
    detrend_method: str = 'mean',
    sinc2_correct: bool = True
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Calculate Top PSD from height profile.

    Top PSD considers only the upper part of the surface (h > 0 after detrending).

    Reference: J. Chem. Phys. 162, 074704 (2025)

    Procedure:
    1. Detrend data (mean = 0)
    2. Create h_top: keep h > 0, set h <= 0 to 0
    3. Calculate phi = N_top / N_total (fraction of points > 0)
    4. Calculate PSD of h_top using same engine as full PSD
    5. Multiply by 1/phi for correction (Persson normalization)

    Returns
    -------
    q, C_top, phi
    """
    n = len(h)
    dx = np.abs(x[1] - x[0])

    # Detrend first
    h_detrended = detrend_profile(h, method=detrend_method)

    # Create h_top: keep h > 0, set h <= 0 to 0
    h_top = np.where(h_detrended > 0, h_detrended, 0.0)

    # Calculate phi = N_top / N_total
    n_top = np.sum(h_detrended > 0)
    phi = n_top / n

    if phi < 0.01:
        warnings.warn(f"Very small phi ({phi:.4f}). Top PSD may be unreliable.")

    # Calculate PSD of h_top using the same engine (no further detrend)
    if window == 'multitaper':
        q, C1d = _compute_1d_psd_multitaper(h_top, dx, sinc2_correct=sinc2_correct)
    elif window == 'welch':
        q, C1d = _compute_1d_psd_welch(h_top, dx, sinc2_correct=sinc2_correct)
    else:
        q, C1d = _compute_1d_psd_single_window(h_top, dx, window=window,
                                                 sinc2_correct=sinc2_correct)

    # Apply 1/phi correction (Persson top-profile normalization)
    C_top = C1d / phi

    return q, C_top, phi


def convert_1d_to_2d_isotropic_psd(
    q: np.ndarray,
    C1d: np.ndarray,
    method: str = 'sqrt',
    hurst: float = 0.80,
    correction_factor: Optional[float] = None
) -> np.ndarray:
    """
    Convert 1D PSD to 2D isotropic PSD.

    Methods (from PSD_SPECIFICATION.md):
      sqrt (default):  C2D = C1D / (π·q) × √(1 + 3H)
      gamma:           C2D = (C1D / q) × Γ(1+H) / [√π · Γ(H+½)]
      standard:        C2D = C1D / (π·q) × corr   (user-supplied corr)
      none:            C2D = C1D  (no conversion)

    Parameters
    ----------
    q : np.ndarray
        Wavenumber array (1/m)
    C1d : np.ndarray
        1D PSD (m³)
    method : str
        Conversion method ('sqrt', 'gamma', 'standard', 'none')
    hurst : float
        Hurst exponent H (default 0.80)
    correction_factor : float, optional
        User-supplied correction for 'standard' method

    Returns
    -------
    C2d : np.ndarray
        2D isotropic PSD (m⁴)
    """
    if method == 'none':
        return C1d.copy()

    with np.errstate(divide='ignore', invalid='ignore'):
        if method == 'gamma':
            corr = gamma_func(1.0 + hurst) / (np.sqrt(np.pi) * gamma_func(hurst + 0.5))
            C2d = (C1d / q) * corr
        elif method == 'standard':
            corr = correction_factor if correction_factor is not None else 1.0
            C2d = C1d / (np.pi * q) * corr
        else:  # 'sqrt' (default)
            corr = np.sqrt(1.0 + 3.0 * hurst)
            C2d = C1d / (np.pi * q) * corr

        C2d = np.where(np.isfinite(C2d), C2d, 0.0)

    return C2d


def self_affine_psd_model(q: np.ndarray, C0: float, q0: float, H: float) -> np.ndarray:
    """
    Self-affine fractal PSD model.

    C(q) = C0 * (q/q0)^(-2(1+H))  for q >= q0
    C(q) = C0                      for q < q0

    Parameters
    ----------
    q : np.ndarray
        Wavenumber array
    C0 : float
        PSD value at rolloff wavenumber q0
    q0 : float
        Rolloff wavenumber
    H : float
        Hurst exponent (0 < H < 1)

    Returns
    -------
    C : np.ndarray
        Model PSD values
    """
    C = np.zeros_like(q)

    # Plateau region (q < q0)
    below_q0 = q < q0
    C[below_q0] = C0

    # Power-law region (q >= q0)
    above_q0 = q >= q0
    C[above_q0] = C0 * (q[above_q0] / q0) ** (-2 * (1 + H))

    return C


def fit_self_affine_model(
    q: np.ndarray,
    C: np.ndarray,
    q_min: Optional[float] = None,
    q_max: Optional[float] = None,
    fit_mode: str = 'slope_only'
) -> Dict[str, Any]:
    """
    Fit PSD data to self-affine fractal model.

    Parameters
    ----------
    q : np.ndarray
        Wavenumber array (1/m)
    C : np.ndarray
        PSD values (m⁴)
    q_min : float, optional
        Minimum q for fitting range
    q_max : float, optional
        Maximum q for fitting range
    fit_mode : str
        'slope_only': Fit only the slope in log-log space to get H
        'full': Fit C0, q0, and H simultaneously

    Returns
    -------
    result : dict
        Fitting results containing:
        - H: Hurst exponent
        - C0: PSD at rolloff (if fitted)
        - q0: Rolloff wavenumber (if fitted)
        - slope: Slope in log-log space
        - r_squared: R² value of fit
        - q_fit: q values used for fitting
        - C_fit: Fitted C values
    """
    # Filter valid data
    valid = (q > 0) & (C > 0) & np.isfinite(q) & np.isfinite(C)

    if q_min is not None:
        valid &= (q >= q_min)
    if q_max is not None:
        valid &= (q <= q_max)

    q_valid = q[valid]
    C_valid = C[valid]

    if len(q_valid) < 3:
        raise ValueError("Not enough valid data points for fitting")

    # Log-log transformation
    log_q = np.log10(q_valid)
    log_C = np.log10(C_valid)

    if fit_mode == 'slope_only':
        # Linear fit in log-log space
        # log(C) = log(C0) + slope * log(q/q0)
        # slope = -2(1+H)
        coeffs = np.polyfit(log_q, log_C, 1)
        slope = coeffs[0]
        intercept = coeffs[1]

        # Calculate H from slope
        H = -slope / 2 - 1

        # R² calculation
        log_C_pred = np.polyval(coeffs, log_q)
        ss_res = np.sum((log_C - log_C_pred)**2)
        ss_tot = np.sum((log_C - np.mean(log_C))**2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Estimate C0 and q0 from fit
        # At q=q0: log(C0) = intercept + slope * log(q0)
        # Choose q0 as the geometric mean of the fit range
        q0_est = 10 ** np.mean(log_q)
        C0_est = 10 ** (intercept + slope * np.log10(q0_est))

        result = {
            'H': H,
            'slope': slope,
            'intercept': intercept,
            'C0': C0_est,
            'q0': q0_est,
            'r_squared': r_squared,
            'q_fit': q_valid,
            'C_fit': 10**log_C_pred
        }

    elif fit_mode == 'full':
        # Full nonlinear fit
        def fit_func(log_q, log_C0, log_q0, H):
            q = 10**log_q
            q0 = 10**log_q0
            C0 = 10**log_C0
            C = self_affine_psd_model(q, C0, q0, H)
            return np.log10(np.maximum(C, 1e-30))

        # Initial guesses
        slope_init, intercept_init = np.polyfit(log_q, log_C, 1)
        H_init = max(0.1, min(0.9, -slope_init / 2 - 1))
        q0_init = q_valid[0]
        C0_init = C_valid[0]

        try:
            popt, pcov = curve_fit(
                fit_func,
                log_q, log_C,
                p0=[np.log10(C0_init), np.log10(q0_init), H_init],
                bounds=([-30, np.log10(q_valid.min()), 0.01],
                        [-5, np.log10(q_valid.max()), 0.99]),
                maxfev=10000
            )

            log_C0, log_q0, H = popt
            C0 = 10**log_C0
            q0 = 10**log_q0

            # R² calculation
            log_C_pred = fit_func(log_q, *popt)
            ss_res = np.sum((log_C - log_C_pred)**2)
            ss_tot = np.sum((log_C - np.mean(log_C))**2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            result = {
                'H': H,
                'C0': C0,
                'q0': q0,
                'slope': -2 * (1 + H),
                'r_squared': r_squared,
                'q_fit': q_valid,
                'C_fit': 10**log_C_pred
            }

        except Exception as e:
            # Fall back to slope-only fit
            return fit_self_affine_model(q, C, q_min, q_max, 'slope_only')

    else:
        raise ValueError(f"Unknown fit_mode: {fit_mode}")

    return result


def find_linear_region(
    q: np.ndarray,
    C: np.ndarray,
    min_points: int = 10,
    r_squared_threshold: float = 0.95
) -> Tuple[float, float]:
    """
    Automatically find the linear region in log-log PSD plot.

    Uses a fast algorithm based on local slope analysis instead of
    exhaustive search.

    Parameters
    ----------
    q : np.ndarray
        Wavenumber array
    C : np.ndarray
        PSD values
    min_points : int
        Minimum number of points for fitting
    r_squared_threshold : float
        Minimum R² to consider as linear region

    Returns
    -------
    q_min, q_max : float
        Bounds of linear region
    """
    valid = (q > 0) & (C > 0) & np.isfinite(q) & np.isfinite(C)
    q_valid = q[valid]
    C_valid = C[valid]

    n = len(q_valid)
    if n < min_points:
        return q_valid[0], q_valid[-1]

    log_q = np.log10(q_valid)
    log_C = np.log10(C_valid)

    # Downsample if too many points (for speed)
    max_points = 200
    if n > max_points:
        step = n // max_points
        indices = np.arange(0, n, step)
        log_q_ds = log_q[indices]
        log_C_ds = log_C[indices]
    else:
        indices = np.arange(n)
        log_q_ds = log_q
        log_C_ds = log_C

    n_ds = len(log_q_ds)

    # Calculate local slopes using rolling window
    window = min(15, n_ds // 4)
    if window < 5:
        window = 5

    slopes = np.zeros(n_ds)
    for i in range(n_ds):
        start = max(0, i - window // 2)
        end = min(n_ds, i + window // 2 + 1)
        if end - start >= 3:
            coeffs = np.polyfit(log_q_ds[start:end], log_C_ds[start:end], 1)
            slopes[i] = coeffs[0]

    # Find region with most consistent slope (lowest variance)
    # Use rolling variance of slope
    var_window = min(20, n_ds // 3)
    if var_window < 5:
        var_window = 5

    slope_variance = np.zeros(n_ds)
    for i in range(n_ds):
        start = max(0, i - var_window // 2)
        end = min(n_ds, i + var_window // 2 + 1)
        if end - start >= 3:
            slope_variance[i] = np.var(slopes[start:end])
        else:
            slope_variance[i] = np.inf

    # Find contiguous region with low slope variance
    # Threshold: variance < 10% of median slope squared
    median_slope = np.median(slopes[slopes != 0])
    var_threshold = 0.1 * median_slope**2

    # Find longest contiguous region below threshold
    below_threshold = slope_variance < var_threshold

    # If no region found, relax threshold
    if not np.any(below_threshold):
        var_threshold = np.percentile(slope_variance[slope_variance > 0], 30)
        below_threshold = slope_variance < var_threshold

    # Find longest contiguous segment
    best_start, best_end = 0, n_ds - 1
    best_length = 0

    start_idx = None
    for i in range(n_ds):
        if below_threshold[i]:
            if start_idx is None:
                start_idx = i
        else:
            if start_idx is not None:
                length = i - start_idx
                if length > best_length:
                    best_length = length
                    best_start = start_idx
                    best_end = i - 1
                start_idx = None

    # Check last segment
    if start_idx is not None:
        length = n_ds - start_idx
        if length > best_length:
            best_length = length
            best_start = start_idx
            best_end = n_ds - 1

    # Map back to original indices
    orig_start = indices[best_start]
    orig_end = indices[best_end]

    # Ensure minimum points
    if orig_end - orig_start < min_points:
        # Fall back to middle 50% of data
        orig_start = n // 4
        orig_end = 3 * n // 4

    return q_valid[orig_start], q_valid[orig_end]


def calculate_surface_parameters(
    q: np.ndarray,
    C: np.ndarray,
    q_min: Optional[float] = None,
    q_max: Optional[float] = None
) -> Dict[str, float]:
    """
    Calculate surface roughness parameters from 2D isotropic PSD.

    Parameters
    ----------
    q : np.ndarray
        Wavenumber array (1/m)
    C : np.ndarray
        2D isotropic PSD (m⁴)
    q_min, q_max : float, optional
        Integration limits

    Returns
    -------
    params : dict
        - h_rms: RMS height (m)
        - h_rms_slope: RMS slope (dimensionless)
        - h_rms_curvature: RMS curvature (1/m)
    """
    # Filter by q range
    mask = np.ones(len(q), dtype=bool)
    if q_min is not None:
        mask &= (q >= q_min)
    if q_max is not None:
        mask &= (q <= q_max)

    q_filt = q[mask]
    C_filt = C[mask]

    valid = (q_filt > 0) & (C_filt > 0) & np.isfinite(C_filt)
    q_filt = q_filt[valid]
    C_filt = C_filt[valid]

    if len(q_filt) < 2:
        return {'h_rms': 0, 'h_rms_slope': 0, 'h_rms_curvature': 0}

    # RMS height: h_rms² = 2π ∫ q C(q) dq
    h_rms_sq = 2 * np.pi * np.trapezoid(q_filt * C_filt, q_filt)
    h_rms = np.sqrt(max(h_rms_sq, 0))

    # RMS slope (h'rms): (h'rms)² = 2π ∫ q³ C(q) dq
    slope_sq = 2 * np.pi * np.trapezoid(q_filt**3 * C_filt, q_filt)
    h_rms_slope = np.sqrt(max(slope_sq, 0))

    # RMS curvature: (h''rms)² = 2π ∫ q⁵ C(q) dq
    curv_sq = 2 * np.pi * np.trapezoid(q_filt**5 * C_filt, q_filt)
    h_rms_curvature = np.sqrt(max(curv_sq, 0))

    return {
        'h_rms': h_rms,
        'h_rms_slope': h_rms_slope,
        'h_rms_curvature': h_rms_curvature
    }


class ProfilePSDAnalyzer:
    """
    Class to analyze surface profile and calculate PSD.

    Provides methods for:
    - Loading profile data
    - Calculating Full and Top PSD
    - Fitting to self-affine model
    - Extracting surface parameters
    """

    def __init__(self):
        self.x = None
        self.h = None
        # Raw (unbinned) PSD data
        self.q_raw = None
        self.C_full_1d_raw = None
        self.C_top_1d_raw = None
        self.C_full_2d_raw = None
        self.C_top_2d_raw = None
        # Binned (smoothed) PSD data - used for display and fitting
        self.q = None
        self.C_full_1d = None
        self.C_top_1d = None
        self.C_full_2d = None
        self.C_top_2d = None
        self.phi = None
        self.fit_result_full = None
        self.fit_result_top = None
        self.surface_params = None
        self.points_per_decade = 20  # Default binning resolution

    def load_data(
        self,
        filepath: str,
        x_col: int = 0,
        h_col: int = 1,
        delimiter: Optional[str] = None,
        skip_header: int = 0,
        x_unit: str = 'm',
        h_unit: str = 'm'
    ):
        """Load profile data from file."""
        self.x, self.h = load_profile_data(
            filepath, x_col, h_col, delimiter, skip_header, x_unit, h_unit
        )
        return self

    def set_data(self, x: np.ndarray, h: np.ndarray):
        """Set profile data directly."""
        self.x = np.asarray(x)
        self.h = np.asarray(h)
        return self

    def calculate_psd(
        self,
        window: str = 'multitaper',
        detrend_method: str = 'mean',
        calculate_top: bool = True,
        apply_binning: bool = True,
        n_bins: int = 88,
        sinc2_correct: bool = True,
        conversion_method: str = 'sqrt',
        hurst: float = 0.80,
        correction_factor: Optional[float] = None,
        # Legacy alias kept for backward compatibility
        points_per_decade: Optional[int] = None,
    ):
        """
        Calculate Full and optionally Top PSD.

        Parameters
        ----------
        window : str
            'multitaper' (default, DPSS NW=4 K=7), 'welch', 'hann',
            'hamming', 'blackman', 'none'
        detrend_method : str
            'mean' (default), 'linear', 'quadratic', 'none'
        calculate_top : bool
            Whether to calculate Top PSD
        apply_binning : bool
            Apply Persson-style log binning (default: True)
        n_bins : int
            Target number of log bins (default: 88, Persson convention)
        sinc2_correct : bool
            Apply sinc² rolloff compensation near Nyquist (default: True)
        conversion_method : str
            1D→2D: 'sqrt' (default), 'gamma', 'standard', 'none'
        hurst : float
            Hurst exponent H for conversion (default: 0.80)
        correction_factor : float, optional
            User correction for 'standard' method
        points_per_decade : int, optional
            Legacy alias — if given, overrides n_bins via n_bins ≈ decades × ppd
        """
        if self.x is None or self.h is None:
            raise ValueError("No data loaded. Call load_data() or set_data() first.")

        # Store scan length for reference
        n = len(self.h)
        dx = np.abs(self.x[1] - self.x[0])
        self.scan_length = n * dx
        self.n_points = n
        self.sample_spacing = dx

        # Legacy: convert points_per_decade → n_bins
        if points_per_decade is not None:
            q_min_est = 2 * np.pi / self.scan_length
            q_max_est = np.pi / dx
            n_decades = np.log10(q_max_est / q_min_est)
            n_bins = max(10, int(n_decades * points_per_decade))

        self.n_bins = n_bins
        self.points_per_decade = points_per_decade or n_bins  # keep attr for compat
        self.conversion_method = conversion_method
        self.hurst = hurst
        self.sinc2_correct = sinc2_correct

        # ── 1. Calculate raw (unbinned) Full 1D PSD ──
        self.q_raw, self.C_full_1d_raw = calculate_1d_psd(
            self.x, self.h, window, detrend_method, sinc2_correct=sinc2_correct
        )
        # ── 2. 1D → 2D conversion (raw) ──
        self.C_full_2d_raw = convert_1d_to_2d_isotropic_psd(
            self.q_raw, self.C_full_1d_raw,
            method=conversion_method, hurst=hurst,
            correction_factor=correction_factor
        )

        # ── 3. Top PSD (raw) ──
        if calculate_top:
            _, self.C_top_1d_raw, self.phi = calculate_top_psd(
                self.x, self.h, window, detrend_method,
                sinc2_correct=sinc2_correct
            )
            self.C_top_2d_raw = convert_1d_to_2d_isotropic_psd(
                self.q_raw, self.C_top_1d_raw,
                method=conversion_method, hurst=hurst,
                correction_factor=correction_factor
            )

        # ── 4. Logarithmic binning ──
        if apply_binning:
            self.q, self.C_full_1d = logarithmic_binning(
                self.q_raw, self.C_full_1d_raw, n_bins
            )
            self.C_full_2d = convert_1d_to_2d_isotropic_psd(
                self.q, self.C_full_1d,
                method=conversion_method, hurst=hurst,
                correction_factor=correction_factor
            )
            if calculate_top and self.C_top_1d_raw is not None:
                _, self.C_top_1d = logarithmic_binning(
                    self.q_raw, self.C_top_1d_raw, n_bins
                )
                self.C_top_2d = convert_1d_to_2d_isotropic_psd(
                    self.q, self.C_top_1d,
                    method=conversion_method, hurst=hurst,
                    correction_factor=correction_factor
                )
        else:
            self.q = self.q_raw
            self.C_full_1d = self.C_full_1d_raw
            self.C_full_2d = self.C_full_2d_raw
            if calculate_top:
                self.C_top_1d = self.C_top_1d_raw
                self.C_top_2d = self.C_top_2d_raw

        return self

    def fit_model(
        self,
        q_min: Optional[float] = None,
        q_max: Optional[float] = None,
        fit_mode: str = 'slope_only',
        auto_range: bool = False,
        use_top_psd: bool = False
    ):
        """
        Fit PSD to self-affine model.

        Parameters
        ----------
        q_min, q_max : float, optional
            Fitting range
        fit_mode : str
            'slope_only' or 'full'
        auto_range : bool
            Automatically find linear region
        use_top_psd : bool
            Use Top PSD for fitting (default: Full PSD)
        """
        if self.q is None:
            raise ValueError("No PSD calculated. Call calculate_psd() first.")

        C = self.C_top_2d if use_top_psd else self.C_full_2d

        if C is None:
            raise ValueError("Selected PSD not available.")

        # Auto-find linear region if requested
        if auto_range:
            q_min, q_max = find_linear_region(self.q, C)

        result = fit_self_affine_model(self.q, C, q_min, q_max, fit_mode)

        if use_top_psd:
            self.fit_result_top = result
        else:
            self.fit_result_full = result

        return result

    def get_surface_parameters(
        self,
        q_min: Optional[float] = None,
        q_max: Optional[float] = None,
        use_top_psd: bool = False
    ) -> Dict[str, float]:
        """Get surface roughness parameters."""
        C = self.C_top_2d if use_top_psd else self.C_full_2d

        if C is None:
            raise ValueError("No PSD available.")

        self.surface_params = calculate_surface_parameters(self.q, C, q_min, q_max)
        return self.surface_params

    def get_psd_for_persson(self, use_top_psd: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get PSD data formatted for Persson friction calculation.

        Returns
        -------
        q : np.ndarray
            Wavenumber array (1/m)
        C : np.ndarray
            2D isotropic PSD (m⁴)
        """
        C = self.C_top_2d if use_top_psd else self.C_full_2d

        if C is None:
            raise ValueError("No PSD available.")

        return self.q.copy(), C.copy()
