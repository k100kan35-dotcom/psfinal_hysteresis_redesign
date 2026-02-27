"""
Data Loading Utilities
=======================

Functions to load measured PSD, DMA, and strain sweep data from files.
Supports txt, csv, and excel formats.
"""

import numpy as np
from scipy import signal
from scipy import interpolate
from typing import Tuple, Optional, Dict, List, Callable, Union
import os
import re


def load_psd_from_text(
    data_text: str,
    skip_header: int = 0,
    is_log_data: Optional[bool] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load PSD data from text string.

    Parameters
    ----------
    data_text : str
        Text data with format: q(1/m)  C(q)(m^4)
        Can be tab or space separated
    skip_header : int, optional
        Number of header lines to skip
    is_log_data : bool or None, optional
        If True, input data is in log10 format (log10(q), log10(C))
        If False, input data is in linear format
        If None (default), auto-detect based on data values

    Returns
    -------
    q : np.ndarray
        Wavenumber values (1/m)
    C_q : np.ndarray
        PSD values (m^4)
    """
    lines = data_text.strip().split('\n')[skip_header:]

    q_list = []
    C_list = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        # Split by tab or multiple spaces
        parts = line.split()
        if len(parts) >= 2:
            try:
                q_val = float(parts[0])
                C_val = float(parts[1])
                q_list.append(q_val)
                C_list.append(C_val)
            except ValueError:
                continue

    q = np.array(q_list)
    C_q = np.array(C_list)

    # Auto-detect log-log format if not specified
    if is_log_data is None:
        # Heuristic: if C values are negative or q values are small (< 10),
        # it's likely log10 data
        # log10(q) for typical PSD: 2~8 (q = 100 ~ 1e8)
        # log10(C) for typical PSD: -20 ~ -5 (C = 1e-20 ~ 1e-5)
        if len(C_q) > 0:
            # Check if C values are mostly negative (log10 of small positive values)
            if np.mean(C_q) < 0 and np.max(q) < 15:
                is_log_data = True
            else:
                is_log_data = False

    # Convert from log10 to linear if input is log-log format
    if is_log_data:
        q = 10**q
        C_q = 10**C_q

    # Sort by q
    sort_idx = np.argsort(q)
    q = q[sort_idx]
    C_q = C_q[sort_idx]

    return q, C_q


def load_psd_from_file(
    filename: str,
    skip_header: int = 0,
    is_log_data: Optional[bool] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load PSD data from file.

    File format (tab or space separated):
    q(1/m)  C(q)(m^4)
    100     1.23e-12
    200     4.56e-13
    ...

    Or if is_log_data=True (log10 format):
    log10(q)  log10(C)
    2         -12
    3         -13
    ...

    Parameters
    ----------
    filename : str
        Path to PSD data file
    skip_header : int, optional
        Number of header lines to skip
    is_log_data : bool or None, optional
        If True, input data is in log10 format
        If False, input data is in linear format
        If None (default), auto-detect based on data values

    Returns
    -------
    q : np.ndarray
        Wavenumber values (1/m)
    C_q : np.ndarray
        PSD values (m^4)
    """
    with open(filename, 'r') as f:
        data_text = f.read()

    return load_psd_from_text(data_text, skip_header, is_log_data)


def load_dma_from_text(
    data_text: str,
    skip_header: int = 0,
    freq_unit: str = 'Hz',
    modulus_unit: str = 'MPa'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load DMA (Dynamic Mechanical Analysis) data from text string.

    Parameters
    ----------
    data_text : str
        Text data with format: frequency  E'  E''
        Can be tab or space separated
    skip_header : int, optional
        Number of header lines to skip
    freq_unit : str, optional
        Frequency unit: 'Hz' or 'rad/s' (default: 'Hz')
    modulus_unit : str, optional
        Modulus unit: 'Pa', 'MPa', or 'GPa' (default: 'MPa')

    Returns
    -------
    omega : np.ndarray
        Angular frequency (rad/s)
    E_storage : np.ndarray
        Storage modulus E' (Pa)
    E_loss : np.ndarray
        Loss modulus E'' (Pa)
    """
    lines = data_text.strip().split('\n')[skip_header:]

    freq_list = []
    E_storage_list = []
    E_loss_list = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        parts = line.split()
        if len(parts) >= 3:
            try:
                freq = float(parts[0])
                E_prime = float(parts[1])
                E_double_prime = float(parts[2])

                freq_list.append(freq)
                E_storage_list.append(E_prime)
                E_loss_list.append(E_double_prime)
            except ValueError:
                continue

    freq = np.array(freq_list)
    E_storage = np.array(E_storage_list)
    E_loss = np.array(E_loss_list)

    # Convert frequency to rad/s
    if freq_unit.lower() == 'hz':
        omega = 2 * np.pi * freq
    else:
        omega = freq

    # Convert modulus to Pa
    if modulus_unit.lower() == 'mpa':
        E_storage *= 1e6
        E_loss *= 1e6
    elif modulus_unit.lower() == 'gpa':
        E_storage *= 1e9
        E_loss *= 1e9
    # else: already in Pa

    # Sort by frequency
    sort_idx = np.argsort(omega)
    omega = omega[sort_idx]
    E_storage = E_storage[sort_idx]
    E_loss = E_loss[sort_idx]

    return omega, E_storage, E_loss


def load_dma_from_file(
    filename: str,
    skip_header: int = 0,
    freq_unit: str = 'Hz',
    modulus_unit: str = 'MPa'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load DMA data from file.

    File format (tab or space separated):
    frequency(Hz)  E'(MPa)  E''(MPa)
    0.01          10.5     1.2
    0.1           15.3     2.5
    ...

    Parameters
    ----------
    filename : str
        Path to DMA data file
    skip_header : int, optional
        Number of header lines to skip
    freq_unit : str, optional
        Frequency unit: 'Hz' or 'rad/s'
    modulus_unit : str, optional
        Modulus unit: 'Pa', 'MPa', or 'GPa'

    Returns
    -------
    omega : np.ndarray
        Angular frequency (rad/s)
    E_storage : np.ndarray
        Storage modulus E' (Pa)
    E_loss : np.ndarray
        Loss modulus E'' (Pa)
    """
    with open(filename, 'r') as f:
        data_text = f.read()

    return load_dma_from_text(data_text, skip_header, freq_unit, modulus_unit)


def create_material_from_dma(
    omega: np.ndarray,
    E_storage: np.ndarray,
    E_loss: np.ndarray,
    material_name: str = "Measured Material",
    reference_temp: float = 20.0
):
    """
    Create ViscoelasticMaterial from DMA data.

    Parameters
    ----------
    omega : np.ndarray
        Angular frequency (rad/s)
    E_storage : np.ndarray
        Storage modulus E' (Pa)
    E_loss : np.ndarray
        Loss modulus E'' (Pa)
    material_name : str, optional
        Name of the material
    reference_temp : float, optional
        Reference temperature (°C)

    Returns
    -------
    ViscoelasticMaterial
        Material object with loaded master curve
    """
    from ..core.viscoelastic import ViscoelasticMaterial

    material = ViscoelasticMaterial(
        frequencies=omega,
        storage_modulus=E_storage,
        loss_modulus=E_loss,
        reference_temp=reference_temp,
        name=material_name
    )

    return material


def create_psd_from_data(
    q: np.ndarray,
    C_q: np.ndarray,
    interpolation_kind: str = 'log-log'
):
    """
    Create PSD model from measured data.

    Parameters
    ----------
    q : np.ndarray
        Wavenumber values (1/m)
    C_q : np.ndarray
        PSD values (m^4)
    interpolation_kind : str, optional
        Interpolation method (default: 'log-log')

    Returns
    -------
    MeasuredPSD
        PSD model object
    """
    from ..core.psd_models import MeasuredPSD

    psd = MeasuredPSD(
        q_data=q,
        C_data=C_q,
        interpolation_kind=interpolation_kind,
        extrapolate=False,
        fill_value=0.0
    )

    return psd


def smooth_dma_data(
    omega: np.ndarray,
    E_storage: np.ndarray,
    E_loss: np.ndarray,
    window_length: int = None,
    polyorder: int = 2,
    remove_outliers: bool = True
) -> Dict[str, np.ndarray]:
    """
    Smooth DMA data using Savitzky-Golay filter in log space.

    Applies smoothing to reduce measurement noise while preserving
    the overall trend of viscoelastic master curves.

    Parameters
    ----------
    omega : np.ndarray
        Angular frequency (rad/s)
    E_storage : np.ndarray
        Storage modulus E' (Pa)
    E_loss : np.ndarray
        Loss modulus E'' (Pa)
    window_length : int, optional
        Window length for Savitzky-Golay filter.
        If None, automatically determined based on data length.
        Must be odd and >= polyorder + 2.
    polyorder : int, optional
        Polynomial order for Savitzky-Golay filter (default: 2)
    remove_outliers : bool, optional
        If True, remove outliers before smoothing (default: True)

    Returns
    -------
    dict
        Dictionary containing:
        - 'omega': angular frequency (same as input)
        - 'E_storage_smooth': smoothed storage modulus
        - 'E_loss_smooth': smoothed loss modulus
        - 'omega_raw': original omega
        - 'E_storage_raw': original E_storage
        - 'E_loss_raw': original E_loss
    """
    n = len(omega)

    # Remove outliers if requested
    if remove_outliers and n > 10:
        # Calculate median absolute deviation in log space
        log_E_storage = np.log10(np.maximum(E_storage, 1e3))
        log_E_loss = np.log10(np.maximum(E_loss, 1.0))

        # Compute rolling median
        from scipy.ndimage import median_filter
        window_size = min(7, n if n % 2 == 1 else n - 1)
        median_E_storage = median_filter(log_E_storage, size=window_size, mode='nearest')
        median_E_loss = median_filter(log_E_loss, size=window_size, mode='nearest')

        # Calculate deviation
        dev_E_storage = np.abs(log_E_storage - median_E_storage)
        dev_E_loss = np.abs(log_E_loss - median_E_loss)

        # Threshold: 3 times median absolute deviation
        mad_E_storage = np.median(dev_E_storage)
        mad_E_loss = np.median(dev_E_loss)
        threshold_storage = 3.0 * mad_E_storage
        threshold_loss = 3.0 * mad_E_loss

        # Mark outliers
        outlier_mask = (dev_E_storage > threshold_storage) | (dev_E_loss > threshold_loss)

        # Replace outliers with median values
        E_storage_clean = E_storage.copy()
        E_loss_clean = E_loss.copy()
        E_storage_clean[outlier_mask] = 10**median_E_storage[outlier_mask]
        E_loss_clean[outlier_mask] = 10**median_E_loss[outlier_mask]
    else:
        E_storage_clean = E_storage
        E_loss_clean = E_loss

    # Determine window length if not specified
    if window_length is None:
        # Use about 30-40% of data points for stronger smoothing
        window_length = max(11, min(71, int(n * 0.35)))
        if window_length % 2 == 0:
            window_length += 1

    # Ensure window length is valid
    window_length = min(window_length, n)
    if window_length % 2 == 0:
        window_length -= 1
    window_length = max(polyorder + 2, window_length)

    # If not enough points, skip smoothing
    if n < polyorder + 2:
        return {
            'omega': omega,
            'E_storage_smooth': E_storage,
            'E_loss_smooth': E_loss,
            'omega_raw': omega,
            'E_storage_raw': E_storage,
            'E_loss_raw': E_loss
        }

    # Smooth in log space for better results
    log_omega = np.log10(omega)
    log_E_storage = np.log10(np.maximum(E_storage_clean, 1e3))
    log_E_loss = np.log10(np.maximum(E_loss_clean, 1.0))

    # Preserve low-frequency slope by fitting initial trend
    # Use first 20% of data or at least 5 points for slope estimation
    n_low_freq = max(5, int(n * 0.2))
    n_low_freq = min(n_low_freq, n // 2)  # Don't use more than half

    # Fit linear trend in log-log space for low frequencies
    if n_low_freq >= 3:
        # Storage modulus low-freq slope
        p_storage = np.polyfit(log_omega[:n_low_freq], log_E_storage[:n_low_freq], 1)
        # Loss modulus low-freq slope
        p_loss = np.polyfit(log_omega[:n_low_freq], log_E_loss[:n_low_freq], 1)
    else:
        p_storage = [0, log_E_storage[0]]
        p_loss = [0, log_E_loss[0]]

    # Apply Savitzky-Golay filter with 'interp' mode for better edge handling
    log_E_storage_smooth = signal.savgol_filter(
        log_E_storage, window_length, polyorder, mode='interp'
    )
    log_E_loss_smooth = signal.savgol_filter(
        log_E_loss, window_length, polyorder, mode='interp'
    )

    # Apply second pass with smaller window for refinement
    window_length_2 = max(7, window_length // 2)
    if window_length_2 % 2 == 0:
        window_length_2 += 1
    if window_length_2 >= polyorder + 2:
        log_E_storage_smooth = signal.savgol_filter(
            log_E_storage_smooth, window_length_2, polyorder, mode='interp'
        )
        log_E_loss_smooth = signal.savgol_filter(
            log_E_loss_smooth, window_length_2, polyorder, mode='interp'
        )

    # Apply third pass for extra smoothness
    window_length_3 = max(5, window_length_2 // 2)
    if window_length_3 % 2 == 0:
        window_length_3 += 1
    if window_length_3 >= polyorder + 2:
        log_E_storage_smooth = signal.savgol_filter(
            log_E_storage_smooth, window_length_3, polyorder, mode='interp'
        )
        log_E_loss_smooth = signal.savgol_filter(
            log_E_loss_smooth, window_length_3, polyorder, mode='interp'
        )

    # Blend with original slope at low frequencies to maintain natural behavior
    # Create smooth transition using sigmoid-like weight
    blend_points = min(n_low_freq, n // 3)
    if blend_points >= 2:
        # Weight decreases from 1 to 0 over blend region
        blend_weights = np.zeros(n)
        blend_region = np.linspace(0, 1, blend_points)
        # Smooth transition using cosine taper
        blend_weights[:blend_points] = 0.5 * (1 + np.cos(blend_region * np.pi))

        # Calculate what the original slope predicts
        log_E_storage_trend = p_storage[0] * log_omega + p_storage[1]
        log_E_loss_trend = p_loss[0] * log_omega + p_loss[1]

        # Blend smoothed data with original trend at low frequencies
        log_E_storage_smooth = (
            blend_weights * log_E_storage_trend +
            (1 - blend_weights) * log_E_storage_smooth
        )
        log_E_loss_smooth = (
            blend_weights * log_E_loss_trend +
            (1 - blend_weights) * log_E_loss_smooth
        )

    # Convert back to linear space
    E_storage_smooth = 10**log_E_storage_smooth
    E_loss_smooth = 10**log_E_loss_smooth

    return {
        'omega': omega,
        'E_storage_smooth': E_storage_smooth,
        'E_loss_smooth': E_loss_smooth,
        'omega_raw': omega,
        'E_storage_raw': E_storage,
        'E_loss_raw': E_loss
    }


def save_example_data(output_dir: str = 'examples/data'):
    """
    Save example measured data files.

    Creates example PSD and DMA data files in the specified directory.

    Parameters
    ----------
    output_dir : str, optional
        Output directory for data files
    """
    os.makedirs(output_dir, exist_ok=True)

    # Example PSD data (rougher surface)
    psd_data = """# Surface PSD Data
# q(1/m)  C(q)(m^4)
2.0e+01  3.0e-09
2.0e+01  2.0e-09
3.0e+01  6.0e-10
5.0e+01  1.0e-10
1.0e+02  5.0e-12
5.0e+02  7.0e-14
1.0e+03  4.0e-15
5.0e+03  6.0e-17
1.0e+04  3.0e-18
5.0e+04  4.0e-20
1.0e+05  2.0e-21
5.0e+05  3.0e-23
1.0e+06  2.0e-24
5.0e+06  2.0e-26
1.0e+07  1.0e-27
5.0e+07  2.0e-29
1.0e+08  1.0e-30
5.0e+08  9.0e-33
1.0e+09  1.0e-33
"""

    psd_file = os.path.join(output_dir, 'example_psd.txt')
    with open(psd_file, 'w') as f:
        f.write(psd_data)

    # Example DMA data
    dma_data = """# DMA Master Curve Data
# Frequency(Hz)  E'(MPa)  E''(MPa)
0.01      6.7     0.7
0.1       7.8     1.0
1.0       8.9     1.2
10        10.0    1.6
100       12.3    2.1
1000      15.0    3.4
10000     20.7    7.0
100000    31.0    13.6
1000000   54.6    31.7
10000000  104     68.5
100000000 239     168
1000000000 613    358
10000000000 1280  606
"""

    dma_file = os.path.join(output_dir, 'example_dma.txt')
    with open(dma_file, 'w') as f:
        f.write(dma_data)

    print(f"Example data files created:")
    print(f"  {psd_file}")
    print(f"  {dma_file}")


# ============================================================================
# Strain Data Loading Functions
# ============================================================================

def load_strain_sweep_file(
    filepath: str,
    file_format: str = 'auto'
) -> Dict[float, List[Tuple[float, float, float, float]]]:
    """
    Load strain sweep data from file (txt, csv, or excel).

    Expected data format: T, f, strain, ReE, ImE
    where:
    - T: Temperature (Celsius)
    - f: Frequency (Hz)
    - strain: Strain amplitude (% or fraction)
    - ReE: Storage modulus E' (real part)
    - ImE: Loss modulus E'' (imaginary part)

    Parameters
    ----------
    filepath : str
        Path to strain sweep data file
    file_format : str, optional
        File format: 'auto', 'txt', 'csv', or 'excel'
        If 'auto', determined from file extension

    Returns
    -------
    data_by_T : dict
        Dictionary mapping temperature to list of (freq, strain, ReE, ImE) tuples
        Sorted by strain within each temperature block
    """
    # Determine file format
    if file_format == 'auto':
        ext = os.path.splitext(filepath)[1].lower()
        if ext in ['.xlsx', '.xls']:
            file_format = 'excel'
        elif ext == '.csv':
            file_format = 'csv'
        else:
            file_format = 'txt'

    if file_format == 'excel':
        return _load_strain_sweep_excel(filepath)
    elif file_format == 'csv':
        return _load_strain_sweep_csv(filepath)
    else:
        return _load_strain_sweep_txt(filepath)


def _load_strain_sweep_txt(filepath: str) -> Dict[float, List[Tuple[float, float, float, float]]]:
    """Load strain sweep data from text file."""
    data_by_T = {}
    float_re = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

    def is_numeric_row(line):
        nums = float_re.findall(line)
        return len(nums) >= 5

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            # Skip marker lines
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

    # Sort each temperature block by strain
    for T in list(data_by_T.keys()):
        rows = data_by_T[T]
        rows.sort(key=lambda x: x[1])
        data_by_T[T] = rows

    return data_by_T


def _load_strain_sweep_csv(filepath: str) -> Dict[float, List[Tuple[float, float, float, float]]]:
    """Load strain sweep data from CSV file."""
    import csv

    data_by_T = {}

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        header_skipped = False

        for row in reader:
            if not row or len(row) < 5:
                continue

            # Skip header row
            if not header_skipped:
                try:
                    float(row[0])
                except ValueError:
                    header_skipped = True
                    continue

            try:
                T = float(row[0])
                freq = float(row[1])
                strain = float(row[2])
                ReE = float(row[3])
                ImE = float(row[4])
                data_by_T.setdefault(T, []).append((freq, strain, ReE, ImE))
            except (ValueError, IndexError):
                continue

    # Sort each temperature block by strain
    for T in list(data_by_T.keys()):
        rows = data_by_T[T]
        rows.sort(key=lambda x: x[1])
        data_by_T[T] = rows

    return data_by_T


def _load_strain_sweep_excel(filepath: str) -> Dict[float, List[Tuple[float, float, float, float]]]:
    """Load strain sweep data from Excel file."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas and openpyxl required for Excel file support. "
                         "Install with: pip install pandas openpyxl")

    df = pd.read_excel(filepath, engine='openpyxl')

    # Assume columns are: T, f, strain, ReE, ImE (first 5 columns)
    data_by_T = {}

    for idx, row in df.iterrows():
        try:
            T = float(row.iloc[0])
            freq = float(row.iloc[1])
            strain = float(row.iloc[2])
            ReE = float(row.iloc[3])
            ImE = float(row.iloc[4])
            data_by_T.setdefault(T, []).append((freq, strain, ReE, ImE))
        except (ValueError, IndexError):
            continue

    # Sort each temperature block by strain
    for T in list(data_by_T.keys()):
        rows = data_by_T[T]
        rows.sort(key=lambda x: x[1])
        data_by_T[T] = rows

    return data_by_T


def load_fg_curve_file(
    filepath: str,
    strain_is_percent: bool = False
) -> Optional[Dict[str, np.ndarray]]:
    """
    Load pre-computed f,g curve data from file.

    File format: strain, f, [g]
    where:
    - strain: Strain amplitude (fraction or %)
    - f: Storage modulus correction factor f(strain)
    - g: Loss modulus correction factor g(strain) [optional]

    Parameters
    ----------
    filepath : str
        Path to f,g curve data file
    strain_is_percent : bool, optional
        If True, strain values are in percent and will be converted to fraction

    Returns
    -------
    dict or None
        Dictionary with keys: 'strain', 'f', 'g' (g may be None)
        Returns None if parsing fails
    """
    float_re = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
    strains, fvals, gvals = [], [], []

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#'):
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
        s = s / 100.0

    f = np.array(fvals, dtype=float)
    g = np.array(gvals, dtype=float) if len(gvals) == len(strains) else None

    # Sort by strain
    idx = np.argsort(s)
    s = s[idx]
    f = f[idx]
    if g is not None:
        g = g[idx]

    return {"strain": s, "f": f, "g": g}


def compute_fg_from_strain_sweep(
    data_by_T: Dict[float, List[Tuple[float, float, float, float]]],
    target_freq: float = 1.0,
    freq_tolerance: float = 0.01,
    freq_mode: str = 'nearest',
    strain_is_percent: bool = True,
    e0_n_points: int = 5,
    clip_leq_1: bool = True
) -> Dict[float, Dict[str, Union[np.ndarray, float]]]:
    """
    Compute f(strain) and g(strain) curves from strain sweep data.

    For each temperature:
    - Extract data at target frequency
    - Normalize by low-strain modulus E0
    - f = ReE / E0_re, g = ImE / E0_im

    Parameters
    ----------
    data_by_T : dict
        Strain sweep data from load_strain_sweep_file
    target_freq : float, optional
        Target frequency (Hz), default 1.0
    freq_tolerance : float, optional
        Frequency tolerance (Hz), default 0.01
    freq_mode : str, optional
        Frequency selection mode: 'nearest' or 'tolerance'
    strain_is_percent : bool, optional
        If True, strain values are in percent (default True)
    e0_n_points : int, optional
        Number of low-strain points to average for E0 (default 5)
    clip_leq_1 : bool, optional
        If True, clip f,g values to <= 1.0 (default True)

    Returns
    -------
    fg_by_T : dict
        Dictionary mapping temperature to dict with:
        - 'strain': strain array (fraction)
        - 'f': f(strain) array
        - 'g': g(strain) array
        - 'E0_re': reference storage modulus
        - 'E0_im': reference loss modulus
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

        # Select data at target frequency
        if freq_mode == 'nearest':
            uniq = np.unique(freqs)
            if uniq.size == 0:
                continue
            fsel = float(uniq[np.argmin(np.abs(uniq - target_freq))])
            mask = np.isclose(freqs, fsel)
        else:
            mask = np.abs(freqs - target_freq) <= freq_tolerance

        if not np.any(mask):
            continue

        s = strains[mask].astype(float)
        reE = ReE[mask].astype(float)
        imE = ImE[mask].astype(float)

        if strain_is_percent:
            s = s / 100.0

        # Filter valid data
        valid = np.isfinite(s) & np.isfinite(reE) & np.isfinite(imE) & (s > 0)
        s, reE, imE = s[valid], reE[valid], imE[valid]

        if s.size < max(3, e0_n_points):
            continue

        # Sort by strain
        idx = np.argsort(s)
        s, reE, imE = s[idx], reE[idx], imE[idx]

        # Calculate E0 from low-strain average
        n0 = int(max(1, min(e0_n_points, s.size)))
        E0_re = float(np.mean(reE[:n0]))
        E0_im = float(np.mean(imE[:n0]))

        # Calculate f and g
        f = reE / E0_re if E0_re != 0 else np.full_like(reE, np.nan)
        g = imE / E0_im if E0_im != 0 else np.full_like(imE, np.nan)

        if clip_leq_1:
            f = np.minimum(f, 1.0)
            # g is NOT clipped: g(ε) can exceed 1.0 (loss modulus overshoot)

        fg_by_T[T] = {
            "strain": s,
            "f": f,
            "g": g,
            "E0_re": E0_re,
            "E0_im": E0_im
        }

    return fg_by_T


def create_fg_interpolator(
    strain: np.ndarray,
    f: np.ndarray,
    g: Optional[np.ndarray] = None,
    interp_kind: str = 'loglog_linear',
    extrap_mode: str = 'hold'
) -> Tuple[Callable[[float], float], Optional[Callable[[float], float]]]:
    """
    Create interpolation functions for f(strain) and g(strain).

    Parameters
    ----------
    strain : np.ndarray
        Strain values (fraction)
    f : np.ndarray
        f(strain) values
    g : np.ndarray, optional
        g(strain) values
    interp_kind : str, optional
        Interpolation kind: 'linear', 'loglog_linear', 'loglog_cubic'
    extrap_mode : str, optional
        Extrapolation mode: 'hold', 'linear', 'none'

    Returns
    -------
    f_interp : callable
        Function f(strain) -> float
    g_interp : callable or None
        Function g(strain) -> float, or None if g not provided
    """
    def _safe_log(x):
        return np.log(np.clip(x, 1e-300, None))

    def _create_interp_func(x, y, kind, extrap):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        # Filter valid data
        mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
        x, y = x[mask], y[mask]

        if len(x) < 2:
            return lambda s: np.nan

        idx = np.argsort(x)
        x, y = x[idx], y[idx]
        xmin, xmax = x[0], x[-1]

        def interp_func(xq):
            xq = np.atleast_1d(np.asarray(xq, dtype=float))
            yq = np.full_like(xq, np.nan, dtype=float)

            inside = (xq >= xmin) & (xq <= xmax)

            if kind == 'linear':
                yq[inside] = np.interp(xq[inside], x, y)
            elif kind == 'loglog_linear':
                if np.any(y <= 0):
                    yq[inside] = np.interp(xq[inside], x, y)
                else:
                    lx = _safe_log(x)
                    ly = _safe_log(y)
                    yq[inside] = np.exp(np.interp(_safe_log(xq[inside]), lx, ly))
            elif kind == 'loglog_cubic':
                if np.any(y <= 0) or len(x) < 4:
                    yq[inside] = np.interp(xq[inside], x, y)
                else:
                    from scipy.interpolate import CubicSpline
                    lx = _safe_log(x)
                    ly = _safe_log(y)
                    cs = CubicSpline(lx, ly, extrapolate=False)
                    lxq = _safe_log(xq[inside])
                    yq[inside] = np.exp(cs(lxq))

            # Handle extrapolation
            left = xq < xmin
            right = xq > xmax

            if extrap == 'hold':
                yq[left] = y[0]
                yq[right] = y[-1]
            elif extrap == 'linear':
                if np.any(left) and len(x) >= 2:
                    slope = (y[1] - y[0]) / (x[1] - x[0])
                    yq[left] = y[0] + slope * (xq[left] - x[0])
                if np.any(right) and len(x) >= 2:
                    slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
                    yq[right] = y[-1] + slope * (xq[right] - x[-1])

            # Safety: replace any remaining NaN with 1.0 (no correction)
            # But preserve NaN when extrap='none' so nanmean can exclude them
            if extrap != 'none':
                yq = np.nan_to_num(yq, nan=1.0)

            return float(yq[0]) if len(yq) == 1 else yq

        return interp_func

    f_interp = _create_interp_func(strain, f, interp_kind, extrap_mode)
    g_interp = _create_interp_func(strain, g, interp_kind, extrap_mode) if g is not None else None

    return f_interp, g_interp


DEFAULT_STRAIN_SPLIT = {
    'threshold': 0.142,
    'f_low':  {0.02: 0.10, 29.9: 0.90},
    'f_high': {29.9: 0.30, 49.99: 0.70},
    'g_low':  {0.02: 0.35, 49.99: 0.65},
    'g_high': {29.9: 0.55, 49.99: 0.45},
}


def _find_nearest_temp(available_temps, target_temp, tol=5.0):
    """Find the nearest available temperature within tolerance."""
    best, best_diff = None, float('inf')
    for t in available_temps:
        diff = abs(t - target_temp)
        if diff < best_diff:
            best, best_diff = t, diff
    return best if best_diff <= tol else None


def _weighted_avg_1d(values_by_T, temp_weights, available_temps):
    """Compute weighted average of values from selected temperatures.

    Handles NaN per-point: if a temperature has NaN at a given strain,
    it is excluded from the average at that point and weights are
    renormalized among the remaining temperatures.

    Parameters
    ----------
    values_by_T : dict
        {temp: array} of interpolated values at grid_strain points
    temp_weights : dict
        {temp: weight} requested temperatures and weights
    available_temps : list
        Temperatures actually available in values_by_T

    Returns
    -------
    result : np.ndarray
        Weighted average array
    """
    matched = []
    for t_req, w in temp_weights.items():
        t_act = _find_nearest_temp(available_temps, t_req)
        if t_act is not None and t_act in values_by_T:
            matched.append((values_by_T[t_act], w))

    if not matched:
        return None

    n = len(matched[0][0])
    result = np.full(n, np.nan)

    for i in range(n):
        total_w = 0.0
        val = 0.0
        for vals, w in matched:
            if np.isfinite(vals[i]):
                val += vals[i] * w
                total_w += w
        if total_w > 0:
            result[i] = val / total_w

    return result


def average_fg_curves(
    fg_by_T: Dict[float, Dict[str, np.ndarray]],
    selected_temps: List[float],
    grid_strain: np.ndarray,
    interp_kind: str = 'loglog_linear',
    avg_mode: str = 'mean',
    n_min: int = 1,
    clip_leq_1: bool = True,
    strain_split: Optional[Dict] = None,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Average f,g curves from multiple temperatures onto a common strain grid.

    Parameters
    ----------
    fg_by_T : dict
        f,g curves by temperature from compute_fg_from_strain_sweep
    selected_temps : list
        List of temperatures to include in averaging
    grid_strain : np.ndarray
        Common strain grid for output
    interp_kind : str, optional
        Interpolation kind: 'linear', 'loglog_linear', 'loglog_cubic'
    avg_mode : str, optional
        Averaging mode: 'mean', 'median', 'max'
    n_min : int, optional
        Minimum number of temperature curves required at each point
    clip_leq_1 : bool, optional
        If True, clip f,g values to <= 1.0
    strain_split : dict or None, optional
        Strain-split weighted averaging configuration. When provided,
        f and g are independently averaged with different temperature
        weights for low/high strain regions. Format::

            {
                'threshold': 0.142,
                'f_low':  {temp: weight, ...},
                'f_high': {temp: weight, ...},
                'g_low':  {temp: weight, ...},
                'g_high': {temp: weight, ...},
            }

        Use DEFAULT_STRAIN_SPLIT for the optimized defaults.
        When None, uses equal-weight mean of selected_temps (legacy).

    Returns
    -------
    result : dict or None
        Dictionary with:
        - 'strain': strain grid
        - 'f_avg': averaged f(strain)
        - 'g_avg': averaged g(strain)
        - 'Ts_used': list of temperatures used
        - 'n_eff': effective number of curves at each point
    """
    # Collect temperatures: only use user-selected temperatures
    selected_set = set(selected_temps)
    if strain_split is not None:
        # strain_split 모드에서도 사용자가 선택한 온도만 사용
        Ts = [T for T in fg_by_T if T in selected_set]
    else:
        Ts = [T for T in selected_temps if T in fg_by_T]

    if len(Ts) == 0:
        return None

    # Interpolate each temperature's f,g onto grid_strain
    F_by_T = {}  # {temp: f_array}
    G_by_T = {}  # {temp: g_array}

    for T in Ts:
        s = fg_by_T[T]['strain']
        f = fg_by_T[T]['f']
        g = fg_by_T[T]['g']

        if len(s) < 2:
            continue

        f_interp, g_interp = create_fg_interpolator(
            s, f, g, interp_kind=interp_kind, extrap_mode='none'
        )

        fq = np.array([f_interp(sv) for sv in grid_strain])
        gq = np.array([g_interp(sv) for sv in grid_strain]) if g_interp else np.full_like(grid_strain, np.nan)

        F_by_T[T] = fq
        G_by_T[T] = gq

    if len(F_by_T) == 0:
        return None

    available_temps = list(F_by_T.keys())

    # Per-temperature forward-fill for strain_split mode:
    # fill NaN at trailing edge with last valid value so that
    # weighted averages use consistent data across temperatures.
    if strain_split is not None:
        for T in available_temps:
            for arr in (F_by_T[T], G_by_T[T]):
                last_valid = np.nan
                for i in range(len(arr)):
                    if np.isfinite(arr[i]):
                        last_valid = arr[i]
                    elif np.isfinite(last_valid):
                        arr[i] = last_valid

    # ── Strain-split weighted averaging ─────────────────────
    if strain_split is not None:
        threshold = strain_split['threshold']
        lo_mask = grid_strain < threshold
        hi_mask = ~lo_mask

        f_avg = np.full(len(grid_strain), np.nan)
        g_avg = np.full(len(grid_strain), np.nan)

        # f: low strain
        if np.any(lo_mask) and 'f_low' in strain_split:
            f_lo = _weighted_avg_1d(
                {T: F_by_T[T][lo_mask] for T in available_temps},
                strain_split['f_low'], available_temps
            )
            if f_lo is not None:
                f_avg[lo_mask] = f_lo

        # f: high strain
        if np.any(hi_mask) and 'f_high' in strain_split:
            f_hi = _weighted_avg_1d(
                {T: F_by_T[T][hi_mask] for T in available_temps},
                strain_split['f_high'], available_temps
            )
            if f_hi is not None:
                f_avg[hi_mask] = f_hi

        # g: low strain
        if np.any(lo_mask) and 'g_low' in strain_split:
            g_lo = _weighted_avg_1d(
                {T: G_by_T[T][lo_mask] for T in available_temps},
                strain_split['g_low'], available_temps
            )
            if g_lo is not None:
                g_avg[lo_mask] = g_lo

        # g: high strain
        if np.any(hi_mask) and 'g_high' in strain_split:
            g_hi = _weighted_avg_1d(
                {T: G_by_T[T][hi_mask] for T in available_temps},
                strain_split['g_high'], available_temps
            )
            if g_hi is not None:
                g_avg[hi_mask] = g_hi

        n_eff = np.ones(len(grid_strain), dtype=int)

    # ── Legacy: equal-weight mean/median/max ────────────────
    else:
        F = np.vstack([F_by_T[T] for T in available_temps])
        G = np.vstack([G_by_T[T] for T in available_temps])

        if np.all(np.isnan(F)) or np.all(np.isnan(G)):
            return None

        with np.errstate(all='ignore'):
            if avg_mode == 'mean':
                f_avg = np.nanmean(F, axis=0)
                g_avg = np.nanmean(G, axis=0)
            elif avg_mode == 'median':
                f_avg = np.nanmedian(F, axis=0)
                g_avg = np.nanmedian(G, axis=0)
            elif avg_mode == 'max':
                f_avg = np.nanmax(F, axis=0)
                g_avg = np.nanmax(G, axis=0)
            else:
                raise ValueError(f"Unknown avg_mode: {avg_mode}")

        n_eff = np.sum(np.isfinite(F), axis=0)

    # Handle NaN values with forward-then-backward fill
    valid_mask = np.isfinite(f_avg) & np.isfinite(g_avg) & (n_eff >= n_min)

    if not np.any(valid_mask):
        f_avg = np.ones_like(f_avg)
        g_avg = np.ones_like(g_avg)
    else:
        last_valid_f, last_valid_g = 1.0, 1.0
        for i in range(len(grid_strain)):
            if valid_mask[i]:
                last_valid_f = float(f_avg[i])
                last_valid_g = float(g_avg[i])
            elif not np.isfinite(f_avg[i]) or not np.isfinite(g_avg[i]):
                f_avg[i] = last_valid_f
                g_avg[i] = last_valid_g

        first_valid_idx = np.argmax(valid_mask)
        if first_valid_idx > 0:
            first_f = float(f_avg[first_valid_idx])
            first_g = float(g_avg[first_valid_idx])
            for i in range(first_valid_idx):
                f_avg[i] = first_f
                g_avg[i] = first_g

    if clip_leq_1:
        f_avg = np.minimum(f_avg, 1.0)
        # g_avg is NOT clipped: g(ε) can exceed 1.0 (loss modulus overshoot)

    return {
        'strain': grid_strain.copy(),
        'f_avg': f_avg,
        'g_avg': g_avg,
        'Ts_used': Ts,
        'n_eff': n_eff
    }


def persson_strain_grid(max_strain_fraction: float, start_strain: float = 1.49e-4, ratio: float = 1.5) -> np.ndarray:
    """
    Generate Persson-style discrete strain grid.

    Starts at start_strain and multiplies by ratio at each step.

    Parameters
    ----------
    max_strain_fraction : float
        Maximum strain (fraction, not %)
    start_strain : float, optional
        Starting strain value (default 1.49e-4)
    ratio : float, optional
        Multiplication ratio between grid points (default 1.5)

    Returns
    -------
    grid : np.ndarray
        Strain grid array
    """
    out = []
    s = start_strain
    while s <= max_strain_fraction * 1.000001:
        out.append(s)
        s *= ratio
    return np.array(out, dtype=float)


def create_strain_grid(
    n_points: int,
    max_strain_fraction: float,
    use_persson_grid: bool = True
) -> np.ndarray:
    """
    Create strain grid for f,g curve calculations.

    Parameters
    ----------
    n_points : int
        Number of grid points (used if not using Persson grid)
    max_strain_fraction : float
        Maximum strain (fraction)
    use_persson_grid : bool, optional
        If True, use Persson-style discrete grid (default True)

    Returns
    -------
    grid : np.ndarray
        Strain grid array
    """
    if use_persson_grid:
        g = persson_strain_grid(max_strain_fraction)
        if g.size >= 2:
            return g

    # Fallback to log-spaced grid
    smin = 1e-4
    smax = max(1.1 * smin, max_strain_fraction)
    return np.logspace(np.log10(smin), np.log10(smax), max(5, n_points))
