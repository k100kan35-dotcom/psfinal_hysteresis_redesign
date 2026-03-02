"""
Viscoelastic Material Properties
=================================

Handles viscoelastic material characterization:
- Master curves (storage and loss moduli)
- WLF time-temperature superposition
- Frequency-dependent complex modulus
"""

import numpy as np
from scipy import interpolate
from typing import Optional, Tuple, Union
import warnings


class ViscoelasticMaterial:
    """
    Viscoelastic material with frequency-dependent properties.

    Stores and interpolates master curve data for E'(ω) and E''(ω).
    """

    def __init__(
        self,
        frequencies: Optional[np.ndarray] = None,
        storage_modulus: Optional[np.ndarray] = None,
        loss_modulus: Optional[np.ndarray] = None,
        reference_temp: float = 20.0,
        name: str = "Material"
    ):
        """
        Initialize viscoelastic material.

        Parameters
        ----------
        frequencies : np.ndarray, optional
            Angular frequencies (rad/s) for master curve
        storage_modulus : np.ndarray, optional
            Storage modulus E' (Pa) at each frequency
        loss_modulus : np.ndarray, optional
            Loss modulus E'' (Pa) at each frequency
        reference_temp : float, optional
            Reference temperature (°C) for master curve (default: 20)
        name : str, optional
            Material name
        """
        self.name = name
        self.reference_temp = reference_temp

        if frequencies is not None and storage_modulus is not None:
            self.set_master_curve(frequencies, storage_modulus, loss_modulus)
        else:
            self._frequencies = None
            self._storage_modulus = None
            self._loss_modulus = None
            self._E_prime_interp = None
            self._E_double_prime_interp = None
            self._E_abs_interp = None

        # WLF parameters (can be set later)
        self.C1 = None
        self.C2 = None
        self.T_ref_wlf = None

    def set_master_curve(
        self,
        frequencies: np.ndarray,
        storage_modulus: np.ndarray,
        loss_modulus: Optional[np.ndarray] = None
    ):
        """
        Set master curve data.

        Parameters
        ----------
        frequencies : np.ndarray
            Angular frequencies (rad/s)
        storage_modulus : np.ndarray
            Storage modulus E' (Pa)
        loss_modulus : np.ndarray, optional
            Loss modulus E'' (Pa). If None, assumed to be zero.
        """
        self._frequencies = np.asarray(frequencies)
        self._storage_modulus = np.asarray(storage_modulus)

        if loss_modulus is not None:
            self._loss_modulus = np.asarray(loss_modulus)
        else:
            self._loss_modulus = np.zeros_like(self._storage_modulus)

        # Sort by frequency
        sort_idx = np.argsort(self._frequencies)
        self._frequencies = self._frequencies[sort_idx]
        self._storage_modulus = self._storage_modulus[sort_idx]
        self._loss_modulus = self._loss_modulus[sort_idx]

        # Filter valid data: positive frequency and finite modulus values
        valid_mask = (
            (self._frequencies > 0) &
            np.isfinite(self._storage_modulus) &
            np.isfinite(self._loss_modulus) &
            (self._storage_modulus > 0)
        )

        if np.sum(valid_mask) < 2:
            raise ValueError("Not enough valid data points for interpolation")

        # Create interpolators in log-log space
        log_freq = np.log10(self._frequencies[valid_mask])
        log_E_prime = np.log10(
            np.maximum(self._storage_modulus[valid_mask], 1e3)
        )
        log_E_double_prime = np.log10(
            np.maximum(self._loss_modulus[valid_mask], 1.0)
        )

        # Ensure no NaN in log values
        log_E_prime = np.nan_to_num(log_E_prime, nan=6.0)
        log_E_double_prime = np.nan_to_num(log_E_double_prime, nan=5.0)

        # Choose interpolation method based on number of data points
        # Linear interpolation is most robust - avoids oscillation artifacts
        # that quadratic/cubic can produce between noisy data points
        n_points = len(log_freq)
        interp_kind = 'linear'

        # Store frequency range for later reference
        self._log_freq_min = log_freq.min()
        self._log_freq_max = log_freq.max()

        # E': 저주파는 rubbery plateau (hold), 고주파는 glassy plateau (hold)
        # 주의: extrapolate 사용 시 WLF 시프트된 재료에서 저주파 외삽이
        # rubbery plateau 아래로 비물리적 값을 생성하여 G(q), P(q)를 왜곡시킴
        # → flash temperature 반복 시 μ_hot이 μ_cold보다 커지는 양의 피드백 발생!
        # 따라서 boundary clamping 사용 (물리적으로 올바른 plateau 값 유지)
        self._E_prime_interp = interpolate.interp1d(
            log_freq,
            log_E_prime,
            kind=interp_kind,
            fill_value=(log_E_prime[0], log_E_prime[-1]),
            bounds_error=False
        )

        # E'': 저주파는 hold (외삽 시 비물리적 증가 방지, rubbery에서 E''→0)
        # 고주파는 hold (glassy plateau)
        self._E_double_prime_interp = interpolate.interp1d(
            log_freq,
            log_E_double_prime,
            kind=interp_kind,
            fill_value=(log_E_double_prime[0], log_E_double_prime[-1]),
            bounds_error=False
        )

        # Complex modulus magnitude |E*| = √(E'² + E''²)
        # 저주파/고주파 모두 plateau → boundary clamping이 물리적으로 올바름
        E_abs = np.sqrt(self._storage_modulus**2 + self._loss_modulus**2)
        log_E_abs = np.log10(np.maximum(E_abs[self._frequencies > 0], 1e3))

        self._E_abs_interp = interpolate.interp1d(
            log_freq,
            log_E_abs,
            kind=interp_kind,
            fill_value=(log_E_abs[0], log_E_abs[-1]),
            bounds_error=False
        )

    def set_wlf_parameters(
        self,
        C1: float,
        C2: float,
        T_ref: Optional[float] = None
    ):
        """
        Set WLF (Williams-Landel-Ferry) parameters for time-temperature superposition.

        log10(aT) = -C1 * (T - T_ref) / (C2 + T - T_ref)

        Parameters
        ----------
        C1 : float
            WLF parameter C1
        C2 : float
            WLF parameter C2 (K)
        T_ref : float, optional
            Reference temperature (°C). If None, uses master curve reference.
        """
        self.C1 = C1
        self.C2 = C2
        self.T_ref_wlf = T_ref if T_ref is not None else self.reference_temp

    def wlf_shift_factor(self, temperature: float) -> float:
        """
        Calculate WLF shift factor.

        Parameters
        ----------
        temperature : float
            Temperature (°C)

        Returns
        -------
        float
            Shift factor aT
        """
        if self.C1 is None or self.C2 is None:
            return 1.0

        T_diff = temperature - self.T_ref_wlf
        log_aT = -self.C1 * T_diff / (self.C2 + T_diff)
        return 10**log_aT

    def get_modulus(
        self,
        frequency: Union[float, np.ndarray],
        temperature: Optional[float] = None
    ) -> Union[complex, np.ndarray]:
        """
        Get complex modulus E*(ω) at given frequency.

        E*(ω) = E'(ω) + i*E''(ω)

        Parameters
        ----------
        frequency : float or np.ndarray
            Angular frequency (rad/s)
        temperature : float, optional
            Temperature (°C). If provided, applies WLF shift.

        Returns
        -------
        complex or np.ndarray
            Complex modulus (Pa)
        """
        if self._E_prime_interp is None:
            raise ValueError("Master curve not set. Call set_master_curve first.")

        # Apply temperature shift if needed
        if temperature is not None and temperature != self.reference_temp:
            aT = self.wlf_shift_factor(temperature)
            frequency = frequency * aT

        # Handle scalar or array
        scalar_input = np.isscalar(frequency)
        freq_array = np.atleast_1d(frequency)

        # Interpolate in log space
        log_freq = np.log10(np.maximum(freq_array, 1e-10))

        log_E_prime = self._E_prime_interp(log_freq)
        log_E_double_prime = self._E_double_prime_interp(log_freq)

        # Safety clamp: 저주파 외삽 시 비물리적 값 방지
        # E' >= 0.01 MPa (10^4 Pa), E'' >= 0.001 MPa (10^3 Pa)
        log_E_prime = np.clip(log_E_prime, 4.0, None)
        log_E_double_prime = np.clip(log_E_double_prime, 3.0, None)

        # Handle NaN/Inf
        if np.any(~np.isfinite(log_E_prime)):
            log_E_prime = np.nan_to_num(log_E_prime, nan=6.0, posinf=12.0, neginf=4.0)
        if np.any(~np.isfinite(log_E_double_prime)):
            log_E_double_prime = np.nan_to_num(log_E_double_prime, nan=5.0, posinf=10.0, neginf=3.0)

        E_prime = 10**log_E_prime
        E_double_prime = 10**log_E_double_prime

        # Create complex modulus
        E_complex = E_prime + 1j * E_double_prime

        if scalar_input:
            return E_complex[0]
        else:
            return E_complex

    def get_storage_modulus(
        self,
        frequency: Union[float, np.ndarray],
        temperature: Optional[float] = None
    ) -> Union[float, np.ndarray]:
        """Get storage modulus E'(ω)."""
        E_complex = self.get_modulus(frequency, temperature)
        return np.real(E_complex)

    def get_loss_modulus(
        self,
        frequency: Union[float, np.ndarray],
        temperature: Optional[float] = None
    ) -> Union[float, np.ndarray]:
        """Get loss modulus E''(ω)."""
        E_complex = self.get_modulus(frequency, temperature)
        return np.imag(E_complex)

    def get_modulus_magnitude(
        self,
        frequency: Union[float, np.ndarray],
        temperature: Optional[float] = None
    ) -> Union[float, np.ndarray]:
        """
        Get complex modulus magnitude |E*(ω)|.

        |E*| = √(E'² + E''²)

        Parameters
        ----------
        frequency : float or np.ndarray
            Angular frequency (rad/s)
        temperature : float, optional
            Temperature (°C)

        Returns
        -------
        float or np.ndarray
            Modulus magnitude (Pa)
        """
        E_complex = self.get_modulus(frequency, temperature)
        return np.abs(E_complex)

    def get_loss_tangent(
        self,
        frequency: Union[float, np.ndarray],
        temperature: Optional[float] = None
    ) -> Union[float, np.ndarray]:
        """
        Get loss tangent tan(δ) = E''/E'.

        Parameters
        ----------
        frequency : float or np.ndarray
            Angular frequency (rad/s)
        temperature : float, optional
            Temperature (°C)

        Returns
        -------
        float or np.ndarray
            Loss tangent (dimensionless)
        """
        E_prime = self.get_storage_modulus(frequency, temperature)
        E_double_prime = self.get_loss_modulus(frequency, temperature)
        return E_double_prime / E_prime

    @staticmethod
    def create_example_sbr() -> 'ViscoelasticMaterial':
        """
        Create example SBR (Styrene-Butadiene Rubber) material.

        Returns
        -------
        ViscoelasticMaterial
            Example SBR material with typical properties
        """
        # Typical SBR master curve at 20°C
        # Frequencies from 10^-6 to 10^8 rad/s
        log_freq = np.linspace(-6, 8, 100)
        frequencies = 10**log_freq

        # Storage modulus: transitions from rubbery to glassy
        E_rubbery = 1e6  # 1 MPa
        E_glassy = 1e9   # 1 GPa
        freq_transition = 1e4  # rad/s
        width = 2.0

        storage_modulus = E_rubbery + (E_glassy - E_rubbery) / (
            1 + (freq_transition / frequencies)**width
        )

        # Loss modulus: peak at transition
        loss_peak_height = 3e8  # Pa
        loss_width = 1.5
        loss_modulus = loss_peak_height * np.exp(
            -(np.log10(frequencies) - np.log10(freq_transition))**2 / loss_width**2
        )

        material = ViscoelasticMaterial(
            frequencies=frequencies,
            storage_modulus=storage_modulus,
            loss_modulus=loss_modulus,
            reference_temp=20.0,
            name="SBR (example)"
        )

        # Set typical WLF parameters for SBR
        material.set_wlf_parameters(C1=17.4, C2=51.6, T_ref=20.0)

        return material

    @staticmethod
    def create_example_pdms() -> 'ViscoelasticMaterial':
        """
        Create example PDMS (Polydimethylsiloxane) material.

        Returns
        -------
        ViscoelasticMaterial
            Example PDMS material
        """
        log_freq = np.linspace(-4, 6, 80)
        frequencies = 10**log_freq

        # PDMS is less frequency-dependent than rubber
        E_rubbery = 5e5  # 0.5 MPa
        E_glassy = 1e9   # 1 GPa
        freq_transition = 1e3
        width = 1.5

        storage_modulus = E_rubbery + (E_glassy - E_rubbery) / (
            1 + (freq_transition / frequencies)**width
        )

        loss_peak_height = 1e8
        loss_width = 1.0
        loss_modulus = loss_peak_height * np.exp(
            -(np.log10(frequencies) - np.log10(freq_transition))**2 / loss_width**2
        )

        material = ViscoelasticMaterial(
            frequencies=frequencies,
            storage_modulus=storage_modulus,
            loss_modulus=loss_modulus,
            reference_temp=20.0,
            name="PDMS (example)"
        )

        material.set_wlf_parameters(C1=15.0, C2=40.0, T_ref=20.0)

        return material

    def __repr__(self) -> str:
        return f"ViscoelasticMaterial(name='{self.name}', T_ref={self.reference_temp}°C)"
