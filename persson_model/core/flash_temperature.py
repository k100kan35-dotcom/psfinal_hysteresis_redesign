"""
Flash Temperature Calculator (Greenwood Interpolation)
======================================================

Implements the flash temperature calculation for rubber friction using
Greenwood's interpolation formula. This avoids the complex 3D thermal
integral by using Peclet number-based approximation.

Algorithm (Two-Pass):
    Pass 1 (Cold): Calculate mu_cold and (A/A0)_cold at base temperature T
    Pass 2 (Hot):  Calculate flash temperature ΔT, then recalculate
                   mu_hot and (A/A0)_hot at T_hot = T + ΔT

Key Formulas:
    1. Heat Flux:     q_dot = mu × sigma_0 × v  (Persson 2006, no P(q_m) division)
    2. Peclet Number: Jd = v × d_macro / D_th
    3. Persson 2006:  ΔT = (q_dot × d) / (8κ × sqrt(1 + (π/2) × Jd))  [circular contact]
    4. Hot Temp:      T_hot = T_base + ΔT

References:
    - Greenwood, J.A. (1991) "An interpolation formula for flash temperatures"
    - Persson, B.N.J. (2006) "Rubber friction: role of the flash temperature"
"""

import numpy as np
from typing import Optional, Dict


class FlashTemperatureCalculator:
    """
    Calculator for flash temperature using Greenwood interpolation formula.

    The flash temperature arises from frictional heating at real contact
    asperities during sliding. It affects both the friction coefficient
    (mu_visc) and the real contact area ratio (A/A0) through the
    temperature-dependent viscoelastic modulus via WLF shift.

    Parameters
    ----------
    rho : float
        Rubber density (kg/m³). Default: 1150
    C_v : float
        Specific heat capacity (J/(kg·K)). Default: 1500
    kappa_th : float
        Thermal conductivity (W/(m·K)). Default: 0.3
    d_macro : float
        Macroscopic asperity contact diameter (m). Default: 0.001 (1 mm)
    """

    def __init__(
        self,
        rho: float = 1150.0,
        C_v: float = 1500.0,
        kappa_th: float = 0.3,
        d_macro: float = 0.001
    ):
        self.rho = rho
        self.C_v = C_v
        self.kappa_th = kappa_th
        self.d_macro = d_macro

        # Derived: thermal diffusivity D_th = kappa / (rho * Cv)
        self.D_th = kappa_th / (rho * C_v)

    @property
    def thermal_diffusivity(self) -> float:
        """Thermal diffusivity D_th (m²/s)."""
        return self.D_th

    def update_properties(
        self,
        rho: Optional[float] = None,
        C_v: Optional[float] = None,
        kappa_th: Optional[float] = None,
        d_macro: Optional[float] = None
    ):
        """Update thermal properties and recompute D_th."""
        if rho is not None:
            self.rho = rho
        if C_v is not None:
            self.C_v = C_v
        if kappa_th is not None:
            self.kappa_th = kappa_th
        if d_macro is not None:
            self.d_macro = d_macro
        self.D_th = self.kappa_th / (self.rho * self.C_v)

    def heat_flux(
        self,
        A_A0: float,
        mu: float,
        sigma_0: float,
        velocity: float
    ) -> float:
        """
        Calculate frictional heat flux at the macroscopic contact.

        q_dot = mu × sigma_0 × v

        This is the total frictional dissipation per unit nominal contact area.
        The macroscopic Greenwood formula then calculates the temperature rise
        for a heat source of diameter d_macro moving at velocity v.

        Note: A_A0 is accepted for interface compatibility but NOT used in
        the heat flux calculation. Per Persson (2006), the macroscopic
        flash temperature uses q = μ·σ₀·v without the A/A₀ factor.

        Parameters
        ----------
        A_A0 : float
            Real contact area ratio (A/A0) from cold pass (reserved)
        mu : float
            Friction coefficient from cold pass
        sigma_0 : float
            Nominal contact pressure (Pa)
        velocity : float
            Sliding velocity (m/s)

        Returns
        -------
        float
            Heat flux q_dot (W/m²)
        """
        return mu * sigma_0 * velocity

    def peclet_number(self, velocity: float) -> float:
        """
        Calculate Peclet number (dimensionless).

        Jd = v × d_macro / D_th

        Ratio of advective thermal transport to diffusive transport.
        - Jd → 0: Low speed, thermal diffusion dominates (quasi-static)
        - Jd >> 1: High speed, heat accumulation dominates

        Parameters
        ----------
        velocity : float
            Sliding velocity (m/s)

        Returns
        -------
        float
            Peclet number Jd (dimensionless)
        """
        return velocity * self.d_macro / self.D_th

    def delta_T(self, q_dot: float, Jd: float) -> float:
        """
        Calculate flash temperature rise using Persson (2006) circular contact formula.

        ΔT = (q̇·d) / (8κ) × 1 / √(1 + (π/2)·Jd)

        where Jd = v·d/D_th is the Peclet number.

        This formula is for 3D circular contact patches (appropriate for
        rubber friction on rough surfaces). Interpolates between:
        - Low speed (Jd→0): ΔT ≈ q̇·d / (8κ)  (steady-state, circular source)
        - High speed (Jd>>1): ΔT ∝ 1/√v  (moving heat source)

        Note: Greenwood (1991) band formula (2κ, π/16) gives ~11x larger ΔT
        and is NOT appropriate for 3D circular contact patches.

        Reference: Persson, B.N.J. (2006) J. Chem. Phys. 124, 054703

        Parameters
        ----------
        q_dot : float
            Heat flux (W/m²)
        Jd : float
            Peclet number (dimensionless), Jd = v·d/D_th

        Returns
        -------
        float
            Flash temperature rise ΔT (K or °C)
        """
        if q_dot <= 0 or not np.isfinite(q_dot):
            return 0.0
        if not np.isfinite(Jd) or Jd < 0:
            Jd = 0.0
        # Persson 2006 circular contact: ΔT = (q̇·d)/(8κ) / √(1 + (π/2)Jd)
        return (q_dot * self.d_macro) / (8.0 * self.kappa_th) / np.sqrt(
            1.0 + (np.pi / 2.0) * Jd
        )

    def delta_T_at_scale(self, q_dot_local: float, d_local: float, velocity: float) -> float:
        """
        Calculate flash temperature rise at a specific length scale.

        Uses the Greenwood interpolation formula with a scale-dependent
        characteristic length d = 2π/q instead of the fixed d_macro.

        This enables per-wavenumber flash temperature accumulation:
            ΔT_total = Σ_i δT(q_i)
        where each scale contributes its own temperature rise based on
        its characteristic contact patch size d(q) = 2π/q.

        Parameters
        ----------
        q_dot_local : float
            Local heat flux at this scale (W/m²)
        d_local : float
            Characteristic contact diameter at this scale (m), typically 2π/q
        velocity : float
            Sliding velocity (m/s)

        Returns
        -------
        float
            Flash temperature rise ΔT (K or °C) at this scale
        """
        if q_dot_local <= 0 or not np.isfinite(q_dot_local):
            return 0.0
        if d_local <= 0 or not np.isfinite(d_local):
            return 0.0
        Jd_local = velocity * d_local / self.D_th
        if not np.isfinite(Jd_local) or Jd_local < 0:
            Jd_local = 0.0
        return (q_dot_local * d_local) / (8.0 * self.kappa_th) / np.sqrt(
            1.0 + (np.pi / 2.0) * Jd_local
        )

    def calculate(
        self,
        A_A0_cold: float,
        mu_cold: float,
        sigma_0: float,
        velocity: float,
        T_base: float
    ) -> Dict:
        """
        Full flash temperature calculation for a single velocity.

        Parameters
        ----------
        A_A0_cold : float
            Contact area ratio from cold (Pass 1) calculation
        mu_cold : float
            Friction coefficient from cold (Pass 1) calculation
        sigma_0 : float
            Nominal contact pressure (Pa)
        velocity : float
            Sliding velocity (m/s)
        T_base : float
            Base (background) temperature (°C)

        Returns
        -------
        dict
            'delta_T': temperature rise (°C)
            'T_hot': hot temperature (°C)
            'q_dot': heat flux (W/m²)
            'Jd': Peclet number
        """
        q_dot = self.heat_flux(A_A0_cold, mu_cold, sigma_0, velocity)
        Jd = self.peclet_number(velocity)
        dT = self.delta_T(q_dot, Jd)
        T_hot = T_base + dT

        return {
            'delta_T': dT,
            'T_hot': T_hot,
            'q_dot': q_dot,
            'Jd': Jd
        }

    def calculate_multi_velocity(
        self,
        A_A0_cold_arr: np.ndarray,
        mu_cold_arr: np.ndarray,
        sigma_0: float,
        v_arr: np.ndarray,
        T_base: float
    ) -> Dict:
        """
        Calculate flash temperature for multiple velocities.

        Parameters
        ----------
        A_A0_cold_arr : np.ndarray
            Contact area ratio array from cold pass (one per velocity)
        mu_cold_arr : np.ndarray
            Friction coefficient array from cold pass (one per velocity)
        sigma_0 : float
            Nominal contact pressure (Pa)
        v_arr : np.ndarray
            Velocity array (m/s)
        T_base : float
            Base temperature (°C)

        Returns
        -------
        dict
            'delta_T': array of temperature rises (°C)
            'T_hot': array of hot temperatures (°C)
            'q_dot': array of heat fluxes (W/m²)
            'Jd': array of Peclet numbers
            'T_base': base temperature (°C)
        """
        v_arr = np.asarray(v_arr)
        A_A0_cold_arr = np.asarray(A_A0_cold_arr)
        mu_cold_arr = np.asarray(mu_cold_arr)
        n_v = len(v_arr)

        delta_T = np.zeros(n_v)
        T_hot = np.zeros(n_v)
        q_dot = np.zeros(n_v)
        Jd = np.zeros(n_v)

        for j in range(n_v):
            result = self.calculate(
                A_A0_cold_arr[j], mu_cold_arr[j],
                sigma_0, v_arr[j], T_base
            )
            delta_T[j] = result['delta_T']
            T_hot[j] = result['T_hot']
            q_dot[j] = result['q_dot']
            Jd[j] = result['Jd']

        return {
            'delta_T': delta_T,
            'T_hot': T_hot,
            'q_dot': q_dot,
            'Jd': Jd,
            'T_base': T_base,
            'v': v_arr
        }

    def get_summary(self) -> Dict:
        """Get current thermal parameters summary."""
        return {
            'rho': self.rho,
            'C_v': self.C_v,
            'kappa_th': self.kappa_th,
            'd_macro': self.d_macro,
            'D_th': self.D_th,
            'D_th_formatted': f'{self.D_th:.2e} m²/s'
        }

    def __repr__(self) -> str:
        return (
            f"FlashTemperatureCalculator("
            f"ρ={self.rho} kg/m³, "
            f"Cv={self.C_v} J/(kg·K), "
            f"κ={self.kappa_th} W/(m·K), "
            f"d={self.d_macro*1000:.1f} mm, "
            f"D={self.D_th:.2e} m²/s)"
        )
