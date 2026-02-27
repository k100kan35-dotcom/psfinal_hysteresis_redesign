"""
Friction Coefficient Calculator for Persson Model
===================================================

Implements the viscoelastic friction coefficient (mu_visc) calculation
based on Persson's contact mechanics theory.

Mathematical Definition:
    mu_visc = (1/2) * integral[q0->q1] dq * q^3 * C(q) * P(q) * S(q)
              * integral[0->2pi] dphi * cos(phi) * Im[E(q*v*cos(phi))] / ((1-nu^2)*sigma0)

where:
    - P(q) = erf(1 / (2*sqrt(G(q)))) : contact area ratio
    - S(q) = gamma + (1-gamma) * P(q) : contact correction factor (gamma ~ 0.6)
    - Im[E(omega)] : loss modulus
    - omega = q * v * cos(phi) : excitation frequency

Key Implementation Notes:
    1. The angle integral uses symmetry: integrate [0, pi/2] and multiply by 4
       because cos(phi) and |qv*cos(phi)| are symmetric
    2. The prefactor 1/((1-nu^2)*sigma0) is applied to the angle integral result
    3. Temperature correction via shift factor is applied inside get_loss_modulus
    4. Nonlinear correction (Payne effect) is optionally applied via f,g curves

References:
    - RubberFriction Manual [Source 70, 936]
    - Hankook Tire Paper [Source 355]
    - Persson, B.N.J. (2001, 2006)
"""

import numpy as np
from scipy.special import erf
from scipy.integrate import simpson
from typing import Callable, Optional, Tuple, Union, Dict
import warnings


def get_effective_modulus(
    omega: float,
    temperature: float,
    loss_modulus_func: Callable[[float, float], float],
    strain: Optional[float] = None,
    g_interpolator: Optional[Callable[[float], float]] = None
) -> float:
    """
    Get effective loss modulus with optional strain-dependent correction.

    This function implements the modulus acquisition step from the work instruction:
    1. Get linear loss modulus at the given frequency and temperature
    2. If nonlinear correction is enabled, apply g(strain) correction

    Parameters
    ----------
    omega : float
        Angular frequency (rad/s)
    temperature : float
        Temperature (Celsius)
    loss_modulus_func : callable
        Function that returns linear loss modulus Im[E(omega, T)]
    strain : float, optional
        Local strain amplitude (fraction, not %). If None, no correction applied.
    g_interpolator : callable, optional
        Function g(strain) for loss modulus correction. If None, no correction.

    Returns
    -------
    float
        Effective loss modulus (Pa)

    Notes
    -----
    The effective modulus is: ImE_eff = ImE_linear(omega, T) * g(strain)
    where g(strain) <= 1 accounts for the Payne effect (strain softening).
    """
    # Ensure omega is positive
    omega = max(abs(omega), 1e-10)

    # Get linear loss modulus
    ImE_linear = loss_modulus_func(omega, temperature)

    # Apply nonlinear correction if available
    if strain is not None and g_interpolator is not None:
        strain = np.clip(strain, 0.0, 1.0)  # Limit to 0-100%
        g_val = g_interpolator(strain)

        # Handle NaN or invalid g_val - fallback to linear (g=1.0)
        if not np.isfinite(g_val):
            g_val = 1.0  # No correction if g is invalid
        else:
            # Clip g to reasonable range: minimum 0.01 to prevent zero
            # g should be > 0 for physical meaning (complete loss of modulus is unrealistic)
            g_val = np.clip(g_val, 0.01, None)  # g can exceed 1.0

        ImE_eff = ImE_linear * g_val
    else:
        ImE_eff = ImE_linear

    return ImE_eff


class FrictionCalculator:
    """
    Calculator for viscoelastic friction coefficient (mu_visc).

    This class handles the double integral calculation for friction:
    - Inner integral: integration over angle phi from 0 to 2*pi (using symmetry)
    - Outer integral: integration over wavenumber q from q0 to q1

    The calculation follows the work instruction:
    1. For each wavenumber q, calculate the angle integral
    2. Compute the integrand: q^3 * C(q) * P(q) * S(q) * angle_integral
    3. Integrate over q and multiply by 1/2
    """

    def __init__(
        self,
        psd_func: Callable[[np.ndarray], np.ndarray],
        loss_modulus_func: Callable[[float, float], float],
        sigma_0: float,
        velocity: float,
        temperature: float = 20.0,
        poisson_ratio: float = 0.5,
        gamma: float = 0.6,
        n_angle_points: int = 72,
        g_interpolator: Optional[Callable[[float], float]] = None,
        strain_estimate: float = 0.01,
        p_exponent: int = 2
    ):
        """
        Initialize friction calculator.

        Parameters
        ----------
        psd_func : callable
            Function C(q) that returns PSD values for given wavenumbers
        loss_modulus_func : callable
            Function that returns loss modulus Im[E(omega, T)] for given
            angular frequency and temperature
        sigma_0 : float
            Nominal contact pressure (Pa)
        velocity : float
            Sliding velocity (m/s)
        temperature : float, optional
            Temperature (Celsius), default 20.0
        poisson_ratio : float, optional
            Poisson's ratio of the material (default: 0.5)
        gamma : float, optional
            Contact correction factor (default: 0.5)
        n_angle_points : int, optional
            Number of points for angle integration (default: 72)
        g_interpolator : callable, optional
            Function g(strain) for nonlinear correction
        strain_estimate : float, optional
            Default strain estimate for nonlinear correction (default: 0.01 = 1%)
        p_exponent : int, optional
            Exponent for P in S(q) formula: S(q) = γ + (1-γ)·P^p_exponent
            1 for P¹, 2 for P² (default: 2)
        """
        self.psd_func = psd_func
        self.loss_modulus_func = loss_modulus_func
        self.sigma_0 = sigma_0
        self.velocity = velocity
        self.temperature = temperature
        self.poisson_ratio = poisson_ratio
        self.gamma = gamma
        self.n_angle_points = n_angle_points
        self.g_interpolator = g_interpolator
        self.strain_estimate = strain_estimate
        self.p_exponent = p_exponent

        # Precompute constant factor: 1 / ((1 - nu^2) * sigma0)
        self.prefactor = 1.0 / ((1 - poisson_ratio**2) * sigma_0)

    def calculate_P_from_G(self, G: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate contact area ratio P(q) from G(q).

        P(q) = erf(1 / (2 * sqrt(G(q))))

        Parameters
        ----------
        G : float or np.ndarray
            G(q) values (dimensionless)

        Returns
        -------
        float or np.ndarray
            P(q) contact area ratio (0 to 1)
        """
        G = np.asarray(G, dtype=float)
        P = np.zeros_like(G)

        # Handle NaN and negative G values - treat as full contact
        invalid_mask = ~np.isfinite(G) | (G < 0)
        P[invalid_mask] = 1.0

        # Handle G close to zero (full contact)
        small_G_mask = (G >= 0) & (G < 1e-10) & np.isfinite(G)
        P[small_G_mask] = 1.0

        # Normal calculation for valid G > 0
        valid_mask = np.isfinite(G) & (G >= 1e-10)
        if np.any(valid_mask):
            sqrt_G = np.sqrt(G[valid_mask])
            # Prevent overflow for very large G
            arg = 1.0 / (2.0 * sqrt_G)
            arg = np.minimum(arg, 10.0)  # erf(10) ~ 1.0
            P[valid_mask] = erf(arg)

        return P

    def calculate_S_from_P(self, P: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate contact correction factor S(q) from P(q).

        S(q) = gamma + (1 - gamma) * P(q)^p_exponent

        Parameters
        ----------
        P : float or np.ndarray
            Contact area ratio P(q)

        Returns
        -------
        float or np.ndarray
            S(q) contact correction factor
        """
        P = np.asarray(P)
        return self.gamma + (1 - self.gamma) * P ** self.p_exponent

    def _angle_integral_friction(
        self,
        q: float,
        strain: Optional[float] = None,
        return_details: bool = False
    ) -> Union[float, Tuple[float, Dict]]:
        """
        Compute inner integral over angle phi for friction.

        Integrates: integral[0->2pi] dphi * cos(phi) * Im[E(q*v*cos(phi))] / ((1-nu^2)*sigma0)

        Due to symmetry:
        - For phi in [0, pi/2]: cos(phi) > 0, omega > 0, contribution positive
        - For phi in [pi/2, pi]: cos(phi) < 0, omega < 0, Im[E] also changes sign
          due to causality (Im[E(-w)] = -Im[E(w)]), so (-cos) * (-Im) = positive
        - We use symmetry: integrate 0 to pi/2 and multiply by 4

        Parameters
        ----------
        q : float
            Wavenumber (1/m)
        strain : float, optional
            Local strain for nonlinear correction
        return_details : bool, optional
            If True, return detailed calculation info

        Returns
        -------
        float or (float, dict)
            Result of angle integration, optionally with details
        """
        # Use symmetry: integrate over [0, pi/2] and multiply by 4
        # This avoids numerical cancellation issues
        phi = np.linspace(0, np.pi / 2, self.n_angle_points)
        dphi = phi[1] - phi[0] if len(phi) > 1 else np.pi / 2

        # Calculate frequencies for each angle
        # omega = q * v * cos(phi), always positive in [0, pi/2]
        omega_arr = q * self.velocity * np.cos(phi)

        # Get cos(phi) values (always positive in [0, pi/2])
        cos_phi = np.cos(phi)

        # Calculate integrand: cos(phi) * Im[E(omega)] / ((1-nu^2)*sigma0)
        integrand = np.zeros_like(phi)
        ImE_values = np.zeros_like(phi)

        # Use strain estimate if not provided
        if strain is None:
            strain = self.strain_estimate

        for i, (w, c) in enumerate(zip(omega_arr, cos_phi)):
            # Handle very small frequencies
            omega_val = max(w, 1e-10)

            # Get effective loss modulus (with optional nonlinear correction)
            ImE = get_effective_modulus(
                omega_val,
                self.temperature,
                self.loss_modulus_func,
                strain=strain if self.g_interpolator else None,
                g_interpolator=self.g_interpolator
            )
            ImE_values[i] = ImE

            # Calculate integrand term: cos(phi) * Im[E] * prefactor
            integrand[i] = c * ImE * self.prefactor

        # Numerical integration using Simpson's rule (more accurate than trapezoidal)
        # Multiply by 4 due to symmetry (0 to pi/2 -> 0 to 2pi)
        if len(phi) >= 3:
            result = 4.0 * simpson(integrand, x=phi)
        else:
            result = 4.0 * np.trapezoid(integrand, phi)

        if return_details:
            details = {
                'phi': phi,
                'omega': omega_arr,
                'cos_phi': cos_phi,
                'ImE': ImE_values,
                'integrand': integrand,
                'integral_value': result,
                'q': q,
                'velocity': self.velocity,
                'prefactor': self.prefactor
            }
            return result, details

        return result

    def calculate_mu_visc(
        self,
        q_array: np.ndarray,
        G_array: np.ndarray,
        C_q_array: Optional[np.ndarray] = None,
        progress_callback: Optional[Callable[[int], None]] = None,
        strain_array: Optional[np.ndarray] = None
    ) -> Tuple[float, dict]:
        """
        Calculate viscoelastic friction coefficient mu_visc.

        mu_visc = (1/2) * integral[q0->q1] dq * q^3 * C(q) * P(q) * S(q) * (angle_integral)

        This follows the work instruction:
        1. For each q, compute P(q) = erf(1/(2*sqrt(G(q))))
        2. Compute S(q) = gamma + (1-gamma)*P(q)
        3. Compute angle integral with optional strain correction
        4. Form integrand: q^3 * C(q) * P(q) * S(q) * angle_integral
        5. Integrate over q and multiply by 1/2

        Parameters
        ----------
        q_array : np.ndarray
            Array of wavenumbers (1/m) in ascending order
        G_array : np.ndarray
            Array of G(q) values (dimensionless, already divided by sigma0^2)
        C_q_array : np.ndarray, optional
            Array of PSD values C(q). If None, uses self.psd_func
        progress_callback : callable, optional
            Function to call with progress updates (0-100)
        strain_array : np.ndarray, optional
            Array of strain values for each q (for nonlinear correction)

        Returns
        -------
        mu_visc : float
            Viscoelastic friction coefficient
        details : dict
            Dictionary containing intermediate values for debugging
        """
        q_array = np.asarray(q_array)
        G_array = np.asarray(G_array)
        n = len(q_array)

        # Get PSD values
        if C_q_array is None:
            C_q_array = self.psd_func(q_array)
        else:
            C_q_array = np.asarray(C_q_array)

        # Prepare strain array if not provided
        if strain_array is None:
            strain_array = np.full(n, self.strain_estimate)
        else:
            strain_array = np.asarray(strain_array)

        # Calculate P(q) from G(q): P = erf(1 / (2*sqrt(G)))
        P_array = self.calculate_P_from_G(G_array)

        # Calculate S(q) from P(q): S = gamma + (1-gamma)*P
        S_array = self.calculate_S_from_P(P_array)

        # Calculate angle integral for each q
        angle_integral_array = np.zeros(n)
        integrand_array = np.zeros(n)

        # Store intermediate values for debugging
        q3_values = np.zeros(n)
        qCPS_values = np.zeros(n)

        for i, q in enumerate(q_array):
            # Calculate angle integral with strain-dependent correction
            strain_i = strain_array[i] if i < len(strain_array) else self.strain_estimate
            angle_integral_array[i] = self._angle_integral_friction(q, strain=strain_i)

            # Calculate components for debugging
            q3_values[i] = q**3
            qCPS_values[i] = q**3 * C_q_array[i] * P_array[i] * S_array[i]

            # Calculate full integrand: q^3 * C(q) * P(q) * S(q) * angle_integral
            integrand_array[i] = qCPS_values[i] * angle_integral_array[i]

            # Progress callback
            if progress_callback and i % max(1, n // 20) == 0:
                progress_callback(int((i + 1) / n * 100))

        # Numerical integration over q using Simpson's rule
        if n >= 3:
            integral = simpson(integrand_array, x=q_array)
        else:
            integral = np.trapezoid(integrand_array, q_array)

        # Apply prefactor 1/2
        mu_visc = 0.5 * integral

        # Calculate cumulative friction contribution
        cumulative_mu = np.zeros(n)
        cumsum = 0.0
        for i in range(1, n):
            delta = 0.5 * (integrand_array[i-1] + integrand_array[i]) * (q_array[i] - q_array[i-1])
            cumsum += 0.5 * delta
            cumulative_mu[i] = cumsum

        details = {
            'q': q_array,
            'G': G_array,
            'P': P_array,
            'S': S_array,
            'C_q': C_q_array,
            'q3': q3_values,
            'q3CPS': qCPS_values,
            'angle_integral': angle_integral_array,
            'integrand': integrand_array,
            'cumulative_mu': cumulative_mu,
            'velocity': self.velocity,
            'sigma0': self.sigma_0,
            'temperature': self.temperature,
            'gamma': self.gamma,
            'prefactor': self.prefactor,
            'strain_estimate': self.strain_estimate,
            'total_integral': integral,
            'mu_visc': mu_visc
        }

        return mu_visc, details

    def calculate_mu_visc_multi_velocity(
        self,
        q_array: np.ndarray,
        G_matrix: np.ndarray,
        v_array: np.ndarray,
        C_q_array: Optional[np.ndarray] = None,
        progress_callback: Optional[Callable[[int], None]] = None,
        strain_estimator: Optional[Callable[[np.ndarray, np.ndarray, float], np.ndarray]] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Calculate mu_visc for multiple velocities.

        Parameters
        ----------
        q_array : np.ndarray
            Array of wavenumbers (1/m)
        G_matrix : np.ndarray
            2D array G(q, v) with shape (len(q), len(v))
        v_array : np.ndarray
            Array of velocities (m/s)
        C_q_array : np.ndarray, optional
            Array of PSD values C(q)
        progress_callback : callable, optional
            Function to call with progress updates (0-100)
        strain_estimator : callable, optional
            Function to estimate local strain: strain_estimator(q_array, G_array, velocity)

        Returns
        -------
        mu_array : np.ndarray
            Array of mu_visc values for each velocity
        details : dict
            Dictionary containing results for each velocity
        """
        v_array = np.asarray(v_array)
        n_v = len(v_array)

        mu_array = np.zeros(n_v)
        all_details = []

        original_velocity = self.velocity

        for j, v in enumerate(v_array):
            # Update velocity
            self.velocity = v

            # Get G(q) for this velocity
            G_q = G_matrix[:, j]

            # Estimate strain if estimator provided
            if strain_estimator is not None:
                strain_array = strain_estimator(q_array, G_q, v)
            else:
                strain_array = None

            # Calculate mu_visc
            mu_visc, details = self.calculate_mu_visc(
                q_array, G_q, C_q_array, strain_array=strain_array
            )

            mu_array[j] = mu_visc
            details['velocity'] = v
            all_details.append(details)

            # Progress callback
            if progress_callback:
                progress_callback(int((j + 1) / n_v * 100))

        # Restore original velocity
        self.velocity = original_velocity

        # Add summary stats
        summary = {
            'velocities': v_array,
            'mu_min': np.min(mu_array),
            'mu_max': np.max(mu_array),
            'mu_mean': np.mean(mu_array),
            'peak_idx': np.argmax(mu_array),
            'peak_velocity': v_array[np.argmax(mu_array)],
            'peak_mu': np.max(mu_array),
            'details': all_details
        }

        return mu_array, summary

    def update_parameters(
        self,
        sigma_0: Optional[float] = None,
        velocity: Optional[float] = None,
        temperature: Optional[float] = None,
        poisson_ratio: Optional[float] = None,
        gamma: Optional[float] = None
    ):
        """
        Update calculation parameters.

        Parameters
        ----------
        sigma_0 : float, optional
            New nominal contact pressure (Pa)
        velocity : float, optional
            New sliding velocity (m/s)
        temperature : float, optional
            New temperature (Celsius)
        poisson_ratio : float, optional
            New Poisson's ratio
        gamma : float, optional
            New contact correction factor
        """
        if sigma_0 is not None:
            self.sigma_0 = sigma_0
            self.prefactor = 1.0 / ((1 - self.poisson_ratio**2) * sigma_0)

        if velocity is not None:
            self.velocity = velocity

        if temperature is not None:
            self.temperature = temperature

        if poisson_ratio is not None:
            self.poisson_ratio = poisson_ratio
            self.prefactor = 1.0 / ((1 - self.poisson_ratio**2) * self.sigma_0)

        if gamma is not None:
            self.gamma = gamma


def calculate_mu_visc_simple(
    q_array: np.ndarray,
    C_q_array: np.ndarray,
    G_array: np.ndarray,
    v: float,
    sigma0: float,
    func_ImE: Callable[[float, float], float],
    temperature: float = 20.0,
    nu: float = 0.5,
    gamma: float = 0.6,
    n_phi: int = 72,
    g_interpolator: Optional[Callable[[float], float]] = None,
    strain: float = 0.01
) -> Tuple[float, dict]:
    """
    Simplified function to calculate mu_visc without class instantiation.

    This is a convenience function for quick calculations.

    Parameters
    ----------
    q_array : np.ndarray
        Array of wavenumbers (1/m) from q0 to q1
    C_q_array : np.ndarray
        Array of PSD values C(q) at each wavenumber
    G_array : np.ndarray
        Array of G(q) values (dimensionless, already divided by sigma0^2)
    v : float
        Sliding velocity (m/s)
    sigma0 : float
        Nominal contact pressure (Pa)
    func_ImE : callable
        Function that returns loss modulus Im[E(omega, T)]
    temperature : float, optional
        Temperature (Celsius), default 20.0
    nu : float, optional
        Poisson's ratio, default 0.5
    gamma : float, optional
        Contact correction factor, default 0.5
    n_phi : int, optional
        Number of angle integration points, default 72
    g_interpolator : callable, optional
        Function g(strain) for nonlinear correction
    strain : float, optional
        Strain estimate for nonlinear correction (default: 0.01 = 1%)

    Returns
    -------
    mu_visc : float
        Viscoelastic friction coefficient
    details : dict
        Dictionary with intermediate calculation values
    """
    q_array = np.asarray(q_array)
    C_q_array = np.asarray(C_q_array)
    G_array = np.asarray(G_array)
    n = len(q_array)

    # Prefactor: 1 / ((1 - nu^2) * sigma0)
    prefactor = 1.0 / ((1 - nu**2) * sigma0)

    # Calculate P(q) from G(q): P = erf(1 / (2*sqrt(G)))
    P_array = np.zeros_like(G_array, dtype=float)
    small_G_mask = G_array < 1e-10
    P_array[small_G_mask] = 1.0
    valid_mask = ~small_G_mask
    if np.any(valid_mask):
        sqrt_G = np.sqrt(G_array[valid_mask])
        arg = np.minimum(1.0 / (2.0 * sqrt_G), 10.0)
        P_array[valid_mask] = erf(arg)

    # Calculate S(q) from P(q): S = gamma + (1-gamma)*P
    S_array = gamma + (1 - gamma) * P_array

    # Angle array - use symmetry: integrate 0 to pi/2 and multiply by 4
    phi = np.linspace(0, np.pi / 2, n_phi)

    # Calculate angle integral and full integrand for each q
    angle_integral_array = np.zeros(n)
    integrand_array = np.zeros(n)
    q3CPS_array = np.zeros(n)

    for i, q in enumerate(q_array):
        # Calculate omega = q * v * cos(phi), always positive in [0, pi/2]
        omega = q * v * np.cos(phi)
        cos_phi = np.cos(phi)

        # Calculate integrand: cos(phi) * Im[E(omega)] * prefactor
        integrand_phi = np.zeros_like(phi)
        for j, (w, c) in enumerate(zip(omega, cos_phi)):
            omega_val = max(w, 1e-10)

            # Get effective modulus with optional nonlinear correction
            ImE = get_effective_modulus(
                omega_val, temperature, func_ImE,
                strain=strain if g_interpolator else None,
                g_interpolator=g_interpolator
            )
            integrand_phi[j] = c * ImE * prefactor

        # Angle integral (multiply by 4 due to symmetry), use Simpson's rule
        if n_phi >= 3:
            angle_integral_array[i] = 4.0 * simpson(integrand_phi, x=phi)
        else:
            angle_integral_array[i] = 4.0 * np.trapezoid(integrand_phi, phi)

        # Precompute q^3 * C * P * S for debugging
        q3CPS_array[i] = q**3 * C_q_array[i] * P_array[i] * S_array[i]

        # Full integrand: q^3 * C(q) * P(q) * S(q) * angle_integral
        integrand_array[i] = q3CPS_array[i] * angle_integral_array[i]

    # Numerical integration over q using Simpson's rule
    if n >= 3:
        integral = simpson(integrand_array, x=q_array)
    else:
        integral = np.trapezoid(integrand_array, q_array)

    mu_visc = 0.5 * integral

    # Cumulative friction
    cumulative_mu = np.zeros(n)
    cumsum = 0.0
    for i in range(1, n):
        delta = 0.5 * (integrand_array[i-1] + integrand_array[i]) * (q_array[i] - q_array[i-1])
        cumsum += 0.5 * delta
        cumulative_mu[i] = cumsum

    details = {
        'q': q_array,
        'P': P_array,
        'S': S_array,
        'C_q': C_q_array,
        'G': G_array,
        'q3CPS': q3CPS_array,
        'angle_integral': angle_integral_array,
        'integrand': integrand_array,
        'cumulative_mu': cumulative_mu,
        'velocity': v,
        'sigma0': sigma0,
        'temperature': temperature,
        'gamma': gamma,
        'prefactor': prefactor,
        'total_integral': integral,
        'mu_visc': mu_visc
    }

    return mu_visc, details


def apply_nonlinear_strain_correction(
    E_prime: np.ndarray,
    E_double_prime: np.ndarray,
    strain: float,
    f_curve: Callable[[float], float],
    g_curve: Callable[[float], float]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply nonlinear strain correction to modulus values.

    When high local strain occurs at contact asperities, the linear
    viscoelastic moduli need to be corrected using the f,g functions
    from strain sweep experiments.

    E'_nonlinear = E' * f(strain)
    E''_nonlinear = E'' * g(strain)

    Parameters
    ----------
    E_prime : np.ndarray
        Storage modulus E' (Pa)
    E_double_prime : np.ndarray
        Loss modulus E'' (Pa)
    strain : float
        Local strain amplitude (fraction, not %)
    f_curve : callable
        Function f(strain) for storage modulus correction
        Returns value in range [0, 1]
    g_curve : callable
        Function g(strain) for loss modulus correction
        Returns value in range [0, 1]

    Returns
    -------
    E_prime_corrected : np.ndarray
        Corrected storage modulus
    E_double_prime_corrected : np.ndarray
        Corrected loss modulus
    """
    f_val = f_curve(strain)
    g_val = g_curve(strain)

    # Clip to valid range
    f_val = np.clip(f_val, 0.0, 1.0)
    g_val = np.clip(g_val, 0.01, None)  # g can exceed 1.0

    E_prime_corrected = E_prime * f_val
    E_double_prime_corrected = E_double_prime * g_val

    return E_prime_corrected, E_double_prime_corrected


def estimate_local_strain(
    G_area: float,
    C_q: float,
    q: float,
    sigma0: float,
    E_prime: float,
    method: str = 'persson'
) -> float:
    """
    Estimate local strain at contact asperities.

    The local strain depends on the contact geometry and material stiffness.
    Higher roughness and softer materials lead to higher local strains.

    Parameters
    ----------
    G_area : float
        G(q) area function value (dimensionless)
    C_q : float
        PSD value C(q) at wavenumber q
    q : float
        Wavenumber (1/m)
    sigma0 : float
        Nominal contact pressure (Pa)
    E_prime : float
        Storage modulus E' (Pa)
    method : str, optional
        Estimation method: 'persson' or 'simple'

    Returns
    -------
    strain : float
        Estimated local strain (fraction)
    """
    if method == 'persson':
        # Persson's approach: strain related to surface slope
        # Local strain ~ sqrt(C(q) * q^4) / (E'/sigma0)
        if E_prime < 1e3:
            E_prime = 1e3  # Minimum modulus

        slope_rms = np.sqrt(C_q * q**4) if C_q > 0 else 0
        strain = slope_rms * sigma0 / E_prime

    elif method == 'simple':
        # Simple estimate: strain ~ sqrt(G) * sigma0/E
        if G_area > 0:
            strain = np.sqrt(G_area) * sigma0 / max(E_prime, 1e3)
        else:
            strain = 0.0
    else:
        raise ValueError(f"Unknown method: {method}")

    # Limit to physically reasonable range
    strain = np.clip(strain, 0.0, 1.0)  # 0 to 100%

    return float(strain)


def calculate_rms_slope_profile(
    q_array: np.ndarray,
    C_q_array: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate cumulative RMS slope (xi) profile from PSD data.

    The RMS slope squared is computed as:
        xi²(q) = 2π ∫[q0→q] k³ C(k) dk

    This represents the cumulative contribution of surface roughness
    up to wavenumber q.

    Parameters
    ----------
    q_array : np.ndarray
        Array of wavenumbers (1/m), should be in ascending order
    C_q_array : np.ndarray
        Array of PSD values C(q) at each wavenumber (m⁴)

    Returns
    -------
    xi_squared_array : np.ndarray
        Cumulative RMS slope squared at each q
    xi_array : np.ndarray
        Cumulative RMS slope (sqrt of xi_squared) at each q

    Notes
    -----
    Based on Persson theory [Source 21, 119]:
    - The integrand is k³ × C(k)
    - Factor of 2π comes from angular integration in 2D PSD
    """
    q_array = np.asarray(q_array)
    C_q_array = np.asarray(C_q_array)
    n = len(q_array)

    # Calculate integrand: k³ × C(k)
    integrand = q_array**3 * C_q_array

    # Cumulative integration using trapezoidal rule
    xi_squared_array = np.zeros(n)

    for i in range(1, n):
        # Integrate from q[0] to q[i]
        xi_squared_array[i] = 2 * np.pi * np.trapezoid(integrand[:i+1], q_array[:i+1])

    # RMS slope is sqrt of xi_squared
    xi_array = np.sqrt(np.maximum(xi_squared_array, 0))

    return xi_squared_array, xi_array


def calculate_strain_profile(
    q_array: np.ndarray,
    C_q_array: np.ndarray,
    strain_factor: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate local strain profile from PSD data using RMS slope.

    The local strain at each wavenumber is estimated as:
        ε(q) = strain_factor × ξ(q)

    where ξ(q) is the cumulative RMS slope up to wavenumber q.

    Parameters
    ----------
    q_array : np.ndarray
        Array of wavenumbers (1/m)
    C_q_array : np.ndarray
        Array of PSD values C(q) at each wavenumber (m⁴)
    strain_factor : float, optional
        Factor to convert RMS slope to strain (default: 0.5)
        Persson theory suggests ε ≈ 0.5 × ξ [Source 197]

    Returns
    -------
    strain_array : np.ndarray
        Local strain at each wavenumber (fraction, not %)
    xi_array : np.ndarray
        RMS slope at each wavenumber
    xi_squared_array : np.ndarray
        RMS slope squared at each wavenumber

    Notes
    -----
    The strain estimate follows Persson's approach where the local
    deformation at contact asperities is related to the surface slope.
    """
    xi_squared_array, xi_array = calculate_rms_slope_profile(q_array, C_q_array)

    # Strain estimate: ε = strain_factor × ξ
    strain_array = strain_factor * xi_array

    # Clip to physically reasonable range (0 to 100%)
    strain_array = np.clip(strain_array, 0.0, 1.0)

    return strain_array, xi_array, xi_squared_array


def calculate_hrms_profile(
    q_array: np.ndarray,
    C_q_array: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate cumulative RMS height (hrms) profile from PSD data.

    The RMS height squared is computed as:
        h²_rms(q) = 2π ∫[q0→q] k C(k) dk

    Parameters
    ----------
    q_array : np.ndarray
        Array of wavenumbers (1/m)
    C_q_array : np.ndarray
        Array of PSD values C(q) at each wavenumber (m⁴)

    Returns
    -------
    hrms_squared_array : np.ndarray
        Cumulative RMS height squared at each q (m²)
    hrms_array : np.ndarray
        Cumulative RMS height at each q (m)
    """
    q_array = np.asarray(q_array)
    C_q_array = np.asarray(C_q_array)
    n = len(q_array)

    # Integrand: k × C(k)
    integrand = q_array * C_q_array

    # Cumulative integration
    hrms_squared_array = np.zeros(n)

    for i in range(1, n):
        hrms_squared_array[i] = 2 * np.pi * np.trapezoid(integrand[:i+1], q_array[:i+1])

    hrms_array = np.sqrt(np.maximum(hrms_squared_array, 0))

    return hrms_squared_array, hrms_array


class RMSSlopeCalculator:
    """
    Calculator for RMS slope and local strain profiles.

    This class provides methods to calculate surface roughness statistics
    from PSD data and estimate local strains for friction calculations.
    """

    def __init__(
        self,
        q_array: np.ndarray,
        C_q_array: np.ndarray,
        strain_factor: float = 0.5
    ):
        """
        Initialize RMS slope calculator.

        Parameters
        ----------
        q_array : np.ndarray
            Array of wavenumbers (1/m)
        C_q_array : np.ndarray
            Array of PSD values C(q) at each wavenumber (m⁴)
        strain_factor : float, optional
            Factor to convert RMS slope to strain (default: 0.5)
        """
        self.q_array = np.asarray(q_array)
        self.C_q_array = np.asarray(C_q_array)
        self.strain_factor = strain_factor

        # Calculate profiles
        self._calculate_profiles()

    def _calculate_profiles(self):
        """Calculate all roughness profiles."""
        # RMS slope
        self.xi_squared, self.xi = calculate_rms_slope_profile(
            self.q_array, self.C_q_array
        )

        # Local strain: ε(q) = strain_factor × ξ(q)
        self.strain = self.strain_factor * self.xi

        # RMS height
        self.hrms_squared, self.hrms = calculate_hrms_profile(
            self.q_array, self.C_q_array
        )

    def get_strain_at_q(self, q: float) -> float:
        """
        Get interpolated strain value at specific wavenumber.

        Parameters
        ----------
        q : float
            Wavenumber (1/m)

        Returns
        -------
        float
            Strain value at q (fraction)
        """
        from scipy.interpolate import interp1d

        if q <= self.q_array[0]:
            return self.strain[0]
        if q >= self.q_array[-1]:
            return self.strain[-1]

        # Log-log interpolation for better accuracy
        log_q = np.log10(self.q_array)
        log_strain = np.log10(np.maximum(self.strain, 1e-10))

        interp_func = interp1d(log_q, log_strain, kind='linear', fill_value='extrapolate')
        return 10**interp_func(np.log10(q))

    def get_summary(self) -> Dict:
        """
        Get summary statistics of the roughness profiles.

        Returns
        -------
        dict
            Summary containing key statistics
        """
        return {
            'q_min': self.q_array[0],
            'q_max': self.q_array[-1],
            'n_points': len(self.q_array),
            'xi_max': np.max(self.xi),
            'xi_at_qmax': self.xi[-1],
            'strain_max': np.max(self.strain),
            'strain_at_qmax': self.strain[-1],
            'hrms_total': self.hrms[-1],
            'strain_factor': self.strain_factor
        }

    def get_profiles(self) -> Dict:
        """
        Get all calculated profiles.

        Returns
        -------
        dict
            Dictionary containing all profile arrays
        """
        return {
            'q': self.q_array.copy(),
            'C_q': self.C_q_array.copy(),
            'xi_squared': self.xi_squared.copy(),
            'xi': self.xi.copy(),
            'strain': self.strain.copy(),
            'hrms_squared': self.hrms_squared.copy(),
            'hrms': self.hrms.copy()
        }
