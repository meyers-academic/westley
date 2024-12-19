import numpy as np
import bilby

from pygwb.baseline import Baseline
from .fitter import SmoothCurveDataObj
from scipy.interpolate import interp1d
import pandas as pd

H0 = 2e-18 # seconds^-1


def get_sigma_from_noise_curves(detector_names, freqs, obs_T):
    """
    Make noise curves given names of detectors, frequencies and an observation time (in seconds)

    Parameters:
    -----------
    detector_names: list
        list of detector names
    freqs: numpy.ndarray
        frequency array at which to evaluate noise curve
    obs_T: float
        observation time (in seconds)

    Returns:
    --------
    sigma : numpy.ndarray
        1-sigma sensitivity noise curve for Omega_gw(f) based on two-detector search.
    """

    detectors = []

    if len(detector_names) == 1:
        if 'CE' in detector_names:
            CE = bilby.gw.detector.get_empty_interferometer('CE')
            CE1 = bilby.gw.detector.get_empty_interferometer('H1')
            CE2 = bilby.gw.detector.get_empty_interferometer('L1')
            CE.strain_data.frequency_array = freqs
            CE1.strain_data.frequency_array = freqs
            CE2.strain_data.frequency_array = freqs
            CE1.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(frequency_array = freqs, psd_array=CE.power_spectral_density_array)
            CE2.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(frequency_array = freqs, psd_array=CE.power_spectral_density_array)
            detectors.append(CE1)
            detectors.append(CE2)
        else:
            raise ValueError("Only one detector is supported for CE.")
    elif len(detector_names) == 2:
        detectors.append(bilby.gw.detector.get_empty_interferometer(detector_names[0]))
        detectors.append(bilby.gw.detector.get_empty_interferometer(detector_names[1]))
        for det in detectors: 
            det.strain_data.frequency_array = freqs
    else:
        raise ValueError("Only two detectors are supported. One name can be supplied if it is Cosmic Explorer ('CE').")

    # make baseline & calculate ORF
    duration = 1/np.abs(freqs[1] - freqs[0])
    BL = Baseline('BL', detectors[0], detectors[1],
                  frequencies=freqs, duration=duration)
    BL.orf_polarization = 'tensor'

    # calculate sigma^2
    df = np.abs(freqs[1] - freqs[0])

    S0 = 3 * H0 ** 2 / (10 * np.pi ** 2 * freqs ** 3)
    sigma_2 = (
        (detectors[0].amplitude_spectral_density_array) ** 2
        * (detectors[1].amplitude_spectral_density_array) ** 2
        / (2 * obs_T * df * BL.overlap_reduction_function ** 2 * S0 ** 2)
    )

    return np.sqrt(sigma_2)


def simulate_broken_powerlaw(freqs, sigma, omega_break,
                             f_break_turnover=60,
                             alpha_pre=2, alpha_post=-2):
    """
    Generate a broken power law signal
    """
    signal = np.zeros(freqs.size)
    idx_break = np.argmin(np.abs(freqs - f_break_turnover))

    signal[:idx_break] = omega_break * (freqs[:idx_break] / f_break_turnover)**(alpha_pre)
    signal[idx_break:] = omega_break * (freqs[idx_break] / f_break_turnover)**(alpha_post) * (freqs[idx_break:] / freqs[idx_break])**(alpha_post)

    data = signal + sigma * np.random.randn(freqs.size)
    data_obj = SmoothCurveDataObj(freqs, data, sigma)

    return signal, data, data_obj


def simulate_squiggly(freqs, sigma, sine_amplitude, offset_amplitude, fref=10):
    """
    Generate a squiggly signal
    """
    signal = sine_amplitude * np.sin(freqs / fref) + offset_amplitude
    data = signal + sigma * np.random.randn(freqs.size)
    data_obj = SmoothCurveDataObj(freqs, data, sigma)

    return signal, data, data_obj


def simulate_sachdev(freqs, sigma, fref=10):
    """
    Generate a Sachdev signal
    """
    file_path = 'Plotdata.csv'
    data = pd.read_csv(file_path)
    x = data['x'] + 15
    y = data[' y'] * 100
    interp_func = interp1d(x, y, kind='linear', fill_value='extrapolate')
    signal = interp_func(freqs)

    data = signal + sigma * np.random.randn(freqs.size)
    data_obj = SmoothCurveDataObj(freqs, data, sigma)

    return signal, data, data_obj


def simulate_fopts(freqs, sigma, Omega_star=1.8e-10, f_star=30,
                   alpha1=3, alpha2=-4, delta=2):
    """
    Generate a first-order phase transition signal (FOPTS)

    Parameters:
    -----------
    freqs : numpy.ndarray
        Frequency array
    sigma : numpy.ndarray
        Noise curve
    Omega_star : float, optional
        Amplitude of the signal (default: 1.8e-10)
    f_star : float, optional
        Frequency where knee break occurs [Hz] (default: 30)
    alpha1 : float, optional
        Pre-break spectral index (default: 3)
    alpha2 : float, optional
        Post-break spectral index (default: -4)
    delta : float, optional
        Smoothing parameter (default: 2)

    Returns:
    --------
    signal : numpy.ndarray
        The generated FOPTS signal
    data : numpy.ndarray
        Signal plus noise
    data_obj : SmoothCurveDataObj
        Data object containing the signal, noise, and frequencies
    """
    signal = (Omega_star * (freqs / f_star)**alpha1 * 
              (1 + (freqs / f_star)**delta)**((alpha2-alpha1)/delta))
    data = signal + sigma * np.random.randn(freqs.size)
    data_obj = SmoothCurveDataObj(freqs, data, sigma)

    return signal, data, data_obj