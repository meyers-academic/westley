import numpy as np
from westley.simulate import simulate_squiggly, simulate_fopts


def test_simulate_squiggly():
    """Test the simulate_squiggly function"""
    # Set up test parameters
    freqs = np.linspace(10, 100, 91)
    sigma = np.ones_like(freqs) * 1e-6
    sine_amplitude = 1e-8
    offset_amplitude = 1e-7
    fref = 10

    # Run the simulation
    signal, data, data_obj = simulate_squiggly(
        freqs, sigma, sine_amplitude, offset_amplitude, fref
    )

    # Test signal shape and properties
    assert signal.shape == freqs.shape, "Signal shape should match frequency array"
    assert data.shape == freqs.shape, "Data shape should match frequency array"

    # Test that the signal follows expected pattern
    expected_signal = sine_amplitude * np.sin(freqs / fref) + offset_amplitude
    np.testing.assert_array_almost_equal(signal, expected_signal)

    # Test that noise is properly scaled
    noise = data - signal
    assert np.abs(np.std(noise) -
                  1e-6) < 2e-7, "Noise standard deviation should be close to 1"

    # Test data object properties
    assert np.array_equal(data_obj.data_xvals,
                          freqs), "Data object frequencies should match input"
    assert np.array_equal(data_obj.data_yvals,
                          data), "Data object data should match output"
    assert np.array_equal(data_obj.data_errors,
                          sigma), "Data object sigma should match input"


def test_simulate_fopts():
    """Test the simulate_fopts function"""
    # Set up test parameters
    freqs = np.linspace(10, 100, 91)
    sigma = np.ones_like(freqs) * 1e-9
    Omega_star = 1.8e-10
    f_star = 30
    alpha1 = 3
    alpha2 = -4
    delta = 2

    # Run the simulation
    signal, data, data_obj = simulate_fopts(
        freqs, sigma, Omega_star, f_star, alpha1, alpha2, delta
    )

    # Test signal shape and properties
    assert signal.shape == freqs.shape, "Signal shape should match frequency array"
    assert data.shape == freqs.shape, "Data shape should match frequency array"

    # Test that the signal follows expected pattern
    expected_signal = (Omega_star * (freqs / f_star)**alpha1 *
                       (1 + (freqs / f_star)**delta)**((alpha2-alpha1)/delta))
    np.testing.assert_array_almost_equal(signal, expected_signal)

    # Test that noise is properly scaled
    noise = data - signal
    assert np.abs(np.std(noise) -
                  1e-9) < 2e-10, "Noise standard deviation should be close to 1"

    # Test data object properties
    assert np.array_equal(
        data_obj.data_xvals, freqs), "Data object frequencies should match input"
    assert np.array_equal(
        data_obj.data_yvals, data), "Data object data should match output"
    assert np.array_equal(
        data_obj.data_errors, sigma), "Data object sigma should match input"

    # Test with default parameters
    signal_default, data_default, _ = simulate_fopts(freqs, sigma)
    assert signal_default.shape == freqs.shape, "Default signal shape should match frequency array"
