from src.simulation.earthquake import EarthquakeSimulator


def test_inactive_returns_zero_offset():
    eq = EarthquakeSimulator()
    assert eq.get_offset(0.0, 0.0, current_time=0.0) == (0.0, 0.0, 0.0)


def test_zero_offset_before_wave_arrival():
    eq = EarthquakeSimulator()
    eq.trigger(current_time=0.0, epicenter=(0.0, 0.0), magnitude=8.0)
    # Ponto distante: a onda ainda não teve tempo de chegar.
    dx, dy, dz = eq.get_offset(x=500.0, z=0.0, current_time=0.01)
    assert (dx, dy, dz) == (0.0, 0.0, 0.0)


def test_nonzero_offset_after_wave_arrival():
    eq = EarthquakeSimulator()
    eq.trigger(current_time=0.0, epicenter=(0.0, 0.0), magnitude=8.0)
    # No epicentro, a onda chega em t=0 (dist=0); um instante depois já deve haver deslocamento.
    dx, dy, dz = eq.get_offset(x=0.0, z=0.0, current_time=0.05)
    assert (dx, dy, dz) != (0.0, 0.0, 0.0)


def test_magnitude_scaling_and_floor():
    eq = EarthquakeSimulator()
    eq.trigger(current_time=0.0, magnitude=5.0)
    assert eq.magnitude == max(0.4, 5.0 * 0.42)

    eq.trigger(current_time=0.0, magnitude=0.0)
    assert eq.magnitude == 0.4


def test_damping_richter_threshold():
    eq = EarthquakeSimulator()
    eq.trigger(current_time=0.0, magnitude=7.0)
    assert eq.damping == 0.15

    eq.trigger(current_time=0.0, magnitude=6.9)
    assert eq.damping == 0.25


def test_crack_intensity_below_threshold_is_zero():
    eq = EarthquakeSimulator()
    eq.trigger(current_time=0.0, magnitude=3.0)
    assert eq.get_crack_intensity() == 0.0


def test_crack_intensity_scales_and_clamps():
    eq = EarthquakeSimulator()
    eq.trigger(current_time=0.0, magnitude=6.0)
    expected = (6.0 - 3.8) / 4.2
    assert eq.get_crack_intensity() == expected

    eq.trigger(current_time=0.0, magnitude=9.0)
    assert eq.get_crack_intensity() == 1.0


def test_reset_clears_max_richter_stop_does_not():
    eq = EarthquakeSimulator()
    eq.trigger(current_time=0.0, magnitude=8.0)

    eq.stop()
    assert eq.active is False
    assert eq.max_richter == 8.0

    eq.reset()
    assert eq.max_richter == 0.0
