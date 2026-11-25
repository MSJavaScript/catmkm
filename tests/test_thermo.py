import numpy as np

from catmkm.thermo import shomate_gas, shomate_gas_by_name, vibration


def test_vibration_zero_frequencies():
    assert vibration(300, []) == 0.0


def test_shomate_free_energy_is_finite():
    value = shomate_gas_by_name(500, "CO", [2170])
    assert value is not None
    assert np.isfinite(value)


def test_shomate_temperature_range_lookup():
    # CO uses the 298-1300 K range at 500 K.
    assert shomate_gas_by_name(500, "CO", [2170]) is not None
    # Outside all ranges -> None.
    assert shomate_gas_by_name(7000, "CO", [2170]) is None


def test_shomate_reference_consistency():
    # At the reference temperature the enthalpy part vanishes; only ZPE and
    # entropy terms remain, so the result is finite and small.
    value = shomate_gas(298.15, [25.56759, 6.096130, 4.054656, -2.671201,
                                 0.131021, -118.0089, 227.3665, -110.5271], [])
    assert np.isfinite(value)
