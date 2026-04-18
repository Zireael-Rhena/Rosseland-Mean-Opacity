"""
test_constants.py
-----------------
Sanity checks for CGS physical constants against NIST CODATA 2018.
"""

import math
import pytest
from hydrogen_opacity.constants import load_constants


@pytest.fixture(scope="module")
def const():
    return load_constants()


class TestConstantValues:
    def test_kB_value(self, const):
        assert abs(const.k_B - 1.380649e-16) / 1.380649e-16 < 1e-6

    def test_h_value(self, const):
        assert abs(const.h - 6.62607015e-27) / 6.62607015e-27 < 1e-6

    def test_hbar_equals_h_over_2pi(self, const):
        expected = const.h / (2.0 * math.pi)
        assert abs(const.hbar - expected) / expected < 1e-12

    def test_c_value(self, const):
        assert abs(const.c - 2.99792458e10) / 2.99792458e10 < 1e-9

    def test_m_e_value(self, const):
        assert abs(const.m_e - 9.1093837015e-28) / 9.1093837015e-28 < 1e-6

    def test_m_H_greater_than_proton_mass(self, const):
        # m_H = m_p + m_e  > 1.67e-24 g
        assert const.m_H > 1.67e-24

    def test_e_cgs_value(self, const):
        assert abs(const.e_cgs - 4.80320427e-10) / 4.80320427e-10 < 1e-6

    def test_sigma_T_value(self, const):
        assert abs(const.sigma_T - 6.6524587321e-25) / 6.6524587321e-25 < 1e-6

    def test_ev_to_erg_exact(self, const):
        # 1 eV = 1.602176634e-12 erg (exact since 2019 SI redefinition)
        assert abs(const.ev_to_erg - 1.602176634e-12) / 1.602176634e-12 < 1e-12

    def test_chi_H(self, const):
        assert const.chi_H_ev == pytest.approx(13.6, rel=1e-3)

    def test_chi_Hminus(self, const):
        assert const.chi_Hminus_ev == pytest.approx(0.754, rel=1e-4)

    def test_lambda0_Hminus_is_1p64_micron(self, const):
        assert const.lambda0_Hminus_micron == pytest.approx(1.64, rel=1e-6)

    def test_sigma_SB_value(self, const):
        assert abs(const.sigma_SB - 5.670374419e-5) / 5.670374419e-5 < 1e-6


class TestConstantRelations:
    def test_hbar_positive(self, const):
        assert const.hbar > 0.0

    def test_all_positive(self, const):
        for field_name in [
            "k_B", "h", "hbar", "c", "m_e", "m_H", "e_cgs",
            "sigma_T", "ev_to_erg", "sigma_SB",
        ]:
            val = getattr(const, field_name)
            assert val > 0.0, f"{field_name} not positive"

    def test_m_H_m_e_ordering(self, const):
        # Proton much heavier than electron
        assert const.m_H / const.m_e > 1800

    def test_hbar_less_than_h(self, const):
        assert const.hbar < const.h
