import numpy as np
import scipy.constants


class ExpModel:

    def __init__(
            self,
            bz_amp=1e-6,
            count_rate=30e3, odmr_contrast=0.3,
            pwr_to_rabi_freq=1e6,
            r1=30, r2=50, r3=20, x0=100, y0=100
    ):

        # Profile and magnitude of Bz field pattern
        # - Center coordinates
        self._x0 = x0
        self._y0 = y0

        # - Inner and outer radii of the zero region, decay length
        self._r1 = r1
        self._r2 = r2
        self._r3 = r3

        # - Bz field value
        self._bz_amp = bz_amp
        self._bz = None

        # Current position of confocal scanner
        self._x_loc = None
        self._y_loc = None

        # Current mw source settings
        self._mw_freq = None
        self._mw_pwr = None

        # - conversion of power to Rabi frequency
        self._pwr_to_rabi_freq = pwr_to_rabi_freq
        self._rabi_freq = None

        # ODMR parameters
        self._count_rate = count_rate
        self._odmr_contrast = odmr_contrast

    # Confocal scanner interface methods

    def scan_set_loc(self, x, y):
        self._x_loc = x
        self._y_loc = y

        self._bz = self._bz_amp * self._bz_profile(
            r=np.sqrt(
                (x - self._x0)**2 + (y - self._y0)**2
            ),
            r1=self._r1,
            r2=self._r2,
            r3=self._r3
        )

        return 0

    def scan_get_loc(self):
        return self._x_loc, self._y_loc

    # MW source interface methods

    def mw_set_pwr(self, pwr):
        self._mw_pwr = pwr
        self._rabi_freq = self._pwr_to_rabi_freq * np.sqrt(pwr)

        return 0

    def mw_get_pwr(self):
        return self._mw_pwr

    def mw_set_freq(self, freq):
        self._mw_freq = freq

        return 0

    def mw_get_freq(self):
        return self._mw_freq

    # Photon counter interface methods

    def get_count(self, tau=1e-3):

        # Central frequency
        freq_0 = 2.87e9

        # Calculate line shift
        mu_b = scipy.constants.physical_constants['Bohr magneton'][0]
        h = scipy.constants.h

        freq_shift = mu_b * self._bz / h

        # Calculate current count rate
        count_rate_factor = self._double_dip(
            x=self._mw_freq,
            x1=freq_0 - freq_shift,
            x2=freq_0 + freq_shift,
            a=self._rabi_freq,
            alpha=self._odmr_contrast
        )
        count_rate = self._count_rate * count_rate_factor

        # Generate random count number
        count_number = np.random.poisson(
            lam=count_rate * tau
        )

        return count_number

    # Technical methods

    @staticmethod
    def _bz_profile(r, r1, r2, r3):

        if r <= r1:
            return np.sqrt(
                1 - (r / r1) ** 2
            )

        elif r <= r2:
            return 0

        else:
            return -(r3 / r)**3 * np.sqrt(r - r2)

    @staticmethod
    def _double_dip(x, x1, x2, a, alpha):

        def lorentz_peak(x):
            return 1 / (1 + x ** 2)

        peak_1 = lorentz_peak(
            (x - x1) / a
        )

        peak_2 = lorentz_peak(
            (x - x2) / a
        )

        return 1 - alpha * (peak_1 + peak_2)
