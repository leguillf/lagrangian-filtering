"""Inertial filter objects.

This definition allows for the definition of inertial filters. These
may be as simple as constant frequency, or may vary depending on
latitude or even arbitrary conditions like vorticity.

"""

import dask.array as da
from scipy import signal


class Filter(object):
    """The base class for inertial filters.

    This holds the filter state, and provides an interface for
    applying the filter to advected particle data.

    Args:
        frequency (float): The high-pass cutoff frequency of the filter.
        fs (float): The sampling frequency of the data over which the
            filter is applied.

    """

    def __init__(self, frequency, fs):
        self._filter = signal.butter(4, frequency, "highpass", fs=fs)

    def apply_filter(self, data, time_index):
        """Apply the filter to an array of data.

        Args:
            data: An array of (time x particle) of advected particle data.
                This can be a dask array of lazily-loaded temporary data.
            time_index (int): The index along the time dimension corresponding
                to the central point, to extract after filtering.

        Returns:
            An array of (particle) of the filtered particle data, restricted
            to the specified time index.

        """

        def filter_select(x):
            return signal.filtfilt(*self._filter, x)[..., time_index]

        # apply scipy filter as a ufunc
        # mapping an array to scalar over the first axis, automatically vectorize execution
        # and allow rechunking (since we have a chunk boundary across the first axis)
        filtered = da.apply_gufunc(
            filter_select,
            "(i)->()",
            data,
            axis=0,
            output_dtypes=data.dtype,
            allow_rechunk=True,
        )

        return filtered.compute()
