import pytest

import numpy as np
from scipy import signal

import filtering


def test_spatial_filter():
    """Test creation and frequency response of a latitude-dependent filter."""

    lats = np.array([1, 2])

    # construct filter
    f = lambda lon, lat: lat * 0.1
    filt = filtering.filter.SpatialFilter(f, 1, np.array([0]), lats)

    # compute frequency response of filter
    for lat, filter_obj in zip(lats, filt._filter):
        w, h = signal.freqz(*filter_obj)
        assert np.all(abs(h)[w < f(0, lat)] < 0.1)
