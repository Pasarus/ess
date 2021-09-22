# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)

import scipp as sc
import scippneutron as scn

from benchmarking.bench_function import bench_scipp_filter, bench_mantid_filter
from ess.diffraction.filter import filter_bad_pulses
from tests.utils import with_mantid_only


@with_mantid_only
def test_result_of_bad_pulse():
    from mantid.simpleapi import Load, FilterBadPulses

    # Example script
    filename = "PG3_4866_event.nxs"
    ws = Load(filename)
    ds = scn.from_mantid(ws)

    # Perform scipp filter
    scipp_centre_filtered_ds = filter_bad_pulses(ds.copy(), True)["good_pulse", 1]
    scipp_centre_sum = sc.sum(scipp_centre_filtered_ds.bins.sum())
    scipp_left_filtered_ds = filter_bad_pulses(ds.copy(), False)["good_pulse", 1]
    scipp_left_sum = sc.sum(scipp_left_filtered_ds.bins.sum())

    # Perform mantid filter
    ws = FilterBadPulses(InputWorkspace=ws)
    mantid_filtered_ds = scn.from_mantid(ws)
    mantid_sum = sc.sum(mantid_filtered_ds.bins.sum())

    scipp_centre_diff = abs(scipp_centre_sum.value - mantid_sum.value)
    scipp_left_diff = abs(scipp_left_sum.value - mantid_sum.value)

    assert scipp_centre_diff == 524.0
    assert scipp_left_diff == 619484.0


@with_mantid_only
def test_result_of_bad_pulse_performance_mantid_comparison():
    from mantid.simpleapi import FilterBadPulses

    scipp_centre = bench_scipp_filter(filter_bad_pulses, filename="PG3_4866_event.nxs", repeat_times=5, args=True)
    scipp_left = bench_scipp_filter(filter_bad_pulses, filename="PG3_4866_event.nxs", repeat_times=5)
    mantid_centre = bench_mantid_filter(FilterBadPulses, filename="PG3_4866_event.nxs", repeat_times=5)

    average_scipp = (float(scipp_left) + float(scipp_centre)) / 2.

    # Assert that Mantid is no more than 3 times faster than the average of a Scipp equivalent operation for filtering
    # bad pulses, this is for stability as we expect no more than 2 times as more Scipp operations need to occur.
    assert mantid_centre / average_scipp <= 3


