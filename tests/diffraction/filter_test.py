# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
import numpy as np
import scipp as sc
import scippneutron as scn

import pytest

from ess.diffraction.filter import filter_bad_pulses, filter_by_time, filter_attribute_by_time


def _test_filter_sums_for_bad_pulses(filtered_da, bad_sum, good_sum):
    good_filtered_da = filtered_da["pulse_slices", 1]
    bad_filtered_da = filtered_da["pulse_slices", 0]
    good_da_sum = sc.sum(good_filtered_da.bins.sum()).value
    bad_da_sum = sc.sum(bad_filtered_da.bins.sum()).value

    np.testing.assert_almost_equal(good_da_sum, good_sum)
    np.testing.assert_almost_equal(bad_da_sum, bad_sum)


def test_results_of_bad_pulse_defaults():
    da = scn.data.tutorial_event_data()

    filtered_da = filter_bad_pulses(da)

    _test_filter_sums_for_bad_pulses(filtered_da, 51799.703125, 6282525.0)


def test_results_of_bad_pulse_centred():
    da = scn.data.tutorial_event_data()

    filtered_da = filter_bad_pulses(da, True)

    _test_filter_sums_for_bad_pulses(filtered_da, 120990.6640625, 6213332.0)


def test_results_of_bad_pulse_left():
    da = scn.data.tutorial_event_data()

    filtered_da = filter_bad_pulses(da, False)

    _test_filter_sums_for_bad_pulses(filtered_da, 51799.703125, 6282525.0)


def test_results_of_bad_pulse_min_threshold():
    da = scn.data.tutorial_event_data()

    filtered_da = filter_bad_pulses(da, minimum_threshold=.5)

    _test_filter_sums_for_bad_pulses(filtered_da, 23172.6582031, 6311146.5)


def test_result_of_filter_by_time_first_half():
    da = scn.data.tutorial_event_data()

    # Run scipp filter by time
    run_start = sc.scalar(value=np.datetime64(da.attrs["run_start"].value).astype("datetime64[ns]"))
    run_end = run_start + sc.to_unit(
        sc.scalar(value=da.attrs["duration"].value * 0.5, dtype=sc.dtype.int64, unit=sc.units.s),
        unit="ns")
    filtered_da = filter_by_time(da.copy(), run_start, run_end)
    da_sum = sc.sum(filtered_da.bins.sum())

    assert da_sum.value == 3317887.25


def test_result_of_filter_by_time():
    da = scn.data.tutorial_event_data()

    # Run scipp filter by time
    run_start = sc.scalar(value=np.datetime64(da.attrs["run_start"].value).astype("datetime64[ns]"))
    half_duration = sc.to_unit(sc.scalar(value=da.attrs["duration"].value * 0.5, dtype=sc.dtype.int64, unit=sc.units.s),
                               unit="ns")
    run_end = run_start + half_duration
    filtered_da = filter_by_time(da.copy(), run_start, run_end)
    da_sum = sc.sum(filtered_da.bins.sum())

    assert da_sum.value == 3317887.25


def test_result_of_filter_by_time_different_event_time_coord():
    da = scn.data.tutorial_event_data()

    # Rename time in the data
    new_dim_name = "time"
    da.bins.coords[new_dim_name] = da.bins.coords.pop('pulse_time')

    # Run scipp filter by time
    run_start = sc.scalar(value=np.datetime64(da.attrs["run_start"].value).astype("datetime64[ns]"))
    half_duration = sc.to_unit(sc.scalar(value=da.attrs["duration"].value * 0.5, dtype=sc.dtype.int64, unit=sc.units.s),
                               unit="ns")
    run_end = run_start + half_duration
    filtered_da = filter_by_time(da.copy(), run_start, run_end, new_dim_name)
    da_sum = sc.sum(filtered_da.bins.sum())

    assert da_sum.value == 3317887.25


def test_result_of_filter_by_time_incorrect_event_time_coord():
    da = scn.data.tutorial_event_data()

    # Run scipp filter by time
    run_start = sc.scalar(value=np.datetime64(da.attrs["run_start"].value).astype("datetime64[ns]"))
    half_duration = sc.to_unit(
        sc.scalar(value=da.attrs["duration"].value * 0.5, dtype=sc.dtype.int64, unit=sc.units.s),
        unit="ns")
    run_end = run_start + half_duration
    with pytest.raises(sc.core.NotFoundError):
        filter_by_time(da.copy(), run_start, run_end, "this_is_not_a_valid_time_coord_for_the_event_data")


def test_result_of_filter_by_time_different_unit_variable_times():
    da = scn.data.tutorial_event_data()

    # Change unit to seconds in the event data
    da.events.coords["pulse_time"] = sc.to_unit(da.events.coords["pulse_time"], unit="s")

    # Run scipp filter by time
    run_start = sc.scalar(value=np.datetime64(da.attrs["run_start"].value).astype("datetime64[s]"))
    half_duration = sc.to_unit(
        sc.scalar(value=da.attrs["duration"].value * 0.5, dtype=sc.dtype.int64, unit=sc.units.s),
        unit="s")
    run_end = run_start + half_duration
    filtered_da = filter_by_time(da.copy(), run_start, run_end, datetime_unit="s")
    da_sum = sc.sum(filtered_da.bins.sum())

    # Expect a different result as we are filtering out more due to the reduced pulse_time resolution
    assert da_sum.value == 3317864.75


def test_result_of_filter_by_time_string_times():
    da = scn.data.tutorial_event_data()

    # Run scipp filter by time
    filtered_da = filter_by_time(da.copy(), "2011-08-12T15:50:17", "2011-08-12T17:22:06")
    da_sum = sc.sum(filtered_da.bins.sum())

    # Should just be everything given it is passed the start - 1 second and end + 1 second
    assert da_sum.value == 6334321.5
    assert da_sum.value == sc.sum(da.bins.sum()).value


def test_result_of_filter_by_time_incorrect_unit_string_times():
    da = scn.data.tutorial_event_data()

    # Run scipp filter by time
    with pytest.raises(sc.core.UnitError):
        filter_by_time(da.copy(), "2011-08-12T15:50:17", "2011-08-12T17:22:05", datetime_unit="s")


def test_result_of_filter_attribute_by_time_variable():
    da = scn.data.tutorial_event_data()

    run_start = sc.scalar(value=np.datetime64(da.attrs["run_start"].value).astype("datetime64[ns]"))
    half_duration = sc.to_unit(
        sc.scalar(value=da.attrs["duration"].value * 0.5, dtype=sc.dtype.int64, unit=sc.units.s),
        unit="ns")
    run_end = run_start + half_duration

    filter_result_first_half = filter_attribute_by_time(da, "SampleTemp", run_start, run_end)

    assert len(filter_result_first_half) == 257
    np.testing.assert_almost_equal(sc.sum(filter_result_first_half).value, 77123.02212524414)


def test_result_of_filter_attribute_by_time_string():
    da = scn.data.tutorial_event_data()

    filter_result_full = filter_attribute_by_time(da, "SampleTemp", da.attrs["run_start"].value,
                                                  da.attrs["end_time"].value)

    assert len(filter_result_full) == 465
    np.testing.assert_almost_equal(sc.sum(filter_result_full).value, 139523.16213989258)


def test_result_of_filter_attribute_by_time_none_default_time():
    da = scn.data.tutorial_event_data()

    # Change SampleTemp time dimension name
    new_dim_time = "new_time"
    da.attrs["SampleTemp"].value.coords[new_dim_time] = da.attrs["SampleTemp"].value.coords.pop("time")

    filter_result_full = filter_attribute_by_time(da, "SampleTemp", da.attrs["run_start"].value,
                                                  da.attrs["end_time"].value, time_name=new_dim_time)

    assert len(filter_result_full) == 465
    np.testing.assert_almost_equal(sc.sum(filter_result_full).value, 139523.16213989258)




# # Equivalent to ensuring that FilterLogByTime is possible
# @with_mantid_only
# def test_result_of_filter_attribute_by_time():
#     from mantid.simpleapi import FilterLogByTime, Load
#
#     filename = "PG3_4866_event.nxs"
#     ws = Load(filename)
#     da = scn.from_mantid(ws)
#
#     scipp_run_start = sc.scalar(value=np.datetime64(da.attrs["run_start"].value).astype("datetime64[ns]"))
#     scipp_run_end = scipp_run_start + sc.to_unit(
#         sc.scalar(value=da.attrs["duration"].value * 0.5, dtype=sc.dtype.int64, unit=sc.units.s),
#         unit="ns")
#     da_sum = _filter_attribute_by_time(da, "SampleTemp", scipp_run_start, scipp_run_end)
#     scipp_mean = sc.mean(da_sum.data)
#
#     mantid_end = da.attrs["duration"].value * 0.5
#     mantid_results = FilterLogByTime(InputWorkspace="ws", LogName="SampleTemp", StartTime=0, EndTime=mantid_end)
#     mantid_mean = np.mean(np.mean(mantid_results))
#
#     np.testing.assert_almost_equal(scipp_mean.value, mantid_mean)
#
#
# # Equivalent to ensuring that it's possible to filter a log by value
# @with_mantid_only
# def test_result_of_filter_attribute_by_value():
#     from mantid.simpleapi import Load
#
#     filename = "PG3_4866_event.nxs"
#     ws = Load(filename)
#     da = scn.from_mantid(ws)
#
#     scipp_result = _filter_attribute_by_value(da, "SampleTemp", sc.scalar(299.), sc.scalar(300.))
#
#     assert len(scipp_result.data.values) == 467
#     np.testing.assert_almost_equal(sc.mean(scipp_result.data).value, 299.9960808)


# Equivalent to ensuring that

# @with_mantid_only
# def test_result_of_bad_pulse_performance_mantid_comparison():
#     from mantid.simpleapi import FilterBadPulses
#
#     scipp_centre = bench_scipp_filter(filter_bad_pulses, filename="PG3_4866_event.nxs", repeat_times=5, args=True)
#     scipp_left = bench_scipp_filter(filter_bad_pulses, filename="PG3_4866_event.nxs", repeat_times=5)
#     mantid_centre = bench_mantid_filter(FilterBadPulses, filename="PG3_4866_event.nxs", repeat_times=5)
#
#     average_scipp = (float(scipp_left) + float(scipp_centre)) / 2.
#
#     # Assert that Mantid is no more than 3 times faster than the average of a Scipp equivalent operation for filtering
#     # bad pulses, this is for stability as we expect no more than 2 times as more Scipp operations need to occur.
#     assert mantid_centre / average_scipp <= 3


# def _filter_by_time(da, start, end):
#     edges = sc.concatenate(start, end, 'pulse_time')
#     return sc.bin(da, edges=[edges])



