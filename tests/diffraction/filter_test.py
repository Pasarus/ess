# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
import numpy as np
import scipp as sc
import scippneutron as scn

from ess.diffraction.filter import filter_bad_pulses, filter_by_value, filter


def _test_filter_sums_for_bad_pulses(filtered_da, bad_sum, good_sum):
    good_filtered_da = filtered_da["pulse_slices", 1]
    bad_filtered_da = filtered_da["pulse_slices", 0]
    good_da_sum = sc.sum(good_filtered_da.bins.sum()).value
    bad_da_sum = sc.sum(bad_filtered_da.bins.sum()).value

    np.testing.assert_almost_equal(good_da_sum, good_sum)
    np.testing.assert_almost_equal(bad_da_sum, bad_sum)


def test_results_of_bad_pulse_defaults():
    da = scn.data.tutorial_event_data()
    proton_charge = da.attrs["proton_charge"].value

    filtered_da = filter_bad_pulses(da, proton_charge)

    _test_filter_sums_for_bad_pulses(filtered_da, 51821.69921875, 6282503.0)


def test_results_of_bad_pulse_min_threshold():
    da = scn.data.tutorial_event_data()
    proton_charge = da.attrs["proton_charge"].value

    filtered_da = filter_bad_pulses(da, proton_charge, minimum_threshold=.5)

    _test_filter_sums_for_bad_pulses(filtered_da, 23194.6582031, 6311124.5)


def test_results_of_bad_pulse_max_threshold():
    da = scn.data.tutorial_event_data()
    proton_charge = da.attrs["proton_charge"].value

    filtered_da = filter_bad_pulses(da, proton_charge, maximum_threshold=.975)

    _test_filter_sums_for_bad_pulses(filtered_da, 2562825.75, 3771492.0)


def test_result_of_filter_attribute_by_value():
    da = scn.data.tutorial_event_data()
    da_to_filter = da.attrs["SampleTemp"].value

    filter_result = filter_by_value(da_to_filter, sc.scalar(299.), sc.scalar(300.))

    assert len(filter_result) == 202
    np.testing.assert_almost_equal(sc.sum(filter_result).value, 60591.3738098)
    assert sc.min(filter_result.data).value >= 299.
    assert sc.max(filter_result.data).value <= 300.

    # Assert the cleanup occured, and only time is left in original da, and result.
    assert len(da_to_filter.coords) == 0
    assert len(filter_result.coords) == 0


def test_result_of_filter_data_array_on_attribute():
    da = scn.data.tutorial_event_data()

    max_temp = sc.scalar(300.)
    min_temp = sc.scalar(299.)
    sample_temp = da.attrs["SampleTemp"].value

    # Rename sampletemp time dimension so mapping is possible
    sample_temp = sample_temp.rename_dims({'time': 'pulse_time'})
    sample_temp.coords['pulse_time'] = sample_temp.coords.pop('time')

    good_data_lut = (sample_temp >= min_temp) & (sample_temp <= max_temp)
    filter_result = filter(da, good_data_lut, "slice_data")

    inside_temp_range = filter_result["slice_data", 1].copy()
    outside_temp_range = filter_result["slice_data", 0].copy()

    inside_temp_range_sum = sc.sum(
        sc.sum(inside_temp_range.astype("float64").bins.sum())).data.value
    outside_temp_range_sum = sc.sum(
        sc.sum(outside_temp_range.astype("float64").bins.sum())).data.value
    assert len(inside_temp_range.data) == 10694
    assert len(outside_temp_range.data) == 10694
    # Caution if not float64 converted before summation, a rounding error of 4.75 will
    # occur and impact your results, whilst the filter performed fine.
    assert inside_temp_range_sum + outside_temp_range_sum == \
           sc.sum(da.astype("float64").bins.sum()).value
