# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)

import scipp as sc
import numpy as np


def filter_bad_pulses(da: sc.DataArray,
                      proton_charge: sc.DataArray,
                      minimum_threshold: float = .95,
                      maximum_threshold: float = 1.0,
                      data_time_coord: str = "pulse_time"):
    """
    Filters out any bad pulses based on an attribute named "proton_charge" if the
     "proton_charge" value is less than 95% of the mean (by default, the percentage can
     be changed using "minimum_threshold") it will be filtered out, if a proton charge
     is 10% higher than the maximum (This won't actually make any difference in
     comparison to just the max value) based on how Mantid did filtering of bad pulses.

    :param da: sc.DataArray, the data array that is to be filtered.

    :param proton_charge: sc.DataArray, the DataArray that contains the proton charges,
     e.g. da.attrs["proton_charge"].value. Expected to have

    :param minimum_threshold: float, the minimum threshold is a percentage used to
     calculate the bad proton_charge values. It is used in this equation, (min =
     mean(data) * minimum_threshold), the min is then used for filtering out what is
     thought to be bad proton_charge values as a minimum in the filter.

    :param maximum_threshold: float, the maximum threshold is a percentage used to
     calculate the bad proton_charge values. It is used in this equation,
     (max = max(data) * maximum_threshold), the max is then used for filtering out
     what is though to be bad proton_charge values as a minimum in the filter.

    :param data_time_coord: str, the name of the time coord in your DataArray, da,
     defaults to "pulse_time"

    :return: A DataArray that contains a dimension called "pulse_slices" in which "good"
     data in slice 1, and "bad" data in slice "0". To access the good data you would
     slice this return result like this: da["pulse_slices", 1].
    """
    # Auto find the time dim name from attribute, assume 1D DataArray
    proton_charge_time_dim_initial_name = proton_charge.dims[0]
    proton_charge = proton_charge \
        .rename_dims({proton_charge_time_dim_initial_name: data_time_coord})
    proton_charge.coords[data_time_coord] = \
        proton_charge.coords.pop(proton_charge_time_dim_initial_name)

    max_charge = sc.max(proton_charge.data)
    max_charge *= maximum_threshold
    min_charge = sc.mean(proton_charge.data)
    min_charge *= minimum_threshold

    good_pulse = (proton_charge >= min_charge) & (proton_charge < max_charge)

    return filter(da=da, condition_intervals=good_pulse, dim="pulse_slices")


def filter_by_value(da: sc.DataArray, min_value: sc.Variable, max_value: sc.Variable):
    """
    Return all data points between a minimum and maximum values of da in a DataArray

    :param da: sc.DataArray, the sc.DataArray that is being filtered

    :param min_value: sc.Variable, the lower bound of the filtering for the attribute

    :param max_value: sc.Variable, the upper bound of the filtering for the attribute

    :return: sc.DataArray or sc.Variable, whichever the attribute is in the original da
     that is passed.
    """
    filter_coord_name = '_'.join(da.coords)
    da.coords[filter_coord_name] = da.data
    data_in_range = da.groupby(filter_coord_name,
                               bins=sc.concatenate(min_value, max_value,
                                                   filter_coord_name)).copy(0)
    del data_in_range.coords[filter_coord_name]
    del da.coords[filter_coord_name]
    return data_in_range


def _find_edges(condition_intervals):
    mapping_coord = condition_intervals.dims[0]
    group_data_coord = condition_intervals.coords[mapping_coord]
    edge = condition_intervals.data
    edge = edge[mapping_coord, :-1] ^ edge[mapping_coord, 1:]
    edges = sc.Dataset(data={
        mapping_coord: group_data_coord[mapping_coord, 1:],
        'good': condition_intervals.data[mapping_coord, :-1]
    },
                       coords={'edge': edge})
    edges = edges.groupby(group='edge').copy(True)
    data = sc.DataArray(data=edges['good'].data
                        ^ condition_intervals[mapping_coord, 0].data,
                        coords={mapping_coord: edges[mapping_coord].data})
    data = sc.concatenate(condition_intervals[mapping_coord, 0], data, mapping_coord)
    data = sc.concatenate(data, condition_intervals[mapping_coord, -1], mapping_coord)
    return data


def filter(da: sc.DataArray, condition_intervals: sc.DataArray, dim: str):
    """
    Filter a data array based on passed condition_intervals. Example input:
     filter(da, ((temp >= min_temp) & (temp < max_temp)), "dim") where temp is the a
     sc.Variable or sc.DataArray.

    :param da: sc.DataArray, the data that is being filtered
    :param condition_intervals: sc.DataArray, a lookup table for each event time, each
     event time coordinate should have true or false as the corresponding data of the
     DataArray, to determine whether or not it should be filtered into slice 0 (False),
     or 1 (True).
    :param dim: str, the name of the coordinate, for which the slices will be
     attributed.
    :return: The original da, with  slices of given dim, sliced based on the event bands
     in condition_intervals. Use this to access the results of your filter.
    """
    edges = _find_edges(condition_intervals)
    coord_name = edges.dims[-1]
    da.bins.coords[dim] = sc.lookup(edges, coord_name)[da.bins.coords[coord_name]]

    unique_values = np.unique(edges.data.values)
    groups = sc.array(dims=[dim], values=unique_values)
    grouped = sc.bin(da, groups=[groups])

    return grouped
