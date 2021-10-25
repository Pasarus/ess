# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)

import scipp as sc


def filter_bad_pulses(da: sc.DataArray,
                      proton_charge: sc.DataArray,
                      center: bool = False,
                      minimum_threshold: float = .95,
                      maximum_threshold: float = 1.1,
                      data_time_coord: str = "pulse_time"):
    """
    Filters out any bad pulses based on an attribute named "proton_charge" if the
     "proton_charge" value is less than 95% of the mean (by default, the percentage can
     be changed using "minimum_threshold") it will be filtered out, if a proton charge
     is 10% higher than the maximum (This won't actually make any difference in
     comparison to just the max value) based on how Mantid did filtering of bad pulses.

    :param da: sc.DataArray, the data array that is to be filtered.

    :param proton_charge: sc.DataArray, the dataarray that contains the proton charges,
     e.g. da.attrs["proton_charge"].value. Expected to have

    :param center: bool, whether or not to center the data of the "proton_charge" as if
     it were binned, this is done with SNS data.

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

    if center:
        proton_charge.data.values[:-1] = 0.5 * (proton_charge.data.values[1:] +
                                                proton_charge.data.values[:-1])

    max_charge = sc.max(proton_charge.data)
    max_charge *= maximum_threshold
    min_charge = sc.mean(proton_charge.data)
    min_charge *= minimum_threshold

    good_pulse = (proton_charge >= min_charge) & (proton_charge < max_charge)

    return filter_by_attribute_edges(da=da,
                                     good_data_lut=good_pulse,
                                     data_slices_name="pulse_slices")


def filter_by_value(da: sc.DataArray, min_value: sc.Variable, max_value: sc.Variable):
    """
    Return all data points between a minimum and maximum values of da in a DataArray

    :param da: sc.DataArray, the sc.DataArray that is being filtered

    :param min_value: sc.Variable, the lower bound of the filtering for the attribute

    :param max_value: sc.Variable, the upper bound of the filtering for the attribute

    :return: sc.DataArray or sc.Variable, whichever the attribute is in the original da
     that is passed.
    """
    # Define what data is in range
    greater_than = da >= min_value
    less_than = da < max_value
    in_range = greater_than == less_than

    # Group up data based on the in range data that is present and clean up groupby
    # coordinate
    filter_coord_name = '_'.join(da.coords)
    da.coords[filter_coord_name] = in_range.data
    data_in_range = da.groupby(filter_coord_name).copy(True)
    del data_in_range.coords[filter_coord_name]
    del da.coords[filter_coord_name]

    return data_in_range


def _find_edges_based_on_lut(good_data_lut, mapping_coord):
    good_data_coord = good_data_lut.coords[mapping_coord]
    edge = good_data_lut.data
    edge = edge[mapping_coord, :-1] ^ edge[mapping_coord, 1:]
    good_edges = sc.Dataset(data={
        mapping_coord: good_data_coord[mapping_coord, 1:],
        'good': good_data_lut.data[mapping_coord, :-1]
    },
                            coords={'edge': edge})
    return good_edges.groupby(group='edge').copy(True)


def _find_da_coords_for_slicing(da, good_data_lut):
    dim = good_data_lut.dims[-1]
    good_edges = _find_edges_based_on_lut(good_data_lut, dim)
    good_data = sc.DataArray(data=good_edges['good'].data ^ good_data_lut[dim, 0].data,
                             coords={dim: good_edges[dim].data})
    good_data = sc.concatenate(good_data_lut[dim, 0], good_data, dim)
    good_data = sc.concatenate(good_data, good_data_lut[dim, -1], dim)
    good_pulse_lut = sc.lookup(good_data, dim)

    return good_pulse_lut[da.bins.coords[dim]]


def filter_by_attribute_edges(da: sc.DataArray,
                              good_data_lut,
                              data_slices_name: str = "slice_data"):
    """
    Filter the data based on the passed good_data_lut, output the passed DataArray with
     a new dimension called the string passed to data_slices_name, where the data in
     slice 0, is "bad" data and the data in slice 1 is the "good" data.

    :param da: sc.DataArray, the sc.DataArray that is being filtered

    :param good_data_lut: Any, a definition of what is good data, using the boolean
      operators on a set of variables. For example this is achieved in
      "filter_bad_pulses" by (proton_charge >= min_charge) & (proton_charge <
      max_charge) where proton_charge is the attribute (sc.Variable) being filtered.

    :param data_slices_name: str, The name of the dimension that will be added to the
      passed sc.DataArray, da, that can then be sliced later for good and bad data.

    :return: The originally passed sc.DataArray, da, with an extra dimension that can
      then be sliced for good data on slice 1, and sliced for bad data on slice 0. For
      example, da["slice_data", 1] for good data.
    """
    da.bins.coords[data_slices_name] = _find_da_coords_for_slicing(da, good_data_lut)

    groups = sc.array(dims=[data_slices_name], values=[False, True])
    grouped = sc.bin(da, groups=[groups])
    return grouped
