# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)

import scipp as sc
import string
from random import random

import numpy as np


def filter_bad_pulses(da: sc.DataArray, centre: bool = False, minimum_threshold: float = .95):
    """
    Filters out any bad pulses based on an attribute named "proton_charge" if the "proton_charge" value is less than 95%
     of the mean (by default, the percentage can be changed using "minimum_threshold") it will be filtered out, if it is
     10% higher than the maximum (This won't actually make any difference in comparison to just the max value) based on
     how Mantid did filtering of bad pulses.

    Args:
        da: sc.DataArray, the data array that is to be filtered.

        centre: bool, whether or not to centre the data of the "proton_charge" as if it were binned, this is done with
         SNS data.

        minimum_threshold: float, the minimum threshold is a percentage used calculate at what point is determined as a
         bad proton charge, this value is multiplied by the mean value of the proton_charge data, and that is the minimum
         by which filtering will occur.

    Returns:
    A DataArray that contains a dimension called "pulse_slices" in which "good" data in slice 1, and "bad" data in slice
    "0". To access the good data you would slice this return result like this: da["pulse_slices", 1].
    """
    proton_charge = da.attrs['proton_charge'].value
    proton_charge = proton_charge.rename_dims({'time': 'pulse_time'})
    proton_charge.coords['pulse_time'] = proton_charge.coords.pop('time')

    if centre:
        proton_charge.data.values[:-1] = 0.5 * (proton_charge.data.values[1:] + proton_charge.data.values[:-1])

    max_charge = sc.max(proton_charge.data)
    max_charge *= 1.1  # determine true max (max_value * 1.1)
    min_charge = sc.mean(proton_charge.data)
    min_charge *= minimum_threshold  # determine true min (mean_value * .01)

    good_pulse = (proton_charge >= min_charge) & (proton_charge < max_charge)

    return filter_data_array_on_attribute(da=da, good_data_lut=good_pulse, mapping_coord="pulse_time",
                                          data_slices_name="pulse_slices")


def _input_processing_time_conversion(start, end, datetime_unit):
    if not isinstance(start, sc.Variable):
        start = sc.scalar(value=np.datetime64(start).astype(f"datetime64[{datetime_unit}]"))
    if not isinstance(end, sc.Variable):
        end = sc.scalar(value=np.datetime64(end).astype(f"datetime64[{datetime_unit}]"))
    return start, end


def filter_by_time(da: sc.DataArray, start, end, time_dim_name: str = "pulse_time", datetime_unit: str = "ns"):
    """
    Snip out or slice out a part of a DataArray of binned events, based on time, given start and end

    Args:
        da: sc.DataArray, the sc.DataArray that is being filtered

        start: Any convertible to a datetime64 Scalar variable e.g. str or sc.Variable, input using absolute time format
         similar to Mantid, e.g. 2011-08-13T17:11:30 or any other format that can be converted into a numpy datetime64
         variable. The start of the time_dim_name dimension, in the output

        end: Any, input format the same as start. The end of the time_dim_name dimension in the output

        time_dim_name: str, The time dimension name, defaults to "pulse_time"

        datetime_unit: str, The time unit used in the time_dim_name dimension, e.g. nanoseconds = "ns" or seconds = "s",
          defaults to "ns". Only needed if you pass not already constructed start or end i.e. a string instead of a
          Scalar variable containing datetime64.

    Returns:
        The output of the sc.bin after filtering out everything between the start and end args from the input da
    """
    start, end = _input_processing_time_conversion(start, end, datetime_unit)
    edges = sc.concatenate(start, end, time_dim_name)
    return sc.bin(da, edges=[edges])


# Example of how you would filter an attribute by time on a given DataArray
def filter_attribute_by_time(da: sc.DataArray, attribute_name: str, start, end, time_name: str = "time",
                             datetime_unit: str = "ns"):
    """
    Return all data points between a minimum and maximum time of an attribute in a DataArray

    Args:
        da: sc.DataArray, the sc.DataArray that is being filtered

        attribute_name: str, The name of the attribute to be filtered from the DataArray da.

        start: Any convertible to a datetime64 Scalar variable e.g. str or sc.Variable, input using absolute time format
         similar to Mantid, e.g. 2011-08-13T17:11:30 or any other format that can be converted into a numpy datetime64
         variable. The start of the time_dim_name dimension, in the output

        end: Any, input format the same as start. The end of the time_dim_name dimension in the output

        time_name: str, The name of the time coordinate in the attribute, defaults to "time"

        datetime_unit: str, The time unit used in the time_dim_name dimension, e.g. nanoseconds = "ns" or seconds = "s",
          defaults to "ns". Only needed if you pass not already constructed start or end i.e. a string instead of a
          Scalar variable containing datetime64.

    Returns:
        sc.DataArray or sc.Variable, whichever the attribute is in the original da that is passed.
    """
    start, end = _input_processing_time_conversion(start, end, datetime_unit)
    attribute_data = da.attrs[attribute_name].value
    return attribute_data[time_name, start:end]


def _generate_random_string(length: int = 20):
    charecters = string.ascii_letters + string.digits + string.punctuation
    return "".join(random.choice(charecters) for ii in range(length))


def filter_attribute_by_value(da: sc.DataArray, attribute_name: str, min_value: sc.Variable, max_value: sc.Variable):
    """
    Return all data points between a minimum and maximum values of an attribute in a DataArray

    Args:
        da: sc.DataArray, the sc.DataArray that is being filtered

        attribute_name: str, The name of the attribute to be filtered from the DataArray da.

        min_value: sc.Variable, the lower bound of the filtering for the attribute

        max_value: sc.Variable, the upper bound of the filtering for the attribute

    Returns:
         sc.DataArray or sc.Variable, whichever the attribute is in the original da that is passed.
    """
    # Define what data is in range
    attribute = da.attrs[attribute_name].value
    greater_than = attribute > min_value
    less_than = attribute < max_value
    in_range = greater_than == less_than

    # Group up data based on the in range data that is present and clean up groupby coordinate
    filter_coord_name = _generate_random_string()
    attribute.coords[filter_coord_name] = in_range.data
    data_in_range = attribute.groupby(filter_coord_name).copy(1)
    del data_in_range.coords[filter_coord_name]
    del attribute.coords[filter_coord_name]

    return data_in_range


def filter_data_array_on_attribute(da: sc.DataArray, good_data_lut, mapping_coord: str,
                                   data_slices_name: str = "slice_data"):
    """
    Filter the data based on the passed good_data_lut, output the passed DataArray with a new dimension called the
    string passed to data_slices_name, where the data in slice 0, is "bad" data and the data in slice 1 is the "good"
    data.

    Args:
        da: sc.DataArray, the sc.DataArray that is being filtered

        good_data_lut: Any, a definition of what is good data, using the boolean operators on a set of variables. For
         example this is achieved in "filter_bad_pulses" by (proton_charge >= min_charge) & (proton_charge < max_charge)
         where proton_charge is the attribute (sc.Variable) being filtered.

        mapping_coord: str, The coordinate that connects the attribute, and the data together, this is often the,
         pulse_time, you may need to rename the attribute's coordinate to support this.

        data_slices_name: str, The name of the dimension that will be added to the passed sc.DataArray, da, that can
         then be sliced later for good and bad data.

    Returns:
        The originally passed sc.DataArray, da, with an extra dimension that can then be sliced for good data on slice
        1, and sliced for bad data on slice 0. For example, da["slice_data", 1] for good data.
    """
    good_data_coord = good_data_lut.coords[mapping_coord]
    edge = good_data_lut.data
    edge = edge[mapping_coord, :-1] ^ edge[mapping_coord, 1:]
    good_edges = sc.Dataset(
        data={mapping_coord: good_data_coord[mapping_coord, 1:], 'good': good_data_lut.data[mapping_coord, :-1]},
        coords={'edge': edge})
    good_edges = good_edges.groupby(group='edge').copy(1)

    good_data = sc.DataArray(data=good_edges['good'].data ^ good_data_lut[mapping_coord, 0].data,
                             coords={mapping_coord: good_edges[mapping_coord].data})
    good_data = sc.concatenate(good_data_lut[mapping_coord, 0], good_data, mapping_coord)
    good_data = sc.concatenate(good_data, good_data_lut[mapping_coord, -1], mapping_coord)
    good_pulse_lut = sc.lookup(good_data, mapping_coord)

    da.bins.coords[data_slices_name] = good_pulse_lut[da.bins.coords[mapping_coord]]
    groups = sc.array(dims=[data_slices_name], values=[False, True])
    grouped = sc.bin(da, groups=[groups])
    return grouped
