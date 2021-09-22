# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)

import scipp as sc


# def filter_dataset_on_attribute(da: sc.DataArray, boundaries_lut, connecting_dim: str,
#                                 filtered_output_slices: str = "filtered_slices"):
#     """
#
#     Args:
#         da: DataArray to perform the filter on
#         boundaries_lut: the boundaries of where is considered "good" in the dataset based on the corresponding attribute
#         connecting_dim: The dimension by which the data is related to the attribute for example "time"
#         filtered_output_slices: name of the
#
#     Returns: filtered_data of slices accessible by slicing the areas between boundaries. For example: if the boundaries
#     are [a, b, c] and I slice this returned data, on slice 0 such as filtered_data["filtered_slices", 0] I will get all
#     the data in the dataset between boundary a and b, based on the attribute and connecting_dim.
#
#     """
#
#     # attribute_ = ds.attrs[attribute].values
#     #
#     # ds.bins.coords[filtered_output_slices] = sc.buckets.map(attribute_, ds.data, connecting_dim)
#     #
#     # filtered_variable_unit = attribute_.unit
#     # filtered_variable_dtype = attribute_.dtype
#     # filtered_variable = sc.Variable(dims=[filtered_output_slices], unit=filtered_variable_unit,
#     #                                 values=np.asarray(boundaries), dtype=filtered_variable_dtype)
#     #
#     # binned_ds = sc.bin(ds, [filtered_variable])
#     #
#     # return binned_ds.copy()
#     da.bins.coords[filtered_output_slices] = boundaries_lut[da.bins.coords[connecting_dim]]
#     groups = sc.array(dims=[connecting_dim], values=[False, True])
#     grouped = sc.bin(da, groups=[groups])
#
#     return grouped.copy()

"""
good_pulse = (proton_charge >= min_charge) & (proton_charge < max_charge)
good_pulse_lut = sc.lookup(good_pulse, 'pulse_time')
da.bins.coords['good_pulse'] = good_pulse_lut[da.bins.coords['pulse_time']]
groups = sc.array(dims=['good_pulse'], values=[False, True])
grouped = sc.bin(da, groups=[groups])
"""


# def filter_bad_pulses(ds: sc.Dataset, centre: bool = False):
#     proton_charges = ds.attrs["proton_charge"]
#     proton_charges.rename_dims({'time': 'pulse_time'})  # rename time to pulse_time as this is useful later
#
#     # Hacky procedure here to generate average proton-charge from neighbouring pulses to give Centre behaviour with log
#     # filtering (same as Mantid)!
#     if centre:
#         proton_charges.data.values[:-1] = 0.5 * (proton_charges.data.values[1:] + proton_charges.data.values[:-1])
#         # proton_charges -= sc.mean(0.5 * (proton_charges["pulse_time", 1:] - proton_charges["pulse_time", :-1]))
#
#     max_charge = sc.max(proton_charges.data)
#     max_charge *= 1.1  # determine true max (max_value * 1.1)
#     min_charge = sc.mean(proton_charges.data)
#     min_charge *= .95  # determine true min (mean_value * .01)
#
#     good_pulse = (proton_charges >= min_charge) & (proton_charges <= max_charge)
#     good_pulse_lut = sc.lookup(good_pulse, 'pulse_time')
#
#     return filter_dataset_on_attribute(ds, attribute="proton_charge",
#                                        boundaries_lut=good_pulse_lut, connecting_dim="pulse_time")
#
#
# def filter_bad_pulses(da: sc.DataArray):
#     proton_charge = da.attrs['proton_charge'].value
#     proton_charge = proton_charge.rename_dims({'time': 'pulse_time'})
#     proton_charge.coords['pulse_time'] = proton_charge.coords.pop('time')
#
#     max_charge = sc.max(proton_charge.data)
#     max_charge *= 1.1  # determine true max (max_value * 1.1)
#     min_charge = sc.mean(proton_charge.data)
#     min_charge *= .95  # determine true min (mean_value * .01)
#
#     good_pulse = (proton_charge >= min_charge) & (proton_charge < max_charge)
#     pulse_time = good_pulse.coords['pulse_time']
#     edge = good_pulse.data
#     edge = edge['pulse_time', :-1] ^ edge['pulse_time', 1:]
#     good_edges = sc.Dataset(
#         data={'pulse_time': pulse_time['pulse_time', 1:], 'good': good_pulse.data['pulse_time', :-1]},
#         coords={'edge': edge})
#     good_edges = good_edges.groupby(group='edge').copy(1)
#     good_pulse2 = sc.DataArray(data=good_edges['good'].data ^
#                                     good_pulse['pulse_time', 0].data,
#                                coords={'pulse_time': good_edges['pulse_time'].data})
#     good_pulse2 = sc.concatenate(good_pulse['pulse_time', 0], good_pulse2,
#                                  'pulse_time')
#     good_pulse2 = sc.concatenate(good_pulse2, good_pulse['pulse_time', -1],
#                                  'pulse_time')
#     good_pulse_lut = sc.lookup(good_pulse2, 'pulse_time')
#     da.bins.coords['good_pulse'] = good_pulse_lut[da.bins.coords['pulse_time']]
#     groups = sc.array(dims=['good_pulse'], values=[False, True])
#     grouped = sc.bin(da, groups=[groups])
#     return grouped


def filter_bad_pulses(da: sc.DataArray, centre=False):
    proton_charge = da.attrs['proton_charge'].value
    proton_charge = proton_charge.rename_dims({'time': 'pulse_time'})
    proton_charge.coords['pulse_time'] = proton_charge.coords.pop('time')

    if centre:
        proton_charge.data.values[:-1] = 0.5 * (proton_charge.data.values[1:] + proton_charge.data.values[:-1])
        # proton_charges -= sc.mean(0.5 * (proton_charges["pulse_time", 1:] - proton_charges["pulse_time", :-1]))

    max_charge = sc.max(proton_charge.data)
    max_charge *= 1.1  # determine true max (max_value * 1.1)
    min_charge = sc.mean(proton_charge.data)
    min_charge *= .95  # determine true min (mean_value * .01)

    good_pulse = (proton_charge >= min_charge) & (proton_charge < max_charge)

    return filter_data_array_on_attribute(da=da, good_data_lut=good_pulse, mapping_coord="pulse_time",
                                          good_data_slice_name="good_pulse")


def filter_data_array_on_attribute(da: sc.DataArray, good_data_lut, mapping_coord: str,
                                   good_data_slice_name="good_data"):
    good_data_coord = good_data_lut.coords[mapping_coord]
    edge = good_data_lut.data
    edge = edge[mapping_coord, :-1] ^ edge[mapping_coord, 1:]
    good_edges = sc.Dataset(
        data={mapping_coord: good_data_coord[mapping_coord, 1:], 'good': good_data_lut.data[mapping_coord, :-1]},
        coords={'edge': edge})
    good_edges = good_edges.groupby(group='edge').copy(1)
    good_pulse2 = sc.DataArray(data=good_edges['good'].data ^
                                    good_data_lut[mapping_coord, 0].data,
                               coords={mapping_coord: good_edges[mapping_coord].data})
    good_pulse2 = sc.concatenate(good_data_lut[mapping_coord, 0], good_pulse2,
                                 mapping_coord)
    good_pulse2 = sc.concatenate(good_pulse2, good_data_lut[mapping_coord, -1],
                                 mapping_coord)
    good_pulse_lut = sc.lookup(good_pulse2, mapping_coord)
    da.bins.coords[good_data_slice_name] = good_pulse_lut[da.bins.coords[mapping_coord]]
    groups = sc.array(dims=[good_data_slice_name], values=[False, True])
    grouped = sc.bin(da, groups=[groups])
    return grouped
