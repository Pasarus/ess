# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)

import scipp as sc


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
