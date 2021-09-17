# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# flake8: noqa: E501
"""
This module is focused on enabling different binning for reflectometry data.
"""

# author: Andrew R. McCluskey (arm61)

import numpy as np
import scipp as sc
import scippneutron as scn


def q_bin(data, bins):
    """
    Return data that has been binned in the q-bins passed.

    :param data: reflectometry data to be binned
    :type data: Union[ess.reflectometry.ReflData.data, ess.amor.AmorData.data, ess.amor.AmorReference.data]
    :param bins: q-bin edges 
    :type bins: scipp._scipp.core.Variable 

    :return: Data array binned into qz with resolution
    :rtype: scipp._scipp.core.DataArray 
    :raises: NotFoundError is qz or tof coordinate cannot be found 
    """
    if 'qz' in data.events.coords and 'tof' in data.events.coords:
        erase = ['tof'] + data.dims
        # to_unit now supports binned data
        data.events.coords['qz'] = sc.to_unit(data.events.coords['qz'], bins.unit)
        binned = sc.bin(data, erase=erase, edges=[bins])
        if 'sigma_qz_by_qz' in data.events.coords:
            qzr = np.array([])
            for i in binned.data.values:
                try:
                    qzr = np.append(qzr, i.coords['sigma_qz_by_qz'].values.max())
                except ValueError:
                    qzr = np.append(qzr, 0)
            binned.coords['sigma_qz_by_qz'] = sc.Variable(values=qzr, dims=['qz'])
    else:
        raise sc.NotFoundError('qz or tof coordinate cannot be found.')
    # Why are we dividing by the number of events?
    return binned / (data.events.shape[0] * sc.units.dimensionless)


def two_dimensional_bin(data, bins):
    """
    Perform some arbitrary two-dimensional binning.

    :param data: reflectometry data to be binned
    :type data: Union[ess.reflectometry.ReflData.data, ess.amor.AmorData.data, ess.amor.AmorReference.data]
    :param bins: Bin edges
    :type bins: Tuple[scipp._scipp.core.Variable]

    :return: Data array binned into given bin edges
    :rtype: scipp._scipp.core.DataArray 
    """
    for i in bins:
        # to_unit now supports binned data
        data.events.coords[i.dims[0]] = sc.to_unit(data.events.coords[i.dims[0]],
                                                   i.unit)
    return sc.bin(data.bins.concatenate('detector_id'), edges=list(bins))
