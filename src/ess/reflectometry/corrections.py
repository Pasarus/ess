# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
"""
Corrections to be used for neutron reflectometry reduction processes.
"""
import numpy as np
import scipp as sc
from scipy.special import erf
from ess.reflectometry import HDM, G_ACC


def angle_with_gravity(data, pixel_position, sample_position):
    """
    Find the angle of reflection when accounting for the presence of gravity.

    :param data: Reduction data array.
    :type data: `scipp.DataArray`
    :param pixel_position: Detector pixel positions, should be a `vector_3_float64`-type
        object.
    :type pixel_position: `scipp.Variable`
    :param sample_position: Scattered neutron origin position.
    :type sample_position: `scipp.Variable`

    :return: Gravity corrected angle values.
    :rtype: `scipp.Variable`
    """
    # This is a workaround until scipp #1819 is resolved, at which time the following
    # should be used instead
    # velocity = sc.to_unit(HDM / wavelength, 'm/s')
    # At which point the args can be changed to wavelength
    # (where this is
    # data.bins.constituents['data'].coords['wavelength'].astype(sc.dtype.float64)
    # or similar) instead of data
    velocity = sc.to_unit(
        HDM /
        data.bins.constituents["data"].coords["wavelength"].astype(sc.dtype.float64),
        "m/s",
    )
    data.bins.constituents["data"].coords["velocity"] = velocity
    velocity = data.bins.coords["velocity"]
    velocity.events.unit = sc.units.m / sc.units.s
    y_measured = pixel_position.fields.y
    z_measured = pixel_position.fields.z
    z_origin = sample_position.fields.z
    y_origin = sample_position.fields.y
    y_dash = y_dash0(velocity, z_origin, y_origin, z_measured, y_measured)
    intercept = y_origin - y_dash * z_origin
    y_true = z_measured * y_dash + intercept
    angle = sc.to_unit(sc.atan(y_true / z_measured).bins.constituents["data"], 'deg')
    return angle


def y_dash0(velocity, z_origin, y_origin, z_measured, y_measured):
    """
    Evaluation of the first dervative of the kinematic equations for for the trajectory
    of a neutron reflected from a surface.

    Args:
    :param velocity: Neutron velocity.
    :type velocity: `scipp.Variable`
    :param z_origin: The z-origin position for the reflected neutron.
    :type z_origin: `scipp.Variable`
    :param y_origin: The y-origin position for the reflected neutron.
    :type y_origin: `scipp.Variable`
    :param z_measured: The z-measured position for the reflected neutron.
    :type z_measured: `scipp.Variable`
    :param y_measured: The y-measured position for the reflected neutron.
    :type y_measured: `scipp.Variable`

    :return: The gradient of the trajectory of the neutron at the origin position.
    :rtype: `scipp.Variable`
    """
    velocity2 = velocity * velocity
    z_diff = z_measured - z_origin
    y_diff = y_measured - y_origin
    return -0.5 * sc.norm(G_ACC) * z_diff / velocity2 + y_diff / z_diff


def illumination_correction(beam_size, sample_size, theta):
    """
    The factor by which the intensity should be multiplied to account for the
    scattering geometry, where the beam is Gaussian in shape.

    :param beam_size: Width of incident beam.
    :type beam_size: `scipp.Variable`
    :param sample_size: Width of sample in the dimension of the beam.
    :type sample_size: `scipp.Variable`
    :param theta: Incident angle.
    :type theta: `scipp.Variable`

    :return: Correction factor.
    :rtype: `scipp.Variable`
    """
    beam_on_sample = beam_size / sc.sin(theta)
    fwhm_to_std = 2 * np.sqrt(2 * np.log(2))
    scale_factor = erf((sample_size / beam_on_sample * fwhm_to_std).values)
    return sc.Variable(values=scale_factor, dims=theta.dims)


def illumination_of_sample(beam_size, sample_size, theta):
    """
    Determine the illumination of the sample by the beam and therefore the size of this
    illuminated length.

    :param beam_size: Width of incident beam, in metres.
    :type beam_size: `scipp.Variable`
    :param sample_size: Width of sample in the dimension of the beam, in metres.
    :type sample_size: `scipp.Variable`
    :param theta: Incident angle.
    :type theta: `scipp.Variable`

    :return: The size of the beam, for each theta, on the sample.
    :rtype: `scipp.Variable`
    """
    beam_on_sample = beam_size / sc.sin(theta)
    if ((sc.mean(beam_on_sample)) > sample_size).value:
        beam_on_sample = sc.broadcast(sample_size, shape=theta.shape, dims=theta.dims)
    return beam_on_sample
