# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
"""
Corrections to be used for neutron reflectometry reduction processes.
"""
import numpy as np
import scipp as sc
from scipy.special import erf
from ess.reflectometry import HDM, G_ACC


def angle_with_gravity(data: sc.DataArray, pixel_position: sc.Variable,
                       sample_position: sc.Variable) -> sc.Variable:
    """
    Find the angle of reflection when accounting for the presence of gravity.

    :param data: Reduction data array.
    :param pixel_position: Detector pixel positions, should be a `vector_3_float64`-type
        object.
    :param sample_position: Scattered neutron origin position.
    :return: Gravity corrected angle values.
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


def y_dash0(velocity: sc.Variable, z_origin: sc.Variable, y_origin: sc.Variable,
            z_measured: sc.Variable, y_measured: sc.Variable) -> sc.Variable:
    """
    Evaluation of the first dervative of the kinematic equations for for the trajectory
    of a neutron reflected from a surface.

    :param velocity: Neutron velocity.
    :param z_origin: The z-origin position for the reflected neutron.
    :param y_origin: The y-origin position for the reflected neutron.
    :param z_measured: The z-measured position for the reflected neutron.
    :param y_measured: The y-measured position for the reflected neutron.
    :return: The gradient of the trajectory of the neutron at the origin position.
    """
    velocity2 = velocity * velocity
    z_diff = z_measured - z_origin
    y_diff = y_measured - y_origin
    return -0.5 * sc.norm(G_ACC) * z_diff / velocity2 + y_diff / z_diff


def illumination_correction(beam_size: sc.Variable, sample_size: sc.Variable,
                            theta: sc.Variable) -> sc.Variable:
    """
    The factor by which the intensity should be multiplied to account for the
    scattering geometry, where the beam is Gaussian in shape.

    :param beam_size: Width of incident beam.
    :param sample_size: Width of sample in the dimension of the beam.
    :param theta: Incident angle.
    :return: Correction factor.
    """
    beam_on_sample = beam_size / sc.sin(theta)
    fwhm_to_std = 2 * np.sqrt(2 * np.log(2))
    scale_factor = erf((sample_size / beam_on_sample * fwhm_to_std).values)
    return sc.Variable(values=scale_factor, dims=theta.dims)


def illumination_of_sample(beam_size: sc.Variable, sample_size: sc.Variable,
                           theta: sc.Variable) -> sc.Variable:
    """
    Determine the illumination of the sample by the beam and therefore the size of this
    illuminated length.

    :param beam_size: Width of incident beam, in metres.
    :param sample_size: Width of sample in the dimension of the beam, in metres.
    :param theta: Incident angle.
    :return: The size of the beam, for each theta, on the sample.
    """
    beam_on_sample = beam_size / sc.sin(theta)
    if ((sc.mean(beam_on_sample)) > sample_size).value:
        beam_on_sample = sc.broadcast(sample_size, shape=theta.shape, dims=theta.dims)
    return beam_on_sample
