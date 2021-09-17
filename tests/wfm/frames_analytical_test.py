# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
import ess.wfm as wfm
import scipp as sc
from .common import allclose


def _frames_from_slopes(data):
    detector_pos_norm = sc.norm(data.meta["position"])

    # Get the number of WFM frames
    choppers = data.meta["choppers"].value
    nframes = choppers["WFMC1"].opening_angles_open.sizes["frame"]

    # Now find frame boundaries
    frames = sc.Dataset()
    frames["time_min"] = sc.zeros(dims=["frame"], shape=[nframes], unit=sc.units.us)
    frames["time_max"] = sc.zeros_like(frames["time_min"])
    frames["delta_time_min"] = sc.zeros_like(frames["time_min"])
    frames["delta_time_max"] = sc.zeros_like(frames["time_min"])
    frames["wavelength_min"] = sc.zeros(dims=["frame"],
                                        shape=[nframes],
                                        unit=sc.units.angstrom)
    frames["wavelength_max"] = sc.zeros_like(frames["wavelength_min"])
    frames["delta_wavelength_min"] = sc.zeros_like(frames["wavelength_min"])
    frames["delta_wavelength_max"] = sc.zeros_like(frames["wavelength_min"])

    frames["time_correction"] = sc.zeros(dims=["frame"],
                                         shape=[nframes],
                                         unit=sc.units.us)

    near_wfm_chopper = choppers["WFMC1"]
    far_wfm_chopper = choppers["WFMC2"]

    # Distance between WFM choppers
    dz_wfm = sc.norm(far_wfm_chopper.position - near_wfm_chopper.position)
    # Mid-point between WFM choppers
    z_wfm = 0.5 * (near_wfm_chopper.position + far_wfm_chopper.position)
    # Distance between detector positions and wfm chopper mid-point
    zdet_minus_zwfm = sc.norm(data.meta["position"] - z_wfm)
    # Neutron mass to Planck constant ratio
    # TODO: would be nice to use physical constants in scipp or scippneutron
    alpha = 2.5278e+2 * (sc.Unit('us') / sc.Unit('angstrom') / sc.Unit('m'))

    near_t_open = near_wfm_chopper.time_open
    near_t_close = near_wfm_chopper.time_close
    far_t_open = far_wfm_chopper.time_open

    for i in range(nframes):
        dt_lambda_max = near_t_close["frame", i] - near_t_open["frame", i]
        slope_lambda_max = dz_wfm / dt_lambda_max
        intercept_lambda_max = sc.norm(
            near_wfm_chopper.position) - slope_lambda_max * near_t_close["frame", i]
        t_lambda_max = (detector_pos_norm - intercept_lambda_max) / slope_lambda_max

        slope_lambda_min = sc.norm(near_wfm_chopper.position) / (
            near_t_close["frame", i] -
            (data.meta["source_pulse_length"] + data.meta["source_pulse_t_0"]))
        intercept_lambda_min = sc.norm(
            far_wfm_chopper.position) - slope_lambda_min * far_t_open["frame", i]
        t_lambda_min = (detector_pos_norm - intercept_lambda_min) / slope_lambda_min

        t_lambda_min_plus_dt = (detector_pos_norm -
                                (sc.norm(near_wfm_chopper.position) - slope_lambda_min *
                                 near_t_close["frame", i])) / slope_lambda_min
        dt_lambda_min = t_lambda_min_plus_dt - t_lambda_min

        # Compute wavelength information
        lambda_min = (t_lambda_min + 0.5 * dt_lambda_min -
                      far_t_open["frame", i]) / (alpha * zdet_minus_zwfm)
        lambda_max = (t_lambda_max - 0.5 * dt_lambda_max -
                      far_t_open["frame", i]) / (alpha * zdet_minus_zwfm)
        dlambda_min = dz_wfm * lambda_min / zdet_minus_zwfm
        dlambda_max = dz_wfm * lambda_max / zdet_minus_zwfm

        frames["time_min"]["frame", i] = t_lambda_min
        frames["delta_time_min"]["frame", i] = dt_lambda_min
        frames["time_max"]["frame", i] = t_lambda_max
        frames["delta_time_max"]["frame", i] = dt_lambda_max
        frames["wavelength_min"]["frame", i] = lambda_min
        frames["wavelength_max"]["frame", i] = lambda_max
        frames["delta_wavelength_min"]["frame", i] = dlambda_min
        frames["delta_wavelength_max"]["frame", i] = dlambda_max
        frames["time_correction"]["frame", i] = far_t_open["frame", i]

    frames["wfm_chopper_mid_point"] = z_wfm
    return frames


def _check_against_reference(ds, frames):
    reference = _frames_from_slopes(ds)
    for key in frames:
        # TODO: once scipp 0.8 is released, use sc.allclose here which also works on
        # vector_3_float64.
        # assert allclose(reference[key].data, frames[key].data)
        if frames[key].dtype == sc.dtype.vector_3_float64:
            for xyz in "xyz":
                assert allclose(getattr(reference[key].data.fields, xyz),
                                getattr(frames[key].data.fields, xyz))
        else:
            assert allclose(reference[key].data, frames[key].data)
    for i in range(frames.sizes['frame'] - 1):
        assert allclose(frames["delta_time_max"]["frame", i].data,
                        frames["delta_time_min"]["frame", i + 1].data)


def test_frames_analytical():
    ds = sc.Dataset(coords=wfm.make_fake_beamline())
    frames = wfm.get_frames(ds)
    _check_against_reference(ds, frames)


def test_frames_analytical_large_dz_wfm():
    ds = sc.Dataset(coords=wfm.make_fake_beamline(
        chopper_positions={
            "WFMC1": sc.vector(value=[0.0, 0.0, 6.0], unit='m'),
            "WFMC2": sc.vector(value=[0.0, 0.0, 8.0], unit='m')
        }))
    frames = wfm.get_frames(ds)
    _check_against_reference(ds, frames)


def test_frames_analytical_short_pulse():
    ds = sc.Dataset(coords=wfm.make_fake_beamline(
        pulse_length=sc.to_unit(sc.scalar(1.86e+03, unit='us'), 's')))
    frames = wfm.get_frames(ds)
    _check_against_reference(ds, frames)


def test_frames_analytical_large_t_0():
    ds = sc.Dataset(coords=wfm.make_fake_beamline(
        pulse_t_0=sc.to_unit(sc.scalar(300., unit='us'), 's')))
    frames = wfm.get_frames(ds)
    _check_against_reference(ds, frames)


def test_frames_analytical_6_frames():
    ds = sc.Dataset(coords=wfm.make_fake_beamline(nframes=6))
    frames = wfm.get_frames(ds)
    _check_against_reference(ds, frames)


def test_frames_analytical_short_lambda_min():
    ds = sc.Dataset(coords=wfm.make_fake_beamline(
        lambda_min=sc.scalar(0.5, unit='angstrom')))
    frames = wfm.get_frames(ds)
    _check_against_reference(ds, frames)