# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
"""
This is the super-class for raw reflectometry data for reduction.
From this super-class, instrument specfic sub-classes may be created
(see for example the :py:class:`ess.amor.AmorData` class).
"""

# author: Andrew R. McCluskey (arm61)

import numpy as np
import scipp as sc
import scippneutron as scn
from ess.reflectometry import corrections, resolution, binning, orso, write
from typing import Union


class ReflData:
    def __init__(
        self,
        data: Union[sc.DataArray, str],
        sample_angle_offset: sc.Variable = 0 * sc.units.deg,
        gravity: bool = True,
        beam_size: sc.Variable = 0.001 * sc.units.m,
        sample_size: sc.Variable = 0.01 * sc.units.m,
        detector_spatial_resolution: sc.Variable = 0.0025 * sc.units.m,
        data_file: str = None,
    ):
        """
        The general reflectometry data class.
        This will be used by the instrument specific sub-class for data storage and
        reduction.

        :param data: The data to be reduced or the path to the file to be reduced.
        :param sample_angle_offset: Correction for omega or possibly misalignment of
            sample. Optional, default :code:`0 degrees of arc`.
        :param gravity: Should gravity be accounted for. Optional, default :code:`True`.
        :param beam_size: Size of the beam perpendicular to the scattering surface. Optional, default :code:`0.001 m`.
        :param sample_size: Size of the sample in direction of the beam. Optional, default :code:`0.01 m`.
        :param detector_spatial_resolution: Spatial resolution of the detector. Optional, default :code:`2.5 mm`.
        :param data_file: If a :py:class:`scipp._scipp.core.DataArray` is given as the :py:attr:`data`, a :py:attr:`data_file` should be defined for output in the file. Optional, default :code:`None`.

        """
        if isinstance(data, str):
            self.data_file = data
            self.data = scn.load_nexus(self.data_file)
        else:
            self.data_file = data_file
            self.data = data
        self.data.bins.constituents["data"].variances = np.ones_like(
            self.data.bins.constituents["data"].values)
        self.sample_angle_offset = sample_angle_offset
        self.gravity = gravity
        self.beam_size = beam_size
        self.sample_size = sample_size
        self.detector_spatial_resolution = detector_spatial_resolution
        self.orso = orso.Orso(orso.Creator(name='scipp'), orso.DataSource(),
                              orso.Reduction(), [])
        experiment = orso.Experiment(self.data.attrs['instrument_name'].value,
                                     'neutron')
        self.orso.data_source.experiment = experiment

    @property
    def event(self):
        """
        Return the event data buffer.

        Returns
            (:py:class:`scipp._scipp.core.DataArrayView`): Event data information.
        """
        return self.data.bins.constituents["data"]

    def q_bin(self, bins):
        """
        Return data that has been binned in the q-bins passed.

        Args:
            bins (:py:attr:`array_like`): q-bin edges.
            unit (:py:class:`scipp._scipp.core.Unit`): Unit for q. Defaults to 1/Å.

        Returns:
            (:py:class:`scipp._scipp.core.DataArray`): Data array binned into qz with resolution.
        """
        return binning.q_bin(self.data, bins).bins.sum()

    def wavelength_theta_bin(self, bins):
        """
        Return data that has been binned in the wavelength and theta bins passed.

        :param bins: wavelength and theta edges
        :type bins: Tuple(scipp._scipp.core.Variable)

        :return: Data array binned into wavelength and theta
        :rtype: scipp._scipp.core.DataArray
        """
        return binning.two_dimensional_bin(
            self.data, bins) / (self.event.shape[0] * sc.units.dimensionless)

    def q_theta_bin(self, bins):
        """
        Return data that has been binned in the q and theta bins passed.

        :param bins: q and theta edges
        :type: Tuple(scipp._scipp.core.Variable)

        :return: Data array binned into q and theta
        :rtype: scipp._scipp.core.DataArray
        """
        return binning.two_dimensional_bin(
            self.data, bins) / (self.event.shape[0] * sc.units.dimensionless)

    def wavelength_q_bin(self, bins):
        """
        Return data that has been binned in the wavelength and q bins passed.

        :param bins: q and theta edges
        :type: Tuple(scipp._scipp.core.Variable)

        :return: Data array binned into wavelength and q
        :rtype: scipp._scipp.core.DataArray
        """
        return binning.two_dimensional_bin(
            self.data, bins) / (self.event.shape[0] * sc.units.dimensionless)

    def find_wavelength(self):
        """
        From the time-of-flight data, find the wavelength for each neutron event.
        """
        scn.convert(self.data, origin="tof", target="wavelength", scatter=True)
        self.data.bins.constituents["data"].coords["wavelength"] = (scn.convert(
            self.data, origin="tof", target="wavelength",
            scatter=True).bins.constituents["data"].coords["wavelength"])

    def find_theta(self):
        """
        Calculate the theta angle (and resolution) with or without a gravity correction.
        """
        if self.gravity:
            nu_angle = corrections.angle_with_gravity(
                self.data,
                self.data.coords["position"],
                self.data.attrs["sample_position"],
            )
            theta = -self.sample_angle_offset + nu_angle
            self.data.bins.constituents["data"].coords["theta"] = theta
            # Check if the beam size on the sample is overilluminating the
            # sample. Using the beam on sample size from the value of theta
            # where the neutrons scatter from a point
            half_beam_on_sample = (corrections.illumination_of_sample(
                self.beam_size, self.sample_size, theta) / 2.0)
            # Find the range of possible positions that the neutron could
            # strike, this range of theta values is taken to be the full
            # width half maximum for the theta distribution
            offset_positive = resolution.z_offset(self.data.attrs["sample_position"],
                                                  half_beam_on_sample)
            offset_negative = resolution.z_offset(self.data.attrs["sample_position"],
                                                  half_beam_on_sample)
            self.data.bins.constituents['data'].attrs[
                'offset_positive'] = offset_positive
            self.data.bins.constituents['data'].attrs[
                'offset_negative'] = offset_negative
            angle_max = corrections.angle_with_gravity(
                self.data, self.data.coords["position"],
                self.data.bins.attrs['offset_positive'])
            angle_min = corrections.angle_with_gravity(
                self.data,
                self.data.coords["position"],
                self.data.bins.attrs['offset_negative'],
            )
            del self.data.bins.constituents['data'].attrs['offset_positive']
            del self.data.bins.constituents['data'].attrs['offset_negative']
            fwhm_to_std = 2 * np.sqrt(2 * np.log(2))
            sigma_theta_position = (angle_max - angle_min) / fwhm_to_std
            self.data.bins.constituents["data"].attrs[
                "sigma_theta_position"] = sigma_theta_position
            # Then find the full width half maximum of the theta distribution
            # due to the detector's spatial resolution, which we will call
            # sigma_gamma
            sigma_gamma = resolution.detector_resolution(
                self.detector_spatial_resolution, self.data.coords["position"].fields.z,
                self.data.attrs["sample_position"].fields.z)
            self.data.attrs["sigma_gamma"] = sigma_gamma
            sigma_theta = sc.sqrt(
                (self.data.attrs["sigma_gamma"] / self.data.bins.coords["theta"]) *
                (self.data.attrs["sigma_gamma"] / self.data.bins.coords["theta"]) +
                (self.data.bins.attrs["sigma_theta_position"] /
                 self.data.bins.coords["theta"]) *
                (self.data.bins.attrs["sigma_theta_position"] /
                 self.data.bins.coords["theta"]))
            self.data.bins.constituents["data"].coords[
                "sigma_theta_by_theta"] = sigma_theta.bins.constituents["data"]
        else:
            raise NotImplementedError

    def find_qz(self):
        """
        Calculate the scattering vector (and resolution).
        """
        self.data.bins.constituents["data"].coords["qz"] = (
            4.0 * np.pi * sc.sin(self.data.bins.constituents["data"].coords["theta"]) /
            self.data.bins.constituents["data"].coords["wavelength"])
        self.data.coords["s_qz_bins"] = sc.zeros(
            dims=self.data.coords["detector_id"].dims,
            shape=self.data.coords["detector_id"].shape,
        )
        for i in list(self.data.coords.keys()):
            if "sigma" in i:
                self.data.coords["s_qz_bins"] += (self.data.coords[i] *
                                                  self.data.coords[i])
        self.data.bins.constituents["data"].coords["s_qz_events"] = sc.zeros(
            dims=self.data.bins.constituents["data"].coords["qz"].dims,
            shape=self.data.bins.constituents["data"].coords["qz"].shape,
        )
        for i in list(self.data.bins.constituents["data"].coords.keys()):
            if "sigma" in i:
                self.data.bins.constituents["data"].coords["s_qz_events"] += (
                    self.data.bins.coords[i] *
                    self.data.bins.coords[i]).bins.constituents["data"]
        self.data.bins.constituents["data"].coords["sigma_qz_by_qz"] = sc.sqrt(
            (self.data.coords["s_qz_bins"] +
             self.data.bins.coords["s_qz_events"]).bins.constituents["data"])

    def illumination(self):
        """
        Perform illumination correction.
        """
        self.event.coords["illumination"] = corrections.illumination_correction(
            self.beam_size,
            self.sample_size,
            self.event.coords["theta"],
        )
        self.data /= self.data.bins.coords["illumination"]

    def detector_masking(
        self,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
        z_min=None,
        z_max=None,
    ):
        """
        Masking on detector pixels.

        Args:
            x_min (:py:class:`scipp._scipp.core.Variable`, optional): Minimum x-dimension to be used. Optional, default no minimum mask.
            x_max (:py:class:`scipp._scipp.core.Variable`, optional): Maximum x-dimension to be used. Optional, default no maximum mask.
            y_min (:py:class:`scipp._scipp.core.Variable`, optional): Minimum y-dimension to be used. Optional, default no minimum mask.
            y_max (:py:class:`scipp._scipp.core.Variable`, optional): Maximum y-dimension to be used. Optional, default no maximum mask.
            z_min (:py:class:`scipp._scipp.core.Variable`, optional): Minimum z-dimension to be used. Optional, default no minimum mask.
            z_max (:py:class:`scipp._scipp.core.Variable`, optional): Maximum z-dimension to be used. Optional, default no maximum mask.
        """
        x_position = self.data.coords["position"].fields.x
        y_position = self.data.coords["position"].fields.y
        z_position = self.data.coords["position"].fields.z
        if x_min is None:
            x_min = sc.min(x_position)
        if x_max is None:
            x_max = sc.max(x_position)
        if y_min is None:
            y_min = sc.min(y_position)
        if y_max is None:
            y_max = sc.max(y_position)
        if z_min is None:
            z_min = sc.min(z_position)
        if z_max is None:
            z_max = sc.max(z_position)
        self.data.masks["x_mask"] = (x_position < x_min) | (x_position > x_max)
        self.data.masks["y_mask"] = (y_position < y_min) | (y_position > y_max)
        self.data.masks["z_mask"] = (z_position < z_min) | (z_position > z_max)

    def theta_masking(self, theta_min=None, theta_max=None):
        """
        Masking data based on reflected angle.

        Args:
            theta_min (:py:class:`scipp._scipp.core.Variable`, optional): Minimum theta to be used. Optional, default no minimum mask.
            theta_max (:py:class:`scipp._scipp.core.Variable`, optional): Maximum theta to be used. Optional, default no maximum mask.
        """
        if theta_min is None:
            theta_min = sc.min(self.data.bins.constituents['data'].coords["theta"])
        if theta_max is None:
            theta_max = sc.max(self.data.bins.constituents['data'].coords["theta"])
        theta_max = sc.to_unit(theta_max,
                               self.data.bins.constituents['data'].coords['theta'].unit)
        wavelength_min = sc.to_unit(
            theta_min, self.data.bins.constituents['data'].coords['theta'].unit)
        range = [
            sc.min(self.data.bins.constituents['data'].coords['theta']).value,
            theta_min.value, theta_max.value,
            sc.max(self.data.bins.constituents['data'].coords['theta']).value
        ]
        theta = sc.array(dims=['theta'],
                         unit=self.data.bins.constituents['data'].coords['theta'].unit,
                         values=range)
        self.data = sc.bin(self.data, edges=[theta])
        self.data.masks['theta'] = sc.array(dims=['theta'], values=[True, False, True])

    def wavelength_masking(self, wavelength_min=None, wavelength_max=None):
        """
        Masking data based on wavelength.

        Args:
            wavelength_min (:py:class:`scipp._scipp.core.Variable`, optional): Minimum wavelength to be used. Optional, default no minimum mask.
            wavelength_max (:py:class:`scipp._scipp.core.Variable`, optional): Maximum wavelength to be used. Optional, default no maximum mask.
        """
        if wavelength_min is None:
            wavelength_min = sc.min(
                self.data.bins.constituents['data'].coords["wavelength"])
        if wavelength_max is None:
            wavelength_max = sc.max(
                self.data.bins.constituents['data'].coords["wavelength"])
        wavelength_max = sc.to_unit(
            wavelength_max,
            self.data.bins.constituents['data'].coords['wavelength'].unit)
        wavelength_min = sc.to_unit(
            wavelength_min,
            self.data.bins.constituents['data'].coords['wavelength'].unit)
        range = [
            sc.min(self.data.bins.constituents['data'].coords['wavelength']).value,
            wavelength_min.value, wavelength_max.value,
            sc.max(self.data.bins.constituents['data'].coords['wavelength']).value
        ]
        wavelength = sc.array(
            dims=['wavelength'],
            unit=self.data.bins.constituents['data'].coords['wavelength'].unit,
            values=range)
        self.data = sc.bin(self.data, edges=[wavelength])
        self.data.masks['wavelength'] = sc.array(dims=['wavelength'],
                                                 values=[True, False, True])

    def write_reflectometry(self, filename, bins, header=None):
        """
        Write the reflectometry intensity data to a file.

        Args:
            filename (:py:attr:`str`): The file path for the file to be saved to.
            bin_kwargs (:py:attr:`dict`, optional): A dictionary of keyword arguments to be passed to the :py:func:`q_bin` class method. Optional, default is that default :py:func:`q_bin` keywords arguments are used.
            header (:py:class:`ess.reflectometry.Orso`, optional): ORSO-compatible header object. Optional defaults to :py:attr:`ReflData.orso`.
        """
        write.reflectometry(self, filename, bins, header)

    def write_wavelength_theta(self, filename, bins, header=None):
        """
        Write the reflectometry intensity data as a function of wavelength-theta to a file.

        Args:
            filename (:py:attr:`str`): The file path for the file to be saved to.
            bins (:py:attr:`tuple` of :py:attr:`array_like`): wavelength and theta edges.
            header (:py:class:`ess.reflectometry.Orso`, optional): ORSO-compatible header object. Optional defaults to :py:attr:`ReflData.orso`.
        """
        write.wavelength_theta(self, filename, bins, header)
