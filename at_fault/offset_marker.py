import numpy as np


import attr
from attr.validators import instance_of, optional

#from .stats import *


def validate_offset_units(instance, offset_units, value):
    '''
    Validator for `offset_units` attribute in `OffsetMarker` class
    '''
    acceptable_offset_units = ['y', 'ky', 'My']
    if not value in acceptable_offset_units:
        raise ValueError(
            "{} not acceptable unit; only {}".format(value,
                                                     acceptable_offset_units))


def validate_time_units(instance, time_units, value):
    '''
    Validator for `time_units` attribute in `OffsetMarker` class
    '''
    acceptable_time_units = ['y', 'ky', 'My']
    if not value in acceptable_time_units:
        raise ValueError(
            "{} not acceptable unit; only {}".format(value,
                                                     acceptable_time_units))


acceptable_distribution_types = ['unspecified', 'normal', 'laplacian',
                                 'uniform', 'arbitrary']

def validate_time_dist_type(instance, time_dist_type, value):
    '''
    Validator for `time_units` attribute in `OffsetMarker` class
    '''
    if not value in acceptable_distribution_types:
        raise ValueError(
            "{} not acceptable unit; only {}".format(value,
                                               acceptable_distribution_types))


def validate_offset_dist_type(instance, offset_dist_type, value):
    '''
    Validator for `offset_units` attribute in `OffsetMarker` class
    '''
    if not value in acceptable_distribution_types:
        raise ValueError(
            "{} not acceptable unit; only {}".format(value,
                                               acceptable_distribution_types))


@attr.s
class OffsetMarker(object):
    
    source = attr.ib()
    metadata = attr.ib()

    # Offset parameters
    offset_units = attr.ib(default='m', validator=validate_offset_units)
    offset_dist_type = attr.ib(default='unspecified', 
                               validator=validate_offset_dist_type)

    offset_mean = attr.ib(default=None, validator=optional(instance_of(float)))
    offset_median= attr.ib(default=None,validator=optional(instance_of(float)))
    offset_sd = attr.ib(default=None, validator=optional(instance_of(float)))
    offset_mad = attr.ib(default=None, validator=optional(instance_of(float)))
    offset_min = attr.ib(default=None, validator=optional(instance_of(float)))
    offset_max = attr.ib(default=None, validator=optional(instance_of(float)))
    
    offset_vals = attr.ib(default=attr.Factory(np.array), convert=np.array,
                          validator=instance_of(np.array))
    offset_probs = attr.ib(default=attr.Factory(np.array), convert=np.array,
                           validator=instance_of(np.array))

    # Time parameters
    time_units = attr.ib(default='ky', validator=validate_time_units)
    time_dist_type = attr.ib(default='unspecified', 
                             validator=validate_time_dist_type)

    time_mean = attr.ib(default=None, validator=optional(instance_of(float)))
    time_median= attr.ib(default=None,validator=optional(instance_of(float)))
    time_sd = attr.ib(default=None, validator=optional(instance_of(float)))
    time_mad = attr.ib(default=None, validator=optional(instance_of(float)))
    time_min = attr.ib(default=None, validator=optional(instance_of(float)))
    time_max = attr.ib(default=None, validator=optional(instance_of(float)))
    
    time_vals = attr.ib(default=attr.Factory(np.array), convert=np.array,
                        validator=instance_of(np.array))
    time_probs = attr.ib(default=attr.Factory(np.array), convert=np.array,
                         validator=instance_of(np.array))

    #if self.
