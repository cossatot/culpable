import numpy as np


import attr
from attr.validators import instance_of, optional

from .stats import inverse_transform_sample, sample_from_bounded_normal


def validate_offset_units(instance, offset_units, value):
    '''
    Validator for `offset_units` attribute in `OffsetMarker` class
    '''
    acceptable_offset_units = ['y', 'ky', 'My']
    if not value in acceptable_offset_units:
        raise ValueError(
            "{} not acceptable unit; only {}".format(value,
                                                     acceptable_offset_units))


def validate_age_units(instance, age_units, value):
    '''
    Validator for `age_units` attribute in `OffsetMarker` class
    '''
    acceptable_age_units = ['y', 'cal_yr', 'ky', 'My']
    if not value in acceptable_age_units:
        raise ValueError(
            "{} not acceptable unit; only {}".format(value,
                                                     acceptable_age_units))


acceptable_distribution_types = ['unspecified', 'normal', 'laplacian',
                                 'uniform', 'arbitrary']

def validate_age_dist_type(instance, age_dist_type, value):
    '''
    Validator for `age_units` attribute in `OffsetMarker` class
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
    
    source = attr.ib(default=None)
    metadata = attr.ib(default=None)
    name = attr.ib(default=None, validator=instance_of(str))

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

    # Age parameters
    age_units = attr.ib(default='ky', validator=validate_age_units)
    age_dist_type = attr.ib(default='unspecified', 
                             validator=validate_age_dist_type)

    age_mean = attr.ib(default=None, validator=optional(instance_of(float)))
    age_median = attr.ib(default=None,validator=optional(instance_of(float)))
    age_sd = attr.ib(default=None, validator=optional(instance_of(float)))
    age_mad = attr.ib(default=None, validator=optional(instance_of(float)))
    age_min = attr.ib(default=None, validator=optional(instance_of(float)))
    age_max = attr.ib(default=None, validator=optional(instance_of(float)))
    
    age_vals = attr.ib(default=attr.Factory(np.array), convert=np.array,
                        validator=instance_of(np.array))
    age_probs = attr.ib(default=attr.Factory(np.array), convert=np.array,
                         validator=instance_of(np.array))


    def check_age_dist_type(self):
        if self.age_dist_type == 'unspecified':
            if age_mean is not None and age_sd is not None:
                self.age_dist_type = 'normal'
            elif (age_min is not None and age_max is not None
                  and age_sd == None):
                self.age_dist_type = 'uniform'
            elif age_probs is not None and age_vals is not None:
                self.age_dist_type = 'arbitrary'

    def check_offset_dist_type(self):
        if self.age_dist_type == 'unspecified':
            if offset_mean is not None and offset_sd is not None:
                self.offset_dist_type = 'normal'
            elif (offset_min is not None and offset_max is not None
                  and offset_sd == None):
                self.offset_dist_type = 'uniform'
            elif offset_probs is not None and offset_vals is not None:
                self.offset_dist_type = 'arbitrary'

    def check_dist_types(self):
        check_age_dist_type()
        check_offset_dist_type()

    # sampling
    def sample_offset_from_normal(self, n):
        """Generates n-length sample from normal distribution of offsets"""

        return sample_from_bounded_normal(self.offset_mean, self.offset_sd, n,
                                          self.offset_min, self.offset_max)
    
    def sample_offset_from_uniform(self, n):
        """Generates n-length sample from uniform distribution of offsets"""
        
        return np.random.uniform(self.offset_min, self.offset_max, n)
    
    def sample_offset_from_arbitrary(self, n):
        """Generates n-length sample from arbitrary distribution of offsets"""
        offset_sample = inverse_transform_sample(self.offset_vals,
                                                 self.offset_probs, n)
        return offset_sample
    
    def sample_offset(self, n):
        """Generates n-length array of samples from distribution"""

        check_offset_dist_type()

        if self.offset_dist_type == 'normal':
            offset_sample = self.sample_offset_from_normal(n)
        
        elif self.offset_dist_type == 'uniform':
            offset_sample = self.sample_offset_from_uniform(n)
        
        elif self.offset_dist_type == 'arbitrary':
            offset_sample = self.sample_offset_from_arbitrary(n)
        
        return offset_sample    
    
    def sample_age_from_normal(self, n):
        """Generates n-length sample from normal distribution of ages"""
        if self.age_min:
            age_min = self.age_min
        else:
            age_min = 0.

        age_sample = sample_from_bounded_normal(self.age_mean, self.age_sd, n,
                                           age_min, self.age_max)

        return age_sample
    
    def sample_age_from_uniform(self, n):
        """Generates n-length sample from uniform distribution of ages"""
        return np.random.uniform(self.age_min, self.age_max, n)
        
    def sample_age_from_arbitrary(self, n):
        """Generates n-length sample from uniform distribution of ages"""
        return inverse_transform_sample(self.age_vals, self.age_probs, n)

    def sample_age(self, n):
        """Generates n-length array of samples from distribution"""
        check_age_dist_type()

        if self.age_dist_type == 'normal':
            age_sample = self.sample_age_from_normal(n)
        
        elif self.age_dist_type == 'uniform':
            age_sample = self.sample_age_from_uniform(n)
        
        elif self.age_dist_type == 'arbitrary':
            age_sample = self.sample_age_from_arbitrary(n)
        
        return age_sample

    def sample(self, n):
        age_sample = self.sample_age(n)
        offset_sample = self.sample_offset(n)
        
        asl = len(age_sample)
        osl = len(offset_sample)
        
        if asl > osl:
            age_sample = age_sample[0:osl]
        elif osl > asl:
            offset_sample = offset_sample[0:asl]
        
        return age_sample, offset_sample
