import numpy as np


import attr
from attr.validators import instance_of, optional

from .stats import inverse_transform_sample, sample_from_bounded_normal

def validate_offset_units(instance, offset_units, value):
    '''
    Validator for `offset_units` attribute in `OffsetMarker` class
    '''
    acceptable_offset_units = ['mm', 'm', 'km']
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
    
    offset_vals = attr.ib(default=attr.Factory(list), #convert=np.array,
                          #validator=instance_of(np.array)
                          )
    offset_probs = attr.ib(default=attr.Factory(list), 
                           #convert=np.array,
                           #validator=instance_of(np.array)
                           )

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
    
    age_vals = attr.ib(default=attr.Factory(list), 
                       #convert=np.array,
                       #validator=instance_of(np.array)
                       )
    age_probs = attr.ib(default=attr.Factory(list), 
                        #convert=np.array,
                        #validator=instance_of(np.array)
                        )

    # Dip parameters
    dip_dist_type = attr.ib(default='unspecified', 
                            validator=None#validate_dip_dist_type)
                            )

    dip_mean = attr.ib(default=None, validator=optional(instance_of(float)))
    dip_median= attr.ib(default=None,validator=optional(instance_of(float)))
    dip_sd = attr.ib(default=None, validator=optional(instance_of(float)))
    dip_mad = attr.ib(default=None, validator=optional(instance_of(float)))
    dip_min = attr.ib(default=None, validator=optional(instance_of(float)))
    dip_max = attr.ib(default=None, validator=optional(instance_of(float)))
    
    dip_vals = attr.ib(default=attr.Factory(list), #convert=np.array,
                          #validator=instance_of(np.array)
                          )
    dip_probs = attr.ib(default=attr.Factory(list), 
                           #convert=np.array,
                           #validator=instance_of(np.array)
                           )

    # Strike parameters
    strike_dist_type = attr.ib(default='unspecified', 
                               validator=None#validate_strike_dist_type)
                               )

    strike_mean = attr.ib(default=None, validator=optional(instance_of(float)))
    strike_median= attr.ib(default=None,validator=optional(instance_of(float)))
    strike_sd = attr.ib(default=None, validator=optional(instance_of(float)))
    strike_mad = attr.ib(default=None, validator=optional(instance_of(float)))
    strike_min = attr.ib(default=None, validator=optional(instance_of(float)))
    strike_max = attr.ib(default=None, validator=optional(instance_of(float)))
    
    strike_vals = attr.ib(default=attr.Factory(list), #convert=np.array,
                          #validator=instance_of(np.array)
                          )
    strike_probs = attr.ib(default=attr.Factory(list), 
                           #convert=np.array,
                           #validator=instance_of(np.array)
                           )

    # Rake parameters
    rake_dist_type = attr.ib(default='unspecified', 
                             #validator=validate_rake_dist_type)
                             )

    rake_mean = attr.ib(default=None, validator=optional(instance_of(float)))
    rake_median= attr.ib(default=None,validator=optional(instance_of(float)))
    rake_sd = attr.ib(default=None, validator=optional(instance_of(float)))
    rake_mad = attr.ib(default=None, validator=optional(instance_of(float)))
    rake_min = attr.ib(default=None, validator=optional(instance_of(float)))
    rake_max = attr.ib(default=None, validator=optional(instance_of(float)))
    
    rake_vals = attr.ib(default=attr.Factory(list), #convert=np.array,
                          #validator=instance_of(np.array)
                          )
    rake_probs = attr.ib(default=attr.Factory(list), 
                           #convert=np.array,
                           #validator=instance_of(np.array)
                           )

    # Heave parameters
    heave_units = attr.ib(default='m', validator=validate_heave_units)
    heave_dist_type = attr.ib(default='unspecified', 
                              #validator=validate_heave_dist_type)
                              )

    heave_mean = attr.ib(default=None, validator=optional(instance_of(float)))
    heave_median= attr.ib(default=None,validator=optional(instance_of(float)))
    heave_sd = attr.ib(default=None, validator=optional(instance_of(float)))
    heave_mad = attr.ib(default=None, validator=optional(instance_of(float)))
    heave_min = attr.ib(default=None, validator=optional(instance_of(float)))
    heave_max = attr.ib(default=None, validator=optional(instance_of(float)))
    
    heave_vals = attr.ib(default=attr.Factory(list), #convert=np.array,
                          #validator=instance_of(np.array)
                          )
    heave_probs = attr.ib(default=attr.Factory(list), 
                           #convert=np.array,
                           #validator=instance_of(np.array)
                           )

    # Throw parameters
    throw_dist_type = attr.ib(default='unspecified', 
                              #validator=validate_throw_dist_type)
                              )

    throw_mean = attr.ib(default=None, validator=optional(instance_of(float)))
    throw_median= attr.ib(default=None,validator=optional(instance_of(float)))
    throw_sd = attr.ib(default=None, validator=optional(instance_of(float)))
    throw_mad = attr.ib(default=None, validator=optional(instance_of(float)))
    throw_min = attr.ib(default=None, validator=optional(instance_of(float)))
    throw_max = attr.ib(default=None, validator=optional(instance_of(float)))
    
    throw_vals = attr.ib(default=attr.Factory(list), #convert=np.array,
                          #validator=instance_of(np.array)
                          )
    throw_probs = attr.ib(default=attr.Factory(list), 
                           #convert=np.array,
                           #validator=instance_of(np.array)
                           )


    def check_age_dist_type(self):
        if self.age_dist_type == 'unspecified':
            if self.age_mean is not None and self.age_sd is not None:
                self.age_dist_type = 'normal'
            elif (self.age_min is not None and self.age_max is not None
                  and self.age_sd == None):
                self.age_dist_type = 'uniform'
            elif self.age_probs is not None and self.age_vals is not None:
                self.age_dist_type = 'arbitrary'

    def check_offset_dist_type(self):
        if self.offset_dist_type == 'unspecified':
            if self.offset_mean is not None and self.offset_sd is not None:
                self.offset_dist_type = 'normal'
            elif (self.offset_min is not None and self.offset_max is not None
                  and self.offset_sd == None):
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

        self.check_offset_dist_type()

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

        self.check_age_dist_type()

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
