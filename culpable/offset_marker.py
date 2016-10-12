import numpy as np

import attr
from attr.validators import instance_of, optional

from .stats import inverse_transform_sample, sample_from_bounded_normal
from .fault_projections import *


def validate_time_units(instance, time_units, value):
    '''
    Validator for `age_units` attribute in `OffsetMarker` class
    '''
    acceptable_time_units = ['y', 'cal_yr', 'ky', 'My']
    if not value in acceptable_time_units:
        raise ValueError(
            "{} not acceptable unit; only {}".format(value,
                                                     acceptable_time_units))


def validate_distance_units(instance, distance_units, value):
    '''
    Validator for `offset_units` attribute in `OffsetMarker` class
    '''
    acceptable_offset_units = ['mm', 'm', 'km']
    if not value in acceptable_offset_units:
        raise ValueError(
            "{} not acceptable unit; only {}".format(value,
                                                     acceptable_offset_units))


acceptable_distribution_types = ['unspecified', 'normal', 'laplacian',
                                 'uniform', 'arbitrary', 'scalar']

def validate_dist_type(instance, age_dist_type, value):
    '''
    Validator for `age_units` attribute in `OffsetMarker` class
    '''
    if not value in acceptable_distribution_types:
        raise ValueError(
            "{} not acceptable unit; only {}".format(value,
                                               acceptable_distribution_types))


@attr.s
class OffsetMarker(object):
    
    source = attr.ib(default=None)
    metadata = attr.ib(default=None)
    name = attr.ib(default=None, validator=optional(instance_of(str)))

    # Offset parameters
    offset_units = attr.ib(default='m', validator=validate_distance_units)
    offset_dist_type = attr.ib(default='unspecified', 
                               validator=validate_dist_type)

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
    age_units = attr.ib(default='ky', validator=validate_time_units)
    age_dist_type = attr.ib(default='unspecified', 
                             validator=validate_dist_type)

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
                            validator=validate_dist_type)

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
                               validator=validate_dist_type)

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
                             validator=validate_dist_type)
                             

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

    # Horizontal separation parameters
    hor_separation_units = attr.ib(default='m', 
                                   validator=validate_distance_units)
                                   
    hor_separation_dist_type = attr.ib(default='unspecified', 
                                       validator=validate_dist_type)

    hor_separation_mean = attr.ib(default=None,
                                  validator=optional(instance_of(float)))
    hor_separation_median= attr.ib(default=None,
                                   validator=optional(instance_of(float)))
    hor_separation_sd = attr.ib(default=None,
                                validator=optional(instance_of(float)))
    hor_separation_mad = attr.ib(default=None,
                                 validator=optional(instance_of(float)))
    hor_separation_min = attr.ib(default=None,
                                 validator=optional(instance_of(float)))
    hor_separation_max = attr.ib(default=None,
                                 validator=optional(instance_of(float)))
    
    hor_separation_vals = attr.ib(default=attr.Factory(list),
                                  #convert=np.array,
                                  #validator=instance_of(np.array)
                                  )
    hor_separation_probs = attr.ib(default=attr.Factory(list), 
                                   #convert=np.array,
                                   #validator=instance_of(np.array)
                                   )

    # Throw parameters
    vert_separation_units = attr.ib(default='m', 
                                    validator=validate_distance_units)
                                   
    vert_separation_dist_type = attr.ib(default='unspecified', 
                                        validator=validate_dist_type)

    vert_separation_mean = attr.ib(default=None, 
                                   validator=optional(instance_of(float)))
    vert_separation_median= attr.ib(default=None,
                                    validator=optional(instance_of(float)))
    vert_separation_sd = attr.ib(default=None,
                                 validator=optional(instance_of(float)))
    vert_separation_mad = attr.ib(default=None,
                                  validator=optional(instance_of(float)))
    vert_separation_min = attr.ib(default=None,
                                  validator=optional(instance_of(float)))
    vert_separation_max = attr.ib(default=None,
                                  validator=optional(instance_of(float)))
    
    vert_separation_vals = attr.ib(default=attr.Factory(list), 
                                   #convert=np.array,
                                   #validator=instance_of(np.array)
                                   )
    vert_separation_probs = attr.ib(default=attr.Factory(list), 
                                    #convert=np.array,
                                    #validator=instance_of(np.array)
                                    )


    # Strike-slip parameters
    strike_slip_units = attr.ib(default='m', validator=validate_distance_units)
    strike_slip_dist_type = attr.ib(default='unspecified', 
                                    validator=validate_dist_type)

    strike_slip_mean = attr.ib(default=None, 
                               validator=optional(instance_of(float)))
    strike_slip_median= attr.ib(default=None,
                                validator=optional(instance_of(float)))
    strike_slip_sd = attr.ib(default=None, 
                             validator=optional(instance_of(float)))
    strike_slip_mad = attr.ib(default=None, 
                              validator=optional(instance_of(float)))
    strike_slip_min = attr.ib(default=None, 
                              validator=optional(instance_of(float)))
    strike_slip_max = attr.ib(default=None, 
                              validator=optional(instance_of(float)))
    
    strike_slip_vals = attr.ib(default=attr.Factory(list), #convert=np.array,
                          #validator=instance_of(np.array)
                          )
    strike_slip_probs = attr.ib(default=attr.Factory(list), 
                           #convert=np.array,
                           #validator=instance_of(np.array)
                           )

    # Dip-slip parameters
    dip_slip_units = attr.ib(default='m', validator=validate_distance_units)
    dip_slip_dist_type = attr.ib(default='unspecified', 
                                 validator=validate_dist_type)

    dip_slip_mean = attr.ib(default=None, 
                            validator=optional(instance_of(float)))
    dip_slip_median= attr.ib(default=None,
                             validator=optional(instance_of(float)))
    dip_slip_sd = attr.ib(default=None, 
                          validator=optional(instance_of(float)))
    dip_slip_mad = attr.ib(default=None, 
                           validator=optional(instance_of(float)))
    dip_slip_min = attr.ib(default=None, 
                           validator=optional(instance_of(float)))
    dip_slip_max = attr.ib(default=None, 
                           validator=optional(instance_of(float)))
    
    dip_slip_vals = attr.ib(default=attr.Factory(list), #convert=np.array,
                          #validator=instance_of(np.array)
                          )
    dip_slip_probs = attr.ib(default=attr.Factory(list), 
                           #convert=np.array,
                           #validator=instance_of(np.array)
                           )

    # Heave parameters
    heave_units = attr.ib(default='m', validator=validate_distance_units)
    heave_dist_type = attr.ib(default='unspecified', 
                               validator=validate_dist_type)

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
            #TODO: add some check for 'scalar'

    #TODO: check dist type of all offsets entered

    def check_dist_types(self):
        check_age_dist_type()
        check_offset_dist_type()


    def check_offset_consistency(self):
        pass

    def fill_all_offsets(self):
        '''
        Fills values for all offsets given dip, rake, and any offset type
        provided.
        '''
        # scalar only, right now
        pass

    def _find_entered_slip_val(self):
        
        comp_dict = {'offset_mean' :           self.offset_mean,      
                     'offset_max' :            self.offset_max,       
                     'hor_separation_mean' :   self.hor_separation_mean,     
                     'hor_separation_max' :    self.hor_separation_max,      
                     'vert_separation_mean':   self.vert_separation_mean,    
                     'vert_separation_max' :   self.vert_separation_max,     
                     'dip_slip_mean' :         self.dip_slip_mean,    
                     'dip_slip_max' :          self.dip_slip_max,     
                     'strike_slip_mean' :      self.strike_slip_mean, 
                     'strike_slip_max' :       self.strike_slip_max,  
                     'heave_mean' :            self.heave_mean,       
                     'heave_max' :             self.heave_max,        
                     'offset_median' :         self.offset_median,     
                     'offset_probs' :          self.offset_probs,      
                     'hor_separation_median' : self.hor_separation_median,     
                     'hor_separation_probs' :  self.hor_separation_probs,      
                     'vert_separation_median': self.vert_separation_median,    
                     'vert_separation_probs' : self.vert_separation_probs,     
                     'dip_slip_median' :       self.dip_slip_median,   
                     'dip_slip_probs' :        self.dip_slip_probs,    
                     'strike_slip_median' :    self.strike_slip_median,  
                     'strike_slip_probs' :     self.strike_slip_probs, 
                     'heave_median' :          self.heave_median,      
                     'heave_probs' :           self.heave_probs
                     }

        return {k: v for k, v in comp_dict.items() if v not in ([], None)}
   

    def propagate_scalar_slip_components(self):
        comp, comp_val = list(self._find_entered_slip_val())[0]

        if comp == 'offset_mean':
            slip_comps = slip_comps_from_offset(comp_val, self.dip_mean,
                                                self.rake_mean)
        elif comp == 'offset_median':
            slip_comps = slip_comps_from_offset(comp_val, self.dip_median,
                                                self.rake_median)
        elif comp == 'hor_separation_mean':
            slip_comps = slip_comps_from_hor_separation(comp_val, 
                                                        self.dip_mean,
                                                        self.rake_mean)
        elif comp == 'hor_separation_median':
            slip_comps = slip_comps_from_hor_separation(comp_val, 
                                                        self.dip_median,
                                                        self.rake_median)
        elif comp == 'vert_separation_mean':
            slip_comps = slip_comps_from_vert_separation(comp_val, 
                                                         self.dip_mean,
                                                         self.rake_mean)
        elif comp == 'vert_separation_median':
            slip_comps = slip_comps_from_vert_separation(comp_val, 
                                                         self.dip_median,
                                                         self.rake_median)
        elif comp == 'strike_slip_mean':
            slip_comps = slip_comps_from_strike_slip(comp_val, self.dip_mean,
                                                     self.rake_mean)
        elif comp == 'strike_slip_median':
            slip_comps = slip_comps_from_strike_slip(comp_val, self.dip_median,
                                                     self.rake_median)
        elif comp == 'dip_slip_mean':
            slip_comps = slip_comps_from_dip_slip(comp_val, self.dip_mean,
                                                  self.rake_mean)
        elif comp == 'dip_slip_median':
            slip_comps = slip_comps_from_dip_slip(comp_val, self.dip_median,
                                                  self.rake_median)
        elif comp == 'heave_mean':
            slip_comps = slip_comps_from_heave(comp_val, self.dip_mean,
                                                self.rake_mean)
        elif comp == 'heave_median':
            slip_comps = slip_comps_from_heave(comp_val, self.dip_median,
                                                self.rake_median)
        
        if comp.split('_')[-1] == 'mean':
            propagate_slip_comps_from_offset_mean(slip_comps)
        
        elif comp.split('_')[-1] == 'median':
            propagate_slip_comps_from_offset_median(slip_comps)
        
    def propagate_slip_comps_from_offset_mean(self, slip_comps):

        self.hor_separation_mean = slip_comps['hor_sep']
        self.vert_separation_mean = slip_comps['vert_sep']
        self.dip_slip_mean = slip_comps['dip_slip']
        self.strike_slip_mean = slip_comps['strike_slip']
        self.heave_mean = slip_comps['heave']

    def propagate_slip_comps_from_offset_median(self, slip_comps):

        self.hor_separation_median = slip_comps['hor_sep']
        self.vert_separation_median = slip_comps['vert_sep']
        self.dip_slip_median = slip_comps['dip_slip']
        self.strike_slip_median = slip_comps['strike_slip']
        self.heave_median = slip_comps['heave']

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
