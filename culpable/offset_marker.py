import numpy as np

import attr
from attr.validators import instance_of, optional

from .stats import inverse_transform_sample, sample_from_bounded_normal #, kde
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


def validate_angle(instance, attribute, value):
    if (value < 0.) or (value > 90.):
        raise ValueError("Only angles between 0 and 90 acceptable")


def _sample(self, n=1000, return_scalar_array=False):
    '''
    Main sampling function. It is defined as a function for use in multiple
    classes as a method without inheritance, and is not meant to be called
    as a separate function; nonetheless, it can if a class w/ the requisite
    attributes (dist_type, mean, sd, max, min, etc.) are present.
    '''
    if self.dist_type == 'normal':
        return sample_from_bounded_normal(self.mean, self.sd, n,
                                          sample_min=self.min, 
                                          sample_max=self.max)
    elif self.dist_type == 'uniform':
        return np.random.uniform(self.min, self.max, n)
    elif self.dist_type == 'arbitrary':
        return inverse_transform_sample(self.vals, self.probs, n)
    elif self.dist_type == 'scalar':
        if return_scalar_array == True:
            return np.ones(n) * self.mean
        else:
            return self.mean


def _check_dist_types(self):
    # Consider raising exception instead of changing state
    # or at least issue a warning
    if self.mean is not None and self.sd is not None:
        self.dist_type = 'normal'
    elif (self.min is not None and self.max is not None
          and self.sd == None):
        self.dist_type = 'uniform'
    elif self.probs is not None and self.vals is not None:
        self.dist_type = 'arbitrary'

    


@attr.s
class SlipComponent(object):
    units = attr.ib(default='m', validator=validate_distance_units)
    dist_type = attr.ib(default='unspecified', validator=validate_dist_type)
    mean = attr.ib(default=None, convert=float,
                   validator=optional(instance_of(float)))
    median= attr.ib(default=None,convert=float, 
                    validator=optional(instance_of(float)))
    sd = attr.ib(default=None, convert=float, 
                 validator=optional(instance_of(float)))
    mad = attr.ib(default=None, convert=float, 
                  validator=optional(instance_of(float)))
    min = attr.ib(default=None, convert=float, 
                  validator=optional(instance_of(float)))
    max = attr.ib(default=None, convert=float, 
                  validator=optional(instance_of(float)))
    vals = attr.ib(default=attr.Factory(list), #convert=np.array,
                   #validator=instance_of(np.array)
                   )
    probs = attr.ib(default=attr.Factory(list), 
                    #convert=np.array,
                    #validator=instance_of(np.array)
                    )

    def check_dist_types(self):
        _check_dist_types(self)

    def sample(self, n=1000, return_scalar_array=False):
        return _sample(self, **kwargs)


@attr.s
class FaultAngle(object):
    # maybe need to do more to incorporate strike
    dist_type = attr.ib(default='unspecified', validator=validate_dist_type)
    mean = attr.ib(default=None, convert=float, 
                   validator=optional(validate_angle))
    median= attr.ib(default=None, convert=float, 
                    validator=optional(validate_angle))
    sd = attr.ib(default=None, convert=float, 
                 validator=optional(validate_angle))
    mad = attr.ib(default=None, convert=float, 
                  validator=optional(validate_angle))
    min = attr.ib(default=None, convert=float, 
                  validator=optional(validate_angle))
    max = attr.ib(default=None, convert=float, 
                  validator=optional(validate_angle))
    
    vals = attr.ib(default=attr.Factory(list), #convert=np.array,
                   #validator=instance_of(np.array)
                   )
    probs = attr.ib(default=attr.Factory(list), 
                    #convert=np.array,
                    #validator=instance_of(np.array)
                    )

    def check_dist_types(self):
        _check_dist_types(self)

    def sample(self, n=1000, return_scalar_array=False):
        return _sample(self, **kwargs)


@attr.s
class Age(object):
    units = attr.ib(default='m', validator=validate_distance_units)
    dist_type = attr.ib(default='unspecified', convert=float, 
                        validator=validate_dist_type)
    mean = attr.ib(default=None, convert=float, 
                   validator=optional(instance_of(float)))
    median= attr.ib(default=None,convert=float, 
                    validator=optional(instance_of(float)))
    sd = attr.ib(default=None, convert=float, 
                 validator=optional(instance_of(float)))
    mad = attr.ib(default=None, convert=float, 
                  validator=optional(instance_of(float)))
    min = attr.ib(default=None, convert=float, 
                  validator=optional(instance_of(float)))
    max = attr.ib(default=None, convert=float, 
                  validator=optional(instance_of(float)))
    vals = attr.ib(default=attr.Factory(list), #convert=np.array,
                   #validator=instance_of(np.array)
                   )
    probs = attr.ib(default=attr.Factory(list), 
                    #convert=np.array,
                    #validator=instance_of(np.array)
                    )

    def check_dist_types(self):
        _check_dist_types(self)

    def sample(self, n=1000, return_scalar_array=False):
        return _sample(self, **kwargs)





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


    # attribute class initializations
    def init_slip_components(self):
        self.init_offset()
        self.init_hor_separation()
        self.init_vert_separation()
        self.init_strike_slip()
        self.init_dip_slip()
        self.init_heave()


    def _init_age(self):
        if not self.age:
            self.age = Age(dist_type=self.age_dist_type,
                           mean=self.age_mean,
                           median=self.age_median,
                           sd=self.age_sd,
                           mad=self.age_mad,
                           min=self.age_min,
                           max=self.age_max,
                           vals=self.age_vals,
                           probs=self.age_probs)


    def _init_rake(self):
        if not self.rake:
            self.rake = FaultAngle(dist_type=self.rake_dist_type,
                                   mean=self.rake_mean,
                                   median=self.rake_median,
                                   sd=self.rake_sd,
                                   mad=self.rake_mad,
                                   min=self.rake_min,
                                   max=self.rake_max,
                                   vals=self.rake_vals,
                                   probs=self.rake_probs)

    def _init_dip(self):
        if not self.dip:
            self.dip = FaultAngle(dist_type=self.dip_dist_type,
                                  mean=self.dip_mean,
                                  median=self.dip_median,
                                  sd=self.dip_sd,
                                  mad=self.dip_mad,
                                  min=self.dip_min,
                                  max=self.dip_max,
                                  vals=self.dip_vals,
                                  probs=self.dip_probs)

    def _init_offset(self):
        if not self.offset:
            self.offset = SlipComponent(dist_type=self.offset_dist_type,
                                        mean=self.offset_mean,
                                        median=self.offset_median,
                                        sd=self.offset_sd,
                                        mad=self.offset_mad,
                                        min=self.offset_min,
                                        max=self.offset_max,
                                        vals=self.offset_vals,
                                        probs=self.offset_probs)

    def _init_hor_separation(self):
        if not self.hor_separation:
            self.hor_separation = SlipComponent(
                                       dist_type=self.hor_separation_dist_type,
                                        mean=self.hor_separation_mean,
                                        median=self.hor_separation_median,
                                        sd=self.hor_separation_sd,
                                        mad=self.hor_separation_mad,
                                        min=self.hor_separation_min,
                                        max=self.hor_separation_max,
                                        vals=self.hor_separation_vals,
                                        probs=self.hor_separation_probs)

    def _init_vert_separation(self):
        if not self.vert_separation:
            self.vert_separation = SlipComponent(
                                      dist_type=self.vert_separation_dist_type,
                                        mean=self.vert_separation_mean,
                                        median=self.vert_separation_median,
                                        sd=self.vert_separation_sd,
                                        mad=self.vert_separation_mad,
                                        min=self.vert_separation_min,
                                        max=self.vert_separation_max,
                                        vals=self.vert_separation_vals,
                                        probs=self.vert_separation_probs)

    def _init_dip_slip(self):
        if not self.dip_slip:
            self.dip_slip = SlipComponent(dist_type=self.dip_slip_dist_type,
                                        mean=self.dip_slip_mean,
                                        median=self.dip_slip_median,
                                        sd=self.dip_slip_sd,
                                        mad=self.dip_slip_mad,
                                        min=self.dip_slip_min,
                                        max=self.dip_slip_max,
                                        vals=self.dip_slip_vals,
                                        probs=self.dip_slip_probs)

    def _init_strike_slip(self):
        if not self.strike_slip:
            self.strike_slip = SlipComponent(
                                        dist_type=self.strike_slip_dist_type,
                                        mean=self.strike_slip_mean,
                                        median=self.strike_slip_median,
                                        sd=self.strike_slip_sd,
                                        mad=self.strike_slip_mad,
                                        min=self.strike_slip_min,
                                        max=self.strike_slip_max,
                                        vals=self.strike_slip_vals,
                                        probs=self.strike_slip_probs)

    def _init_heave(self):
        if not self.heave:
            self.heave = SlipComponent(dist_type=self.heave_dist_type,
                                        mean=self.heave_mean,
                                        median=self.heave_median,
                                        sd=self.heave_sd,
                                        mad=self.heave_mad,
                                        min=self.heave_min,
                                        max=self.heave_max,
                                        vals=self.heave_vals,
                                        probs=self.heave_probs)

    ### slip component propagation bullshit
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

    def _get_entered_slip_component(self):
        comp, comp_val = self._find_entered_slip_val().popitem()

        component = comp.split('_')[:-1]
        return component

    def propagate_scalar_slip_components(self):
        comp, comp_val = self._find_entered_slip_val().popitem()

        if comp == 'offset_mean':
            slip_comps = slip_components_from_offset(comp_val, self.dip_mean,
                                                     self.rake_mean)
        elif comp == 'offset_median':
            slip_comps = slip_components_from_offset(comp_val, self.dip_median,
                                                     self.rake_median)
        elif comp == 'hor_separation_mean':
            slip_comps = slip_components_from_hor_sep(comp_val, 
                                                      self.dip_mean,
                                                      self.rake_mean)
        elif comp == 'hor_separation_median':
            slip_comps = slip_components_from_hor_sep(comp_val, 
                                                      self.dip_median,
                                                      self.rake_median)
        elif comp == 'vert_separation_mean':
            slip_comps = slip_components_from_vert_sep(comp_val, 
                                                       self.dip_mean,
                                                       self.rake_mean)
        elif comp == 'vert_separation_median':
            slip_comps = slip_components_from_vert_sep(comp_val, 
                                                       self.dip_median,
                                                       self.rake_median)
        elif comp == 'strike_slip_mean':
            slip_comps = slip_components_from_strike_slip(comp_val, 
                                                          self.dip_mean,
                                                          self.rake_mean)
        elif comp == 'strike_slip_median':
            slip_comps = slip_components_from_strike_slip(comp_val, 
                                                          self.dip_median,
                                                          self.rake_median)
        elif comp == 'dip_slip_mean':
            slip_comps = slip_components_from_dip_slip(comp_val, 
                                                       self.dip_mean,
                                                       self.rake_mean)
        elif comp == 'dip_slip_median':
            slip_comps = slip_components_from_dip_slip(comp_val, 
                                                       self.dip_median,
                                                       self.rake_median)
        elif comp == 'heave_mean':
            slip_comps = slip_components_from_heave(comp_val, self.dip_mean,
                                                    self.rake_mean)
        elif comp == 'heave_median':
            slip_comps = slip_components_from_heave(comp_val, self.dip_median,
                                                    self.rake_median)
        
        if comp.split('_')[-1] == 'mean':
            self.propagate_slip_comps_from_offset_mean(slip_comps)
        
        elif comp.split('_')[-1] == 'median':
            self.propagate_slip_comps_from_offset_median(slip_comps)
        
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

    def propagate_slip_comps_from_offset_probs(self, offsets, dips, rakes):

        #self.hor_separation_vals, self.hor_separation_probs = kde(
        #                             hor_sep_from_offset(offsets, dips, rakes)
        raise NotImplementedError


    def _sample_entered_comp(self, n=1000):
        # this just needs to be for initial slip comp propagation.
        # it could have unintended consequences if multiple slip comps exist.
        # need to define other functions for sampling slip comps, even if they
        # share a lot of function(alitie)s.
        comp, comp_val = self._find_entered_slip_val().popitem() #if offest in?

        if comp.split('_')[0] == 'offset':
            return self.sample_offset(self, n)

        elif comp == 'hor_separation_mean':
            hor_seps = np.random.normal(comp_val, self.hor_separation_sd, n)
            rakes = self.sample_rakes(n)
            dips = self.sample_dips(n)

            offsets = offset_from_hor_sep(hor_seps, dips, rakes)

            self.propagate_slip_comps_from_offset_probs(offsets)



        elif comp == 'hor_separation_median':
            raise NotImplementedError
        
        elif comp == 'hor_separation_max':
            raise NotImplementedError

        elif comp == 'hor_separation_probs':
            raise NotImplementedError

        elif comp == 'vert_separation_mean':
            slip_comps = slip_components_from_vert_sep(comp_val, 
                                                      self.dip_mean,
                                                      self.rake_mean)
        elif comp == 'vert_separation_median':
            slip_comps = slip_components_from_vert_sep(comp_val, 
                                                      self.dip_median,
                                                      self.rake_median)
        elif comp == 'strike_slip_mean':
            slip_comps = slip_components_from_strike_slip(comp_val, 
                                                          self.dip_mean,
                                                         self.rake_mean)
        elif comp == 'strike_slip_median':
            slip_comps = slip_components_from_strike_slip(comp_val, 
                                                          self.dip_median,
                                                          self.rake_median)
        elif comp == 'dip_slip_mean':
            slip_comps = slip_components_from_dip_slip(comp_val, 
                                                       self.dip_mean,
                                                       self.rake_mean)
        elif comp == 'dip_slip_median':
            slip_comps = slip_components_from_dip_slip(comp_val, 
                                                       self.dip_median,
                                                       self.rake_median)
        elif comp == 'heave_mean':
            slip_comps = slip_components_from_heave(comp_val, self.dip_mean,
                                                    self.rake_mean)
        elif comp == 'heave_median':
            slip_comps = slip_components_from_heave(comp_val, self.dip_median,
                                                    self.rake_median)


    def mc_slip_comp_propagation(self):
        comp, comp_val = self._find_entered_slip_val().popitem()
        # sample slip comp
        # sample rakes, dips
        # calc offset dist
        # propagate other slip comps

        pass




    # sampling


    # consider sampling strategy for components
    # where offset (or other component)
    # is sampled and then transformed through those functions.
    # Not sure of importance of mc_slip_comp_propagation.

    def sample_rakes(self, n):
        if not self.rake:
            _init_rake(self)

        return self.rake.sample(n)

    def sample_dips(self, n):
        if not self.dip:
            _init_dip(self)

        return self.dip.sample(n)
    
    def sample_ages(self, n):
        if not age.dip:
            _init_age(self)

        return self.dip.sample(n)

    def sample_offsets(self, n):

        # convert whatever to offsets first?

        if not self.offset:

            
            _init_offset(self)

        return self.offset.sample(n)



    def sample(self, n):
        age_sample = self.sample_ages(n)
        offset_sample = self.sample_offsets(n)
        
        asl = len(age_sample)
        osl = len(offset_sample)
        
        if asl > osl:
            age_sample = age_sample[0:osl]
        elif osl > asl:
            offset_sample = offset_sample[0:asl]
        
        return age_sample, offset_sample

    # IO
    def to_dict(self, exclude=(None, 'unspecified', [])):
        out_dict = attr.asdict(self)

        out_dict = {k:v for k, v in out_dict.items() if v not in exclude}

        return out_dict
