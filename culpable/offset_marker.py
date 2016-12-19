from collections import namedtuple
import numpy as np

import attr
from attr.validators import instance_of, optional

from .stats import (inverse_transform_sample, sample_from_bounded_normal,
                    pdf_from_samples, trim_pdf)
from .fault_projections import *

def opt(convert):
    """Invoke the subconverter only if the value is present."""
    def optional_converter(val):
        if val is None:
            return None
        return convert(val)
    return optional_converter


def validate_age_units(instance, time_units, value):
    '''
    Validator for `age_units` attribute in `OffsetMarker` class
    '''
    acceptable_age_units = ['a', 'cal_yr_CE', 'cal_yr_BP', 'ka', 'Ma']
    if not value in acceptable_age_units:
        raise ValueError(
            "{} not acceptable unit; only {}".format(value,
                                                     acceptable_age_units))


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

def validate_dip_rake(instance, attribute, value):
    if (value < 0.) or (value > 90.):
        raise ValueError("Only angles between 0 and 90 acceptable")


def validate_strike(instance, attribute, value):
    if (value < 0.) or (value > 360.):
        raise ValueError("Only angles between 0 and 90 acceptable")


def validate_angle(instance, attribute, value):
    pass


def validate_age(instance, attribute, value):
    pass


def validate_measured_offset(instance, attribute, value):
    pass


def validate_offset_comp(instance, attribute, value):
    acceptable_offset_comps = ['offset', 'vert_separation', 'hor_separation',
                               'strike_slip', 'dip_slip']
    if value not in acceptable_offset_comps:
        raise ValueError("{} not acceptable offset component")


def _sample(self, n=1000, return_scalar_array=False):
    '''
    Main sampling function. It is defined as a function for use in multiple
    classes as a method without inheritance, and is not meant to be called
    as a separate function; nonetheless, it can if a class w/ the requisite
    attributes (dist_type, mean, sd, max, min, etc.) are present.

    Parameters
    ----------
    n : int 
        Number of samples to be returned
    return_scalar_array: Bool, default=False
        Determines whether a single scalar (e.g., 4) or an array of
        equal scalars (e.g., [4, 4, 4, 4, 4]) is returned when there
        is no uncertainty in the distribution, i.e. when all inputs
        are scalar.
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
    '''Checks to consistency among arguments and distribution types.
    May not be up to date.'''
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
    '''Class for a component of fault slip. It contains methods for
    sampling as well as attributes for the values and categories of
    slip.'''
    units = attr.ib(default='m', 
                    validator=validate_distance_units)
    dist_type = attr.ib(default='unspecified', 
                        validator=validate_dist_type)
    mean = attr.ib(default=None, 
                   convert=opt(float),
                   validator=optional(instance_of(float)))
    median= attr.ib(default=None,
                    convert=opt(float), 
                    validator=optional(instance_of(float)))
    sd = attr.ib(default=None, 
                 convert=opt(float), 
                 validator=optional(instance_of(float)))
    mad = attr.ib(default=None, 
                  convert=opt(float), 
                  validator=optional(instance_of(float)))
    min = attr.ib(default=None, 
                  convert=opt(float), 
                  validator=optional(instance_of(float)))
    max = attr.ib(default=None, 
                  convert=opt(float), 
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
        return _sample(self, n=n, return_scalar_array=return_scalar_array)


@attr.s
class FaultAngle(object):
    '''Basic class for angular fault geometry parameters.'''
    # maybe need to do more to incorporate strike
    dist_type = attr.ib(default='unspecified', 
                        validator=validate_dist_type)
    mean = attr.ib(default=None, 
                   convert=opt(float), 
                   validator=optional(validate_angle))
    median= attr.ib(default=None, 
                    convert=opt(float), 
                    validator=optional(validate_angle))
    sd = attr.ib(default=None, 
                 convert=opt(float), 
                 validator=optional(validate_angle))
    mad = attr.ib(default=None, 
                  convert=opt(float), 
                  validator=optional(validate_angle))
    min = attr.ib(default=None, 
                  convert=opt(float), 
                  validator=optional(validate_angle))
    max = attr.ib(default=None, 
                  convert=opt(float), 
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
        return _sample(self, n=n, return_scalar_array=return_scalar_array)


@attr.s
class Age(object):
    '''Class for the age of an offset marker.  Contains functionality
    for sampling.'''
    units = attr.ib(default='ka', 
                    validator=validate_age_units)
    dist_type = attr.ib(default='unspecified', 
                        validator=validate_dist_type)
    mean = attr.ib(default=None, 
                   convert=opt(float), 
                   validator=optional(instance_of(float)))
    median= attr.ib(default=None,
                    convert=opt(float), 
                    validator=optional(instance_of(float)))
    sd = attr.ib(default=None, 
                 convert=opt(float), 
                 validator=optional(instance_of(float)))
    mad = attr.ib(default=None, 
                  convert=opt(float), 
                  validator=optional(instance_of(float)))
    min = attr.ib(default=None, 
                  convert=opt(float), 
                  validator=optional(instance_of(float)))
    max = attr.ib(default=None, 
                  convert=opt(float), 
                  validator=optional(instance_of(float)))
    vals = attr.ib(default=attr.Factory(list), 
                   #convert=opt(np.float_),
                   #validator=instance_of(np.array)
                   )
    probs = attr.ib(default=attr.Factory(list), 
                    #convert=np.array,
                    #validator=instance_of(np.array)
                    )

    def check_dist_types(self):
        _check_dist_types(self)

    def sample(self, n=1000, return_scalar_array=False):
        return _sample(self, n=n, return_scalar_array=return_scalar_array)


@attr.s
class OffsetMarker(object):
    '''Class describing a faulted geologic unit. Contains many attributes
    for geometry, slip/offset, and age, as well as sampling.'''
    source = attr.ib(default=None)
    metadata = attr.ib(default=None)
    name = attr.ib(default=None, validator=optional(instance_of(str)))

    # offset stuff
    measured_offset = attr.ib(default=None, 
                              convert=opt(np.float_),
                              validator=optional(validate_measured_offset))
    
    measured_offset_err = attr.ib(default=None, 
                                  convert=opt(np.float_),
                                  validator=optional(validate_measured_offset))

    measured_offset_component = attr.ib(default='offset',
                                        validator=validate_offset_comp)

    measured_offset_units = attr.ib(default='m', 
                                    validator=validate_distance_units)
    
    measured_offset_dist_type = attr.ib(default='unspecified',
                                        validator=validate_dist_type)
    
    # age stuff
    age = attr.ib(default=None, 
                  convert=opt(np.float_),
                  validator=optional(validate_age))
    
    age_err = attr.ib(default=None, 
                      convert=opt(np.float_),
                      validator=optional(validate_age))

    age_units = attr.ib(default='ka', 
                        validator=validate_age_units)
    
    age_dist_type = attr.ib(default='unspecified',
                            validator=validate_dist_type)

    # strike stuff
    strike = attr.ib(default=None, 
                     convert=opt(np.float_),
                     validator=optional(validate_strike))
    
    strike_err = attr.ib(default=None, 
                         convert=opt(np.float_),
                         validator=optional(validate_strike))

    strike_dist_type = attr.ib(default='unspecified',
                               validator=validate_dist_type)

    # dip stuff
    dip = attr.ib(default=None, 
                  convert=opt(np.float_),
                  validator=optional(validate_dip_rake))
    
    dip_err = attr.ib(default=None, 
                      convert=opt(np.float_),
                      validator=optional(validate_dip_rake))

    dip_dist_type = attr.ib(default='unspecified',
                               validator=validate_dist_type)

    # rake stuff
    rake = attr.ib(default=None, 
                   convert=opt(np.float_),
                   validator=optional(validate_dip_rake))
    
    rake_err = attr.ib(default=None, 
                       convert=opt(np.float_),
                       validator=optional(validate_dip_rake))

    rake_dist_type = attr.ib(default='unspecified',
                               validator=validate_dist_type)

    #TODO: add trend


    #####
    # misc methods
    #####
    def trim_ages(self, min=None, max=None):
        # TODO: Make some logic for non-arbitrary distributions

        if self.age_dist_type == 'arbitrary':
            self.age, self.age_err = trim_pdf(self.age, self.age_err, min, max)
            self.ages.vals, self.ages.probs = self.age, self.age_err

        elif self.age_dist_type == 'uniform':
            age_min = self.age - self.age_err
            age_max = self.age + self.age_err

            if min is not None:
            
                if age_min < min:
                    age_min = min

            if max is not None:
                if age_max > max:
                    age_max = max

            age_mean = (age_min + age_max) / 2
            age_err = age_mean - age_min

            self.age = age_mean
            self.age_err = age_err
            self.ages = Age(min=age_min,
                            max=age_max,
                            units=self.age_units,
                            dist_type='uniform')

    
    #####
    # attribute class initializations
    #####
    def init(self):
        '''Post __init__() call that does additional logic, creating
        classes for age, fault geometry and measured offsets.
        Will be called automatically as a post-init hook once the
        `attrs` library has that capability; for now needs to be
        called manually before doing any sampling or other work.'''
        if self.age is not None:
            self._init_age()
        if self.strike is not None:
            self._init_strike()
        if self.dip is not None:
            self._init_dip()
        if self.rake is not None:
            self._init_rake()
        if self.measured_offset is not None:
            self._init_obs_offset()
    
    def _init_age(self):
        if not hasattr(self, 'ages'):
       #    self.ages
       #except AttributeError:
            if self.age_dist_type == 'normal':
                self.ages = Age(mean=self.age,
                                sd=self.age_err,
                                units=self.age_units,
                                dist_type=self.age_dist_type)

            elif self.age_dist_type == 'uniform':
                self.ages = Age(min=self.age-self.age_err,
                                max=self.age+self.age_err,
                                units=self.age_units,
                                dist_type=self.age_dist_type)

            elif self.age_dist_type == 'laplacian':
                self.ages = Age(median=self.age,
                                mad=self.age_err,
                                units=self.age_units,
                                dist_type=self.age_dist_type)

            elif self.age_dist_type == 'arbitrary':
                self.ages = Age(vals=self.age,
                                probs=self.age_err,
                                units=self.age_units,
                                dist_type=self.age_dist_type)

            elif self.age_dist_type == 'scalar':
                self.ages = Age(mean=self.age,
                                sd=0.,
                                units=self.age_units,
                                dist_type=self.age_dist_type)

            elif self.age_dist_type == 'unspecified':
                #TODO: Put some inference here
                raise Exception('Please specify age distribution type')

    def _init_rake(self):
        if not hasattr(self, 'rakes'):
            if self.rake_dist_type == 'normal':
                self.rakes = FaultAngle(mean=self.rake,
                                        sd=self.rake_err,
                                        dist_type=self.rake_dist_type)

            elif self.rake_dist_type == 'uniform':
                self.rakes = FaultAngle(min=self.rake-self.rake_err,
                                        max=self.rake+self.rake_err,
                                        dist_type=self.rake_dist_type)

            elif self.rake_dist_type == 'laplacian':
                self.rakes = FaultAngle(median=self.rake,
                                        mad=self.rake_err,
                                        dist_type=self.rake_dist_type)

            elif self.rake_dist_type == 'arbitrary':
                self.rakes = FaultAngle(vals=self.rake,
                                    probs=self.rake_err,
                                    dist_type=self.rake_dist_type)

            elif self.rake_dist_type == 'scalar':
                self.rakes = FaultAngle(mean=self.rake,
                                        sd=0.,
                                        dist_type=self.rake_dist_type)

            elif self.rake_dist_type == 'unspecified':
                #TODO: Put some inference here
                raise Exception('Please specify rake distribution type')

    def _init_strike(self):
        if not hasattr(self, 'strikes'):
            if self.strike_dist_type == 'normal':
                self.strikes = FaultAngle(mean=self.strike,
                                          sd=self.strike_err,
                                          dist_type=self.strike_dist_type)

            elif self.strike_dist_type == 'uniform':
                self.strikes = FaultAngle(min=self.strike-self.strike_err,
                                          max=self.strike+self.strike_err,
                                          dist_type=self.strike_dist_type)

            elif self.strike_dist_type == 'laplacian':
                self.strikes = FaultAngle(median=self.strike,
                                          mad=self.strike_err,
                                          dist_type=self.strike_dist_type)

            elif self.strike_dist_type == 'arbitrary':
                self.strikes = FaultAngle(vals=self.strike,
                                          probs=self.strike_err,
                                          dist_type=self.strike_dist_type)

            elif self.strike_dist_type == 'scalar':
                self.strikes = FaultAngle(mean=self.strike,
                                          sd=0.,
                                          dist_type=self.strike_dist_type)

            elif self.strike_dist_type == 'unspecified':
                #TODO: Put some inference here
                if self.strike is not None:
                    raise Exception('Please specify strike distribution type')

    def _init_dip(self):
        if not hasattr(self, 'dips'):
            if self.dip_dist_type == 'normal':
                self.dips = FaultAngle(mean=self.dip,
                                       sd=self.dip_err,
                                       dist_type=self.dip_dist_type)

            elif self.dip_dist_type == 'uniform':
                self.dips = FaultAngle(min=self.dip-self.dip_err,
                                       max=self.dip+self.dip_err,
                                       dist_type=self.dip_dist_type)

            elif self.dip_dist_type == 'laplacian':
                self.dips = FaultAngle(median=self.dip,
                                       mad=self.dip_err,
                                       dist_type=self.dip_dist_type)

            elif self.dip_dist_type == 'arbitrary':
                self.dips = FaultAngle(vals=self.dip,
                                       probs=self.dip_err,
                                       dist_type=self.dip_dist_type)

            elif self.dip_dist_type == 'scalar':
                self.dips = FaultAngle(mean=self.dip,
                                       sd=0.,
                                       dist_type=self.dip_dist_type)

            elif self.dip_dist_type == 'unspecified':
                #TODO: Put some inference here
                raise Exception('Please specify dip distribution type')

    def _init_obs_offset(self):
        if not hasattr(self, 'offset'):
           if self.measured_offset_dist_type == 'normal':
               self.obs_offsets = SlipComponent(
                                     mean=self.measured_offset,
                                     sd=self.measured_offset_err,
                                     units=self.measured_offset_units,
                                     dist_type=self.measured_offset_dist_type)
          
           elif self.measured_offset_dist_type == 'uniform':
               self.obs_offsets = SlipComponent(
                                     min=(self.measured_offset - 
                                          self.measured_offset_err),
                                     max=(self.measured_offset + 
                                          self.measured_offset_err),
                                     units=self.measured_offset_units,
                                     dist_type=self.measured_offset_dist_type)

           elif self.measured_offset_dist_type == 'laplacian':
               self.obs_offsets = SlipComponent(
                                     median=self.measured_offset,
                                     mad=self.measured_offset_err,
                                     units=self.measured_offset_units,
                                     dist_type=self.measured_offset_dist_type)

           elif self.measured_offset_dist_type == 'arbitrary':
               self.obs_offsets = SlipComponent(
                                     vals=self.measured_offset,
                                     probs=self.measured_offset_err,
                                     units=self.measured_offset_units,
                                     dist_type=self.measured_offset_dist_type)

           elif self.measured_offset_dist_type == 'scalar':
               self.obs_offsets = SlipComponent(
                                     mean=self.measured_offset,
                                     sd=0.,
                                     units=self.measured_offset_units,
                                     dist_type=self.measured_offset_dist_type)

           elif self.measured_offset_dist_type == 'unspecified':
               #TODO: Put some inference here
               raise Exception(
                           'Please specify measured_offset distribution type')

           self.obs_offset_to_offset()

    def obs_offset_to_offset(self):
        '''Takes obs_offset class from OffsetMarker initialization and
           converts values to the offset class.'''

        if self.measured_offset_component == 'offset':
            self.offsets = self.obs_offsets

        else:
            try:
                dip_samples = self.dips.sample()
                rake_samples = self.rakes.sample()
                meas_off_samples = self.obs_offsets.sample()
                
            except Exception as e:
                print(e)
            
            if self.measured_offset_component == 'vert_separation':
                off_samples = offset_from_vert_sep(meas_off_samples,
                                                   dip_samples,
                                                   rake_samples)

            elif self.measured_offset_component == 'hor_separation':
                off_samples = offset_from_hor_sep(meas_off_samples,
                                                   dip_samples,
                                                   rake_samples)

            elif self.measured_offset_component == 'dip_slip':
                off_samples = offset_from_dip_slip(meas_off_samples,
                                                   dip_samples,
                                                   rake_samples)

            elif self.measured_offset_component == 'strike_slip':
                off_samples = offset_from_hor_sep(meas_off_samples,
                                                   dip_samples,
                                                   rake_samples)

            else:
                raise NameError("{} not acceptable measured offset component"
                                .format(self.measured_offset_component))

            if np.isscalar(off_samples):
                self.offsets = SlipComponent(mean=off_samples,
                                             sd=0.,
                                             units=self.measured_offset_units,
                                             dist_type='scalar')
            else:

                off_vals, off_probs = pdf_from_samples(off_samples)

                self.offsets = SlipComponent(mean=np.mean(off_samples),
                                             vals=off_vals,
                                             probs=off_probs,
                                             units=self.measured_offset_units,
                                             dist_type='arbitrary')

    ######
    # sampling
    # These all default to arrays of scalars in that instance, unlike
    # the base class methods.
    ######

    def sample_rakes(self, n, return_scalar_array=True):
        '''Samples rakes from the rake distribution parameters. 

        Parameters
        ----------
        n : int 
            Number of samples to be returned
        return_scalar_array: Bool, default=False
            Determines whether a single scalar (e.g., 4) or an array of
            equal scalars (e.g., [4, 4, 4, 4, 4]) is returned when there
            is no uncertainty in the distribution, i.e. when all inputs
            are scalar.
        '''
        if not hasattr(self, 'rakes'):
            self._init_rake(self)
        return self.rakes.sample(n, return_scalar_array)

    def sample_dips(self, n, return_scalar_array=True):
        '''Samples dips from the dip distribution parameters. 

        Parameters
        ----------
        n : int 
            Number of samples to be returned
        return_scalar_array: Bool, default=False
            Determines whether a single scalar (e.g., 4) or an array of
            equal scalars (e.g., [4, 4, 4, 4, 4]) is returned when there
            is no uncertainty in the distribution, i.e. when all inputs
            are scalar.
        '''
        if not hasattr(self, 'dips'):
            self._init_dip(self)
        return self.dips.sample(n, return_scalar_array)
    
    def sample_ages(self, n, return_scalar_array=True):
        '''Samples ages from the age distribution parameters. 

        Parameters
        ----------
        n : int 
            Number of samples to be returned
        return_scalar_array: Bool, default=False
            Determines whether a single scalar (e.g., 4) or an array of
            equal scalars (e.g., [4, 4, 4, 4, 4]) is returned when there
            is no uncertainty in the distribution, i.e. when all inputs
            are scalar.
        '''
        if not hasattr(self, 'ages'):
            self._init_age(self)
        return self.ages.sample(n, return_scalar_array)

    def sample_offsets(self, n, return_scalar_array=True):
        '''Samples offsets from the offset distribution parameters. 

        Parameters
        ----------
        n : int 
            Number of samples to be returned
        return_scalar_array: Bool, default=False
            Determines whether a single scalar (e.g., 4) or an array of
            equal scalars (e.g., [4, 4, 4, 4, 4]) is returned when there
            is no uncertainty in the distribution, i.e. when all inputs
            are scalar.
        '''
        if not self.offsets:
            self._init_offset(self)
        return self.offsets.sample(n, return_scalar_array)

    def sample_vert_separations(self, n, return_scalar_array=True):
        '''Samples vertical separations from the vertical separation 
           distribution parameters, or offset data and fault geometry.

        Parameters
        ----------
        n : int 
            Number of samples to be returned
        return_scalar_array: Bool, default=False
            Determines whether a single scalar (e.g., 4) or an array of
            equal scalars (e.g., [4, 4, 4, 4, 4]) is returned when there
            is no uncertainty in the distribution, i.e. when all inputs
            are scalar.
        '''
        if self.measured_offset_component == 'vert_separation':
            return self.obs_offsets.sample(n, return_scalar_array)
        else:
            if not hasattr(self, 'offsets'):
                self.obs_offset_to_offset()
            return vert_sep_from_offset(self.offsets.sample(n, 
                                                          return_scalar_array),
                                        self.dips.sample(n, 
                                                          return_scalar_array),
                                        self.rakes.sample(n, 
                                                          return_scalar_array))
                                                        
    def sample_hor_separations(self, n, return_scalar_array=True):
        '''Samples horizontal separations from the horizontal separation 
           distribution parameters, or offset data and fault geometry.

        Parameters
        ----------
        n : int 
            Number of samples to be returned
        return_scalar_array: Bool, default=False
            Determines whether a single scalar (e.g., 4) or an array of
            equal scalars (e.g., [4, 4, 4, 4, 4]) is returned when there
            is no uncertainty in the distribution, i.e. when all inputs
            are scalar.
        '''
        if self.measured_offset_component == 'hor_separation':
            return self.obs_offsets.sample(n, return_scalar_array)
        else:
            if not hasattr(self, 'offsets'):
                self.obs_offset_to_offset()
            return hor_sep_from_offset(self.offsets.sample(n, 
                                                          return_scalar_array),
                                       self.dips.sample(n, 
                                                          return_scalar_array),
                                       self.rakes.sample(n, 
                                                          return_scalar_array))

    def sample_dip_slips(self, n, return_scalar_array=True):
        '''Samples dip slips from the dip slip 
           distribution parameters, or offset data and fault geometry.

        Parameters
        ----------
        n : int 
            Number of samples to be returned
        return_scalar_array: Bool, default=False
            Determines whether a single scalar (e.g., 4) or an array of
            equal scalars (e.g., [4, 4, 4, 4, 4]) is returned when there
            is no uncertainty in the distribution, i.e. when all inputs
            are scalar.
        '''
        if self.measured_offset_component == 'dip_slip':
            return self.obs_offsets.sample(n, return_scalar_array)
        else:
            if not hasattr(self, 'offsets'):
                self.obs_offset_to_offset()
            return dip_slip_from_offset(self.offsets.sample(n, 
                                                          return_scalar_array),
                                        self.dips.sample(n, 
                                                          return_scalar_array),
                                        self.rakes.sample(n, 
                                                          return_scalar_array))

    def sample_strike_slips(self, n, return_scalar_array=True):
        '''Samples strike slips from the strike slip 
           distribution parameters, or offset data and fault geometry.

        Parameters
        ----------
        n : int 
            Number of samples to be returned
        return_scalar_array: Bool, default=False
            Determines whether a single scalar (e.g., 4) or an array of
            equal scalars (e.g., [4, 4, 4, 4, 4]) is returned when there
            is no uncertainty in the distribution, i.e. when all inputs
            are scalar.
        '''
        if self.measured_offset_component == 'strike_slip':
            return self.obs_offsets.sample(n, return_scalar_array)
        else:
            if not hasattr(self, 'offsets'):
                self.obs_offset_to_offset()
            return strike_slip_from_offset(self.offsets.sample(n, 
                                                          return_scalar_array),
                                           self.dips.sample(n, 
                                                          return_scalar_array),
                                           self.rakes.sample(n, 
                                                          return_scalar_array))

    def sample(self, n, component='offset', return_scalar_array=True):
        '''Samples ages and offsets from the distribution parameters and
        fault geometry.

        Parameters
        ----------
        n : int 
            Number of samples to be returned
        component: str
            Desired component of slip (e.g., dip_slip, vert_separation).
            Slip components are transformed from offset and fault geometry
            parameters if the desired component is not what the OffsetMarker
            was initialized with.
        return_scalar_array: Bool, default=False
            Determines whether a single scalar (e.g., 4) or an array of
            equal scalars (e.g., [4, 4, 4, 4, 4]) is returned when there
            is no uncertainty in the distribution, i.e. when all inputs
            are scalar.
        '''
        age_sample = self.sample_ages(n, return_scalar_array)

        if component == 'offset':
            offset_sample = self.sample_offsets(n, return_scalar_array)
        elif component == 'vert_separation':
            offset_sample = self.sample_vert_separation(n, return_scalar_array)
        elif component == 'hor_separation':
            offset_sample = self.sample_hor_separation(n, return_scalar_array)
        elif component == 'dip_slip':
            offset_sample = self.sample_dip_slip(n, return_scalar_array)
        elif component == 'strike_slip':
            offset_sample = self.sample_strike_slip(n, return_scalar_array)
        else:
            raise NameError('{} not acceptable slip component'
                            .format(component))

        # make namedtuple class w/ ages, offsets and units annotated
        OffsetMarkerSample = namedtuple('OffsetMarkerSample',
                                        ['ages_{}'.format(self.ages.units),
                                         'offsets_{}'
                                         .format(self.offsets.units)])

        if np.isscalar(age_sample) and np.isscalar(offset_sample):
            return OffsetMarkerSample(age_sample, offset_sample)
        else:
            asl = len(age_sample)
            osl = len(offset_sample)
           
            while asl < n:
                age_sample = np.append(age_sample, self.sample_ages(n))
                age_sample = age_sample[:n]
                asl = len(age_sample)

            while osl < n:
                offset_sample = np.append(offset_sample,
                                          self.sample_offsets(n))
                offset_sample = offset_sample[:n]
                osl = len(offset_sample)

            return OffsetMarkerSample(age_sample, offset_sample)

    # IO
    def to_dict(self, exclude=(None, 'unspecified', [])):
        '''Returns a dictionary of initial attributes, with most default
        or empty values stripped out.'''
        out_dict = attr.asdict(self)

        out_dict = {k:v for k, v in out_dict.items() if v not in exclude}

        return out_dict
