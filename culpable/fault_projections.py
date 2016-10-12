import numpy as np
from numpy import sin, cos, tan, degrees, radians, arctan

# Slip projections
## To/From offset
def offset_from_vert_sep(vert_sep, dip, rake=90.):
    dip_slip = dip_slip_from_vert_sep(vert_sep, dip, rake)
    return offset_from_dip_slip(dip_slip, dip, rake)


def vert_sep_from_offset(offset, dip, rake=90.):
    dip_slip = dip_slip_from_offset(offset, dip, rake)
    return vert_sep_from_dip_slip(dip_slip, dip, rake)


def offset_from_hor_sep(hor_sep, dip, rake=90.):
    dip_slip = dip_slip_from_hor_sep(hor_sep, dip, rake)
    return offset_from_dip_slip(dip_slip, dip, rake)


def hor_sep_from_offset(offset, dip, rake=90.):
    dip_slip = dip_slip_from_offset(offset, dip, rake)
    return hor_sep_from_dip_slip(dip_slip, dip, rake)


def offset_from_strike_slip(strike_slip, dip, rake=0.):
    return strike_slip / cos( radians(rake))


def strike_slip_from_offset(offset, dip, rake=0.):
    return offset * cos( radians(rake))


def offset_from_dip_slip(dip_slip, dip, rake=90.):
    return dip_slip / sin( radians(rake))


def dip_slip_from_offset(offset, dip, rake=90.):
    return offset * sin( radians(rake))


def heave_from_offset(offset, dip, rake=90.):
    apparent_dip = apparent_dip_from_dip_rake(dip, rake)
    return offset * cos( radians(apparent_dip))


def offset_from_heave(heave, dip, rake=90.):
    apparent_dip = apparent_dip_from_dip_rake(dip, rake)
    return heave / cos( radians(apparent_dip))


## Others
def beta_from_rake_dip(rake, dip):
    '''
    Returns beta, the angle (in degrees) between the strike and the
    trend of apparent dip.
    '''
    return degrees( arctan( tan(radians(rake)) * cos(radians(dip))))


def apparent_dip_from_dip_rake(rake, dip):
    return degrees( arctan( tan(radians(rake)) * sin(radians(dip))))


def hor_sep_from_vert_sep(vert_sep, dip, rake=90.):
    offset = offset_from_vert_sep(vert_sep, dip, rake)
    return hor_sep_from_offset(offset, dip, rake)


def vert_sep_from_hor_sep(hor_sep, dip, rake=90.):
    offset = offset_from_hor_sep(hor_sep, dip, rake)
    return vert_sep_from_offset(offset, dip, rake)


def dip_slip_from_vert_sep(vert_sep, dip, rake=90.):
    return vert_sep / sin(radians(dip))


def vert_sep_from_dip_slip(dip_slip, dip, rake=90.):
    return dip_slip * sin(radians(dip))


def strike_slip_from_vert_sep(vert_sep, dip, rake=0.):
    offset = offset_from_vert_sep(vert_sep, dip, rake)
    return strike_slip_from_offset(offset, dip, rake)


def vert_sep_from_strike_slip(strike_slip, dip, rake=0.):
    offset = offset_from_strike_slip(strike_slip, dip, rake)
    return vert_sep_from_offset(offset, dip, rake)


def heave_from_vert_sep(vert_sep, dip, rake=90.):
    offset = offset_from_vert_sep(vert_sep, dip, rake)
    return heave_from_offset(offset, dip, rake)


def vert_sep_from_heave(heave, dip, rake=90.):
    offset = offset_from_heave(heave, dip, rake)
    return vert_sep_from_offset(offset, dip, rake)


def hor_sep_from_dip_slip(dip_slip, dip, rake=90.):
    return dip_slip * cos(radians(dip))


def dip_slip_from_hor_sep(hor_sep, dip, rake=90.):
    return hor_sep / cos(radians(dip))


def hor_sep_from_strike_slip(strike_slip, dip, rake=0.):
    offset = offset_from_strike_slip(strike_slip, dip, rake)
    return hor_sep_from_offset(offset, dip, rake)


def strike_slip_from_hor_sep(hor_sep, dip, rake=0.):
    offset = offset_from_hor_sep(hor_sep, dip, rake)
    return strike_slip_from_offset(offset, dip, rake)


def hor_sep_from_heave(heave, dip, rake=90.):
    offset = offset_from_heave(heave, dip, rake)
    return hor_sep_from_offset(offset, dip, rake)


def heave_from_hor_sep(hor_sep, dip, rake=90.):
    offset = offset_from_hor_sep(hor_sep, dip, rake)
    return heave_from_offset(offset, dip, rake)


def dip_slip_from_heave(heave, dip, rake=90.):
    offset = offset_from_heave(heave, dip, rake)
    return dip_slip_from_offset(offset, dip, rake)


def heave_from_dip_slip(dip_slip, dip, rake=90.):
    offset = offset_from_dip_slip(dip_slip, dip, rake)
    return heave_from_offset(offset, dip, rake)


def dip_slip_from_strike_slip(strike_slip, dip, rake):
    offset = offset_from_strike_slip(strike_slip, dip, rake)
    return dip_slip_from_offset(offset, dip, rake)


def strike_slip_from_dip_slip(dip_slip, dip, rake):
    offset = offset_from_dip_slip(dip_slip, dip, rake)
    return strike_slip_from_offset(offset, dip, rake)


def heave_from_strike_slip(strike_slip, dip, rake=0.):
    hs = hor_sep_from_strike_slip(strike_slip, dip, rake)
    return np.sqrt(strike_slip**2 + hs**2)


def strike_slip_from_heave(heave, dip, rake=0.):
    offset = offset_from_heave(heave, dip, rake)
    return strike_slip_from_offset(offset, dip, rake)



