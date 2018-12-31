import sys; sys.path.append('../')
from culpable.fault_projections import *

_strike_slip = 3.
_dip_slip = 4.
_offset = 5.
_dip = 30.
_rake = 53.130102354155987
#_rake = 180 - _rake
_vert_sep = 2.
_hor_sep = 2. * np.sqrt(3)
_heave = np.sqrt(_hor_sep**2 + _strike_slip**2)


# basic tests to check trig w/ non-edge-case values

def test_offset_from_vert_sep():
    offset = offset_from_vert_sep(_vert_sep, _dip, _rake)
    assert np.isclose(offset, _offset)


def test_vert_sep_from_offset():
    vert_sep = vert_sep_from_offset(_offset, _dip, _rake)
    assert np.isclose(vert_sep, _vert_sep)


def test_offset_from_hor_sep():
    offset = offset_from_hor_sep(_hor_sep, _dip, _rake)
    assert np.isclose(offset, _offset)


def test_hor_sep_from_offset():
    hor_sep = hor_sep_from_offset(_offset, _dip, _rake)
    assert np.isclose(hor_sep, _hor_sep)


def test_offset_from_strike_slip():
    offset = offset_from_strike_slip(_strike_slip, _dip, _rake)
    assert np.isclose(offset, _offset)


def test_strike_slip_from_offset():
    strike_slip = strike_slip_from_offset(_offset, _dip, _rake)
    assert np.isclose(strike_slip, _strike_slip)


def test_offset_from_dip_slip():
    offset = offset_from_dip_slip(_dip_slip, _dip, _rake)
    assert np.isclose(offset, _offset)


def test_dip_slip_from_offset():
    dip_slip = dip_slip_from_offset(_offset, _dip, _rake)
    assert np.isclose(dip_slip, _dip_slip)


def test_offset_from_heave():
    offset = offset_from_heave(_heave, _dip, _rake)
    assert np.isclose(offset, _offset, rtol=0.01)


def test_heave_from_offset():
    heave = heave_from_offset(_offset, _dip, _rake)
    assert np.isclose(heave, _heave, rtol=0.01)


def test_beta_from_rake_dip():
    pass


def test_apparent_dip_from_dip_rake():
    dips = [81, 79, 30, 30, 30]
    rakes = [17, 50, 20, 40, 60]
    ads = [17, 49, 10, 19, 26]

    for i, dip in enumerate(dips):
        rake = rakes[i]
        ad_true = ads[i]
        ad = apparent_dip_from_dip_rake(dip, rake)
        assert np.abs(ad - ad_true) < 1


def test_dip_slip_from_vert_sep():
    dip_slip = dip_slip_from_vert_sep(_vert_sep, _dip, _rake)
    assert np.isclose(dip_slip, _dip_slip)


def test_vert_sep_from_dip_slip():
    vert_sep = vert_sep_from_dip_slip(_dip_slip, _dip, _rake)
    assert np.isclose(vert_sep, _vert_sep)


def test_dip_slip_from_hor_sep():
    dip_slip = dip_slip_from_hor_sep(_hor_sep, _dip, _rake)
    assert np.isclose(dip_slip, _dip_slip)


def test_hor_sep_from_dip_slip():
    hor_sep = hor_sep_from_dip_slip(_dip_slip, _dip, _rake)
    assert np.isclose(hor_sep, _hor_sep)


def test_dip_slip_from_strike_slip():
    dip_slip = dip_slip_from_strike_slip(_strike_slip, _dip, _rake)
    assert np.isclose(dip_slip, _dip_slip)


def test_strike_slip_from_dip_slip():
    strike_slip = strike_slip_from_dip_slip(_dip_slip, _dip, _rake)
    assert np.isclose(strike_slip, _strike_slip)


def test_dip_slip_from_heave():
    dip_slip = dip_slip_from_heave(_heave, _dip, _rake)
    assert np.isclose(dip_slip, _dip_slip, rtol=0.01)


def test_heave_from_dip_slip():
    heave = heave_from_dip_slip(_dip_slip, _dip, _rake)
    assert np.isclose(heave, _heave, rtol=0.01)


def test_strike_slip_from_vert_sep():
    strike_slip = strike_slip_from_vert_sep(_vert_sep, _dip, _rake)
    assert np.isclose(strike_slip, _strike_slip)


def test_vert_sep_from_strike_slip():
    vert_sep = vert_sep_from_strike_slip(_strike_slip, _dip, _rake)
    assert np.isclose(vert_sep, _vert_sep)


def test_strike_slip_from_hor_sep():
    strike_slip = strike_slip_from_hor_sep(_hor_sep, _dip, _rake)
    assert np.isclose(strike_slip, _strike_slip)


def test_hor_sep_from_strike_slip():
    hor_sep = hor_sep_from_strike_slip(_strike_slip, _dip, _rake)
    assert np.isclose(hor_sep, _hor_sep)


def test_strike_slip_from_heave():
    strike_slip = strike_slip_from_heave(_heave, _dip, _rake)
    assert np.isclose(strike_slip, _strike_slip, rtol=0.01)


def test_heave_from_strike_slip():
    heave = heave_from_strike_slip(_strike_slip, _dip, _rake)
    assert np.isclose(heave, _heave, rtol=0.01)


def test_vert_sep_from_hor_sep():
    vert_sep = vert_sep_from_hor_sep(_hor_sep, _dip, _rake)
    assert np.isclose(vert_sep, _vert_sep)


def test_hor_sep_from_vert_sep():
    hor_sep = hor_sep_from_vert_sep(_vert_sep, _dip, _rake)
    assert np.isclose(hor_sep, _hor_sep)


def test_vert_sep_from_heave():
    vert_sep = vert_sep_from_heave(_heave, _dip, _rake)
    assert np.isclose(vert_sep, _vert_sep, rtol=0.01)


def test_heave_from_vert_sep():
    heave = heave_from_vert_sep(_vert_sep, _dip, _rake)
    assert np.isclose(heave, _heave, rtol=0.01)


def test_hor_sep_from_heave():
    hor_sep = hor_sep_from_heave(_heave, _dip, _rake)
    assert np.isclose(hor_sep, _hor_sep, rtol=0.01)


def test_heave_from_hor_sep():
    heave = heave_from_hor_sep(_hor_sep, _dip, _rake)
    assert np.isclose(heave, _heave, rtol=0.01)


