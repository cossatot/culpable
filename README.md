Culpable
========

Python module containing functionality for performing quantitative analysis of
faulting. Current applications are in slip rate/history estimation and
paleoseismology.

[![DOI](https://zenodo.org/badge/70557324.svg)](https://zenodo.org/badge/latestdoi/70557324)


# Installation

1. Download the package, and enter the outer `culpable` directory

2. create a new Python environment if desired

3. Install with:

    ```python
    python setup.py install
    ```
    
    or 
    
    ```python
    pip install -e .
    ```



## Earthquake magnitude estimation


```python
import numpy as np

from culpable.magnitudes import (M_from_D, D_from_M, p_M_D, M_from_L, p_M_L, 
                                 p_M_DL, make_p_M)
from culpable.offset_marker import OffsetMarker
from culpable.stats import Pdf

np.random.seed(69)
```

### Deterministic estimates

Basic estimation of magnitude from rupture length
```python
length_1 = 50.
M_from_L_est = M_from_L(length_1)
```

Now let's change the scaling relationship
```python
M_from_L_reverse = M_from_L(length_1, ref='WC_1994_R')

# basic estimation of magnitude from displacement
```python
disp_1 = 2.3
M_from_D_est = M_from_D(disp_1)
```

basic estimation of displacement from magnitude
```python
mag_1 = 6.7
D_from_M_est = D_from_M(mag_1)
```

Now let's use our own scaling coefficients
```python
D_from_M_new_scaling = D_from_M(mag_1, ref=None, a=5., b=1., base='10')
```



### Bayesian estimates

Note that these are all probabilistic, and the priors and posteriors are
`culpable.stats._Pdf` classes made through the `Pdf` function.  This class has
several useful (necessary, even) attributes and methods. The class is
instantiated by passing x and p(x) values. The p(x) values are relative
probabilities, i.e. probability mass values. The resulting Pdf class will have
been converted to a probability distribution by normalizing by the integral of
the PMF.

```python
disp_pdf = Pdf([1., 1.1, 1.3, 1.7, 2.1, 2.2], [0., 0.1, 0.5, 0.7, 0.3, 0.])
```

First, we need to make a prior for M, p(M)
```python
M_prior = make_p_M(p_M_type='uniform', p_M_min=6.0, p_M_max=7.7,
                   M_step=0.05)
```

Calculate p(M|D), the posterior probability of the magnitude given the 
displacement 
```python
p_M_D_est_1 = p_M_D(disp_1, p_M=M_prior)
```

Now the same but with a correction for paleoseismic sampling bias, i.e. that a
paleoseismic measurement is proportionally more likely to be taken from a high
displacement site vs. a low or average displacement site
```python
p_M_D_est_2 = p_M_D(disp_1, p_M=M_prior, sample_bias_corr=True)
```

Now let's work with some uncertainty in the amount of displacement
```python
disp_samples = np.random.uniform(1.7, 2.4, size=1000)
p_M_D_est_3 = p_M_D(disp_samples, p_M=M_prior, sample_bias_corr=True)
```


Here we do a Bayesian inversion for M given L with uncertainty
```python
length_samples = np.random.normal(156, 15, size=1000)
p_M_L_est_1 = p_M_L(length_samples, p_M=M_prior)
```


Now we do an inversion considering both D and L
```python
p_M_DL_est_1 = p_M_DL(disp_samples, length_samples, p_M=M_prior, 
                      sample_bias_corr=True)
```