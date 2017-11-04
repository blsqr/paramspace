# The `paramspace` package

This package is aimed at being able to iterate over a multidimensional parameter space, where at each point a different dictionary can be returned.

The whole parameter space is contained in the `ParamSpace` class, while each dimension is a so-called `ParamSpan`. To couple one value of the parameter space to a dimension, the `CoupledParamSpan` class can be used.

**NOTE:** Documentation of this package is very rudimentary at this point and will need to be corrected and extended for version 1.0.

## Install

```bash
$ python3 setup.py install
```

## Usage

```python
import paramspace as psp
# ... TODO
```
