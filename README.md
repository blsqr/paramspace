# The `paramspace` package

This package is aimed at being able to iterate over a multidimensional parameter space, where at each point a different dictionary can be returned.

The whole parameter space is contained in the `ParamSpace` class, while each dimension is a so-called `ParamSpan`. To couple one value of the parameter space to a dimension, the `CoupledParamSpan` class can be used.

**NOTE:** Documentation of this package is very rudimentary at this point and will need to be corrected and extended for version 1.0.

Also, in the long term there should be a version 2.0 rewritten from scratch, aiming to be more simple, pythonistic, and supporting more powerful iterator functionality.

## Install

For installation, it is best to use `pip` and pass the directory of the cloned repository to it. This will automatically install `paramspace` and its requirements and makes it very easy to uninstall or upgrade later.

```bash
$ pip3 install paramspace/
```

## Usage

```python
import paramspace as psp
# ... TODO
```
