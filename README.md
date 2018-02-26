# The `paramspace` package

[![pipeline status](https://ts-gitlab.iup.uni-heidelberg.de/yunus/paramspace/badges/master/pipeline.svg)](https://ts-gitlab.iup.uni-heidelberg.de/yunus/paramspace/commits/master)
[![coverage report](https://ts-gitlab.iup.uni-heidelberg.de/yunus/paramspace/badges/master/coverage.svg)](https://ts-gitlab.iup.uni-heidelberg.de/yunus/paramspace/commits/master)

This package is aimed at being able to iterate over a multidimensional parameter space, where at each point a different dictionary can be returned.

The whole parameter space is contained in the `ParamSpace` class, while each dimension is a so-called `ParamSpan`. To couple one value of the parameter space to a dimension, the `CoupledParamSpan` class can be used.

**NOTE:** Documentation of this package is very rudimentary at this point and will need to be corrected and extended for version 1.0.  
Also, in the long term there should be a version 2.0 rewritten from scratch, aiming to be more simple, pythonistic, and supporting more powerful iterator functionality.

**Repository avatar:** The avatar of this repository shows a 2d representation of a 6-dimensional hybercube (see [Wikipedia](https://en.wikipedia.org/wiki/Hypercube), image in public domain).

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