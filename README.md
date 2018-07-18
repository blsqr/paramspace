# The `paramspace` package

The `paramspace` package supplies classes that make it easy to iterate over a multi-dimensional parameter space.

A parameter space is an $`n`$-dimensional space, where each dimension corresponds to one of $`n`$ parameters and each point in this space represents a certain combination of parameter values.  
In modelling and simulations, it is often useful to be able to iterate over certain values of multiple parameters, creating a multi-dimensional, discrete parameter space.

To that end, this package supplies the `ParamSpace` class, which is initialised with a Python `dict`; it holds the whole set of parameters that are required by a simulation.
To add a parameter dimension that can be iterated over, an entry in the dictionary can be replaced by a `ParamDim` object, for which the discrete values to iterate over are defined.

After the `ParamSpace` has been initialised, it allows operations like the following:
```python
for params in pspace:
    run_simulation(**params)
```
The `params` dict is then the dictionary that holds the configuration of the simulation at one specific point in parameter space.

Further features of the `paramspace` package:
* With the `default` argument to `ParamDim`, it is possible to define a default position in parameter space that is used when not iterating over the parameter space
* The `order` argument allows ordering the `ParamDim` objects, such that it can be decided which dimensions are iterated over most frequently.
* `ParamDim` values can be created from `range`, `np.linspace`, and `np.logspace`
* `CoupledParamDim` allows coupling one parameter in an iteration to another
* The `yaml_constructor` module supplies constructor functions that can be implemented to create `ParamSpace` objects during the loading of YAML files
* The `tools` module holds diverse helper function, e.g. for recursively updating a dictionary or retrieving values from it

**Repository avatar:** The avatar of this repository shows a 2d representation of a 6-dimensional hybercube (see [Wikipedia](https://en.wikipedia.org/wiki/Hypercube), image in public domain).


## Install
The `paramspace` package is tested for Python 3.6 and 3.7.

For installation, it is best to use `pip` and pass the URL to this repository to it. This will automatically install `paramspace` and its requirements and makes it very easy to uninstall or upgrade later.

```bash
$ pip3 install git+ssh://git@ts-gitlab.iup.uni-heidelberg.de:10022/yunus/paramspace.git
```

You can also clone this repository and install from the local directory:
```bash
$ git clone ssh://git@ts-gitlab.iup.uni-heidelberg.de:10022/yunus/paramspace.git
$ pip3 install paramspace/
```


## Usage

```python
from paramspace import ParamSpace, ParamDim

# Create the parameter dictionary, values for differently shaped cylinders
cylinders = dict(pi=3.14159,
                 r=ParamDim(default=1, values=[1, 2, 3, 5, 10]),
                 h=ParamDim(default=1, linspace=[0, 10, 11]))

# Define the volume calculation function
def calc_cylinder_vol(*, pi, r, h):
    return pi * (r**2) * h 

# Initialise the parameter space
pspace = ParamSpace(cylinders)

# Iterate over it, using the parameters to calculate the cylinder's volume
for params in pspace:
    print("Height: {},   Radius: {}".format(params['h'], params['r']))
    vol = calc_cylinder_vol(**params)  # Really handy way of passing params :)
    print("  --> Volume: {}".format(vol))
```

Please refer to the docstrings for more information on how the `paramspace` package can be used.


## Known issues
* It is currently not possible to use `yaml.safe_dump` and `yaml.safe_load` with `ParamSpace` and `ParamDim`. This is also why the version requirement for `pyyaml` is so strict: as early as [v4.2](https://github.com/yaml/pyyaml/issues/193), safe loading and dumping might become the default.
* Inconsistency: while a `yaml_constructor` module exists, there are no representer functions available that could create a reasonable serialisation of `paramspace` classes.
