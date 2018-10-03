# The `paramspace` package

The `paramspace` package supplies classes that make it easy to iterate over a multi-dimensional parameter space while maintaining a data structure that is convenient for passing arguments around: `dict`s.

A parameter space is an $`n`$-dimensional space, where each dimension corresponds to one of $`n`$ parameters and each point in this space represents a certain combination of parameter values.  
In modelling and simulations, it is often useful to be able to iterate over certain values of multiple parameters, creating a multi-dimensional, discrete parameter space.
For example, having a model with six parameters that are worth varying, an iteration would go over the cartesian product of all possible parameter values.
(This is hinted at in the avatar of this repository, a 2d representation of a 6-dimensional [hybercube](https://en.wikipedia.org/wiki/Hypercube)).

To that end, this package supplies the `ParamSpace` class, which is initialised with a Python `dict`; it holds the whole set of parameters that are required by a simulation (i.e., _not_ only those that correspond to a parameter dimension).
To add a parameter dimension that can be iterated over, an entry in the dictionary can be replaced by a `ParamDim` object, for which the discrete values to iterate over are defined.

After initialsation of such a `ParamSpace` object, this package allows operations like the following:
```python
for params in pspace:
    run_my_simulation(**params)
```
The `params` object is then a dictionary that holds the configuration of the simulation at one specific point in parameter space.  
In other words: each point in this parameter space refers to a specific state of the given dictionary of simulation parameters.

#### Further features of the `paramspace` package
* With the `default` argument to `ParamDim`, it is possible to define a default position in parameter space that is used when not iterating over the parameter space
* The `order` argument allows ordering the `ParamDim` objects, such that it can be decided which dimensions are iterated over most frequently.
* `ParamDim` values can be created from `range`, `np.linspace`, and `np.logspace`
* With `ParamDim.mask` and `ParamSpace.set_mask`, a subspace of a parameter space can be selected for iteration.
* Via `ParamSpace.state_map`, an [`xarray.DataArray`](http://xarray.pydata.org/en/stable/data-structures.html#dataarray) with labelled dimensions and coordinates is returned.
* `CoupledParamDim` objects allow coupling one parameter in an iteration to another parameter dimension.
* The `paramspace.yaml` object (based on [`ruamel.yaml.YAML`](https://yaml.readthedocs.io/en/latest/)) supplies the constructors and representers necessary to load or dump `paramspace` objects in YAML format. Defining parameter spaces via this interface is much more convenient than it is directly in Python.

#### Contents of this README and further reading
* Short [__installation instructions__](#install)
* A few __usage examples__ are given [below](#usage). Note that a full documentation does not yet exist... but the docstrings are quite informative :)
* For an overview over the __changes,__ see the [changelog](CHANGELOG.md).
* A list of [__known issues__](#known-issues) with some classes


## Install
The `paramspace` package is tested for Python 3.6 and 3.7.

For installation, it is best to use `pip` and pass the URL to this repository to it. This will automatically install `paramspace` and its requirements and makes it very easy to uninstall or upgrade later.

```bash
$ pip3 install git+ssh://git@ts-gitlab.iup.uni-heidelberg.de:10022/yunus/paramspace.git
```

You can also clone this repository and install it (in editable mode) from the local directory:
```bash
$ git clone ssh://git@ts-gitlab.iup.uni-heidelberg.de:10022/yunus/paramspace.git
$ pip3 install -e paramspace/
```


## Usage

### Basics
The example below illustrates how `ParamDim` and `ParamSpace` objects can be created and used together.
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

### Using the power of YAML
While the above way is possible, using the capabilities of the `yaml` module make defining `ParamSpace` objects much more convenient.

Say we have a configuration file that is to be given to our simulation function. With the YAML constructors implemented in this package, we can construct `ParamDim` and `ParamSpace` objects right inside the file where we define all the other parameters: just by adding a `!pspace` and `!pdim` tag to a mapping.

```yaml
# This is the configuration file for my simulation
---
sim_name: my_first_sim
out_dir: ~/sim_output/{date:}

sim_params: !pspace    # <- will construct a ParamSpace from what is inside

  # Define a number of simulation seeds
  seed: !pdim          # <- will create a parameter dimension with seeds 0...22
    default: 0
    range: [23]

  some_param: 1.23
  some_params_to_pass_along:
    num_agents: !pdim  # <- creates values: 10, 32, 100, 316, 1000, 3162, ...
      default: 100
      logspace: [1, 5, 9]
      as_type: int

  # ... and so on
```

We can now load this file and will already have the `ParamSpace` constructed:

```python
from paramspace import yaml

with open("path/to/cfg.yml", mode='r') as cfg_file:
    cfg = yaml.load(cfg_file)

# cfg is now a dict with keys: sim_name, out_dir, sim_params, ...

# Get the ParamSpace object and print some information
pspace = cfg['sim_params']
print("Received parameter space with volume", pspace.volume)
print(pspace.get_info_str())

# Now perform the iteration and run the simulations
print("Starting simulation '{}' ...".format(cfg['sim_name']))
for params in pspace:
    run_my_simulation(**params)
```

#### Comments
* The yaml constructors supply full functionality. It is highly recommended to use them. Additional constructors are:
   * `!pdim-default`: returns the default value _instead_ of the `ParamDim` object; convenient to deactivate a dimension completely.
   * `!coupled-pdim` and `!coupled-pdim-default` have the analogue behaviour, just with `CoupledParamDim`.
* The `yaml` object can also be used to `yaml.dump` the configuration into a yaml file again.
* There is the possibility to iterate and get information about the current state of the parameter space alongside the current value. For that, use the `ParamSpace.iterate` method.


## Known issues
* `CoupledParamDim` objects might be implemented a bit inconsistently:
   * They behave in some cases not equivalent to regular `ParamDim` objects, e.g., they cannot be iterated over on their own (in fact, this will lead to an infinite loop).
   * Their `mask` behaviour might be unexpected.
   * Within a `ParamSpace`, they are mostly hidden from the user. The iteration over parameter space works reliably, but they are, e.g., not accessible within the state maps.
