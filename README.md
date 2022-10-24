[![PyPI](https://img.shields.io/pypi/v/paramspace)](https://pypi.org/project/paramspace/)
[![conda-forge](https://img.shields.io/conda/vn/conda-forge/paramspace)](https://anaconda.org/conda-forge/paramspace)
[![Python Versions](https://img.shields.io/pypi/pyversions/paramspace)](https://pypi.org/project/paramspace/)
[![Documentation](https://img.shields.io/readthedocs/paramspace)](https://paramspace.readthedocs.io/en/latest/)
[![License](https://img.shields.io/pypi/l/paramspace)](https://opensource.org/licenses/BSD-2-Clause)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3826470.svg)](https://doi.org/10.5281/zenodo.3826470)

[![Pipeline Status](https://gitlab.com/blsqr/paramspace/badges/master/pipeline.svg)](https://gitlab.com/blsqr/paramspace/commits/master)
[![Coverage](https://gitlab.com/blsqr/paramspace/badges/master/coverage.svg)](https://gitlab.com/blsqr/paramspace/-/blob/master/tox.ini)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Pre-Commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen)](https://github.com/pre-commit/pre-commit)


# The `paramspace` package

The `paramspace` package supplies classes that make it easy to iterate over a multi-dimensional parameter space while maintaining a data structure that is convenient for passing hierarchically structured arguments around: `dict`s.

A parameter space is an `n`-dimensional space, where each dimension corresponds to one of `n` parameters and each point in this space represents a certain combination of parameter values.  
In modelling and simulations, it is often useful to be able to iterate over certain values of multiple parameters, creating a multi-dimensional, discrete parameter space.
For example, having a model with six parameters that are worth varying, an iteration would go over the cartesian product of all possible parameter values.
(This is hinted at in the icon of this repository, a 2D representation of a 6-dimensional [hybercube](https://en.wikipedia.org/wiki/Hypercube)).

To that end, this package supplies the `ParamSpace` class, which is initialised with a Python `dict`; it holds the whole set of parameters that are required by a simulation (i.e., _not_ only those that correspond to a parameter dimension).
To add a parameter dimension that can be iterated over, an entry in the dictionary can be replaced by a `ParamDim` object, for which the discrete values to iterate over are defined.

After initialisation of such a `ParamSpace` object, this package allows operations like the following:

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
* [Information for Developers](#information-for-developers)


## Install
The `paramspace` package is tested for Python 3.6 - 3.11 and is available [on the Python Package Index](https://pypi.org/project/paramspace/) or [via `conda-forge`](https://anaconda.org/conda-forge/paramspace).

You can install it via `pip` or `conda` using the respective command:

```bash
pip install paramspace
conda install -c conda-forge paramspace
```


### For Developers
For installation of versions that are not on the PyPI, `pip` allows specifying a git repository:

```bash
$ pip3 install git+<clone-URL>@<some-branch-name>
```

Here, replace `<clone-URL>` with the repositories clone URL and specify the branch you want to install the package from.
Alternatively, omit the `@` and everything after it.

For local development and testing, it's best to clone this repository and install it (in editable mode) from the local directory:
```bash
$ git clone <clone-URL>
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

  more_params:
    foo: 42
    deep:
      down:
        magic_number: !pdim
          default: 2
          values: [2, 8, 20, 28, 50, 82, 126]

    # Can also have parameter dimensions inside a sequence
    things_to_do:
      - do_this
      - do_that
      - !pdim
        default: do_some_thing
        values:
          - do_foo
          - do_bar
          - do_spam
      - do_final_things

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
* `CoupledParamDim` objects are implemented a bit inconsistently:
   * They behave in some cases not equivalent to regular `ParamDim` objects, e.g., they cannot be iterated over on their own (in fact, this will lead to an infinite loop).
   * Their `mask` behaviour might be unexpected.
   * Within a `ParamSpace`, they are mostly hidden from the user. The iteration over parameter space works reliably, but they are, e.g., not accessible within the state maps.
* YAML representation is quite fragile, especially for `CoupledParamDim`. This
  will require some rewriting sooner or later, quite possibly accompanied by an
  interface change ...
* [#47](https://gitlab.com/blsqr/paramspace/-/issues/47): When defining `ParamDim` inside a YAML mapping or sequence that is also an anchor, it may occur multiple times inside `ParamDim`, thus leading to an infinite iteration. (Workaround introduced in [!44](https://gitlab.com/blsqr/paramspace/-/merge_requests/44).)

For more information, see the [issue tracker](https://gitlab.com/blsqr/paramspace/-/issues).


## Information for Developers
### Dependencies
When developing for `paramspace`, additionally install the test and development dependencies:

```bash
pip install -e .[test,dev]
```


### Tests
Tests are implemented using [`pytest`](https://github.com/pytest-dev/pytest).

Invoke the following command to run tests and see the coverage report:

```bash
python -m pytest -xv tests/ --cov=paramspace --cov-report=term-missing
```

Additionally, [`tox`](https://github.com/tox-dev/tox) is used in the GitLab CI pipeline to test against various Python versions.


### Code Formatting
This package uses [`black`](https://github.com/psf/black) for formatting.
To run it, simply invoke:

```bash
black .
```

(Yes, it's uncompromising.)

_Note:_ Code formatting is part of the [commit hooks](#commit-hooks), so you won't have to invoke this separately.


### Commit Hooks
A number of [`pre-commit`](https://pre-commit.com) hooks are configured to make development more convenient.
With the development dependencies installed as described above, the git commit hook can be installed by invoking:

```bash
pre-commit install
```

Upon invoking `git commit` the first time, the corresponding dependencies will be installed and the hooks will be executed.
If they fail, make sure that no unexpected changes were performed.

You can also run the hooks on all files, prior to committing:

```bash
pre-commit run -a
```

For more details about currently configured hooks, see [`.pre-commit-config.yaml`](.pre-commit-config.yaml).
