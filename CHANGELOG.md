# Changelog

`paramspace` aims to adhere to [semantic versioning](https://semver.org/).

## v2.6.2
- !79 removes left-over `print` statements 🤦

## v2.6.1
Post-release of v2.6.0 due to botched PyPI deployment.

## v2.6.0
This is a maintenance release, adding the following (mostly internal) improvements:

- The repository now uses `main` as its default branch.
- !71 drops version bounds for requirements, making dependency resolution easier.
  This is also meant to promote always working with the latest package versions.
- !71 uses a more modern theme for the documentation.
- !75 extends the CI/CD tests to include Python 3.11 environments.
- !78 outsources YAML tools into a separate package, [`yayaml`](https://gitlab.com/blsqr/yayaml)
    - Functionality is equivalent, but the package makes it much easier to add constructors and representers *after* import time.
    - Prior to outsourcing, !74 added a number of new YAML tags, which were carried over to `yayaml`:
        - Splitting and joining strings via `!split` and `!join`
        - Handling paths via `!expanduser` and `!joinpath`
- !78 also changes the default YAML tag for representing parameter dimensions to `!sweep` (instead of `!pdim`).
  This behaviour is backwards-compatible.

#### Bug fixes
- !78 addresses a bug where `!sweep-default` returned a `Masked` object.

#### Breaking changes & Removals
- !78 changes the default value for the `order` parameter to 0, such that it is easier to define parameter dimensions that should be *last* (addressing #79).
  This is a breaking change, because state number association changes if non-negative `order` values were set!
  However, to retain backwards-compatibility with `ParamSpace` objects dumped from previous versions, objects that are constructed with `!pdim` will still have `inf` as their default value.
  In practice, this should remedy all issues with dumps from before or after changing to version 2.6.
- !78 removes the `!rec-update` YAML tag, which was highly error-prone.
- !78 removes the deprecated arguments `use_coupled_default` and `use_coupled_values` from `CoupledParamDim`.


## v2.5.9
- !63 extends the CI/CD tests to include Python 3.10 environments
- !67 delays the xarray import to reduce the overall import time of the package

## v2.5.8
- !65 includes minor improvements:
    - slightly improves performance by removing unnecessary log messages
    - adds the `!getenv` YAML constructor, allowing to read environment variables

## v2.5.7
- !64 improves performance of `recursive_replace` and `recursive_collect` for hierarchical structures with many string-like objects.

## v2.5.6
- !62 (internal) drops patch release versions from dependencies to simplify dependency resolution in downstream packages

## v2.5.5
- !58 improves the CI/CD to show test coverage in the Merge Request diff view
- !59 extends the CI/CD tests to include Python 3.9 environments
- !61 updates minimal dependencies such that they work on all supported Python versions

## v2.5.4
- !57 adds the `ParamSpace.get_info_dict` method that supplies information about a parameter space using native data types; this can be useful when working with paramspace-structure data on a platform that cannot load this package.

## v2.5.3
- !50 extends the `yaml` module with three list-generating tags: `!linspace`, `!logspace`, `!arange`, based on the corresponding numpy functions.
- !51 improves formatting of `ParamSpace.get_info_str`

## v2.5.2
- !49 adds the `!rec-update` YAML tag to perform recursive updates of mappings, extending the YAML update (`<<:`) feature.

## v2.5.1
- !48 includes the [`LICENSE`](LICENSE) file into the package build, necessary for deployment to conda-forge (see #50).

## v2.5.0
- !43 makes it possible to define `ParamDim`s inside sequence-like parts of the parameter space dictionary (#39).
- !44 adds the `!deepcopy` tag, which can be used as a workaround for #47
- !45 and !46 add YAML constructors for a number of Python built-in functions and operators, see [here](https://paramspace.readthedocs.io/en/latest/yaml/supported_tags.html)

## v2.4.1
- _Internal:_ !41 and !42 improve the CI/CD configuration

## v2.4.0
- !34 reformats the whole code base using `black` (see #40) and implements `pre-commit` hooks to maintain formatting.
- !35 and !38 add a Sphinx documentation and configures deployment to [ReadTheDocs](https://paramspace.readthedocs.io/en/latest) (see #37).
  Furthermore, it applies internal improvements to docstrings and format strings.
- !36 adds safety measures to the PyPI deployment CI job
- !37 updates and extends GitLab Issue and MR templates

## v2.3.1
- !33 adds a `.coveragerc` file for pytest

## v2.3.0
With this release, the paramspace project page becomes publicly available and the package becomes installable via the Python Package Index. :tada:  
This is an infrastructure release; there are no changes to package functionality.

- !32 adds a GitLab CI job to deploy `paramspace` to PyPI
- !31 changes the package license to be the BSD-2-clause license
- !29 single-sources the package version specification
- !28 and !30 extend testing to Python 3.8 and separate jobs to test dependencies with the lowest specified versions.  


## v2.2.3
- !27: Allows `CoupledParamDim.target_pdim` to change

## v2.2.2
- #30: Fixes a bug where `ParamSpace.activate_subspace` failed when specifying string `loc` values.

## v2.2.1
- !29: Fixes a bug where `ParamSpace.activate_subspace` failed when specifying float `loc` values.

## v2.2.0
- !24: Add YAML representers for `slice` and `range` and add a `!listgen` YAML tag which generates a list.

## v2.1.1
- #28: Fix a bug where `CoupledParamDim` only allowed the `values` argument and no others.

## v2.1.0
- #26: To provide a more consistent and convenient interface for `ParamSpace.iterator`, it is now possible to use the function for a zero-volume parameter space, i.e., one where no parameter dimensions where defined. Consequently, this will only return the current state of the dictionary, which is equivalent to the default state of a dictionary where parameter dimensions were defined.

## v2.0.0
- #18: Calculate the `ParamSpace.state_no` instead of incrementing; this leads to higher reliability and allows more flexible iteration schemes.
- #3: Include the ability to restrict `ParamSpace` to a subspace for iteration by introducing `ParamDim.mask`.
   - This required changing the `state` property of the dimension and parameter space classes to include the default value as state 0. It is one of many changes to the public interface of this package that is introduced in version 2.0 and makes the whole state numbering more
   - Improvements going along this:
      - Accessing a parameter dimension by name
      - Calculating the state mapping; indices now relate directly and unambiguously to the state vector of the parameter space.
      - Accessing single states via number or vector
- !18 introduced `xarray.DataArray` functionality for the `ParamSpace`. With this, the state mapping supports not only labelled dimensions but also coordinates. With it, a number of interface changes came about:
   - When initializing `ParamSpace`, each `ParamDim` in it is assigned a unique name, generated from its path. This is used for internal identification instead of the path. (The path is still accessible as fallback, though ...)
   - There are some restrictions on the values a `ParamDim` can take: they now have to be unique and hashable. This is necessary in order to use them as coordinates for the state map. (!21 alleviates the need for hashable values again.)
   - The `yaml` module now supports `!slice!` and `!range` tags.
- #13: Migrate to the better-maintained [`ruamel.yaml`](https://pypi.org/project/ruamel.yaml/) and implement representers for all implemented classes.
   - This leads to a much nicer and future-proof way of storing the objects while remaining human-readable.
   - All this is managed in the new `paramspace.yaml` module, which also supplies the `ruamel.yaml.YAML` object along which the new API revolves.
   - _For packages updating to this version,_ it is recommended to _not_ add custom constructors that trigger on a different tag; this might lead to confusion because the representer can only create mappings with the tag specified in the `paramspace` implementation.
- #12: Test coverage is now up to 99% and the existing tests have been extended in order to more explicitly test the behaviour of the package.
- #19: Update the README
- #20: Add a new argument, `as_type`, to `ParamDim.__init__` to allow a type cast after the values have been parsed.
- #21: Refactor `ParamSpace.all_points` to `ParamSpace.iterator`
- #24: Change iteration order to match the numpy default ("C-style")
- #25: Implement `ParamSpace.activate_subspace` to conveniently select a subspace of the whole parameter space, not only by masks (negative selection) but by indices or coordinate labels (positive selection).
- !21: Alleviate need for values to be hashable.

## v1.1.1
- #17: Fix a bug that prohibited using nested `ParamSpace` objects

## v1.1
- #10: CI expanded to test for multiple Python versions
- #6, #9: Use semantic versioning; clean up tags and branches; add issue and MR templates

Bug fixes:
- #8: Ensure YAML dumping works
- #14: `linspace` and `logspace` evaluation fixed

## v1.0
_(Note that the first version to be kept track of via the changelog is v1.1.)_

This was almost a total rewrite from previous versions and stabilized the public interface of the main `paramspace` objects, `ParamSpace` and `ParamDim`.
