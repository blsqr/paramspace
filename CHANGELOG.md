# Changelog

`paramspace` aims to adhere to [semantic versioning](https://semver.org/).

## v2.0
- #18: Calculate the `ParamSpace.state_no` instead of incrementing; this leads to higher reliability and allows more flexible iteration schemes.
- #3: Include the ability to restrict `ParamSpace` to a subspace for iteration by introducing `ParamDim.mask`.
   - This required changing the `state` property of the dimension and parameter space classes to include the default value as state 0. It is one of many changes to the public interface of this package that is introduced in version 2.0 and makes the whole state numbering more 
   - Improvements going along this:
      - Accessing a parameter dimension by name
      - Calculating the state mapping; indices now relate directly and unambiguously to the state vector of the parameter space.
      - Accessing single states via number or vector
- !18 introduced `xarray.DataArray` functionality for the `ParamSpace`. With this, the state mapping supports not only labelled dimensions but also coordinates. With it, a number of interface changes came about:
   - When initializing `ParamSpace`, each `ParamDim` in it is assigned a unique name, generated from its path. This is used for internal identification instead of the path. (The path is still accessible as fallback, though ...)
   - There are some restrictions on the values a `ParamDim` can take: they now have to be unique and hashable. This is necessary in order to use them as coordinates for the state map.
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
