# Changelog

`paramspace` aims to adhere to [semantic versioning](https://semver.org/).

## v2.0 (WIP)
- #18: Calculate the `ParamSpace.state_no` instead of incrementing; this leads to higher reliability and allows more flexible iteration schemes.
- #3: Include the ability to restrict `ParamSpace` to a subspace for iteration by introducing `ParamDim.mask`.
   - This required changing the `state` property of the dimension and parameter space classes to include the default value as state 0. It is one of many changes to the public interface of this package that is introduced in version 2.0 and makes the whole state numbering more 
   - Improvements going along this:
      - Accessing a parameter dimension by name
      - Calculating the state mapping; indices now relate directly and unambiguously to the state vector of the parameter space.
      - Accessing single states via number or vector
- #12: Test coverage is now up to 99% and the existing tests have been extended in order to more explicitly test the behaviour of the package. 
- #13: Migrate to the better-maintained [`ruamel.yaml`](https://pypi.org/project/ruamel.yaml/) and implement representers for all implemented classes.
   - This leads to a much nicer and future-proof way of storing the objects while remaining human-readable.
   - All this is managed in the new `paramspace.yaml` module, which also supplies the `ruamel.yaml.YAML` object along which the new API revolves.
   - _For packages updating to this version,_ it is recommended to _not_ add custom constructors that trigger on a different tag; this might lead to confusion because the representer can only create mappings with the tag specified in the `paramspace` implementation.

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
