# Changelog

`paramspace` aims to adhere to [semantic versioning](https://semver.org/).

## v1.2
- #18: Calculate the `ParamSpace.state_no` instead of incrementing; this leads to higher reliability and allows more flexible iteration schemes.

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
