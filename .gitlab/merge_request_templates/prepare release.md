<!-- Use this template for MRs that prepare for a paramspace release. -->

<!-- 1 - Set as MR title: Prepare release of vX.Y -->
<!-- 2 - Adjust the following quick commands: -->
/label ~release
/milestone %"Version X.Y"

<!-- 3 - Fill in the MR description and the checklist below. -->

This MR prepares the vX.Y release of paramspace.


### Can this MR be accepted?
- [ ] Set version number in [`paramspace/__init__.py`](paramspace/__init__.py)
   - Removed the pre-release specifier.
   - Version is now: `X.Y.0`
- [ ] Prepared [changelog](CHANGELOG.md) for release
   - Removed "WIP" in section heading
   - If necessary, re-ordered and cleaned-up the corresponding section
- [ ] Pipeline passes without warnings
   - If the `test_minimal_deps` stage creates warnings, inspect the output log and, if necessary, adjust the lower bounds of the dependencies in [`setup.py`](setup.py).
   - If the `test_minimal_deps` stage is not *run* at all, the branch name probably doesn't start with `prepare-release`. You do not need to rename the branch, but you might have to rename the MR to include the `prepare release` string somewhere (the case does not matter).
- [ ] Approved by @  <!-- only necessary if there are substantial changes -->

<!-- 4 - If you are not allowed to merge, assign a maintainer now. -->
/assign @yunus
