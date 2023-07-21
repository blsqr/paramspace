#!/usr/bin/env python3
from setuptools import setup

# Dependencies for paramspace itself
install_deps = [
    "numpy < 2.0",
    "xarray",
    "ruamel.yaml",
    "yayaml",
]

# Dependencies for executing tests
test_deps = [
    "tox",
    "pytest",
    "pytest-cov",
]

# Dependencies for documentation building
doc_deps = [
    "sphinx == 4.*",
    "sphinx-book-theme == 0.3.*",
    "sphinx-togglebutton",
    "ipython",
]

# Dependencies for development, code maintenance, ...
dev_deps = [
    "pre-commit",
    "seed-isort-config",
    "isort[pyproject]",
    "black",
    "pyupgrade",
]

# .............................................................................

DESCRIPTION = "Dictionary-based, multi-dimensional parameter space iteration"
LONG_DESCRIPTION = """
The **paramspace** package is an open-source project and Python package that
makes it possible to conveniently define dictionary-based, multi-dimensional
parameter spaces and iterate over them.

Why?
----

In Python, dictionaries provide a powerful tool to control program behaviour.
Frequently, these configuration structures take the shape of highly nested
dictionaries, where each hierarchical level holds the information required at
that point of the program.

However, it is frequently desired to instantiate some program not with a
*single* set of parameters but with a set of parameters.
Especially for scientific purposes, e.g. numerical simulations, it is often
required to perform many instantiations of the same program with different
parameters, so-called parameter sweeps.
For simple configuration structures, this can be easily achieved by basic
control flow tools; however, this becomes increasingly difficult the more
parameters are desired to be sweeped over or the further nested they are in the
configuration hierarchy.

This is where the paramspace package comes in.

How?
----

At its core, the paramspace package supplies the ``ParamSpace`` class, which
accepts a dictionary-like object. To define parameter dimensions, individual
entries within that dictionary can be replaced by a ``ParamDim`` object,
regardless of the position and nestedness within the dictionary.
The parameter space is then the cartesian product of all parameter dimensions,
each parameter opening a new dimension of the parameter space.

When iterating over the space, each returned value is a dictionary with one
combination of parameters, ready to be passed on to run the desired program.
This allows retaining a hierarchical configuration structure while at the same
time being able to conveniently perform sweeps over parameters, e.g. to spawn
simulations with.

Furthermore, the paramspace package integrates tightly with YAML, making it
very simple to define multidimensional parameter spaces directly in a
configuration file.

Learn More
----------

For more information, visit the project page and have a look at the README:
https://gitlab.com/blsqr/paramspace

For use of paramspace, see `utopya <https://gitlab.com/utopia-project/utopya>`_
or the `Utopia Project <https://utopia-project.org/>`_.

"""


# .............................................................................


# A function to extract version number from __init__.py
def find_version(*file_paths) -> str:
    """Tries to extract a version from the given path sequence"""
    import codecs
    import os
    import re

    def read(*parts):
        """Reads a file from the given path sequence, relative to this file"""
        here = os.path.abspath(os.path.dirname(__file__))
        with codecs.open(os.path.join(here, *parts), "r") as fp:
            return fp.read()

    # Read the file and match the __version__ string
    file = read(*file_paths)
    match = re.search(r"^__version__\s?=\s?['\"]([^'\"]*)['\"]", file, re.M)
    if match:
        return match.group(1)
    raise RuntimeError("Unable to find version string in " + str(file_paths))


# .............................................................................

setup(
    name="paramspace",
    version=find_version("paramspace", "__init__.py"),
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author="Yunus Sevinchan",
    author_email="yunussevinchan@gmail.com",
    url="https://gitlab.com/blsqr/paramspace",
    license="BSD-2-Clause",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        #
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        #
        "Topic :: Utilities",
        "Typing :: Typed",
    ],
    packages=["paramspace"],
    include_package_data=True,
    python_requires=">=3.6",
    install_requires=install_deps,
    tests_require=test_deps,
    test_suite="tox",
    extras_require=dict(
        test=test_deps,
        dev=dev_deps,
        doc=doc_deps,
    ),
    data_files=[("", ["LICENSE", "README.md", "CHANGELOG.md"])],
)
