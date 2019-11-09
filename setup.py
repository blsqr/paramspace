#!/usr/bin/env python3

from setuptools import setup

# Dependencies for paramspace itself
install_deps = [
    'numpy>=1.17.3',
    'xarray>=0.10.9',
    'ruamel.yaml>=0.16.5'
]

# Derive an extra that uses strict versions; allows testing for these via tox
minimal_install_deps = [dep.replace(">=", "==") for dep in install_deps]

# Dependencies for the tests
test_deps = [
    'tox>=3.1',
    'pytest>=3.4',
    'pytest-cov>=2.6'
]

# .............................................................................

# A function to extract version number from __init__.py
def find_version(*file_paths) -> str:
    """Tries to extract a version from the given path sequence"""
    import os, re, codecs

    def read(*parts):
        """Reads a file from the given path sequence, relative to this file"""
        here = os.path.abspath(os.path.dirname(__file__))
        with codecs.open(os.path.join(here, *parts), 'r') as fp:
            return fp.read()

    # Read the file and match the __version__ string
    file = read(*file_paths)
    match = re.search(r"^__version__\s?=\s?['\"]([^'\"]*)['\"]", file, re.M)
    if match:
        return match.group(1)
    raise RuntimeError("Unable to find version string in " + str(file_paths))


# .............................................................................

setup(
    name='paramspace',
    version=find_version('paramspace', '__init__.py'),
    #
    description='Provides tools to define a dict-based multidimensional '
        'parameter space and iterate over it',
    long_description='The parameter space is the cartesian product of those '
        'parameters one desires to iterate over, each parameter opening a new '
        'dimension of parameter space. '
        'Parameter dimensions can be defined for each entry in a dictionary, '
        'even when nested within further dictionaries. When iterating over '
        'the space, each returned value is a dictionary with one combination '
        'of parameters. This allows retaining a hierarchical configuration '
        'structure while at the same time being able to conveniently perform '
        'sweeps over parameters, e.g. to spawn simulations with. '
        'Furthermore, the paramspace package integrates tightly with YAML, '
        'making it very simple to define multidimensional parameter spaces.',
    #
    author='Yunus Sevinchan',
    author_email='Yunus.Sevinchan@iup.uni-heidelberg.de',
    url='https://ts-gitlab.iup.uni-heidelberg.de/yunus/paramspace',
    license='MIT',
    classifiers=[
      'Development Status :: 5 - Production/Stable',
      'Intended Audience :: Science/Research',
      'License :: OSI Approved :: MIT License',
      'Programming Language :: Python :: 3.6',
      'Programming Language :: Python :: 3.7',
      'Programming Language :: Python :: 3.8',
      'Topic :: Utilities'
    ],
    packages=['paramspace'],
    include_package_data=True,
    install_requires=install_deps,
    tests_require=test_deps,
    test_suite='tox',
    extras_require=dict(test_deps=test_deps,
                        minimal_deps=minimal_install_deps)
)
