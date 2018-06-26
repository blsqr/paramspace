#!/usr/bin/env python3

from setuptools import setup

# Dependency lists
install_deps = ['numpy>=1.13', "PyYAML>=3.12"]
test_deps    = ['pytest>=3.4.0', 'pytest-cov>=2.5.1']


setup(name='ParamSpace',
      version='1.0.1',  # NOTE: also change this in __init__.py
      description='Multidimensional parameter space with dictionaries at each point.',
      long_description='Classes that allow easy iteration over a multidimensional parameter space, generating dictionaries at each point in this parameter space.',
      author='Yunus Sevinchan',
      author_email='Yunus.Sevinchan@iup.uni-heidelberg.de',
      url='https://ts-gitlab.iup.uni-heidelberg.de/yunus/paramspace',
      license='MIT',
      classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Topic :: Utilities'
      ],
      packages=['paramspace'],
      include_package_data=True,
      install_requires=install_deps,
      tests_require=test_deps,
      test_suite='py.test',
      extras_require=dict(test_deps=test_deps)
      )
