#!/usr/bin/env python

from setuptools import setup

setup(name='ParamSpace',
      version='0.9.9',
      description='Multidimensional parameter space with dictionaries at each point.',
      long_description='Classes that allow easy iteration over a multidimensional parameter space, generating dictionaries at each point in this parameter space.',
      author='Yunus Sevinchan',
      author_email='Yunus.Sevinchan@iup.uni-heidelberg.de',
      url='https://ts-gitlab.iup.uni-heidelberg.de/yunus/paramspace',
      licence='MIT',
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.6',
          'Topic :: Utilities'
          ],
      packages=['paramspace'],
      include_package_data=True,
      install_requires=[
          'numpy>=1.13',
          ],
     )
