#!/usr/bin/env python

from setuptools import setup

setup(name='ParamSpace',
      version='0.9',
      description='Multidimensional parameter space with dictionaries at each point.',
      long_description='Classes that allow easy iteration over a multidimensional parameter space, generating dictionaries at each point in this parameter space.',
      author='Yunus Sevinchan',
      author_email='YunusSevinchan@gmail.com',
      url='https://github.com/blusquare/pspace',
      packages=['pspace'],
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.5',
          'Topic :: Utilities'
          ],
     )
