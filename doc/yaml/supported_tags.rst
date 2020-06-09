.. _yaml_tags:

Supported YAML Tags
===================

YAML allows defining custom so-called tags which can be distinguished during loading and serialization of objects.
:py:mod:`paramspace` makes heavy use of this possibility, as it greatly simplifies the definition and usage of configuration files.


:py:mod:`paramspace`-related tags
---------------------------------
The :py:mod:`paramspace.yaml` module implements constructors and representers for the following classes:

* ``!pspace`` constructs a :py:class:`~paramspace.paramspace.ParamSpace`
* ``!pdim`` constructs a :py:class:`~paramspace.paramdim.ParamDim`
* ``!coupled-pdim`` constructs a :py:class:`~paramspace.paramdim.CoupledParamDim`

This is a very convenient way of defining these objects.

.. hint::

    For the :py:class:`~paramspace.paramdim.ParamDim` and derived classes, there additionally are the ``!pdim-default`` and ``!coupled-pdim-default`` tags.
    These do not create a :py:class:`~paramspace.paramdim.ParamDim` objects but directly return the default value.
    By adding the ``-default`` in the end, they can be quickly deactivated inside the configuration file (as an alternative to commenting them out).


Python builtins and basic operators
-----------------------------------
:py:mod:`paramspace.yaml` adds YAML constructors for a number of frequently used Python built-in functions and operators.
Having these available while specifying configurations can make the definition of configurations files more versatile.

.. warning::

    The YAML tags provided here are only meant to allow basic operations, i.e. summing two parameters to create a third.
    Don't overdo it.
    Configuration files should remain easy to read.

The tags shown below call the equivalent Python builtin or the operators defined in the ``operator`` Python module.
Example:

.. literalinclude:: ../../tests/test_yaml.py
    :language: yaml
    :start-after: # START -- utility-yaml-tags
    :end-before: # END ---- utility-yaml-tags
    :dedent: 14
