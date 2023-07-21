.. _yaml_tags:

Supported YAML Tags
===================

YAML allows defining custom so-called tags which can be distinguished during loading and serialization of objects.
:py:mod:`paramspace` makes heavy use of this possibility, as it greatly simplifies the definition and usage of configuration files.

.. hint::

    **Want more YAML tags?**

    Under the hood, this packages uses the `yayaml package <https://gitlab.com/blsqr/yayaml>`_, which provides a wide range of other yaml tags.
    See `its documentation <https://yayaml.readthedocs.org/>`_ for a list of added tags and what they do.

.. contents::
    :local:
    :depth: 2

----

Parameter Space Tags
--------------------
The :py:mod:`paramspace.yaml` module implements constructors and representers for the following classes:

* ``!pspace`` constructs a :py:class:`~paramspace.paramspace.ParamSpace`
* ``!pdim`` constructs a :py:class:`~paramspace.paramdim.ParamDim`
* ``!coupled-pdim`` constructs a :py:class:`~paramspace.paramdim.CoupledParamDim`

This is a very convenient and powerful way of defining these objects, right in the YAML file.
For instance, this is used in `the Utopia framework <https://utopia-project.org/>`_ to define sweeps over model parameters.

.. hint::

    For the :py:class:`~paramspace.paramdim.ParamDim` and derived classes, there additionally are the ``!pdim-default`` and ``!coupled-pdim-default`` tags.
    These do not create a :py:class:`~paramspace.paramdim.ParamDim` objects but directly return the default value.
    By adding the ``-default`` in the end, they can be quickly deactivated inside the configuration file (as an alternative to commenting them out).
