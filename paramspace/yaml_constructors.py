"""Defines the yaml constructors for the generation of
:py:class:`~paramspace.paramspace.ParamSpace` and
:py:class:`~paramspace.paramdim.ParamDim` during loading of YAML files.

Note that they are not registered in this module but in the
:py:mod:`paramspace.yaml` module.
"""
import logging
import warnings
from collections import OrderedDict
from typing import Callable, Iterable, Union

import ruamel.yaml

from .paramdim import CoupledParamDim, ParamDim
from .paramspace import ParamSpace
from .tools import create_indices

# Get logger
log = logging.getLogger(__name__)


# Top-level functions for the yaml-module to import ---------------------------


def pspace(loader, node) -> ParamSpace:
    """yaml constructor for creating a ParamSpace object from a mapping.

    Suggested tag: ``!pspace``
    """
    return _pspace_constructor(loader, node)


def pspace_unsorted(loader, node) -> ParamSpace:
    """yaml constructor for creating a ParamSpace object from a mapping.

    Unlike the regular constructor, this one does NOT sort the input before
    instantiating ParamSpace.

    Suggested tag: ``!pspace-unsorted``
    """
    return _pspace_constructor(loader, node, sort_if_mapping=False)


def pdim(loader, node) -> ParamDim:
    """constructor for creating a ParamDim object from a mapping

    Suggested tag: ``!pdim``
    """
    return _pdim_constructor(loader, node)


def pdim_default(loader, node) -> ParamDim:
    """constructor for creating a ParamDim object from a mapping, but only
    return the default value.

    Suggested tag: ``!pdim-default``
    """
    pdim = _pdim_constructor(loader, node)
    log.debug("Returning default value of constructed ParamDim.")
    return pdim.default


def coupled_pdim(loader, node) -> CoupledParamDim:
    """constructor for creating a CoupledParamDim object from a mapping

    Suggested tag: ``!coupled-pdim``
    """
    return _coupled_pdim_constructor(loader, node)


def coupled_pdim_default(loader, node) -> CoupledParamDim:
    """constructor for creating a CoupledParamDim object from a mapping, but
    only return the default value.

    Suggested tag: ``!coupled-pdim-default``
    """
    cpdim = _coupled_pdim_constructor(loader, node)
    log.debug("Returning default value of constructed CoupledParamDim.")
    return cpdim.default


# The actual constructor functions --------------------------------------------


def _pspace_constructor(
    loader, node, sort_if_mapping: bool = True
) -> ParamSpace:
    """Constructor for instantiating ParamSpace from a mapping or a sequence"""
    log.debug("Encountered tag associated with ParamSpace.")

    # get fields as mapping or sequence
    if isinstance(node, ruamel.yaml.nodes.MappingNode):
        log.debug("Constructing mapping from node ...")
        d = loader.construct_mapping(node, deep=True)

        # Recursively order the content to have consistent loading
        if sort_if_mapping:
            log.debug("Recursively sorting the mapping ...")
            d = recursively_sort_dict(OrderedDict(d))

    else:
        raise TypeError(
            f"ParamSpace node can only be constructed from a mapping or a "
            f"sequence, got node of type {type(node)} with value:\n{node}."
        )

    log.debug("Instantiating ParamSpace ...")
    return ParamSpace(d)


def _pdim_constructor(loader, node) -> ParamDim:
    """Constructor for creating a ParamDim object from a mapping

    For it to be incorported into a ParamSpace, one parent (or higher) of this
    node needs to be tagged such that the pspace_constructor is invoked.
    """
    log.debug("Encountered tag associated with ParamDim.")

    if isinstance(node, ruamel.yaml.nodes.MappingNode):
        log.debug("Constructing mapping ...")
        mapping = loader.construct_mapping(node, deep=True)
        pdim = ParamDim(**mapping)

    else:
        raise TypeError(
            f"ParamDim can only be constructed from a mapping node,got node "
            f"of type {type(node)} with value:\n{node}"
        )

    return pdim


def _coupled_pdim_constructor(loader, node) -> ParamDim:
    """Constructor for creating a ParamDim object from a mapping

    For it to be incorported into a ParamSpace, one parent (or higher) of this
    node needs to be tagged such that the pspace_constructor is invoked.
    """
    log.debug("Encountered tag associated with ParamDim.")

    if isinstance(node, ruamel.yaml.nodes.MappingNode):
        log.debug("Constructing mapping ...")
        mapping = loader.construct_mapping(node, deep=True)
        cpdim = CoupledParamDim(**mapping)

    else:
        raise TypeError(
            f"CoupledParamDim can only be constructed from a mapping node, "
            f"got node of type {type(node)} with value:\n{node}"
        )

    return cpdim


# Some other constructors -----------------------------------------------------


def _slice_constructor(loader, node):
    """Constructor for slices"""
    log.debug("Encountered !slice tag.")

    # get slice arguments either from a scalar or from a sequence
    if isinstance(node, ruamel.yaml.nodes.SequenceNode):
        args = loader.construct_sequence(node, deep=True)
    else:
        args = [loader.construct_yaml_int(node)]

    log.debug("  args:  %s", args)
    slc = slice(*args)
    log.debug("  slice object created: %s", slc)

    return slc


def _range_constructor(loader, node):
    """Constructor for range"""
    log.debug("Encountered !range tag.")

    # get range arguments either from a scalar or from a sequence
    if isinstance(node, ruamel.yaml.nodes.SequenceNode):
        args = loader.construct_sequence(node, deep=True)
    else:
        args = [loader.construct_yaml_int(node)]

    log.debug("  args:  %s", args)
    rg = range(*args)
    log.debug("  range object created: %s", rg)

    return rg


def _list_constructor(loader, node):
    """Constructor for lists, where node can be a mapping or sequence"""
    log.debug("Encountered !listgen tag.")

    if isinstance(node, ruamel.yaml.nodes.MappingNode):
        kwargs = loader.construct_mapping(node, deep=True)

    elif isinstance(node, ruamel.yaml.nodes.SequenceNode):
        kwargs = dict(from_range=loader.construct_sequence(node))

    else:
        raise TypeError(
            f"Expected mapping or sequence node for !listgen, but "
            f"got {type(node)}!"
        )

    log.debug("  kwargs:  %s", kwargs)
    return create_indices(**kwargs)


def _func_constructor(loader, node, *, func: Callable, unpack: bool = True):
    """A constructor that constructs a scalar, mapping, or sequence from the
    given node and subsequently applies the given function on it.

    Args:
        loader: The selected YAML loader
        node: The node from which to construct a Python object
        func (Callable): The callable to invoke on the resulting
        unpack (bool, optional): Whether to unpack sequences or mappings into
            the ``func`` call
    """
    if isinstance(node, ruamel.yaml.nodes.MappingNode):
        s = loader.construct_mapping(node, deep=True)
        if unpack:
            return func(**s)

    elif isinstance(node, ruamel.yaml.nodes.SequenceNode):
        s = loader.construct_sequence(node, deep=True)
        if unpack:
            return func(*s)

    else:
        s = loader.construct_scalar(node)

    return func(s)


# Helpers ---------------------------------------------------------------------


def recursively_sort_dict(d: dict) -> OrderedDict:
    """Recursively sorts a dictionary by its keys, transforming it to an
    OrderedDict in the process.

    From: http://stackoverflow.com/a/22721724/1827608

    Args:
        d (dict): The dictionary to be sorted

    Returns:
        OrderedDict: the recursively sorted dict
    """
    # Start with empty ordered dict for this recursion level
    res = OrderedDict()

    # Fill it with the values from the dictionary
    for k, v in sorted(d.items()):
        if isinstance(v, dict):
            res[k] = recursively_sort_dict(v)
        else:
            res[k] = v
    return res
