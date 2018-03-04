"""This module adds yaml constructors for ParamSpace and ParamDim generation"""

import yaml
import logging
import warnings
from collections import OrderedDict
from typing import Iterable, Union

from paramspace import ParamSpace, ParamDim, CoupledParamDim

# Get logger
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# The functions to be imported

def pspace(loader, node) -> ParamSpace:
    """yaml constructor for creating a ParamSpace object from a mapping.

    Suggested tag: !pspace
    """
    return _pspace_constructor(loader, node)

def pspace_sorted(loader, node) -> ParamSpace:
    """yaml constructor for creating a ParamSpace object from a mapping.

    Unlike the regular constructor, this one sorts the input before
    instantiating ParamSpace.

    Suggested tag: !pspace
    """
    return _pspace_constructor(loader, node, sort_if_mapping=True)

def pdim(loader, node) -> ParamDim:
    """constructor for creating a ParamDim object from a mapping

    Suggested tag: !pdim
    """
    return _pdim_constructor(loader, node)

def pdim_enabled_only(loader, node) -> ParamDim:
    """constructor for creating a ParamDim object from a mapping

    If the ParamDim would not be enabled, the default value is returned.

    Suggested tag: !pdim
    """
    pdim = _pdim_constructor(loader, node)

    if pdim.enabled:
        return pdim
    
    log.debug("ParamDim was disabled; returning default.")
    return pdim.default

def pdim_get_default(loader, node) -> ParamDim:
    """constructor for creating a ParamDim object from a mapping, but only return the default value.

    Suggested tag: !pdim-default
    """
    pdim = _pdim_constructor(loader, node)
    log.debug("Returning default value of constructed ParamDim.")
    return pdim.default

def pdim_always_disable(loader, node) -> ParamDim:
    """constructor for creating a ParamDim object from a mapping, but always disabling it.

    Suggested tag: !disabled-pdim
    """
    pdim = _pdim_constructor(loader, node)
    log.debug("Disabling constructed ParamDim.")
    pdim.enabled = False
    return pdim

def coupled_pdim(loader, node) -> CoupledParamDim:
    """constructor for creating a CoupledParamDim object from a mapping

    Suggested tag: !coupled-pdim
    """
    return _coupled_pdim_constructor(loader, node)

def coupled_pdim_enabled_only(loader, node) -> CoupledParamDim:
    """constructor for creating a CoupledParamDim object from a mapping

    If the ParamDim would not be enabled, the default value is returned.

    Suggested tag: !coupled-pdim
    """
    cpdim = _coupled_pdim_constructor(loader, node)

    if cpdim.enabled:
        return cpdim
    
    log.debug("ParamDim was disabled; returning default.")
    return cpdim.default

def coupled_pdim_get_default(loader, node) -> CoupledParamDim:
    """constructor for creating a CoupledParamDim object from a mapping, but only return the default value.

    Suggested tag: !coupled-pdim-default
    """
    cpdim = _coupled_pdim_constructor(loader, node)
    log.debug("Returning default value of constructed ParamDim.")
    return cpdim.default

def coupled_pdim_always_disable(loader, node) -> CoupledParamDim:
    """constructor for creating a CoupledParamDim object from a mapping, but always disabling it

    Suggested tag: !disabled-coupled-pdim
    """
    cpdim = _coupled_pdim_constructor(loader, node)
    log.debug("Disabling constructed ParamDim.")
    cpdim.enabled = False
    return cpdim

# -----------------------------------------------------------------------------

def _pspace_constructor(loader, node, sort_if_mapping: bool=False) -> ParamSpace:
    """constructor for instantiating ParamSpace from a mapping or a sequence"""
    log.debug("Encountered tag associated with ParamSpace.")

    # get fields as mapping or sequence
    if isinstance(node, yaml.nodes.MappingNode):
        log.debug("Constructing mapping from node ...")
        d = loader.construct_mapping(node, deep=True)

        # Recursively order the content to have consistent loading
        if sort_if_mapping:
            log.debug("Recursively sorting the mapping ...")
            d = recursively_sort_dict(OrderedDict(d))

    elif isinstance(node, yaml.nodes.SequenceNode):
        log.debug("Constructing sequence from node ...")
        d = loader.construct_sequence(node, deep=True)
    
    else:
        raise TypeError("ParamSpace node can only be constructed from a "
                        "Mapping or a Sequence, got node of type {} with "
                        "value:\n{}.".format(type(node), node))

    log.debug("Instantiating ParamSpace ...")
    return ParamSpace(d)

def _pdim_constructor(loader, node) -> ParamDim:
    """constructor for creating a ParamDim object from a mapping

    For it to be incorported into a ParamSpace, one parent (or higher) of this node needs to be tagged such that the pspace_constructor is invoked.
    """
    log.debug("Encountered tag associated with ParamDim.")

    if isinstance(node, yaml.nodes.MappingNode):
        log.debug("Constructing mapping ...")
        mapping = loader.construct_mapping(node, deep=True)
        pdim = ParamDim(**mapping)
    else:
        raise TypeError("ParamDim can only be constructed from a mapping node,"
                        " got node of type {} "
                        "with value:\n{}".format(type(node), node))

    return pdim

def _coupled_pdim_constructor(loader, node) -> ParamDim:
    """constructor for creating a ParamDim object from a mapping

    For it to be incorported into a ParamSpace, one parent (or higher) of this node needs to be tagged such that the pspace_constructor is invoked.
    """
    log.debug("Encountered tag associated with ParamDim.")

    if isinstance(node, yaml.nodes.MappingNode):
        log.debug("Constructing mapping ...")
        mapping = loader.construct_mapping(node, deep=True)
        cpdim = CoupledParamDim(**mapping)
    else:
        raise TypeError("CoupledParamDim can only be constructed from a "
                        "mapping node, got node of type {} "
                        "with value:\n{}".format(type(node), node))

    return cpdim

# Helpers ---------------------------------------------------------------------

def recursively_sort_dict(d: dict) -> OrderedDict:
    """Recursively sorts a dictionary by its keys, transforming it to an OrderedDict in the process.
    
    From: http://stackoverflow.com/a/22721724/1827608
    
    Args:
        d (dict): The dictionary to be sorted
    
    Returns:
        OrderedDict: recursively sorted
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
