"""Defines the yaml constructors for the generation of
:py:class:`~paramspace.paramspace.ParamSpace` and
:py:class:`~paramspace.paramdim.ParamDim` during loading of YAML files.
"""
import logging
from collections import OrderedDict

import numpy as np
import ruamel.yaml
import yayaml as yay

from .paramdim import CoupledParamDim, Masked, ParamDim, ParamDimBase
from .paramspace import ParamSpace
from .tools import recursively_sort_dict

log = logging.getLogger(__name__)


# -- Aliases for some constructors --------------------------------------------


@yay.is_constructor("!pspace-unsorted")
def pspace_unsorted(loader, node) -> ParamSpace:
    """yaml constructor for creating a ParamSpace object from a mapping.

    Unlike the regular constructor, this one does NOT sort the input before
    instantiating ParamSpace."""
    return _pspace_constructor(loader, node, sort_if_mapping=False)


@yay.is_constructor("!pdim")
def pdim(loader, node) -> ParamDim:
    """constructor for creating a ParamDim object from a mapping, but only
    return the default value."""
    return _pdim_constructor(loader, node, Cls=ParamDim, default_order=np.inf)


@yay.is_constructor("!coupled-pdim")
def coupled_pdim(loader, node) -> ParamDim:
    """constructor for creating a ParamDim object from a mapping, but only
    return the default value."""
    return _pdim_constructor(loader, node, Cls=CoupledParamDim)


@yay.is_constructor("!sweep-default", aliases=("!pdim-default",))
def pdim_default(loader, node) -> ParamDim:
    """constructor for creating a ParamDim object from a mapping, but only
    return the default value."""
    pdim = _pdim_constructor(loader, node, Cls=ParamDim)
    log.debug("Returning default value of constructed ParamDim.")
    default = pdim.default
    if isinstance(default, Masked):
        default = default.value
    return default


@yay.is_constructor(
    "!coupled-sweep-default", aliases=("!coupled-pdim-default",)
)
def coupled_pdim_default(loader, node) -> CoupledParamDim:
    """Constructor for creating a CoupledParamDim object from a mapping, but
    only return the default value.

    .. note::

        This can only be used for coupled parameter dimensions that do *not*
        rely on the coupling target for their default value.
    """
    cpdim = _pdim_constructor(loader, node, Cls=CoupledParamDim)
    log.debug("Returning default value of constructed CoupledParamDim.")
    default = cpdim.default
    if isinstance(default, Masked):
        default = default.value
    return default


# -- The actual constructor functions -----------------------------------------


def _pspace_constructor(
    loader, node, sort_if_mapping: bool = True, Cls=ParamSpace
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
            f"{Cls} node can only be constructed from a mapping or a "
            f"sequence, got node of type {type(node)} with value:\n{node}."
        )

    log.debug("Instantiating ParamSpace ...")
    return Cls(d)


def _pdim_constructor(
    loader, node, *, Cls=ParamDim, default_order: float = None
) -> ParamDimBase:
    """Constructor for creating a ParamDim object from a mapping

    For it to be incorported into a ParamSpace, one parent (or higher) of this
    node needs to be tagged such that the pspace_constructor is invoked.
    """
    log.debug(
        "Encountered tag '%s' associated with parameter dimension.", node.tag
    )

    if isinstance(node, ruamel.yaml.nodes.MappingNode):
        log.debug("Constructing mapping ...")
        kwargs = loader.construct_mapping(node, deep=True)

        # To be backwards-compatible to <2.6, where ParamDims do not store the
        # order value unless it is explicitly set.
        # This is IMPORTANT because otherwise loading dumps from <2.6 will
        # yield different state numbers and thus different iteration order.
        if default_order is not None:
            if "order" not in kwargs or kwargs.get("order") is None:
                kwargs["order"] = default_order

        # Can now construct the object:
        pdim = Cls(**kwargs)

    else:
        raise TypeError(
            f"{Cls} can only be constructed from a mapping node, got node "
            f"of type {type(node)} with value:\n{node}"
        )

    return pdim
