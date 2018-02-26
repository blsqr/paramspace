"""The ParamSpace class is an extension of a dict, which can be used to iterate over a paramter space."""

import copy
import logging
import pprint
from itertools import chain
from collections import OrderedDict
from typing import Union, Sequence, Tuple

import numpy as np

from .paramdim import ParamDimBase, ParamDim, CoupledParamDim
from .tools import recursive_collect, recursive_update, recursive_replace

# Get logger
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------

class ParamSpace:

    def __init__(self, d):
        """Initialize a ParamSpace object from a given dictionary."""

        # Save a deep copy of the base dictionary. This dictionary will never be changed.
        self._init_dict = copy.deepcopy(d)

        # Initialize a working copy. The parameter dimensions embedded in this copy will change their values
        self._dict = copy.deepcopy(self._init_dict)

        # Initialize attributes that will be used to gather parameter dimensions and coupled parameter dimensions, and call the function that gathers these objects
        self._dims = None
        self._cdims = None
        self._gather_paramdims() # NOTE attributes set within this method

        # Initialize state attributes
        self._state_no = None

        # Initialize caching attributes
        self._imap = None

    def _gather_paramdims(self):
        """Gathers the ParamDim objects by recursively going through the dictionary."""
        log.debug("Gathering ParamDim objects ...")

        # Traverse the dict and look for ParamDim objects; collect them as (order, key, value) tuples
        pdims = recursive_collect(self._dict,
                                  isinstance, ParamDim,
                                  prepend_info=('info_func', 'keys'),
                                  info_func=lambda ps: ps.order)

        # Sort them -- very important for consistency!
        # This looks at the info first, which is the order entry, and then at the keys. If a ParamDim does not provide an order, it has entry np.inf there, such that those without order get sorted by the key.
        pdims.sort()

        # Now need to reduce the list items to 2-tuples, ditching the order, to allow to initialise the OrderedDict
        self._dims = OrderedDict([tpl[1:] for tpl in pdims])
        log.debug("Found %d ParamDim objects.", self.num_dims)


        log.debug("Gathering CoupledParamDim objects ...")
        # Also collect the coupled ParamDims and continue with the same procedure
        cpdims = recursive_collect(self._dict,
                                   isinstance, CoupledParamDim,
                                   prepend_info=('info_func', 'keys'),
                                   info_func=lambda ps: ps.order)
        cpdims.sort() # same sorting rules as above, but not as crucial here because they do not change the iteration order through state space
        self._cdims = OrderedDict([tpl[1:] for tpl in cpdims])

        # Now resolve the coupling targets and add them to CoupledParamDim instances. Also, let the target ParamDim objects know which CoupledParamDim couples to them
        for cpdim in self.coupled_dims.values():
            # Try to get the coupling target by name
            try:
                c_target = self._dim_by_name(cpdim.coupled_to)
            except NameError as err:
                # Could not resolve it
                # TODO informative error message here
                raise

            # Set attribute of the coupled ParamDim
            cpdim.coupled_obj = c_target

            # And inform the target ParamDim about it being the target of the coupled param span, if it is not already included there
            if cpdim not in c_target.target_of:
                c_target.target_of.append(cpdim)
            
            # Done with this coupling
        else:
            log.debug("Found %d CoupledParamDim objects.",
                      self.num_coupled_dims)

        log.debug("Finished gathering.")

    # Properties ..............................................................

    @property
    def default(self) -> dict:
        """Returns the dictionary with all parameter dimensions resolved to their default values."""
        raise NotImplementedError

    @property
    def current_value(self) -> dict:
        """Returns the dictionary with all parameter dimensions resolved to the values, depending on their states."""
        raise NotImplementedError

    @property
    def volume(self) -> int:
        """Returns the volume of the parameter space, not counting coupled parameter dimensions."""
        vol = 1
        for pdim in self.dims.values():
            vol *= len(pdim)
        return vol

    @property
    def full_volume(self) -> int:
        """Returns the full volume of the parameter space, including coupled parameter dimensions."""
        vol = 1
        for pdim in chain(self.dims.values(), self.coupled_dims.values()):
            vol *= len(pdim)
        return vol

    @property
    def shape(self) -> Tuple[int]:
        """Returns the shape of the parameter space"""
        raise NotImplementedError

    @property
    def state_no(self) -> int:
        """Returns the current state number."""
        return self._state_no
    
    @property
    def state_vector(self) -> Tuple[int]:
        """ """
        return tuple([s.state for s in self.dims.values()])

    @property
    def full_state_vector(self) -> OrderedDict:
        """Returns an OrderedDict of all parameter space dimensions, including coupled ones."""
        return OrderedDict((k, v) for k, v in chain(self.dims.items(),
                                                    self.coupled_dims.items()))

    @property
    def num_dims(self) -> int:
        """Returns the number of parameter space dimensions. Coupled dimensions are not counted here!"""
        return len(self.dims)

    @property
    def num_coupled_dims(self) -> int:
        """Returns the number of coupled parameter space dimensions."""
        return len(self.coupled_dims)

    @property
    def dims(self) -> Sequence[ParamDim]:
        """ """
        return self._dims

    @property
    def coupled_dims(self) -> Sequence[CoupledParamDim]:
        """ """
        return self._cdims

    # Magic methods ...........................................................

    def __repr__(self) -> str:
        """ """
        raise NotImplementedError

    def __str__(self) -> str:
        """ """
        raise NotImplementedError

    def __format__(self) -> str:
        """ """
        raise NotImplementedError

    # Iterator functionality ..................................................

    def __next__(self) -> dict:
        """Move to the next valid point in parameter space and return the corresponding dictionary.
        
        Returns:
            The current value of the iteration
        
        Raises:
            StopIteration: When the iteration has finished
        """

        log.debug("__next__ called")


    # Public API ..............................................................

    def all_points(self):
        """Returns iterator over all points of the parameter space."""
        raise NotImplementedError

    def subspace(self, **slices):
        """Returns iterator over a subspace of the parameter space."""
        raise NotImplementedError


    # Non-public API ..........................................................

    def _dim_by_name(self, name: str) -> ParamDimBase:
        """ """
        raise NotImplementedError

    def _dim_by_name(self, name: str) -> ParamDimBase:
        """ """
        raise NotImplementedError