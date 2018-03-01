"""The ParamSpace class is an extension of a dict, which can be used to iterate over a paramter space."""

import copy
import logging
import pprint
from itertools import chain
import collections
from collections import OrderedDict
from typing import Union, Sequence, Tuple, Generator, MutableMapping, MutableSequence

import numpy as np

from .paramdim import ParamDimBase, ParamDim, CoupledParamDim
from .tools import recursive_collect, recursive_update, recursive_replace

# Get logger
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------

class ParamSpace:

    def __init__(self, d: Union[MutableMapping, MutableSequence]):
        """Initialize a ParamSpace object from a given mapping or sequence.
        
        Args:
            d (Union[MutableMapping, MutableSequence]): The mapping or sequence
                that will form the parameter space. It is crucial that this
                object is mutable.
        """

        if not isinstance(d, (collections.abc.MutableMapping,
                              collections.abc.MutableSequence)):
            raise TypeError("Can only generate ParamSpace objects from a mutable sequence or mapping, got "+str(type(d)))

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
        return recursive_replace(copy.deepcopy(self._dict),
                                 select_func=lambda v: isinstance(v, ParamDimBase),
                                 replace_func=lambda pdim: pdim.default)

    @property
    def current_point(self) -> dict:
        """Returns the dictionary with all parameter dimensions resolved to the values, depending on the point in parameter space at which the iteration is."""
        return recursive_replace(copy.deepcopy(self._dict),
                                 select_func=lambda v: isinstance(v, ParamDimBase),
                                 replace_func=lambda pdim: pdim.current_value)

    @property
    def volume(self) -> int:
        """Returns the volume of the parameter space, not counting coupled parameter dimensions."""
        if self.num_dims == 0:
            return 0

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
        raise NotImplementedError

    def all_points(self) -> dict:
        """Returns iterator over all points of the parameter space."""
        
        if self.volume < 1:
            raise ValueError("Cannot iterate over ParamSpace of zero volume.")

        log.debug("Starting iteration over all %d points in ParamSpace ...",
                  self.volume)

        # Prepare parameter dimensions: set them to state 0
        for pdim in self.dims.values():
            pdim.enter_iteration()

        # This corresponds to ParamSpace's state 0
        self._state_no = 0

        # Yield the first state
        yield self.current_point

        # Now yield all the other states, while available.
        while self._next_state():
            yield self.current_point

        else:
            log.debug("Visited every point in ParamSpace.")
            self.reset()
            log.debug("Reset ParamSpace and ParamDims.")
            return

    def _next_state(self) -> bool:
        """Iterates the state of the parameter dimensions managed by this ParamSpace.

        Important: this assumes that the parameter dimensions already have been prepared for an iteration and that self.state_no == 0.
        
        Returns:
            bool: Returns False when iteration finishes
        """
        log.debug("ParamSpace._next_state called")

        for pdim in self.dims.values():
            try:
                pdim.iterate_state()

            except StopIteration:
                # Went through all states of this span -> go to next dimension and start iterating that (similar to the carry bit in addition)
                # Important: prepare pdim such that it is at state zero again
                pdim.enter_iteration()
                continue
            else:
                # Iterated to next step without reaching the last span item
                break
        else:
            # Loop went through -> all states visited
            self._reset()            
            return False

        # Broke out of loop -> Got the next state and not at the end yet
        # Increment state number
        self._state_no += 1

        # Have not reached the end yet; communicate that
        return True

    def _reset(self) -> None:
        """Resets the paramter space and all of its dimensions to the initial state, i.e. where all states are None.
        """
        for pdim in self.dims.values():
            pdim.reset()

        self._state_no = None

    # Public API ..............................................................

    def inverse_mapping(self) -> np.ndarray:
        """ """
        raise NotImplementedError

    def get_subspace(self, **slices):
        """Returns a subspace of this parameter space."""
        # TODO find a way to preserve the state numbers from the parent
        raise NotImplementedError

    # Non-public API ..........................................................

    def _dim_by_name(self, name: str, include_coupled: bool=False) -> ParamDimBase:
        """ """
        raise NotImplementedError