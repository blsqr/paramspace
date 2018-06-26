"""The ParamSpace class is an extension of a dict, which can be used to iterate over a paramter space."""

import copy
import logging
import warnings
import pprint
from itertools import chain
import collections
from collections import OrderedDict
from typing import Union, Sequence, Tuple, Generator, MutableMapping, MutableSequence, Dict

import numpy as np

from .paramdim import ParamDimBase, ParamDim, CoupledParamDim
from .tools import recursive_collect, recursive_update, recursive_replace

# Get logger
log = logging.getLogger(__name__)

# Define an input type for the dictionary
PStype = Union[MutableMapping, MutableSequence]

# -----------------------------------------------------------------------------

class ParamSpace:

    def __init__(self, d: PStype):
        """Initialize a ParamSpace object from a given mapping or sequence.
        
        Args:
            d (Union[MutableMapping, MutableSequence]): The mapping or sequence
                that will form the parameter space. It is crucial that this
                object is mutable.
        """

        # Warn if type is unusual
        if not isinstance(d, (collections.abc.MutableMapping,
                              collections.abc.MutableSequence)):
            warnings.warn("Got unusual type {} for ParamSpace initialisation."
                          "If the given object is not mutable, this might fail"
                          " somewhere unexpected.".format(type(d)),
                          UserWarning)

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
        self._iter = None

    def _gather_paramdims(self):
        """Gathers the ParamDim objects by recursively going through the dictionary."""
        log.debug("Gathering ParamDim objects ...")

        # Traverse the dict and look for ParamDim objects; collect them as (order, key, value) tuples
        pdims = recursive_collect(self._dict,
                                  select_func=lambda p: isinstance(p,ParamDim),
                                  prepend_info=('info_func', 'keys'),
                                  info_func=lambda p: p.order,
                                  stop_recursion_types=(ParamDimBase,))

        # Sort them -- very important for consistency!
        # This looks at the info first, which is the order entry, and then at the keys. If a ParamDim does not provide an order, it has entry np.inf there, such that those without order get sorted by the key.
        pdims.sort()

        # Now need to reduce the list items to 2-tuples, ditching the order, to allow to initialise the OrderedDict
        self._dims = OrderedDict([tpl[1:] for tpl in pdims])
        log.debug("Found %d ParamDim objects.", self.num_dims)


        log.debug("Gathering CoupledParamDim objects ...")
        # Also collect the coupled ParamDims and continue with the same procedure
        cpdims = recursive_collect(self._dict,
                                   select_func=lambda p: isinstance(p, CoupledParamDim),
                                   prepend_info=('info_func', 'keys'),
                                   info_func=lambda p: p.order,
                                   stop_recursion_types=(ParamDimBase,))
        cpdims.sort() # same sorting rules as above, but not as crucial here because they do not change the iteration order through state space
        self._cdims = OrderedDict([tpl[1:] for tpl in cpdims])

        # Now resolve the coupling targets and add them to CoupledParamDim instances. Also, let the target ParamDim objects know which CoupledParamDim couples to them
        for cpdim_key, cpdim in self.coupled_dims.items():
            # Try to get the coupling target by name
            try:
                c_target = self._dim_by_name(cpdim.target_name)
            
            except (KeyError, ValueError) as err:
                # Could not find that name
                raise ValueError("Could not resolve the coupling target for "
                                 "CoupledParamDim at {}. Check the "
                                 "`target_name` specification of that entry "
                                 "and the full traceback of this error."
                                 "".format(cpdim_key)) from err

            # Set attribute of the coupled ParamDim
            cpdim.target_pdim = c_target

            # And inform the target ParamDim about it being the target of the coupled param dim, if it is not already included there
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
        return tuple([len(pd) for pd in self.dims.values()])

    @property
    def state_no(self) -> int:
        """Returns the current state number."""
        return self._state_no
    
    @property
    def state_vector(self) -> Tuple[int]:
        """Returns the state vector of all detected parameter dimensions."""
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
    def dims(self) -> Dict[Tuple[str], ParamDim]:
        """Returns the ParamDim objects found in this ParamSpace"""
        return self._dims

    @property
    def coupled_dims(self) -> Dict[Tuple[str], CoupledParamDim]:
        """Returns the CoupledParamDim objects found in this ParamSpace"""
        return self._cdims

    # Magic methods ...........................................................

    def __str__(self) -> str:
        """Returns a parsed, human-readable information string"""
        return self.get_info_str()

    def __repr__(self) -> str:
        """Returns the raw string representation of the ParamSpace."""
        # TODO should actually be a string from which to re-create the object
        return ("<{} object at {} with {}>"
                "".format(self.__class__.__name__, id(self),
                          repr(dict(volume=self.volume,
                                    shape=self.shape,
                                    dims=self.dims,
                                    coupled_dims=self.coupled_dims
                                    ))
                          )
                )

    def __format__(self, fstr: str) -> str:
        """ """
        raise NotImplementedError

    def get_info_str(self) -> str:
        """Returns a string that gives information about shape and size of this ParamSpace."""
        # Gather lines in a list
        l = ["ParamSpace Information"]

        # General information about the Parameter Space
        l += ["  Dimensions:  {}".format(self.num_dims)]
        l += ["  Volume:      {}".format(self.volume)]
        l += ["  Coupled:     {}".format(self.num_coupled_dims)]

        # ParamDim information
        l += ["", "Parameter Dimensions"]
        l += ["  (First mentioned are iterated over most often)", ""]

        for name, pdim in self.dims.items():
            l += ["  * {}".format(" -> ".join([str(e) for e in name]))]
            l += ["      {}".format(pdim.values)]

            if pdim.order < np.inf:
                l += ["      Order: {}".format(pdim.order)]

            l += [""]

        # CoupledParamDim information
        if self.num_coupled_dims:
            l += ["", "Coupled Parameter Dimensions"]
            l += ["  (Move alongside the state of the coupled ParamDim)", ""]

            for name, cpdim in self.coupled_dims.items():
                l += ["  * {}".format(" -> ".join([str(e) for e in name]))]
                l += ["      Coupled to:  {}".format(cpdim.target_name)]

                # Add resolved target name, if it differs
                for pdim_name, pdim in self.dims.items():
                    if pdim is cpdim.target_pdim:
                        # Found the coupling target object; get the full name
                        resolved_target_name = pdim_name
                        break
                else:
                    raise RuntimeError("Could not find coupling target; this "
                                       "should not have happened!")

                if resolved_target_name != cpdim.target_name:
                    l[-1] += "  [resolves to: {}]".format(resolved_target_name)

                l += ["      Values:      {}".format(cpdim.values)]
                l += [""]

        return "\n".join(l)

    # Item access .............................................................
    # This is a restricted interface for accessing items
    # It ensures that the ParamSpace remains in a valid state: items are only
    # returned by copy or, if popping them, it is ensured that the item was not
    # a parameter dimension.
    
    # FIXME Resolve misconception: storing key sequences as tuples, but a
    #       tuple could be a key itself as it is hashable...

    def get(self, key, default=None):
        """Returns a _copy_ of the item in the underlying dict"""
        return copy.deepcopy(self._dict.get(key, default))

    def pop(self, key, default=None):
        """Pops an item from the underlying dict, if it is not a ParamDim"""
        item = self._dict.get(key, None)
        if item in self.dims.values() or item in self.coupled_dims.values():
            raise KeyError("Cannot remove item with key '{}' as it is part of "
                           "a parameter dimension.".format(key))

        return self._dict.pop(key, default)

    # Iterator functionality ..................................................

    def __iter__(self) -> PStype:
        """Move to the next valid point in parameter space and return the corresponding dictionary.
        
        Returns:
            The current value of the iteration
        
        Raises:
            StopIteration: When the iteration has finished
        """
        if self._iter is None:
            # Associate with the all_points iteration
            self._iter = self.all_points

        # Let generator yield and given the return value, check how to proceed
        return self._iter()
        # NOTE the generator will also raise StopIteration once it ended
        

    def all_points(self, with_info: Sequence=None) -> Generator[PStype, None, None]:
        """Returns a generator yielding all points of the parameter space."""

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
        yield self._gen_info_tuple(self.current_point, with_info=with_info)

        # Now yield all the other states, while available.
        while self._next_state():
            yield self._gen_info_tuple(self.current_point, with_info=with_info)

        else:
            log.debug("Visited every point in ParamSpace.")
            self._reset()
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
                # Went through all states of this dim -> go to next dimension and start iterating that (similar to the carry bit in addition)
                # Important: prepare pdim such that it is at state zero again
                pdim.enter_iteration()
                continue
            else:
                # Iterated to next step without reaching the last dim item
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
        """Returns an inverse mapping of dimension to state numbers."""
        if self._imap is not None:
            # Return the cached result
            log.debug("Using previously created inverse mapping ...")
            return self._imap
        # else: calculate the inverse mapping

        # Create empty n-dimensional array
        shape = self.shape
        imap = np.ndarray(shape, dtype=int)
        imap.fill(-1) # i.e., not set yet

        # Iterate over all points and save the state number to the map
        for _, state_no, state_vec in self.all_points(with_info=['state_no',
                                                      'state_vec']):
            # Need to convert entries in the state vector if there are Nones
            s = [0 if i is None else i for i in state_vec]

            # Save the state number to the mapping
            try:
                imap[tuple(s)] = state_no

            except IndexError as err:
                fstr = ("Creating inverse mapping failed, probably due to a "
                        "change in the parameter dimensions sizes."
                        " Selector: {} -- imap shape: {}")
                raise RuntimeError(fstr.format(s, imap.shape)) from err
        else:
            log.debug("Finished creating inverse mapping. Caching it...")
            self._imap = imap

        return self._imap

    def get_subspace(self, **slices):
        """Returns a subspace of this parameter space."""
        # TODO find a way to preserve the state numbers from the parent
        raise NotImplementedError

    # Non-public API ..........................................................

    def _dim_no_by_name(self, name: str, include_coupled: bool=False) -> int:
        """Tries to find a ParamDim by its name and returns the number it corresponds to in the list of ordered 
        
        Args:
            name (str): the name of the dim, which can be a tuple of strings
                or a string. If name is a tuple of strings, the exact tuple is
                required to find the dim by its dim_name. If name is a
                string, only the last element of the dim_name is considered.
            include_coupled (bool, optional): Whether to include
                CoupledParamDim objects into the search (NotImplemented)
        
        Returns:
            int: the number of the dimension
        
        Raises:
            KeyError: If the ParamDim could not be found
            NotImplementedError: Argument `include_coupled`
            ValueError: If argument name was only a string, there can be
                duplicates. In the case of duplicate entries, a ValueError is
                raised.
        
        """
        if include_coupled:
            raise NotImplementedError("include_coupled")

        dim_no = None

        if isinstance(name, str):
            for n, dim_name in enumerate(self.dims.keys()):
                if dim_name[-1] == name:
                    if dim_no is not None:
                        # Was already set -> there was a duplicate
                        raise ValueError("Duplicate dim name {} encountered "
                                         "during access via the last key of "
                                         "the dim name. To not get an "
                                         "ambiguous result, pass the full dim "
                                         "name as a tuple.".format(name))
                    dim_no = n

        else:
            for n, dim_name in enumerate(self.dims.keys()):
                if dim_name[-len(name):] == name:
                    # The last part of the sequence matches the given name
                    if dim_no is not None:
                        # Was already set -> there was a duplicate
                        raise ValueError("Access via '{}' was ambiguous. Give "
                                         "the full sequence of strings as a "
                                         "dim name to be sure to access the "
                                         "right element.".format(name))
                    dim_no = n

        if dim_no is None:
            raise KeyError("A ParamDim with name {} was not "
                           "found in this ParamSpace.".format(name))

        return dim_no


    def _dim_by_name(self, name: str, include_coupled: bool=False) -> ParamDimBase:
        """Get the ParamDim object with the given name
        
        Args:
            name (str): the name of the dim, which can be a tuple of strings
                or a string. If name is a tuple of strings, the exact tuple is
                required to find the dim by its dim_name. If name is a
                string, only the last element of the dim_name is considered.
            include_coupled (bool, optional): Whether to include
                CoupledParamDim objects into the search (NotImplemented)
        
        Returns:
            int: the number of the dimension
        
        Raises:
            KeyError: If the ParamDim could not be found
            NotImplementedError: Argument `include_coupled`
            ValueError: If argument name was only a string, there can be
                duplicates. In the case of duplicate entries, a ValueError is
                raised.
        
        """
        if include_coupled:
            raise NotImplementedError("include_coupled")

        pdim = None

        if isinstance(name, str):
            for dim_name, _pdim in self.dims.items():
                if dim_name[-1] == name:
                    if pdim is not None:
                        # Was already set -> there was a duplicate
                        raise ValueError("Duplicate dim name {} encountered "
                                         "during access via the last key of "
                                         "the dim name. To not get an "
                                         "ambiguous result, pass the full dim "
                                         "name as a tuple.".format(name))
                    # Found one, save it
                    pdim = _pdim

        else:
            for dim_name, _pdim in self.dims.items():
                if dim_name[-len(name):] == name:
                    # The last part of the sequence matches the given name
                    if pdim is not None:
                        # Was already set -> there was a duplicate
                        raise ValueError("Access via '{}' was ambiguous. Give "
                                         "the full sequence of strings as a "
                                         "dim name to be sure to access the "
                                         "right element.".format(name))
                    # Found one, save it
                    pdim = _pdim

        if pdim is None:
            raise KeyError("A ParamDim with name {} was not "
                           "found in this ParamSpace.".format(name))

        return pdim

    def _dim_no_by_name(self, name: str, include_coupled: bool=False) -> int:
        """Returns the number of the dimension object with that name."""
        raise NotImplementedError

    def _gen_info_tuple(self, pt, *, with_info: Sequence) -> tuple:
        """Is used during iteration to add additional information to the return tuple."""
        if not with_info:
            return pt

        # Parse the tuple and add information
        info_tup = tuple()
        for info in with_info:
            if info in ['state_no']:
                info_tup += (self.state_no,)
            elif info in ['state_vector', 'state_vec']:
                info_tup += (self.state_vector,)
            elif info in ['progress']:
                info_tup += ((self.state_no+1)/self.volume,)
            else:
                raise ValueError("No such information '{}' available. "
                                 "Check the `with_info` argument!".format(info))

        # Concatenate and return
        return (pt,) + info_tup
