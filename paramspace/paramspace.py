"""Implementation of the ParamSpace class"""

import copy
import logging
import warnings
import collections
from collections import OrderedDict
from itertools import chain
from functools import reduce
from typing import Union, Sequence, Tuple, Generator, MutableMapping, MutableSequence, Dict, List

import numpy as np
import numpy.ma

from .paramdim import ParamDimBase, ParamDim, CoupledParamDim
from .tools import recursive_collect, recursive_update, recursive_replace

# Get logger
log = logging.getLogger(__name__)

# Define an input type for the dictionary
PStype = Union[MutableMapping, MutableSequence]

# -----------------------------------------------------------------------------

class ParamSpace:
    """The ParamSpace class holds dict-like data in which some entries are
    ParamDim objects. These objects each define one parameter dimension.

    The ParamSpace class then allows to iterate over the space that is created
    by the parameter dimensions: at each point of the space (created by the
    cartesian product of all dimensions), one manifestation of the underlying
    dict-like data is returned.
    """

    # Define the yaml tag to use
    yaml_tag = u'!pspace'

    # .........................................................................

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

        # Save a deep copy of the base dictionary. This dictionary will never
        # be changed.
        self._init_dict = copy.deepcopy(d)

        # Initialize a working copy. The parameter dimensions embedded in this
        # copy will change their values
        self._dict = copy.deepcopy(self._init_dict)

        # Initialize attributes that will be used to gather parameter
        # dimensions and coupled parameter dimensions, and call the function
        # that gathers these objects
        self._dims = None
        self._cdims = None
        self._gather_paramdims() # NOTE attributes set within this method

        # Initialize caching attributes
        self._smap = None
        self._iter = None

    def _gather_paramdims(self):
        """Gathers ParamDim objects by recursively going through the dict"""
        log.debug("Gathering ParamDim objects ...")

        # Traverse the dict and look for ParamDim objects; collect them as
        # (order, key, value) tuples
        pdims = recursive_collect(self._dict,
                                  select_func=lambda p: isinstance(p,ParamDim),
                                  prepend_info=('info_func', 'keys'),
                                  info_func=lambda p: p.order,
                                  stop_recursion_types=(ParamDimBase,))

        # Sort them -- very important for consistency!
        # This looks at the info first, which is the order entry, and then at
        # the keys. If a ParamDim does not provide an order, it has entry
        # np.inf there, such that those without order get sorted by the key.
        pdims.sort()

        # Now need to reduce the list items to 2-tuples, ditching the order,
        # to allow to initialise the OrderedDict
        self._dims = OrderedDict([tpl[1:] for tpl in pdims])
        log.debug("Found %d ParamDim objects.", self.num_dims)


        log.debug("Gathering CoupledParamDim objects ...")
        # Also collect the coupled ParamDims; continue with the same procedure
        cpdims = recursive_collect(self._dict,
                                   select_func=lambda p: isinstance(p, CoupledParamDim),
                                   prepend_info=('info_func', 'keys'),
                                   info_func=lambda p: p.order,
                                   stop_recursion_types=(ParamDimBase,))
        cpdims.sort()
        # same sorting rules as above, but not as crucial here because they do
        # not change the iteration order through state space
        self._cdims = OrderedDict([tpl[1:] for tpl in cpdims])

        # Now resolve the coupling targets and add them to CoupledParamDim
        # instances. Also, let the target ParamDim objects know which
        # CoupledParamDim couples to them
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

            # And inform the target ParamDim about it being the target of the
            # coupled param dim, if it is not already included there
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
        """Returns the dictionary with all parameter dimensions resolved to
        their default values.
        """
        return recursive_replace(copy.deepcopy(self._dict),
                                 select_func=lambda v: isinstance(v, ParamDimBase),
                                 replace_func=lambda pdim: pdim.default,
                                 stop_recursion_types=(ParamSpace,))

    @property
    def current_point(self) -> dict:
        """Returns the dictionary with all parameter dimensions resolved to
        the values, depending on the point in parameter space at which the
        iteration is.
        """
        return recursive_replace(copy.deepcopy(self._dict),
                                 select_func=lambda v: isinstance(v, ParamDimBase),
                                 replace_func=lambda pdim: pdim.current_value,
                                 stop_recursion_types=(ParamSpace,))

    @property
    def dims(self) -> Dict[Tuple[str], ParamDim]:
        """Returns the ParamDim objects found in this ParamSpace"""
        return self._dims

    @property
    def coupled_dims(self) -> Dict[Tuple[str], CoupledParamDim]:
        """Returns the CoupledParamDim objects found in this ParamSpace"""
        return self._cdims
        
    @property
    def volume(self) -> int:
        """Returns the active volume of the parameter space, i.e. not counting
        coupled parameter dimensions or masked values
        """
        if self.num_dims == 0:
            return 0

        vol = 1
        for pdim in self.dims.values():
            # Need to check whether a dimension is fully masked, in which case
            # the default value is used and the dimension length is 1
            vol *= len(pdim) if pdim.mask is not True else 1
        return vol

    @property
    def full_volume(self) -> int:
        """Returns the full volume, i.e. ignoring whether parameter dimensions
        are masked.
        """
        if self.num_dims == 0:
            return 0

        vol = 1
        for pdim in self.dims.values():
            vol *= pdim.num_values
        return vol

    @property
    def shape(self) -> Tuple[int]:
        """Returns the shape of the parameter space, not counting masked
        values of parameter dimensions. If a dimension is fully masked, it is
        still represented as of length 1, representing the default value
        being used.
        
        Returns:
            Tuple[int]: The iterator shape
        """
        return tuple([len(pdim) if pdim.mask is not True else 1
                      for pdim in self.dims.values()])
    
    @property
    def full_shape(self) -> Tuple[int]:
        """Returns the shape of the parameter space, ignoring masked values
        
        Returns:
            Tuple[int]: The shape of the fully unmasked iterator
        """
        return tuple([pdim.num_values for pdim in self.dims.values()])
    
    @property
    def states_shape(self) -> Tuple[int]:
        """Returns the shape of the parameter space, including default states
        for each parameter dimension and ignoring masked ones.
        
        Returns:
            Tuple[int]: The shape tuple
        """
        return tuple([pdim.num_states for pdim in self.dims.values()])
    
    @property
    def state_vector(self) -> Tuple[int]:
        """Returns a tuple of all current parameter dimension states"""
        return tuple([s.state for s in self.dims.values()])

    @state_vector.setter
    def state_vector(self, vec: Tuple[int]):
        """Sets the state of all parameter dimensions"""
        if len(vec) != self.num_dims:
            raise ValueError("Given vector needs to be of same length as "
                             "there are number of dimensions ({}), was: {}"
                             "".format(self.num_dims, vec))

        for (name, pdim), new_state in zip(self.dims.items(), vec):
            try:
                pdim.state = new_state

            except ValueError as err:
                raise ValueError("Could not set the state of parameter "
                                 "dimension {} to {}!"
                                 "".format(name, new_state)) from err

        log.debug("Successfully set state vector to %s.", vec)

    @property
    def num_dims(self) -> int:
        """Returns the number of parameter space dimensions. Coupled
        dimensions are not counted here!
        """
        return len(self.dims)

    @property
    def num_coupled_dims(self) -> int:
        """Returns the number of coupled parameter space dimensions."""
        return len(self.coupled_dims)

    @property
    def state_no(self) -> Union[int, None]:
        """Returns the current state number by visiting the active parameter
        dimensions and querying their state numbers.
        """
        log.debug("Calculating state number ...")

        # Go over all parameter dimensions and extract the state values
        states = self.state_vector
        log.debug("  states:       %s", states)

        # Now need the full shape of the parameter space, i.e. ignoring masked
        # values but including the default values
        states_shape = self.states_shape
        log.debug("  states shape: %s  (volume: %s)",
                  states_shape, reduce(lambda x, y: x*y, states_shape))

        # The lengths will now be used to calculate the multipliers, starting
        # with 1 for the 0th pdim.
        # For example, given lengths [10, 10,  20,    5], the corresponding
        # multipliers are:           [ 1, 10, 100, 2000]
        mults = [reduce(lambda x, y: x*y, states_shape[:i], 1)
                 for i in range(self.num_dims)]
        log.debug("  multipliers:  %s", mults)

        # Now, calculate the state number
        state_no = sum([(s * m) for s, m in zip(states, mults)])
        log.debug("  state no:     %s", state_no)

        return state_no

    # Magic methods ...........................................................

    def __eq__(self, other) -> bool:
        """Tests the equality of two ParamSpace objects."""
        if not isinstance(other, ParamSpace):
            return False

        # Check for equality of the two objects' underlying __dict__s content,
        # skipping the caching attributes _smap and _iter
        # NOTE it is ok to not check these, because equality of the other
        #      content asserts that the _smap attributes will be equal, too.
        return all([self.__dict__[k] == other.__dict__[k]
                    for k in self.__dict__.keys()
                    if k not in ('_smap', '_iter')])

    def __str__(self) -> str:
        """Returns a parsed, human-readable information string"""
        return ("<{} object at {} with volume {}, shape {}>"
                "".format(self.__class__.__name__, id(self),
                          self.volume, self.shape)
                )

    def __repr__(self) -> str:
        """Returns the raw string representation of the ParamSpace."""
        # TODO should actually be a string from which to re-create the object
        return ("<paramspace.paramdim.{} object at {} with {}>"
                "".format(self.__class__.__name__, id(self),
                          repr(dict(volume=self.volume,
                                    shape=self.shape,
                                    dims=self.dims,
                                    coupled_dims=self.coupled_dims
                                    ))
                          )
                )

    # TODO implement __format__

    def get_info_str(self) -> str:
        """Returns a string that gives information about shape and size of
        this ParamSpace.
        """
        # Gather lines in a list
        l = ["ParamSpace Information"]

        # General information about the Parameter Space
        l += ["  Dimensions:  {}".format(self.num_dims)]
        l += ["  Coupled:     {}".format(self.num_coupled_dims)]
        l += ["  Shape:       {}".format(self.shape)]
        l += ["  Volume:      {}".format(self.volume)]

        # ParamDim information
        l += ["", "Parameter Dimensions"]
        l += ["  (First mentioned are iterated over most often)", ""]

        for name, pdim in self.dims.items():
            l += ["  * {}".format(self._parse_dim_name(name))]
            l += ["      {}".format(pdim.values)]
            # TODO add information on length?!
            if pdim.mask is True:
                l += ["      fully masked -> using default:  {}"
                      "".format(pdim.default)]

            if pdim.order < np.inf:
                l += ["      Order: {}".format(pdim.order)]

            l += [""]

        # CoupledParamDim information
        if self.num_coupled_dims:
            l += ["", "Coupled Parameter Dimensions"]
            l += ["  (Move alongside the state of the coupled ParamDim)", ""]

            for name, cpdim in self.coupled_dims.items():
                l += ["  * {}".format(self._parse_dim_name(name))]
                l += ["      Coupled to:  {}".format(cpdim.target_name)]

                # Add resolved target name, if it differs
                for pdim_name, pdim in self.dims.items():
                    if pdim is cpdim.target_pdim:
                        # Found the coupling target object; get the full name
                        resolved_target_name = pdim_name
                        break

                if resolved_target_name != cpdim.target_name:
                    l[-1] += "  [resolves to: {}]".format(resolved_target_name)

                l += ["      Values:      {}".format(cpdim.values)]
                l += [""]

        return "\n".join(l)
    
    # YAML representation .....................................................

    @classmethod
    def to_yaml(cls, representer, node):
        """In order to dump a ParamSpace as yaml, basically only the _dict
        attribute needs to be saved. It can be plugged into a constructor
        without any issues.
        However, to make the string representation a bit simpler, the
        OrderedDict is resolved to an unordered one.

        Args:
            representer (ruamel.yaml.representer): The representer module
            node (type(self)): The node, i.e. an instance of this class
        
        Returns:
            a yaml mapping that is able to recreate this object
        """
        # Get the objects _dict
        d = copy.deepcopy(node._dict)

        # Recursively go through it and cast dict on all OrderedDict entries
        def to_dict(od: OrderedDict):
            for k, v in od.items():
                if isinstance(v, OrderedDict):
                    od[k] = to_dict(v)
            return dict(od)

        # Can now call the representer
        return representer.represent_mapping(cls.yaml_tag, to_dict(d))

    @classmethod
    def from_yaml(cls, constructor, node):
        """The default constructor for a ParamSpace object"""
        return cls(**constructor.construct_mapping(node, deep=True))

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
        """Move to the next valid point in parameter space and return the
        corresponding dictionary.
        
        Returns:
            The current value of the iteration
        
        Raises:
            StopIteration: When the iteration has finished
        """
        if self._iter is None:
            # Associate with the all_points iterator
            self._iter = self.all_points

        # Let generator yield and given the return value, check how to proceed
        return self._iter()
        # NOTE the generator will also raise StopIteration once it ended
        
    def all_points(self, *, with_info: Union[str, Tuple[str]]=None, omit_pt: bool=False) -> Generator[PStype, None, None]:
        """Returns a generator yielding all points of the parameter space, i.e.
        the space spanned open by the parameter dimensions.
        
        Args:
            with_info (Union[str, Tuple[str]], optional): Can pass strings
                here that are to be returned as the second value. Possible
                values are: 'state_no', 'state_vector'.
                To get multiple, add them to a tuple.
            omit_pt (bool, optional): If true, the current value is omitted and
                only the information is returned.
        
        Returns:
            Generator[PStype, None, None]: yields point after point of the
                ParamSpace and the corresponding information
        
        Raises:
            ValueError: If the ParamSpace volume is zero and no iteration can
                be performed
        """

        if self.volume < 1:
            raise ValueError("Cannot iterate over ParamSpace of zero volume.")

        # Parse the with_info argument, making sure it is a tuple
        if isinstance(with_info, str):
            with_info = (with_info,)

        log.debug("Starting iteration over %d points in ParamSpace ...",
                  self.volume)

        # Prepare parameter dimensions: set them to state 0
        for pdim in self.dims.values():
            pdim.enter_iteration()

        # Yield the first state
        yield self._gen_iter_rv(self.current_point if not omit_pt else None,
                                with_info=with_info)

        # Now yield all the other states, while available.
        while self._next_state():
            yield self._gen_iter_rv((self.current_point if not omit_pt
                                     else None),
                                    with_info=with_info)

        else:
            log.debug("Visited every point in ParamSpace.")
            self.reset()
            return

    def reset(self) -> None:
        """Resets the paramter space and all of its dimensions to the initial
        state, i.e. where all states are None.
        """
        for pdim in self.dims.values():
            pdim.reset()

        log.debug("Reset ParamSpace and ParamDims.")

    def _next_state(self) -> bool:
        """Iterates the state of the parameter dimensions managed by this
        ParamSpace.

        Important: this assumes that the parameter dimensions already have
        been prepared for an iteration and that self.state_no == 0.
        
        Returns:
            bool: Returns False when iteration finishes
        """
        log.debug("ParamSpace._next_state called")

        for pdim in self.dims.values():
            try:
                pdim.iterate_state()

            except StopIteration:
                # Went through all states of this dim -> go to next dimension
                # and start iterating that (similar to the carry bit in
                # addition)
                # Important: prepare pdim such that it is at state zero again
                pdim.enter_iteration()
                continue
            
            else:
                # Iterated to next step without reaching the last dim item
                break
        else:
            # Loop went through
            # -> All states visited.
            #    Now need to reset and communicate that iteration is finished;
            #    do so by returning false, which is more convenient than
            #    raising StopIteration; the iteration is handled by the
            #    all_points method anyway.
            self.reset()
            return False

        # If this point is reached: broke out of loop
        # -> The next state was reached and we are not at the end yet.
        #    Communicate that by returning True.
        return True

    # Public API ..............................................................
    # Mapping . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

    @property
    def state_map(self) -> np.ndarray:
        """Returns an inverse mapping, i.e. an n-dimensional array where the
        indices along the dimensions relate to the states of the parameter
        dimensions and the content of the array relates to the state numbers.
        """
        # Check if the cached result can be returned
        if self._smap is not None:
            log.debug("Using previously cached inverse mapping ...")
            return self._smap
        
        # else: need to calculate the inverse mapping

        # Create empty n-dimensional array which will hold state numbers
        smap = np.ndarray(self.states_shape, dtype=int)
        smap.fill(-1) # i.e., not set yet

        # As .all_points does not allow iterating over default states, iterate
        # over the multi-index of the smap, set the state vector and get the
        # corresponding state number
        for midx in np.ndindex(smap.shape):
            # Set the state vector ( == midx) of the paramspace
            self.state_vector = midx

            # Resolve the corresponding state number and store at this midx
            smap[tuple(midx)] = self.state_no

        # Make sure there are no unset values
        if np.min(smap) < 0:
            raise RuntimeError("Did (somehow) not visit all points during "
                               "iteration over state space!\n state map:\n{}"
                               "".format(smap))
        
        # Cache and make it read-only before returning
        log.debug("Finished creating inverse mapping. Caching it and making "
                  "the cache read-only ...")
        self._smap = smap
        self._smap.flags.writeable = False

        return self._smap

    def get_masked_smap(self) -> np.ma.MaskedArray:
        """Returns a masked array based on the inverse mapping, where masked
        states are also masked in the array.

        Note that this array has to be re-calculated every time, as the mask
        status of the ParamDim objects is not controlled by the ParamSpace and
        can change without notice.
        """
        mmap = np.ma.MaskedArray(data=self.state_map.copy(), mask=True)

        # TODO could improve the below loop by checking what needs fewer
        #      iterations: unmasking falsely masked entries or vice versa ...
        # Unmask the unmasked values
        for s_no, s_vec in self.all_points(with_info=('state_no', 'state_vec'),
                                           omit_pt=True):
            # By assigning any value, the mask is removed
            mmap[s_vec] = s_no
            # TODO isn't there a better way than re-assigning the state no?!

        return mmap

    def get_state_vector(self, *, state_no: int) -> Tuple[int]:
        """Returns the state vector that corresponds to a state number
        
        Args:
            state_no (int): The state number to look for in the inverse mapping

        Returns:
            Tuple[int]: the state vector corresponding to the state number
        """
        try:
            return tuple(np.argwhere(self.state_map == state_no)[0])

        except IndexError as err:
            raise ValueError("Did not find state number {} in inverse "
                             "mapping! Make sure it is an integer in the "
                             "closed interval [0, {}]."
                             "".format(state_no,
                                       reduce(lambda x, y: x*y,
                                              self.states_shape) - 1))

    def get_dim_values(self, *, state_no: int=None, state_vector: Tuple[int]=None) -> OrderedDict:
        """Returns the current parameter dimension values or those of a
        certain state number or state vector.
        """
        if state_no is None and state_vector is None:
            # Return the current value
            return OrderedDict([(name, pdim.current_value)
                                for name, pdim in self.dims.items()])

        # Check that only one of the arguments was given
        if state_no is not None and state_vector is not None:
            raise TypeError("Expected only one of the arguments `state_no` "
                            "and `state_vector`, got both!")

        elif state_no is not None:
            state_vector = self.get_state_vector(state_no=state_no)

        # Can now assume that state_vector is set
        return OrderedDict([(name, pdim.values[s-1])
                            for (name, pdim), s
                            in zip(self.dims.items(), state_vector)])

    # Masking . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

    def set_mask(self, name: Union[str, Tuple[str]], mask: Union[bool, Tuple[bool]], invert: bool=False) -> None:
        """Set the mask value of the parameter dimension with the given name.
        
        Args:
            name (Union[str, Tuple[str]]): the name of the dim, which can be a
                tuple of strings or a string. If name is a string, only the
                last element of the dimension name tuple is considered.
                If name is a tuple of strings, not the whole sequence needs
                to be supplied but the last parts suffice; it just needs to be
                enough to resolve the dimension names unambiguously.
                For names at the root level that could be ambiguous, a leading
                "/" can be added as first segement of the key sequence.
                Also, the ParamDim's custom name attribute can be used to
                identify it.
            mask (Union[bool, Tuple[bool]]): The new mask values. Can also be
                a slice, the result of which defines the True values of the
                mask.
            invert (bool, optional): If set, the mask will be inverted _after_
                application.
        """
        # Resolve the parameter dimension
        pdim = self._dim_by_name(name)

        # Set its mask value
        pdim.mask = mask

        if invert:
            pdim.mask = [(not m) for m in pdim.mask_tuple]

        # Done.
        log.debug("Set mask of parameter dimension %s to %s.", name, pdim.mask)

    def set_masks(self, *mask_specs) -> None:
        """Sets multiple mask specifications after another. Note that the order
        is maintained and that sequential specifications can apply to the same
        parameter dimensions.
        
        Args:
            *mask_specs: Can be tuples/lists or dicts which will be unpacked
                (in the given order) and passed to .set_mask
        """
        log.debug("Setting %d masks ...", len(mask_specs))

        for ms in mask_specs:
            if isinstance(ms, dict):
                self.set_mask(**ms)
            else:
                self.set_mask(*ms)

    # Non-public API ..........................................................

    def _gen_iter_rv(self, pt, *, with_info: Sequence[str]) -> tuple:
        """Is used during iteration to generate the iteration return value,
        adding additional information if specified.

        Note that pt can also be None if all_points is a dry_run
        """
        if not with_info:
            return pt

        # Parse the tuple and add information
        info_tup = tuple()
        for info in with_info:
            if info in ['state_no']:
                info_tup += (self.state_no,)

            elif info in ['state_vector', 'state_vec']:
                info_tup += (self.state_vector,)

            else:
                raise ValueError("No such information '{}' available. "
                                 "Check the `with_info` argument!"
                                 "".format(info))

        # Return depending on whether a point was given or not
        if pt is not None:
            # Concatenate and return
            return (pt,) + info_tup

        elif len(info_tup) == 1:
            # Return only the single info entry
            return info_tup[0]

        # else: return as tuple
        return info_tup

    def _dim_by_name(self, name: Union[str, Tuple[str]]) -> ParamDimBase:
        """Get the ParamDim object with the given name.

        Note that coupled parameter dimensions cannot be accessed via this
        method.
        
        Args:
            name (Union[str, Tuple[str]]): the name of the dim, which can be a
                tuple of strings or a string. If name is a string, only the
                last element of the dimension name tuple is considered.
                If name is a tuple of strings, not the whole sequence needs
                to be supplied but the last parts suffice; it just needs to be
                enough to resolve the dimension names unambiguously.
                For names at the root level that could be ambiguous, a leading
                "/" can be added as first segement of the key sequence.
                Also, the ParamDim's custom name attribute can be used to
                identify it.
        
        Returns:
            int: the number of the dimension
        
        Raises:
            KeyError: If the ParamDim could not be found
            ValueError: If the parameter dimension name was ambiguous
        
        """
        pdim = None

        # Make sure it's a sequence of strings
        if isinstance(name, str):
            name = (name,)        

        # Need to check whether the given key sequence suggests an abs. path
        is_abs = (name[0] == "/" and len(name) > 1)

        # Assume `name` is a sequence of strings
        for dim_name, _pdim in self.dims.items():
            if (   (not is_abs and name == dim_name[-len(name):])
                or (    is_abs and name[1:] == dim_name)
                or (_pdim.name and name[-1] == _pdim.name)):
                # The last part of the key sequence matches the given name
                if pdim is not None:
                    # Already set -> there was already one matching this name
                    raise ValueError("Could not unambiguously find a "
                                     "parameter dimension matching the name "
                                     "{}! Pass a sequence of keys to select "
                                     "the right dimension. To symbolize that "
                                     "the key sequence is absolute, start "
                                     "with an `/` entry in the key sequence.\n"
                                     "Available parameter dimensions:\n"
                                     " * {}"
                                     "".format(name,
                                               "\n * ".join(self._dim_names)))
                # Found one, save it
                pdim = _pdim

        # If still None after all this, no such name was found
        if pdim is None:
            raise KeyError("A parameter dimension with name {} was not "
                           "found in this ParamSpace."
                           "Available parameter dimensions:\n"
                           " * {}"
                           "".format(name, "\n * ".join(self._dim_names)))

        return pdim

    @staticmethod
    def _parse_dim_name(name: Tuple[str]) -> str:
        """Returns a string representation of a parameter dimension name"""
        return " -> ".join([str(e) for e in name])

    @property
    def _dim_names(self) -> List[str]:
        """Returns a sequence of dimension names that can be joined together"""
        return [self._parse_dim_name(n) for n in self.dims.keys()]

