"""The ParamDim classes ...""" # TODO

import copy
import logging
import warnings
from typing import Iterable, Union, Tuple, Hashable, Sequence

import numpy as np

# Get logger
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Small helper classes

class Masked:
    """To indicate a masked value in a ParamDim"""
    def __init__(self, value):
        """Initialize a Masked object that is a placeholder for the given value
        Args:
            value: The value to mask
        """
        self._val = value

    @property
    def value(self):
        return self._val

    def __str__(self) -> str:
        return "{} (masked)".format(self.value)

    def __repr__(self) -> str:
        return "<paramspace.paramdim.Masked object with masked value: {}>".format(repr(self.value))

class MaskedValueError(ValueError):
    """Raised when trying to set the state of a ParamDim to a masked value"""
    pass

# -----------------------------------------------------------------------------

class ParamDimBase:
    """The ParamDim base class."""

    def __init__(self, *, default, values: Iterable=None, order: float=np.inf, name: str=None, mask: Union[bool, Tuple[bool]]=False, **kwargs) -> None:
        """Initialise a parameter dimension object.
        
        Args:
            default: default value of this parameter dimension
            values (Iterable, optional): Which discrete values this parameter
                dimension can take. This argument takes precedence over any
                constructors given in the kwargs (like range, linspace, …).
            order (float, optional): If given, this allows to specify an order
                within a ParamSpace that includes this ParamDim object
            name (str, optional): If given, this is an *additional* name of
                this ParamDim object, and can be used by the ParamSpace to
                access this object.
            mask (Union[bool, Tuple[bool]], optional): Which values of the
                dimension to mask, i.e.: skip in iteration. Note that masked
                values still count to the length of the parameter dimension!
            **kwargs: Constructors for the `values` argument, valid keys are
                `range`, `linspace`, and `logspace`; corresponding values are
                expected to be iterables and are passed to `range(*args)`,
                `np.linspace(*args)`, or `np.logspace(*args)`, respectively.
        
        Raises:
            ValueError: Description
        """
        # Initialize attributes that are not managed
        self.name = name
        self.order = order
        
        # Initialize attributes that are managed by properties or methods
        self._vals = None
        self._state = None
        self._mask_cache = None

        # Carry over arguments
        self._default = default

        # Set the values, first via the `values` argument, and check whether
        # there are enough arguments to set the values
        if values is not None:
            self._set_values(values)

        elif not any([k in kwargs for k in ('range', 'linspace', 'logspace')]):
            raise ValueError("No argument `values` or other `**kwargs` was "
                             "specified, but at least one of these is needed "
                             "for initialisation.")

        # Check again, now including the `kwargs`
        if kwargs is not None and self.values is None:
            # Need to parse the additional keyword arguments to generate the
            # values attribute from it
            if len(kwargs) > 1:
                warnings.warn("{}.__init__ was called with multiple "
                              "additional `**kwargs`; only one of these will "
                              "be used! The order in which the arguments are "
                              "used is: `range`, `linspace`, "
                              "`logspace`.".format(self.__class__.__name__),
                              UserWarning)

            # Set the values
            if 'range' in kwargs:
                self._set_values(range(*kwargs.get('range')))

            elif 'linspace' in kwargs:
                self._set_values(np.linspace(*kwargs.get('linspace')),
                                 as_float=True)

            elif 'logspace' in kwargs:
                self._set_values(np.logspace(*kwargs.get('logspace')),
                                 as_float=True)
            # else: not possible, was checked above

        elif kwargs and self.values is not None:
            warnings.warn("{}.__init__ was called with both the argument "
                          "`values` and additional `**kwargs`: {}. With "
                          "`values` present, the additional keyword arguments "
                          "are ignored.".format(self.__class__.__name__,
                                                kwargs),
                          UserWarning)

        # Can now set the mask
        self.mask = mask

        # Done.
            
    # Properties ..............................................................

    @property
    def default(self):
        """The default value."""
        return self._default

    @property
    def values(self) -> tuple:
        """The values that are iterated over.
        
        Returns:
            tuple: the values this parameter dimension can take. If None, the
                values are not yet set.
        """
        return self._vals

    @property
    def mask(self) -> Union[bool, Tuple[bool]]:
        """Returns False if no value is masked or a tuple of booleans that
        represents the mask
        """        
        m = self._mask_tuple()  # uses a cached value, if available
        
        if not any(m):  # no entry masked
            return False

        elif all(m):  # all entries masked
            return True

        # leave it as a tuple
        return m

    @mask.setter
    def mask(self, mask: Union[bool, Tuple[bool]]):
        """Sets the mask
        
        Args:
            mask (Union[bool, Tuple[bool]]): A bool or an iterable of booleans
        
        Raises:
            ValueError: If the length of the iterable does not match that of
                this parameter dimension
        """
        # Helper function for setting a mask value
        def set_val(mask: bool, val):
            if mask and not isinstance(val, Masked):
                # Should be masked but is not
                return Masked(val)

            elif isinstance(val, Masked) and not mask:
                # Is masked but shouldn't be
                return val.value

            # Already the desired status
            return val

        # Resolve boolean values
        if isinstance(mask, bool):
            mask = [mask] * len(self)

        # Should be a container now. Assert correct length.
        if len(mask) != len(self.values):
            raise ValueError("Given mask needs to be a boolean or a container "
                             "of same length as the values container ({}), "
                             "was:  {}"
                             "".format(len(self), mask))

        # Mark the mask cache as invalid, such that it is re-calculated when
        # the mask getter is accessed the next time
        self._mask_cache = None

        # Now build a new values container and store as attributes
        self._vals = tuple([set_val(m, v) for m, v in zip(mask, self.values)])

    @property
    def state(self) -> Union[int, None]:
        """The current iterator state
        
        Returns:
            Union[int, None]: The state of the iterator; if it is None, the
                ParamDim is not inside an iteration.
        """
        return self._state

    @state.setter
    def state(self, new_state: Union[int, None]):
        """Sets the current iterator state."""
        if new_state is None:
            self._state = None

        elif isinstance(new_state, int):
            # Check for valid state interval
            if new_state < 0 or new_state >= len(self):
                raise ValueError("New state needs to be positive and cannot "
                                 "exceed the highest index of the value "
                                 "container ({}), was {}."
                                 "".format(len(self)-1, new_state))

            elif isinstance(self._mask_tuple()[new_state], Masked):
                raise MaskedValueError("Value at index {} is masked: {}. "
                                       "Cannot set the state to this index."
                                       "".format(new_state,
                                                 self.values[new_state]))

            # Everything ok. Can set the
            self._state = new_state

        else:
            raise TypeError("New state can only be of type int or None, "
                            "was "+str(type(new_state)))

    @property
    def current_value(self):
        """If in an iteration: return the value according to the current
        state. If not in an iteration, return the default value.
        """
        if self.state is None:
            return self.default
        return self.values[self.state]


    # Magic methods ...........................................................

    def __eq__(self, other) -> bool:
        """Check for equality between self and other
        
        Args:
            other (ParamDimBase): the other ParamDim
        
        Returns:
            bool: Whether the two parameter dimensions have the same content
        """
        if not isinstance(other, type(self)):
            return False

        # Check equality of the objects' __dict__s, leaving out _mask_cache
        return all([self.__dict__[k] == other.__dict__[k]
                    for k in self.__dict__.keys()
                    if k not in ('_mask_cache',)])

    def __len__(self) -> int:
        """Returns the length of the parameter dimension.
        
        Returns:
            int: The length of the associated values list. If the parameter
                dimension is not enabled, the length is 1.
        """
        return len(self.values)

    def __repr__(self) -> str:
        """
        Returns:
            str: Returns the string representation of the ParamDimBase-derived
                object
        """
        # TODO should actually be a string from which to re-create the object
        return ("<{} object at {} with {}>"
                "".format(self.__class__.__name__, id(self),
                          repr(dict(default=self.default,
                                    order=self.order,
                                    values=self.values,
                                    name=self.name,
                                    mask=self.mask))))

    def __str__(self) -> str:
        """
        Returns:
            str: Returns the string representation of the ParamDimBase-derived
                object
        """
        return repr(self)


    # Iterator functionality ..................................................

    def __iter__(self):
        return self

    def __next__(self):
        """Move to the next valid state and return the corresponding parameter
        value.
        
        Returns:
            The current value of the iteration
        """
        log.debug("__next__ called")

        # Iterate state and return the new value (i.e., the new current value)
        self.iterate_state()
        return self.current_value


    # Public API ..............................................................
    # These are needed by the ParamSpace class to have more control over the
    # iteration.

    def iterate_state(self) -> None:
        """Iterates the state of the parameter dimension.
        
        Raises:
            StopIteration: Upon end of iteration
        """
        # Set to zero or increment, depending on whether inside or outside of
        # an iteration
        if self.state is None:
            self.enter_iteration()
            # NOTE This will raise StopIteration if all values are masked.
            #      Thus, in the following, it can be assumed that at least one
            #      value is unmasked.
            return
            
        # Else: within iteration
        # Look for further possible states in the shortened mask tuple
        sub_mask = self._mask_tuple()[self.state + 1:]

        if False in sub_mask:
            # There is another possible state, find it via index
            self.state += (sub_mask.index(False) + 1)

        else:
            # No more possible state values
            # Reset the state, allowing to reuse the object (unlike with
            # other Python iterators). Then communicate: iteration should stop.
            self.reset()
            raise StopIteration

    def enter_iteration(self) -> None:
        """Sets the state to the first possible one, symbolising that an
        iteration has started.
        
        Raises:
            StopIteration: If no iteration is possible because all values are
                masked.
        """
        # Need to distinguish mask states
        if self.mask is False:
            # Trivial case, start with 0
            self.state = 0

        elif self.mask is True:
            # Need to communicate that there is nothing to iterate
            raise StopIteration

        else:
            # Find the first unmasked state
            self.state = self.mask.index(False)

    def reset(self) -> None:
        """Resets the state to None, called after the end of an iteration."""
        self.state = None


    # Non-public API ..........................................................
    
    def _set_values(self, values: Iterable, as_float: bool=False):
        """This function sets the values attribute; it is needed for the
        values setter function that is overwritten when changing the property
        in a derived class.
        
        Args:
            values (Iterable): The iterable to set the values with
            as_float (bool, optional): If given, makes sure that values are
                of type float; this is needed for the numpy initializers
        
        Raises:
            AttributeError: If the attribute is already set
            ValueError: If the iterator is invalid
        """
        if self._vals is not None:
            # Was already set
            raise AttributeError("Values already set; cannot be set again!")

        elif not len(values):
            raise ValueError("Argument `values` needs to be an iterable "
                             "of at least length 1, was " + str(values))
        
        # Resolve iterator as tuple, optionally ensuring it is a float
        if as_float:
            values = [float(v) for v in values]

        # Now store it as tuple attribute
        self._vals = tuple(values)

    def _mask_tuple(self) -> Tuple[bool]:
        """Returns a tuple representation of the current mask"""
        if self._mask_cache is None:
            self._mask_cache = tuple([isinstance(v, Masked)
                                      for v in self.values])
        return self._mask_cache


# -----------------------------------------------------------------------------

class ParamDim(ParamDimBase):
    """The ParamDim class."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Additional attributes, needed for coupling.
        self._target_of = []        

    @property
    def target_of(self):
        """Returns the list that holds all the CoupledParamDim objects that
        point to this instance of ParamDim.
        """
        return self._target_of


# -----------------------------------------------------------------------------

class CoupledParamDim(ParamDimBase):
    """A CoupledParamDim object is recognized by the ParamSpace and its state
    moves alongside with another ParamDim's state.
    """

    def __init__(self, *, target_pdim: ParamDim=None, target_name: Union[str, Sequence[str]]=None, use_coupled_default: bool=None, use_coupled_values: bool=None, **kwargs):
        """
        Args:
            target_pdim (ParamDim, optional): The ParamDim object to couple to
            target_name (Union[str, Sequence[str]], optional): The *name* of
                the ParamDim object to couple to; needs to be within the same
                ParamSpace and the ParamSpace needs to be able to resolve it
                using this name.
            use_coupled_default (bool, optional): Whether to use the default
                value of the coupled ParamDim; need not be given: if it is not
                set it is determined by looking at whether argument `default`
                was passed in the kwargs.
            use_coupled_values (bool, optional): Whether to use the values
                of the coupled ParamDim; need not be given: if it is not
                set it is determined by looking at whether argument `values`
                was passed in the kwargs.
            **kwargs: All ParamDim kwargs
        
        Raises:
            ValueError: If neither target_pdim nor target_name were given
        """

        # Determine whether the coupled values will be used or not
        if use_coupled_default is None:
            use_coupled_default = 'default' not in kwargs

        if use_coupled_values is None:
            use_coupled_values = 'values' not in kwargs

        # Set attributes
        # property managed
        self._target_pdim = None  # the object that is coupled to
        self._target_name = None  # the name of it in a ParamSpace

        # others
        self._init_finished = False
        self.use_coupled_default = use_coupled_default
        self.use_coupled_values = use_coupled_values

        # Warn if there is ambiguity regarding which values will be used
        if self.use_coupled_default is True:
            if 'default' in kwargs:
                warnings.warn("argument `default` was given despite "
                              "`use_coupled_default` being set to True. "
                              "Will ignore the given defaults!")
            kwargs['default'] = None

        if self.use_coupled_values is True:
            if 'values' in kwargs:
                warnings.warn("argument `values` was given despite "
                              "`use_coupled_values` being set to True. "
                              "Will ignore the given defaults!")
            kwargs['values'] = [None]
        # NOTE the values passed here will never actually be used and are just
        # placeholders

        # Initialise via parent 
        super().__init__(**kwargs)

        # Carry over further arguments
        if target_pdim:
            self.target_pdim = target_pdim

            if target_name is not None:
                warnings.warn("Got both `target_pdim` and `target_name` "
                              "arguments; will ignore the latter.",
                              UserWarning)
        elif target_name:
            # Save name of the object to couple to. Resolved by ParamSpace
            self.target_name = target_name

        else:
            raise ValueError("Need either argument `target_pdim` or "
                             "`target_name` to ensure coupling, got none of "
                             "those.")

        log.debug("CoupledParamDim initialised.")
        self._init_finished = True

    # Public API ..............................................................
    # These are needed by the ParamSpace class to have more control over the
    # iteration. Here, the parent class' behaviour is overwritten as the
    # CoupledParamDim's state and iteration should depend completely on that of
    # the target ParamDim...

    def iterate_state(self) -> None:
        """Does nothing, as state has no effect for CoupledParamDim"""
        pass

    def enter_iteration(self) -> None:
        """Does nothing, as state has no effect for CoupledParamDim"""
        pass

    def reset(self) -> None:
        """Does nothing, as state has no effect for CoupledParamDim"""
        pass

    # Properties that only the CoupledParamDim has ----------------------------

    @property
    def target_name(self) -> Union[str, Sequence[str]]:
        """The ParamDim object this CoupledParamDim couples to."""
        return self._target_name

    @target_name.setter
    def target_name(self, target_name: Union[str, Sequence[str]]):
        """Sets the target name, ensuring it to be a valid key sequence."""
        if self._target_name is not None:
            raise RuntimeError("Target name cannot be changed!")

        # Make sure it is of valid type
        if not isinstance(target_name, (tuple, list, str)):
            raise TypeError("Argument `target_name` should be a tuple or list "
                            "(i.e.: a key sequence) or a string! "
                            "Was of type: "+str(type(target_name)))

        elif isinstance(target_name, list):
            target_name = tuple(target_name)

        # Check if a pdim is already set
        if self._target_pdim is not None:
            warnings.warn("A target ParamDim was already set; setting the "
                          "target_name will have no effect.", UserWarning)

        self._target_name = target_name

    @property
    def target_pdim(self) -> ParamDim:
        """The ParamDim object this CoupledParamDim couples to."""
        if self._target_pdim is None:
            raise ValueError("The coupling target has not been set! Either "
                             "set the `target_pdim` to a ParamDim object or "
                             "incorporate this CoupledParamDim into a "
                             "ParamSpace to resolve its coupling target using "
                             "the given `target_name` attribute.")

        return self._target_pdim

    @target_pdim.setter
    def target_pdim(self, pdim: ParamDim):
        if self._target_pdim is not None:
            raise RuntimeError("Cannot change target of CoupledParamDim!")

        elif not isinstance(pdim, ParamDim):
            raise TypeError("Target of CoupledParamDim needs to be of type "
                            "ParamDim, was "+str(type(pdim)))

        elif not self.use_coupled_values and len(self) != len(pdim):
            raise ValueError("The lengths of the value sequences of target "
                             "ParamDim and this CoupledParamDim need to "
                             "match, were: {} and {}, respectively."
                             "".format(len(pdim), len(self)))

        self._target_pdim = pdim
        log.debug("Set CoupledParamDim target.")

    # Properties that need to relay to the coupled ParamDim -------------------
    
    @property
    def default(self):
        """The default value.
        
        Returns:
            the default value this parameter dimension can take.
        
        Raises:
            RuntimeError: If no ParamDim was associated yet
        """
        if self.use_coupled_default:
            return self.target_pdim.default

        return self._default

    @property
    def values(self) -> tuple:
        """The values that are iterated over.
        
        If self.use_coupled_values is set, will be those of the coupled pdim.
        
        Returns:
            tuple: The values of this CoupledParamDim or the target ParamDim
        """
        # Before initialisation finished, this cannot access target_pdim yet
        if not self._init_finished:
            return self._vals

        # The regular case, after initialisation finished
        if self.use_coupled_values:
            return self.target_pdim.values

        return self._vals

    @property
    def state(self) -> Union[int, None]:
        """The current iterator state of the target ParamDim
        
        Returns:
            Union[int, None]: The state of the iterator; if it is None, the
                ParamDim is not inside an iteration.
        """
        return self.target_pdim.state

    @state.setter
    def state(self, new_state):
        """The state of a coupled ParamDim is of no importance; don't change
        it.
        """
        pass

    @property
    def current_value(self):
        """If in an iteration: return the value according to the current
        state. If not in an iteration, return the default value.
        """
        if self.state is None:
            return self.default
        return self.values[self.state]
