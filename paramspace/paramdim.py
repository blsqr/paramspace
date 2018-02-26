"""The ParamSpam classes ...""" # TODO

import copy
import logging
import warnings
from typing import Iterable, Union

import numpy as np

# Get logger
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------

class ParamDimBase:
    """The ParamDim base class."""

    def __init__(self, *, default, values: Iterable=None, enabled: bool=True, order: float=np.inf, name: str=None, **kwargs) -> None:
        """Initialise a parameter dimension object.
        
        Args:
            default: default value of this parameter dimension
            values (Iterable, optional): Which discrete values this parameter
                dimension can take. This argument takes precedence over any
                constructors given in the kwargs (like range, linspace, …).
            enabled (bool, optional): Whether this parameter dimension is
                enabled and should be used in a sweep. Default: True
            order (float, optional): # TODO
            name (str, optional): # TODO
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
        self.enabled = enabled
        
        # Initialize attributes that are managed by properties or methods
        self._vals = None
        self._state = None

        # Carry over arguments
        self.default = default

        # Set the values, first via the `values` argument, and check whether there are enough arguments to set the values
        if values is not None:
            self.values = values

        elif not any([k in kwargs for k in ('range', 'linspace', 'logspace')]):
            raise ValueError("No argument `values` or other `**kwargs` was specified, but at least one of these is needed for initialisation.")

        # Check again, now including the `kwargs`
        if kwargs and self.values is None:
            # Need to parse the additional keyword arguments to generate the values attribute from it
            if len(kwargs) > 1:
                warnings.warn("{}.__init__ was called with multiple additional `**kwargs`; only one of these will be used! The order in which the arguments are used is: `range`, `linspace`, `logspace`.".format(self.__class__.__name__),
                              UserWarning)

            if 'range' in kwargs:
                self.values = range(*kwargs.get('range'))
            elif 'linspace' in kwargs:
                self.values = np.linspace(*kwargs.get('linspace'))
            elif 'logspace' in kwargs:
                self.values = np.logspace(*kwargs.get('logspace'))
            # else: not possible

        elif kwargs and self.values is not None:
            warnings.warn("{}.__init__ was called with both the argument `values` and additional `**kwargs`: {}. With `values` present, the additional keyword arguments are ignored.".format(self.__class__.__name__, kwargs),
                          UserWarning)
            
    # Properties ..............................................................
    @property
    def values(self):
        """The values that are iterated over.
        
        Returns:
            tuple: the values this parameter dimension can take. If None, the values are not yet set.
        """
        return self._vals

    @values.setter
    def values(self, values: Iterable):
        """Set the possible parameter values. Can only be done once and converts the given Iterable to an immutable.
        
        Args:
            values (Iterable): Which values to set. Will be converted to tuple.
        
        Raises:
            RuntimeError: Raised when a span value was already set before
            ValueError: Raised when the given iterable was not at least of length 1
        """
        if self._vals is None:
            if not len(values):
                raise ValueError("Argument `values` needs to be an iterable of at least length 1, was " + str(values))

            self._vals = tuple(values)

        else:
            raise AttributeError("Span is already set and cannot be set again.")

    @property
    def state(self) -> Union[int, None]:
        """The current iterator state
        
        Returns:
            Union[int, None]: The state of the iterator; if it is None, the ParamDim is not inside an iteration.
        """
        return self._state

    @property
    def current_value(self):
        """If in an iteration: return the value according to the current state. If not in an iteration and/or disabled, return the default value."""
        if self.enabled and self.state is not None:
            return self.values[self.state]
        else:
            return self.default

    # Magic methods ...........................................................

    def __len__(self) -> int:
        """
        Returns:
            int: The length of the associated values list; if not enabled, returns 1.
        """
        if self.enabled:
            return len(self.values)
        else:
            return 1

    def __repr__(self) -> str:
        """
        Returns:
            str: Returns the string representation of the ParamDimBase-derived object
        """
        return "{}({})".format(self.__class__.__name__,
                               repr(dict(default=self.default,
                                         order=self.order,
                                         values=self.values,
                                         enabled=self.enabled,
                                         name=self.name)))

    def __str__(self) -> str:
        """
        Returns:
            str: Returns the string representation of the ParamDimBase-derived object
        """
        return repr(self)

    # Iterator functionality ..................................................

    def __next__(self):
        """Move to the next valid state and return the corresponding parameter value.
        
        Returns:
            The current value of the iteration
        
        Raises:
            StopIteration: When the iteration has finished
        """

        if not self.enabled:
            log.debug("__next__ called on disabled ParamDim")
            raise StopIteration

        log.debug("__next__ called")

        # Iterate the state and return the 
        self.iterate_state()
        return self.current_value


    # Public API ..............................................................
    # These are needed by the ParamSpace class to more controllably iterate

    def iterate_state(self) -> None:
        """Iterates the state of the parameter dimension.
        
        Raises:
            StopIteration: Upon end of iteration
        """
        # Set to zero or increment, depending on whether inside or outside of an iteration
        if self.state is None:
            self.enter_iteration()
        else:
            self._state += 1

        # Check if end of iteration is reached
        if self.state == len(self):
            # Yes. Reset the state, allowing to reuse the object (unlike with other Python iterators)
            self.reset()
            raise StopIteration


    def enter_iteration(self) -> None:
        """Sets the state to 0, symbolising that an iteration has started."""
        self._state = 0

    def reset(self) -> None:
        """Resets the state to None, called after the end of an iteration."""
        self._state = None

    # Non-public API ..........................................................


    


class ParamDim(ParamDimBase):
    """The ParamDim class."""

    pass


class CoupledParamDim(ParamDimBase):
    """A CoupledParamDim object is recognized by the ParamSpace and its state moves alongside with another ParamDim's state."""

    pass

