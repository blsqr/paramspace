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
        """Initialise the ParamDim.
        
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
        self.name      = name
        self.order     = order
        self.enabled   = enabled
        
        # Initialize attributes that are managed by properties or methods
        self._vals     = None
        self._state    = None

        # Carry over arguments
        self.default   = default
        self.values    = values

        # If applicable, parse other keyword arguments
        if (self.values is None
            and any([k in kwargs for k in ('range', 'linspace', 'logspace')])):
            # Need to parse the additional keyword arguments to generate the values attribute from it
            if len(kwargs) > 1:
                warnings.warn("{}.__init__ was called with multiple additional `**kwargs`; only one of these will be used! The order is: `range`, `linspace`, `logspace`.".format(self.__class__.__name__))

            if 'range' in kwargs:
                self.values = range(*kwargs.get('range'))
            elif 'linspace' in kwargs:
                self.values = np.linspace(*kwargs.get('linspace'))
            elif 'logspace' in kwargs:
                self.values = np.logspace(*kwargs.get('logspace'))
            # else: not possible

        elif kwargs and self.values is not None:
            warnings.warn("{}.__init__ was called with both the argument `values` and additional `**kwargs`: {}. With `values` present, the additional keyword arguments are ignored.".format(self.__class__.__name__, kwargs), warnings.UserWarning)

        else:
            raise ValueError("No argument `values` or other `**kwargs` specified, but at least one of these is needed for initialisation.")


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
        """Set the possible parameter values. Can only be done once.
        
        Args:
            values (Iterable): Description
        
        Raises:
            RuntimeError: Raised when a span value was already set before
            ValueError: Raised when the given iterable was not at least of length 1
        """
        if self._vals is None:
            if not len(values):
                raise ValueError("Argument `values` needs to be an iterable of at least length 1, was " + str(values))

            self._vals = values

        else:
            raise RuntimeError("Span is already set and cannot be set again.")

    @property
    def state(self):
        """The current iterator state"""
        return self._state

    @property
    def current_value(self):
        """If in an iteration: return the value according to the current state. If not in an iteration and/or disabled, return the default value."""
        if self.enabled and self.state is not None:
            return self.values[self.state]
        else:
            return self.default

    # Magic methods ...........................................................
    # TODO str, repr, len

    def __len__(self):
        """ """
        if not self.enabled:
            return 1
        else:
            return len(self.values)

    # Iterator functionality ..................................................

    def __next__(self):
        """Move to the next valid state and return the corresponding parameter value."""

        if not self.enabled:
            log.debug("__next__ called on disabled ParamDim")
            raise StopIteration

        log.debug("__next__ called")

        if self.state is None:
            # Start of iteration: Set state and return first value
            self._state = 0
            return self.current_value

        else:
            # Continuation of iteration: increment state and 
            self._state += 1

            if self.state == len(self):
                # Reached the end of the iteration: reset state
                # NOTE: unlike with other Python iterators, the object is still usable after this.
                self._state = None
                raise StopIteration
            else:
                return self.current_value


    # Public API ..............................................................




    # Non-public API ..........................................................


    


class ParamDim(ParamDimBase):
    """The ParamDim class."""

    pass


class CoupledParamDim(ParamDimBase):
    """A CoupledParamDim object is recognized by the ParamSpace and its state moves alongside with another ParamDim's state."""

    pass

