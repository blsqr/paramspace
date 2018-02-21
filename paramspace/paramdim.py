"""The ParamSpam classes ...""" # TODO

import copy
import logging
from typing import Iterable

import numpy as np

# Get logger
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------

class ParamDimBase:
    """The ParamDim base class."""

    def __init__(self, *, values: Iterable=None, enabled: bool=True, **kwargs) -> None:
        """Initialise the ParamDim.
        
        Args:
            values (Iterable, optional): Which discrete values this parameter dimension can take. This argument takes precedence over any constructors given in the kwargs (like range, linspace, …).
            enabled (bool, optional): Whether this values is enabled and should be used in a sweep. Default: True
            **kwargs: Description
        """
        
        # Initialize attributes that are managed by properties
        self._vals = None
        self._name = None
        self._order = np.inf

        # Carry over arguments
        self.enabled = enabled



        # Parse other keyword arguments


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
            ValueError: Raised when a span value was already set
        """
        if self._vals is None:
            self._vals = values
        else:
            raise ValueError("Span is already set and cannot be set again.")


    # Public API ..............................................................


    # Non-public API ..........................................................


    


class ParamDim(ParamDimBase):
    """The ParamDim class."""

    pass


class CoupledParamDim(ParamDimBase):
    """A CoupledParamDim object is recognized by the ParamSpace and its state moves alongside with another ParamDim's state."""

    pass

