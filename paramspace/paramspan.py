"""The ParamSpam classes ...""" # TODO

import copy
import logging
from typing import Iterable

import numpy as np

# Get logger
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------

class ParamSpanBase:
    """The ParamSpan base class."""

    def __init__(self, *, span: Iterable=None, enabled: bool=True, **kwargs) -> None:
        """Initialise the ParamSpan.
        
        Args:
            span (Iterable, optional): Which values the span will iterate over. This value takes precedence over any constructors given in the kwargs.
            enabled (bool, optional): Whether this span is enabled and should be used in a sweep. Default: True
            **kwargs: Description
        """
        
        # Initialize attributes that are managed by properties
        self._span = None
        self._name = None
        self._order = np.inf

        # Carry over arguments
        self.enabled = enabled



        # Parse other keyword arguments


    # Properties ..............................................................
    @property
    def span(self):
        """The span that is iterated over.
        
        Returns:
            tuple: the span iterable of this object or None if not yet set
        """
        return self._span

    @span.setter
    def span(self, val: Iterable):
        """Set the span to the value. Can only be done once.
        
        Args:
            val (Iterable): The new span value
        
        Raises:
            ValueError: Raised when a span value was already set
        """
        if self._span is None:
            self._span = val
        else:
            raise ValueError("Span is already set and cannot be set again.")


    # Public API ..............................................................


    # Non-public API ..........................................................


    


class ParamSpan(ParamSpanBase):
    """The ParamSpan class."""

    pass


class CoupledParamSpan(ParamSpanBase):
    """A CoupledParamSpan object is recognized by the ParamSpace and its state moves alongside with another ParamSpan's state."""

    pass

