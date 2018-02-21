"""The ParamSpam classes ...""" # TODO

import copy
import logging

import numpy as np

# Get logger
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------

class ParamSpanBase:
    """The ParamSpan base class. This is used so that the actual ParamSpan class can be a child class of this one and thus be distinguished from CoupledParamSpan"""

    def __init__(self, arg):
        """Initialise the ParamSpan from an argument that is a list, tuple or a dict."""

        # Set attributes to default values
        self.enabled    = True
        self.state      = None  # State of the span (idx of the current value or None, if default state)
        self.span       = None  # Filled with values below
        self.name       = None
        self.order      = None

        # Parse the argument and fill the span
        if isinstance(arg, (list, tuple)):
            # Initialise from sequence: first value is default, all are span
            self.default    = arg[0]
            self.span       = arg[:]
            # NOTE the default value is only inside the span, if the pspan is defined via sequence!

            log.debug("Initialised ParamSpan object from sequence.")

        elif isinstance(arg, dict):
            # Get default value
            self.default    = arg['default']

            # Get either of the span constructors
            if 'span' in arg:
                self.span   = list(arg['span'])

            elif 'range' in arg:
                self.span   = list(range(*arg['range']))

            elif 'linspace' in arg:
                # explicit float casting, because else numpy objects are somehow retained
                self.span   = [float(x) for x in np.linspace(*arg['linspace'])]

            elif 'logspace' in arg:
                # explicit float casting, because else numpy objects are somehow retained
                self.span   = [float(x) for x in np.logspace(*arg['logspace'])]

            else:
                raise ValueError("No valid span key (span, range, linspace, logspace) found in init argument, got {}.".format(arg.keys()))

            # Add additional values to the span
            if arg.get('add'):
                add = arg['add']
                if isinstance(add, (list, tuple)):
                    self.span += list(add)
                else:
                    self.span.append(add)

            # Optionally, cast to int or float
            # TODO Make this a key-value argument, not three key-bool pairs
            if arg.get('as_int'):
                self.span   = [int(v) for v in self.span]

            elif arg.get('as_float'):
                self.span   = [float(v) for v in self.span]

            elif arg.get('as_str'):
                self.span   = [str(v) for v in self.span]

            # If the state idx was given, also save this
            if isinstance(arg.get('state'), int):
                log.warning("Setting state of ParamSpan during initialisation. This might lead to unexpected behaviour if iterating over points in ParamSpace.")
                self._state     = arg.get('state')

            # A span can also be not enabled
            self.enabled    = arg.get('enabled', True)

            # And it can have a name
            self.name       = arg.get('name', None)

            # Also set the order
            self.order      = arg.get('order', np.inf) # default value (i.e.: last) if no order is supplied

            log.debug("Initialised ParamSpan object from mapping.")

        else:
            # TODO not strictly necessary; just let it fail...
            raise TypeError("ParamSpan init argument needs to be of type list, tuple or dict, was {}.".format(type(arg)))

        return

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__,
                               repr(dict(default=self.default,
                                         order=self.order,
                                         span=self.span,
                                         state=self.state,
                                         enabled=self.enabled,
                                         name=self.name)))

    def __len__(self):
        """Return how many span values there are, if the span is enabled."""
        if self.enabled:
            return len(self.span)
        else:
            return 1

    def __getitem__(self, idx):
        if not self.enabled:
            log.warning("ParamSpan is not enabled. Still returning item ...")

        if isinstance(idx, int):
            try:
                return self.span[idx]

            except IndexError:
                # reached end of span
                # raise error - is caught by iterators to know that its finished
                # TODO should better be done using iteration and StopIteration error
                raise

        else:
            # Possbile that this is a slice. Try slicing and return as new ParamSpan object
            pspan       = copy.deepcopy(self)
            pspan.span  = self.span[idx]
            return pspan

    def __setitem__(self, idx, val):
        raise NotImplementedError("ParamSpan values are read-only.")

    # Properties

    @property
    def value_list(self):
        return self.span

    # Public methods

    def get_val_by_state(self):
        """Returns the current ParamSpan value according to the state. This is the main method used by the ParamSpace to resolve the dictionary to its correct state."""
        if self.state is None:
            return self.default
        else:
            return self.span[self.state]

    def next_state(self) -> bool:
        """Increments the state by one, if the state is enabled.

        If None, sets the state to 0.

        If reaching the last possible state, it will restart at zero and return False, signalising that all states were looped through. In all other cases it will return True.
        """
        log.debug("ParamSpan.next_state called ...")

        if not self.enabled:
            return False

        if self.state is None:
            self.state  = 0
        else:
            self.state  = (self.state+1)%len(self)

            if self.state == 0:
                # It is 0 after increment, thus it must have hit the wall.
                return False

        return True

    def set_state_to_zero(self) -> bool:
        """Sets the state to zero (necessary before the beginning of an iteration), if the span is enabled."""
        if self.enabled:
            self.state = 0
            return True
        return False

    def apply_slice(self, slc):
        """Applies a slice to the span list."""
        if not self.enabled:
            # Not enabled -> no span -> nothing to do
            return

        new_span    = self.span[slc]
        if len(new_span) > 0:
            self.span   = new_span
        else:
            raise ValueError("Application of slice {} to {}'s span {} resulted in zero-length span, which is illegal.".format(slc, self.__class__.__name__, self.span))

    def squeeze(self):
        """If of length one, returns the remaining value in the span; if not, returns itself."""
        if self.enabled and len(self) == 1:
            # Enabled and have span -> return the first element
            return self.span[0]

        elif not self.enabled:
            # Not enabled -> work on default and return that
            return self.default

        else:
            # Nothing to squeeze
            return self

# .............................................................................

class ParamSpan(ParamSpanBase):
    """The ParamSpan class."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.target_of  = []

    def apply_slice(self, slc):
        super().apply_slice(slc)

        # Also apply to all coupled ParamSpans
        for cps in self.target_of:
            cps.apply_slice(slc)

# .............................................................................

class CoupledParamSpan(ParamSpanBase):
    """A CoupledParamSpan object is recognized by the ParamSpace and its state moves alongside with another ParamSpan's state."""

    def __init__(self, arg):
        # Check if default and/or span were not given; in those cases, the values from the coupled span are to be used upon request
        self.use_coupled_default    = bool('default' not in arg)
        self.use_coupled_span       = bool('span' not in arg)

        # Make sure they are set, so parent init does not get confused
        arg['default']  = arg.get('default')
        if arg.get('span') is None:
            arg['span']     = []

        super().__init__(arg)

        self.coupled_pspan  = None # ParamSpace sets this after initialisation
        self.coupled_to     = arg['coupled_to']

        if not isinstance(self.coupled_to, str):
            # ensure it is a tuple; important for span name lookup
            self.coupled_to     = tuple(self.coupled_to)

        log.debug("CoupledParamSpan initialised.")

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__,
                               repr(dict(default=self.default,
                                         span=self.span,
                                         state=self.state,
                                         enabled=self.enabled,
                                         name=self.name,
                                         coupled_to=self.coupled_to)))

    # Overwrite the properties that need to relay the coupling to the other ParamSpan

    @property
    def default(self):
        """If the CoupledParamSpan was initialised with a default value on its own, returns that. If not, returns the default value of the coupled ParamSpan object. If not yet coupled to that, returns None."""
        if not self.use_coupled_default:
            return self._default
        elif self.coupled_pspan:
            return self.coupled_pspan.default
        else:
            return None

    @default.setter
    def default(self, val):
        self._default = val

    @property
    def span(self):
        """If the CoupledParamSpan was initialised with a span on its own, returns that span. If not, returns the span of the coupled ParamSpan object. If not yet coupled to that, returns None."""
        if not self.use_coupled_span:
            return self._span
        elif self.coupled_pspan:
            return self.coupled_pspan.span
        else:
            return None

    @span.setter
    def span(self, val):
        self._span = val

    @property
    def state(self):
        if self.coupled_pspan:
            return self.coupled_pspan.state
        else:
            return None

    @state.setter
    def state(self, val):
        self._state = val

    @property
    def enabled(self):
        if self.coupled_pspan:
            return self.coupled_pspan.enabled
        else:
            return self._enabled

    @enabled.setter
    def enabled(self, val):
        self._enabled   = val

    # Methods

    def get_val_by_state(self):
        """Adds a try-except clause to the parent method, to give an understandable error message in case of an index error (due to a coupled span with unequal length)."""
        try:
            return super().get_val_by_state()
        except IndexError as err:
            if not self.use_coupled_span and len(self) != len(self.coupled_pspan):
                raise IndexError("The span provided to CoupledParamSpan has not the same length as that of the ParamSpan it couples to. Lengths: {} and {}.".format(len(self), len(self.coupled_pspan))) from err

            raise

