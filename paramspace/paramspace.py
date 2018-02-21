"""The ParamSpace class is an extension of a dict, which can be used to iterate over a paramter space."""

import copy
import logging
import pprint
from collections import OrderedDict, Mapping

import numpy as np

# TODO clean these up

# Get logger
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------

class ParamSpace:

    def __init__(self, d, return_class=dict):
        """Initialise the ParamSpace object from a dictionary.

        Upon init, the dictionary is traversed; when meeting a ParamSpan object, it will be collected and then added to the spans.
        """

        log.debug("Initialising ParamSpace ...")

        self._init(d, return_class=return_class)

        # Done.
        log.info("Initialised ParamSpace object. (%d dimensions, volume %d)", len(self._spans), self._max_state)

    def _init(self, d, return_class=dict):
        """Initialisation helper, which is called from __init__ and from update. It Initialises the base dictionaries (_init_dict and _dict) as well as the spans and some variables regarding state number.
        """

        # Keep the initial dictionary. This will never be messed with (only exception being an update, where this _init method is called again).
        self._init_dict = copy.deepcopy(d)  # includes the ParamSpan objects

        # The current dictionary (in default state as copy from initial dict)
        # This dictionary is what is returned on request and what is worked on.
        self._dict      = copy.deepcopy(self._init_dict)

        # Initialise the self._spans attribute
        self._spans     = None              # ...is defined in _init_spans
        self._init_spans(self._dict)
        # self._spans is an OrderedDict, which includes as keys the name of the span keys (or a tuple with the traversal path to an entry), and as values the ParamSpan objects
        # The current state of the parameter space is saved in the ParamSpan objects and can be incremented there as well.

        # Additionally, the state_id counts the number of the point the current dictionary is in. It is incremented upon self._next_state() until the number self._max_state is reached.
        self._state_no  = None              # None means: default vals
        self._max_state = self.volume

        # The requested ParamSpace points can be cast to a certain class:
        self._return_class = return_class

        # The inverse mapping can be cached
        self._imap      = None

        return

    def _init_spans(self, d):
        """Looks for instances of ParamSpan in the dictionary d, extracts spans from there, and carries them over
        - Their default value stays in the init_dict
        - Their spans get saved in the spans dictionary
        """
        log.debug("Initialising spans ...")

        # Traverse the dict and look for ParamSpan objects; collect them as (order, key, value) tuples
        pspans  = _recursive_collect(d, isinstance, ParamSpan,
                                     prepend_info=('info_func', 'keys'),
                                     info_func=lambda ps: ps.order)

        # Sort them. This looks at the info first, which is the order entry, and then at the keys. If a ParamSpan does not provide an order, it has entry np.inf there, such that those without order get sorted by the key.
        pspans.sort()
        # NOTE very important for consistency

        # Now need to reduce the list items to 2-tuples, ditching the order, to allow to initialise the OrderedDict
        pspans  = [tpl[1:] for tpl in pspans]

        # Cast to an OrderedDict (pspans is a list of tuples -> same data structure as OrderedDict)
        self._spans     = OrderedDict(pspans)

        # Also collect the coupled ParamSpans and continue with the same procedure
        coupled = _recursive_collect(d, isinstance, CoupledParamSpan,
                                     prepend_info=('info_func', 'keys'),
                                     info_func=lambda ps: ps.order)
        coupled.sort() # same sorting rules as above, but not as crucial here because they do not change the iteration order through state space
        self._cpspans   = OrderedDict([tpl[1:] for tpl in coupled])

        # Now resolve the coupling targets and add them to CoupledParamSpan instances ... also let the target ParamSpan objects know which CoupledParamSpan couples to them
        for cpspan in self._cpspans.values():
            c_target_name       = cpspan.coupled_to

            # Try to get it by name
            c_target            = self.get_span_by_name(c_target_name)

            # Set attribute of the coupled ParamSpan
            cpspan.coupled_pspan= c_target

            # And inform the target ParamSpan about it being the target of the coupled param span, if it is not already included there
            if cpspan not in c_target.target_of:
                c_target.target_of.append(cpspan)

        log.debug("Initialised %d spans and %d coupled spans.", len(self._spans), len(self._cpspans))

    # Formatting ..............................................................

    def __str__(self):
        log.debug("__str__ called. Returning current state dict.")
        return pprint.pformat(self._dict)

    def __repr__(self):
        """To reconstruct the ParamSpace object ..."""
        return "ParamSpace("+str(self)+")"

    def __format__(self, spec: str):
        """Returns a formatted string

        The spec argument is the part right of the colon in the '{foo:bar}' of a format string.
        """

        ALLOWED_JOIN_STRS   =  ["_", "__"]

        # Special behaviour
        if len(spec) == 0:
            return ""

        elif spec == 'span_names':
            # Compile output for span names
            return "  (showing max. last 4 keys)\n  " + "\n  ".join([("" if len(s)<=4 else "."*(len(s)-4)+" -> ") + " -> ".join(s[-min(len(s),4):]) for s in self.get_span_names()])
            # ...a bit messy, but well ...


        # Creating span strings
        parts       = []
        spst_fstr   = "" # span state format string
        join_char   = ""

        # First: build the format string that will be used to handle each param space
        for part in spec.split(","):
            part    = part.split("=")

            # Catch changes of the join character
            if len(part) == 1 and part[0] in ALLOWED_JOIN_STRS:
                join_char   = part[0]
                continue

            # Catch span state format
            if len(part) == 2 and part[0] == "states":
                spst_fstr   = part[1].replace("[", "{").replace("]", "}")
                continue

            # Pass all other parsing to the helper
            try:
                parsed  = self._parse_format_spec(part)
            except ValueError:
                print("Invalid format string '{}'.".format(spec))
                raise
            else:
                if parsed:
                    parts.append(parsed)

        if spst_fstr:
            # Evaluate the current values of the ParamSpace
            names   = [key[-1] for key in self.get_span_names()]
            states  = [span.state for span in self.get_spans()]
            vals    = [span.get_val_by_state() for span in self.get_spans()]
            digits  = [len(str(len(span))) for span in self.get_spans()]

            spst_parts = [spst_fstr.format(name=n, state=s, digits=d, val=v)
                          for n, s, d, v in zip(names, states, digits, vals)]

            parts.append("_".join(spst_parts))

        return join_char.join(parts)

    def _parse_format_spec(self, part: list): # TODO

        return None # currently not implementedd

        # if len(part) == 2:
        #   # is a key, value pair
        #   key, val    = part

        #   if key in ["bla"]:
        #       pass # TODO
        #   else:
        #       raise ValueError("Invalid key value pair '{}: {}'.".format(key, val))

        # elif len(part) == 1:
        #   key     = part[0]

        #   if key in ["bla"]:
        #       pass # TODO
        #   else:
        #       raise ValueError("Invalid key '{}'.".format(key))

        # else:
        #   raise ValueError("Part '{}' had more than one '=' as separator.".format("=".join(part)))

        # return None

    def get_info_str(self) -> str:
        """Returns an information string about the ParamSpace"""
        l = ["ParamSpace Information"]

        # General information about the Parameter Space
        l.append("  Dimensions:  {}".format(self.num_dimensions))
        l.append("  Volume:      {}".format(self.volume))

        # Span information
        l += ["", "Parameter Spans"]
        l += ["  (First spans are iterated over first.)", ""]

        for name, span in self.spans.items():
            l.append("  * {}".format(" -> ".join([str(e) for e in name])))
            l.append("      {}".format(span.value_list))
            l.append("")

        # Coupled Span information
        if len(self._cpspans):
            l += ["", "Coupled Parameter Spans"]
            l += ["  (Move alongside the state of Parameter Spans)", ""]

            for name, cspan in self._cpspans.items():
                l.append("  * {}".format(" -> ".join([str(e) for e in name])))
                l.append("      Coupled to:  {}".format(cspan.coupled_to))
                l.append("      Span:        {}".format(cspan.value_list))
                l.append("")

        return "\n".join(l)

    # Retrieving states of the ParamSpace .....................................

    @property
    def num_dimensions(self) -> int:
        """Returns the number of dimensions, i.e. the number of spans."""
        return len(self._spans)

    @property
    def shape(self) -> tuple:
        """The shape of the parameter space"""
        return tuple([len(s) for s in self.get_spans()])

    @property
    def volume(self) -> int:
        """Returns the volume of the parameter space."""
        vol     = 1
        for pspan in self.get_spans():
            vol *= len(pspan)
        return vol

    @property
    def spans(self):
        """Return the OrderedDict that holds the spans."""
        return self._spans

    @property
    def coupled_spans(self):
        """Return the OrderedDict that holds the coupled spans."""
        return self._cpspans

    @property
    def span_names(self) -> list:
        """Get a list of the span names (tuples of strings). If the span was itself named, that name is used rather than the one created from the dictionary key.

        NOTE: CoupledParamSpans are not included here, same as in the other methods."""
        names   = []

        for name, span in self.spans.items():
            if span.name:
                names.append((span.name,))
            else:
                names.append(name)

        return names

    # TODO migrate the following to properties

    def get_default(self):
        """Returns the default state of the ParamSpace"""
        _dd = _recursive_replace(copy.deepcopy(self._init_dict),
                                 lambda pspan: pspan.default,
                                 isinstance, ParamSpanBase)
        return self._return_class(_dd)

    def get_point(self):
        """Return the current point in Parameter Space (i.e. corresponding to the current state)."""
        _pd = _recursive_replace(copy.deepcopy(self._dict),
                                 lambda pspan: pspan.get_val_by_state(),
                                 isinstance, ParamSpanBase)
        return self._return_class(_pd)

    def get_state_no(self) -> int:
        """Returns the state number"""
        return self._state_no

    def get_span(self, dim_no: int) -> ParamSpan:
        try:
            return list(self.get_spans())[dim_no]
        except IndexError:
            log.error("No span corresponding to argument dim_no {}".format(dim_no))
            raise

    def get_spans(self):
        """Return the spans"""
        return self._spans.values()

    def get_coupled_spans(self):
        """Return the coupled spans"""
        return self.coupled_spans.values()

    def get_span_keys(self):
        """Get the iterator over the span keys (tuples of strings)."""
        return self._spans.keys()

    def get_span_names(self) -> list:
        """Get a list of the span names (tuples of strings). If the span was itself named, that name is used rather than the one created from the dictionary key."""
        return self.span_names

    def get_span_states(self):
        """Returns a tuple of the current span states"""
        return tuple([span.state for span in self.get_spans()])

    def get_span_dim_no(self, name: str) -> int:
        """Returns the dimension number of a span, i.e. the index of the ParamSpan object in the list of spans of this ParamSpace. As the spans are held in an ordered data structure, the dimension number can be used to identify the span. This number also corresponds to the index in the inverse mapping of the ParamSpace.

        Args:
            name (tuple, str) : the name of the span, which can be a tuple of strings or a string. If name is a tuple of strings, the exact tuple is required to find the span by its span_name. If name is a string, only the last element of the span_name is considered.

        Returns:
            int     : the number of the dimension
            None    : a span by this name was not found

        Raises:
            ValueError: If argument name was only a string, there can be duplicates. In the case of duplicate entries, a ValueError is raised.
        """
        dim_no  = None

        if isinstance(name, str):
            for n, span_name in enumerate(self.get_span_names()):
                if span_name[-1] == name:
                    if dim_no is not None:
                        # Was already set -> there was a duplicate
                        raise ValueError("Duplicate span name {} encountered during access via the last key of the span name. To not get an ambiguous result, pass the full span name as a tuple.".format(name))
                    dim_no  = n

        else:
            for n, span_name in enumerate(self.get_span_names()):
                if span_name[-len(name):] == name:
                    # The last part of the sequence matches the given name
                    if dim_no is not None:
                        # Was already set -> there was a duplicate
                        raise ValueError("Access via '{}' was ambiguous. Give the full sequence of strings as a span name to be sure to access the right element.".format(name))
                    dim_no  = n

        return dim_no

    def get_span_by_name(self, name: str) -> ParamSpan:
        """Returns the ParamSpan corresponding to this name.

        Args:
            name (tuple, str) : the name of the span, which can be a tuple of strings or a string. If name is a tuple of strings, the exact tuple is required to find the span by its span_name. If name is a string, only the last element of the span_name is considered.

        Returns:
            int     : the number of the dimension
            None    : a span by this name was not found

        Raises:
            ValueError: If argument name was only a string, there can be duplicates. In the case of duplicate entries, a ValueError is raised.
        """

        return self.get_span(self.get_span_dim_no(name))

    def get_inverse_mapping(self) -> np.ndarray:
        """Creates a mapping of the state tuple to a state number and the corresponding span parameters.

        Returns:
            np.ndarray with the shape of the spans and the state number as value
        """

        if hasattr(self, '_imap') and self._imap is not None:
            # Return the cached result
            # NOTE hasattr is needed for legacy reasons: old objects that are loaded from pickles and do not have the attribute ...
            log.debug("Using previously created inverse mapping ...")
            return self._imap
        # else: calculate the inverse mapping

        # Create empty n-dimensional array
        shape   = tuple([len(_span) for _span in self.get_spans()])
        imap    = np.ndarray(shape, dtype=int)
        imap.fill(-1)   # -> Not set yet

        # Iterate over all points and save the state number to the map
        for state_no, _ in self.get_points():
            # Get the span states and convert all Nones to zeros, as these dimensions have no entry
            s = [Ellipsis if i is None else i for i in self.get_span_states()]

            # Save the state number to the mapping
            try:
                imap[tuple(s)]  = state_no
            except IndexError:
                log.error("Getting inverse mapping failed.")
                print("s: ", s)
                print("imap shape: ", imap.shape)
                raise

        # Save the result to attributes
        self._imap  = imap

        return imap

    # Iterating over the ParamSpace ...........................................

    def get_points(self, fstr: str=None, with_span_states: bool=False) -> tuple:
        """Returns a generator of all states in state space, returning (state_no, point in state space).

        If `with_span_states` is True, the span states tuple is returned instead of the state number"""
        if fstr is not None and not isinstance(fstr, str):
            raise TypeError("Argument fstr needs to be a string or None, was "+str(type(fstr)))
        elif fstr is None:
            # No additional return value
            _add_ret    = ()
        # else: will use format string, even if it is empty

        if with_span_states:
            first_tuple_element     = self.get_span_states
        else:
            first_tuple_element     = self.get_state_no

        if self.num_dimensions == 0:
            log.warning("No dimensions in ParamSpace. Returning defaults.")

            if fstr:
                _add_ret    = ('',)

            yield (None, self.get_default()) + _add_ret
            return # not executed further

        # else: there is a volume to iterate over:

        # Prepare pspans: set them to state 0, else they start with the default
        for pspan in self.get_spans():
            pspan.set_state_to_zero()

        # This is the initial state with state number 0
        self._state_no  = 0

        # Determine state string
        _add_ret    = () if not fstr else (fstr.format(self),)

        # Yield the initial state
        yield (first_tuple_element(), self.get_point()) + _add_ret

        # Now yield all the other states
        while self.next_state():
            _add_ret    = () if not fstr else (fstr.format(self),)
            yield (first_tuple_element(), self.get_point()) + _add_ret

        else:
            log.info("Visited every point in ParamSpace.")
            log.info("Resetting to initial state ...")
            self.reset()
            return

    def next_state(self) -> bool:
        """Increments the state variable"""
        log.debug("ParamSpace.next_state called ...")

        for pspan in self.get_spans():
            if pspan.next_state():
                # Iterated to next step without reaching the last span item
                break
            else:
                # Went through all states of this span -> carry one over (as with addition) and increment the next spans in the ParamSpace.
                continue
        else:
            # Loop went through -> all states visited
            self.reset()            # reset back to default
            return False            # i.e. reached the end

        # Broke out of loop -> Got the next state and not at the end yet

        # Increment state number
        if self._state_no is None:
            self._state_no = 0
        else:
            self._state_no += 1

        return True             # i.e. not reached the end

    # Getting a subspace ......................................................

    def get_subspace(self, *slices, squeeze: bool=True, as_dict_if_0d: bool=False):
        """Returns a copy of this ParamSpace with the slices applied to the corresponding ParamSpans.

        If `squeeze`, the size one spans are removed.

         (Not the nicest implementation overall...)"""

        def apply_slice(pspace, *, slc, name: str):
            """Destructively (!) applies a slice to the span with the given name."""
            pspan   = pspace.get_span_by_name(name)
            pspan.apply_slice(slc)


        # Work on a copy of this ParamSpace
        subspace    = copy.deepcopy(self)

        # Check if the length of the provided slices matches the number of dimensions that could possibly be sliced
        if len(slices) <= subspace.num_dimensions:
            # See if there are Ellipses in the slices that indicate where to expand the list
            num_ellipses = sum([s is Ellipsis for s in slices])
            if num_ellipses == 0:
                # No.
                if len(slices) < subspace.num_dimensions:
                    # Add one in the end so that it is clear where to expand.
                    slices.append(Ellipsis)
                # else: there was one slice defined for each dimension; can and need not add an Ellipsis

            elif num_ellipses > 1:
                raise ValueError("More than one Ellipsis object given!")

            # Now expand them so that the slices list has the same length as the source parameter space has dimensions
            _slices     = []
            for slc in slices:
                if slc is Ellipsis:
                    # Put a number of slice(None) in place of the Ellipsis
                    fill_num    = subspace.num_dimensions - len(slices) + 1
                    _slices     += [slice(None) for _ in range(fill_num)]
                else:
                    _slices.append(slc)

            # Use the new slices list
            slices      = _slices

        else:
            raise ValueError("More slices than dimensions that could potentially be sliced given.")

        # Get the list of names
        names       = subspace.get_span_names()

        # For each name, apply the slice
        for name, slc in zip(names, slices):
            apply_slice(subspace, slc=slc, name=name)
            # NOTE this works directly on the ParamSpan objects

        # Have the option to squeeze away the size-1 ParamSpans
        if squeeze:
            subspace    = _recursive_replace(subspace._dict,
                                             lambda pspan: pspan.squeeze(),
                                             isinstance, ParamSpanBase)
        else:
            # Just extract the subspace dictionary
            subspace    = subspace._dict
        # The previous subspace ParamSpace object will go out of scope here. The changes however were applied to the ParamSpan objects that are stored in the dictionaries.

        # Now, a new ParamSpace object should be initialised, because the old one was messed with too much.
        subspace    = ParamSpace(subspace)

        # Only now is it clear how many dimensions the target space will have. If it is 0-dimensional (i.e. no ParamSpans inside) flatten it (if argument is set to do so)
        if as_dict_if_0d and subspace.num_dimensions == 0:
            # Overwrite with the default, which is the same as the current dict. There is no difference, because there are no ParamSpans defined anyway ...
            subspace    = subspace.get_default()

        return subspace

    # Misc ....................................................................

    def reset(self):
        """Resets all state variables, the state id, and the current dictionary back to the initial dictionary (i.e. with the default values)."""

        for pspan in self.get_spans():
            # Reset the pspan state
            pspan.state     = None

        self._state_no  = None

        log.debug("ParamSpace resetted.")

    def add_span(self): # TODO
        """Add a span to the ParamSpace manually, e.g. after initialisation with a regular dict."""
        raise NotImplementedError("Manually adding a span is not implemented yet. Please initialise the ParamSpace object with the ParamSpan objects already in place.")

    def update(self, u, recessively: bool=True):
        """Update the dictionaries of the ParamSpace with the values from u.

        If recessively is True, the update dictionary u will be updated with the values from self._dict.
        For False, the regular dictionary update will be performed, where self._dict is updated with u.
        """

        if self.get_state_no() is not None:
            log.warning("ParamSpace object can only be updated in the default state, but was in state %s. Call .reset() on the ParamSpace to return to the default state.", self.get_state_no())
            return

        if recessively:
            # Recessive behaviour: Old values have priority
            new_d   = _recursive_update(u, self._dict)
        else:
            # Normal update: New values overwrite old ones
            log.info("Performing non-recessive update. Note that the new values have priority over any previous values, possibly overwriting ParamSpan objects with the same keys.")
            new_d   = _recursive_update(self._dict, u)

        # In order to make the changes apply to _dict and _init_dict, the _init method is called again. This makes sure, the ParamSpace object is in a consistent state after the update.
        self._init(new_d, return_class=self._return_class)

        log.info("Updated ParamSpace object.")

        return
