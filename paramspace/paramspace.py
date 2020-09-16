"""Implementation of the ParamSpace class"""
import collections
import copy
import logging
import warnings
from collections import OrderedDict
from functools import reduce
from itertools import chain
from typing import Any, Dict, Generator, List, Sequence, Set, Tuple, Union

import numpy as np
import numpy.ma
import xarray as xr

from .paramdim import CoupledParamDim, Masked, ParamDim, ParamDimBase
from .tools import recursive_collect, recursive_replace, recursive_update

# Get logger
log = logging.getLogger(__name__)

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
    yaml_tag = "!pspace"

    # .........................................................................

    def __init__(self, d: dict):
        """Initialize a ParamSpace object from a given mapping or sequence.

        Args:
            d (Union[MutableMapping, MutableSequence]): The mapping or sequence
                that will form the parameter space. It is crucial that this
                object is mutable.
        """

        # Warn if type is unusual
        if not isinstance(d, collections.abc.MutableMapping):
            warnings.warn(
                f"Got unusual type {type(d)} for ParamSpace initialisation! "
                f"If the given object is not mutable, this might fail at some "
                f"unexpected later point.",
                UserWarning,
            )

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
        self._dims_by_loc = None
        self._cdims = None
        self._cdims_by_loc = None
        self._gather_paramdims()  # NOTE attributes are set within this method

        # Initialize caching attributes
        self._smap = None
        self._iter = None

    def _gather_paramdims(self):
        """Gathers ParamDim objects by recursively going through the dict"""
        log.debug("Gathering ParamDim objects ...")

        # Traverse the dict and look for ParamDim objects; collect them as
        # (order, key, value) tuples
        pdims = recursive_collect(
            self._dict,
            select_func=lambda p: isinstance(p, ParamDim),
            prepend_info=("info_func", "keys"),
            info_func=lambda p: p.order,
            stop_recursion_types=(ParamDimBase,),
        )

        # Parse the dimension names

        # Sort them -- very important for consistency!
        # This looks at the info first, which is the order entry, and then at
        # the keys. If a ParamDim does not provide an order, it has entry
        # np.inf there, such that those without order get sorted by the key.
        pdims.sort()

        # For initializing OrderedDicts, need to reduce the list items to
        # 2-tuples, ditching the first element (order, needed for sorting)
        pdims = [tpl[1:] for tpl in pdims]

        # Now, first save the objects with keys that represent their location
        # inside the dictionary.
        self._dims_by_loc = OrderedDict(pdims)

        # For easier access, save them in another dict, where the keys are pure
        # strings. To that end, a unique string representation is needed.
        self._dims = OrderedDict(self._unique_dim_names(pdims))

        log.debug("Found %d ParamDim objects.", self.num_dims)

        log.debug("Gathering CoupledParamDim objects ...")
        # Also collect the coupled ParamDims; continue with the same procedure
        cpdims = recursive_collect(
            self._dict,
            select_func=lambda p: isinstance(p, CoupledParamDim),
            prepend_info=("info_func", "keys"),
            info_func=lambda p: p.order,
            stop_recursion_types=(ParamDimBase,),
        )

        # Sort and ditch the order, same as with regular ParamDims
        # Note: sorting is not as crucial here because coupled dims do not
        # change the iteration order through state space
        cpdims.sort()
        cpdims = [tpl[1:] for tpl in cpdims]

        # Now store them, equivalent to how the regular dimensions were stored
        self._cdims_by_loc = OrderedDict(cpdims)
        self._cdims = OrderedDict(self._unique_dim_names(cpdims))

        # Now resolve the coupling targets and add them to CoupledParamDim
        # instances. Also, let the target ParamDim objects know which
        # CoupledParamDim couples to them
        for cpdim_key, cpdim in self.coupled_dims.items():
            # Try to get the coupling target by name
            try:
                c_target = self._get_dim(cpdim.target_name)

            except (KeyError, ValueError) as err:
                # Could not find that name
                _dim_info = self._parse_dims(mode="both")
                raise ValueError(
                    f"Could not resolve the coupling target for "
                    f"CoupledParamDim at {cpdim_key}. Check the "
                    f"`target_name` specification of that entry "
                    f"and the full traceback of this error.\n"
                    f"Available parameter dimensions:\n{_dim_info}"
                ) from err

            # Set attribute of the coupled ParamDim
            cpdim.target_pdim = c_target

            # And inform the target ParamDim about it being the target of the
            # coupled param dim, if it is not already included there
            if cpdim not in c_target.target_of:
                c_target.target_of.append(cpdim)

            # Done with this coupling
        else:
            log.debug(
                "Found %d CoupledParamDim objects.", self.num_coupled_dims
            )

        log.debug("Finished gathering.")

    @staticmethod
    def _unique_dim_names(
        kv_pairs: Sequence[Tuple],
    ) -> List[Tuple[str, ParamDim]]:
        """Given a sequence of key-value pairs, tries to create a unique string
        representation of the entries, such that it can be used as a unique
        mapping from names to parameter dimension objects.

        Args:
            kv_pairs (Sequence[Tuple]): Pairs of (path, ParamDim), where the
                path is a Tuple of strings.

        Returns:
            List[Tuple[str, ParamDim]]: The now unique (name, ParamDim) pairs

        Raises:
            ValueError: For invalid names, i.e.: failure to find a unique
                representation.
        """

        def unique(names: List[str]) -> bool:
            """Check for uniqueness of the given list of names"""
            return len(set(names)) == len(names)

        def collisions(names: List[str]) -> Set[Tuple[int]]:
            """For each name, find the collisons with other names and return
            a set of indicies that collide with other names, such that those
            names can be adjusted.
            """

            def collide(a: str, b: str) -> bool:
                """Returns True if two names collide, with collisions defined
                as the following:

                    * The shorter one is part of the longer one, seen from the
                      back, e.g. ``foo`` vs ``spamfoo``
                    * The sorter one is part of the longer one, seen from the
                      front, e.g. ``spamfoo`` vs ``spam``
                """
                L = min(len(a), len(b))
                return (a[-L:] == b[-L:]) or (a[:L] == b[:L])

            # First, determine colliding names for each combination
            colls = [
                [j for j, other in enumerate(names) if collide(name, other)]
                for name in names
            ]

            # Filter out those entries that are only including themselves and
            # create a set, containing the colliding indices
            return {tuple(c) for c in colls if len(c) > 1}

        def join_path(path: Sequence[Union[str, int]]) -> str:
            """Joins a path sequence to a string, handling integer entries"""
            return ".".join([str(seg) for seg in path])

        def initial_name(path: Sequence[Union[str, int]]) -> str:
            """Given a path sequence, returns an initial name, i.e. a guess for
            a good unique name.

            For purely key-based paths, simply start with the last path
            segment.
            For paths that contain some index-based access, start with a longer
            sequence that includes the name of the parent key.

            Examples:
                * ``foo.bar.baz.spam`` becomes ``spam``
                * ``foo.bar.0.baz.1.spam`` becomes ``bar.0.baz.1.spam``

            Args:
                path (Sequence[Union[str, int]]): The path sequence

            Returns:
                str: The joined path sequence that serves as initial name
            """
            key_is_str = [isinstance(seg, str) for seg in path]
            if all(key_is_str):
                return path[-1]

            first_non_str = key_is_str.index(False)
            return join_path(path[max(0, first_non_str - 1) :])

        # Extract paths and pdims
        paths = [("",) + path for path, _ in kv_pairs]
        plens = [len(p) for p in paths]
        pdims = [pdim for _, pdim in kv_pairs]

        # First, check the custom names
        pdim_names = [pdim.name for pdim in pdims if pdim.name]

        if any([not isinstance(name, str) for name in pdim_names]):
            raise TypeError(
                f"Custom parameter dimension names need to be strings, but at "
                f"least one of the custom names was not a string: {pdim_names}"
            )

        elif any(["." in name for name in pdim_names]):
            raise ValueError(
                f"Custom parameter dimension names cannot contain the "
                f"hierarchy-separating character '.'! Please remove it from "
                f"the names it appears in: {pdim_names}"
            )

        elif len(set(pdim_names)) != len(pdim_names):
            raise ValueError(
                f"There were duplicates among the manually set names of "
                f"parameter dimensions!\nList of names: {pdim_names}"
            )

        # Set the custom names; with the others, determine an initial name
        # depending on the contents of the path sequence.
        names = [
            pdim.name if pdim.name else initial_name(paths[cidx])
            for cidx, pdim in enumerate(pdims)
        ]

        # Set a list of locks, which specifies which names are fixed and should
        # not change throughout the rest of the process. These are initialized
        # with locks for the explicitly given names.
        locks = [bool(pdim.name) for pdim in pdims]

        # With the remaining names, use path segments to generate a name,
        # starting in the back and adding more entries, if there are
        # collisions. By requiring at least one iteration, some pathological
        # cases can be resolved.
        i = 0
        while i == 0 or not unique(names):
            # Go over the collisions and resolve them
            for colls in collisions(names):
                for cidx in colls:
                    # Ignore those that are locked
                    if locks[cidx]:
                        continue

                    # else: may change this name
                    # Get the path segement, starting from the back
                    path_seg = paths[cidx][-(i + 1) :]

                    # Make sure the while loop has a break condition
                    if i > max(plens):
                        raise ValueError(
                            f"Could not automatically find a unique string "
                            f"representation for path {paths[cidx]}! You "
                            f"should set a custom name for the parameter "
                            "dimension."
                        )

                    # Check there is no '.' in the (relevant!) path segement
                    elif any(["." in str(seg) for seg in path_seg]):
                        raise ValueError(
                            f"A path segement of {path_seg} contains the '.' "
                            f"character which interferes with automatically "
                            f"creating unambiguous parameter dimension names. "
                            f"Please select a custom name for the object at "
                            f"path {paths[cidx]}."
                        )

                    # If the resulting name would be shorter than the existing
                    # one, discard it. This is to ensure that initial names
                    # that were longer due to an index access segment are not
                    # overwritten by the above path segment selection
                    new_name = join_path(path_seg)

                    if len(new_name) < len(names[cidx]):
                        continue

                    # All checks passed
                    names[cidx] = new_name

            # Done with this iteration. Check for uniqueness again ...
            i += 1

        # Generate the list of (name, ParamDim) tuples
        return list(zip(names, pdims))

    def _get_dim(self, name: Union[str, Tuple[str]]) -> ParamDimBase:
        """Get the ParamDim object with the given name or location.

        Note that coupled parameter dimensions cannot be accessed via this
        method.

        Args:
            name (Union[str, Tuple[str]]): If a string, will look it up by
                that name, which has to match completely. If it is a tuple of
                strings, the location is looked up instead.

        Returns:
            ParamDimBase: the parameter dimension object

        Raises:
            KeyError: If the ParamDim could not be found
            ValueError: If the parameter dimension name was ambiguous

        """
        if isinstance(name, str):
            try:
                return self._dims[name]

            except KeyError as err:
                _dim_info = self._parse_dims(mode="both")
                raise ValueError(
                    f"A parameter dimension with name '{name}' was not found "
                    f"in this ParamSpace. Available parameter dimensions:\n"
                    f"{_dim_info}"
                ) from err

        # else: is assumed to be a path segement, i.e. a sequence of strings
        # Need to check whether the given key sequence suggests an abs. path
        is_abs = len(name) > 1 and name[0] == ""

        # Now go over the dimensions and try to find matching path segements
        pdim = None

        for path, _pdim in self.dims_by_loc.items():
            if (not is_abs and name == path[-len(name) :]) or (
                is_abs and name[1:] == path
            ):
                # Found one.
                if pdim is None:
                    # Save it and continue to check for ambiguity
                    pdim = _pdim
                    continue

                # else: already set -> there was already one matching this name
                _dim_info = self._parse_dims(mode="both")
                raise ValueError(
                    f"Could not unambiguously find a parameter dimension "
                    f"matching the path segment {name}! "
                    f"Pass a longer path segement to select the right "
                    f"parameter dimension. To symbolize that the key sequence "
                    f"should be regarded as absolute, start with an empty "
                    f"string entry in the key sequence.\nAvailable parameter "
                    f"dimensions:\n{_dim_info}"
                )

        # If still None after all this, no such name was found
        if pdim is None:
            _dim_info = self._parse_dims(mode="both")
            raise ValueError(
                f"A parameter dimension matching location {name} was not "
                f"found in this ParamSpace. Available parameter dimensions:\n"
                f"{_dim_info}"
            )

        return pdim

    # Properties ..............................................................
    # Resolving a state . . . . . . . . . . . . . . . . . . . . . . . . . . . .

    @property
    def default(self) -> dict:
        """Returns the dictionary with all parameter dimensions resolved to
        their default values.

        If an object is Masked, it will resolve it.
        """

        def get_unmasked_default(pdim):
            if isinstance(pdim.default, Masked):
                return pdim.default.value
            return pdim.default

        return recursive_replace(
            copy.deepcopy(self._dict),
            select_func=lambda v: isinstance(v, ParamDimBase),
            replace_func=get_unmasked_default,
            stop_recursion_types=(ParamSpace,),
        )

    @property
    def current_point(self) -> dict:
        """Returns the dictionary with all parameter dimensions resolved to
        the values, depending on the point in parameter space at which the
        iteration is.

        Note that unlike .default, this does not resolve the value if it is
        Masked.
        """
        return recursive_replace(
            copy.deepcopy(self._dict),
            select_func=lambda v: isinstance(v, ParamDimBase),
            replace_func=lambda pdim: pdim.current_value,
            stop_recursion_types=(ParamSpace,),
        )

    # Dimensions: by names or locations . . . . . . . . . . . . . . . . . . . .

    @property
    def dims(self) -> Dict[str, ParamDim]:
        """Returns the ParamDim objects of this ParamSpace. The keys of this
        dictionary are the unique names of the dimensions, created during
        initialization."""
        return self._dims

    @property
    def dims_by_loc(self) -> Dict[Tuple[str], ParamDim]:
        """Returns the ParamDim objects of this ParamSpace, keys being the
        paths to the objects in the dictionary.
        """
        return self._dims_by_loc

    @property
    def coupled_dims(self) -> Dict[str, CoupledParamDim]:
        """Returns the CoupledParamDim objects of this ParamSpace. The keys of
        this dictionary are the unique names of the dimensions, created during
        initialization.
        """
        return self._cdims

    @property
    def coupled_dims_by_loc(self) -> Dict[Tuple[str], CoupledParamDim]:
        """Returns the CoupledParamDim objects found in this ParamSpace, keys
        being the paths to the objects in the dictionary."""
        return self._cdims_by_loc

    # Coordinates . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

    @property
    def coords(self) -> Dict[str, tuple]:
        """Returns the coordinates of all parameter dimensions as dict.
        This does not include the coupled dimensions!

        As the coordinates are merely collected from the parameter dimensions,
        they may include Masked objects.

        Note that the coordinates are converted to lists to make interfacing
        with xarray.DataArray easier.
        """
        return {name: list(pdim.coords) for name, pdim in self.dims.items()}

    @property
    def pure_coords(self) -> Dict[str, tuple]:
        """Returns the pure coordinates of all parameter dimensions as dict.
        This does not include the coupled dimensions!

        Unlike the .coords property, the pure coordinates are cleaned of any
        Masked values.

        Note that the coordinates are converted to lists to make interfacing
        with xarray.DataArray easier.
        """
        return {
            name: list(pdim.pure_coords) for name, pdim in self.dims.items()
        }

    # TODO coupled coordinates?

    @property
    def current_coords(self) -> OrderedDict:
        """Returns the current coordinates of all parameter dimensions.

        This is a shortcut for the get_dim_values method without arguments.
        """
        return self.get_dim_values()

    # Shape, volume, states . . . . . . . . . . . . . . . . . . . . . . . . . .

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
        return tuple(
            [
                len(pdim) if pdim.mask is not True else 1
                for pdim in self.dims.values()
            ]
        )

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
    def max_state_no(self) -> int:
        """Returns the highest possible state number"""
        if self.states_shape:
            return reduce(lambda x, y: x * y, self.states_shape) - 1
        return 0

    @property
    def state_vector(self) -> Tuple[int]:
        """Returns a tuple of all current parameter dimension states"""
        return tuple([s.state for s in self.dims.values()])

    @state_vector.setter
    def state_vector(self, vec: Tuple[int]):
        """Sets the state of all parameter dimensions"""
        if len(vec) != self.num_dims:
            raise ValueError(
                f"Given vector needs to be of same length as there are number "
                f"of dimensions ({self.num_dims}), was: {vec}"
            )

        for (name, pdim), new_state in zip(self.dims.items(), vec):
            try:
                pdim.state = new_state

            except ValueError as err:
                raise ValueError(
                    f"Could not set the state of parameter dimension {name} "
                    f"to {new_state}!"
                ) from err

        log.debug("Successfully set state vector to %s.", vec)

    @property
    def state_no(self) -> Union[int, None]:
        """Returns the current state number by visiting the active parameter
        dimensions and querying their state numbers.
        """
        return self._calc_state_no(self.state_vector)

    @state_no.setter
    def state_no(self, state_no: int):
        """Set the state number.

        This will first calculate the state vector from the number and then
        apply it.
        """
        self.state_vector = self.get_state_vector(state_no=state_no)

    # Magic methods ...........................................................

    def __eq__(self, other) -> bool:
        """Tests the equality of two ParamSpace objects."""
        if not isinstance(other, ParamSpace):
            return False

        # Check for equality of the two objects' underlying __dict__s content,
        # skipping the caching attributes _smap and _iter
        # NOTE it is ok to not check these, because equality of the other
        #      content asserts that the _smap attributes will be equal, too.
        return all(
            [
                self.__dict__[k] == other.__dict__[k]
                for k in self.__dict__.keys()
                if k not in ("_smap", "_iter")
            ]
        )

    def __str__(self) -> str:
        """Returns a parsed, human-readable information string"""
        return (
            f"<{self.__class__.__name__} object at {id(self)} with "
            f"volume {self.volume}, shape {self.shape}>"
        )

    def __repr__(self) -> str:
        """Returns the raw string representation of the ParamSpace."""
        # TODO should actually be a string from which to re-create the object
        return "<paramspace.paramdim.{} object at {} with {}>" "".format(
            self.__class__.__name__,
            id(self),
            repr(
                dict(
                    volume=self.volume,
                    shape=self.shape,
                    dims=self.dims,
                    coupled_dims=self.coupled_dims,
                )
            ),
        )

    # TODO implement __format__

    # Information .............................................................

    def get_info_dict(self) -> dict:
        """Returns a dict with information about this ParamSpace object.

        The returned dict contains similar information as
        :py:meth:`~paramspace.paramspace.ParamSpace.get_info_str`.
        Furthermore, it uses only native data types (scalars, sequences, and
        mappings) such that it is easily serializable and usable in scenarios
        where the paramspace package is not available.

        .. note::

            This information is not meant to fully recreate the ParamSpace
            object, but merely to provide essential metadata like the volume
            or shape of the parameter space and the coordinates of each of its
            dimensions.

        Raises:
            NotImplementedError: If any of the parameter dimensions is masked.
        """

        def prepare_pdim_info(
            pdim: Union[ParamDim, ParamDimBase],
            *,
            name: str,
            keyseq: Tuple[str],
        ) -> dict:
            """Helper function to gather relevant ParamDim information"""
            if pdim.mask:
                raise NotImplementedError(
                    "Retrieving information of ParamSpace objects with masked "
                    "parameter dimensions is not yet possible."
                )
                # NOTE This is a safety measure, as it is is currently unclear
                #      how to clearly and robustly communicate a masked
                #      parameter space via this metadata method.

            info = dict()
            info["name"] = name
            info["full_path"] = list(keyseq)
            info["values"] = list(pdim.values)

            if isinstance(pdim, CoupledParamDim):
                target_name = pdim.target_name
                if isinstance(target_name, str):
                    info["target_name"] = target_name
                else:
                    info["target_name"] = list(target_name)

            return info

        d = dict()

        # ParamSpace information
        d["shape"] = self.shape
        d["volume"] = self.volume
        d["num_dims"] = self.num_dims
        d["num_coupled_dims"] = self.num_coupled_dims

        # Information of individual ParamDim objects
        pdim_iter = zip(self.dims.items(), self.dims_by_loc.keys())
        d["dims"] = [
            prepare_pdim_info(pdim, name=name, keyseq=keyseq)
            for (name, pdim), keyseq in pdim_iter
        ]

        # Information of individual CoupledParamDim objects
        cpdim_iter = zip(
            self.coupled_dims.items(), self.coupled_dims_by_loc.keys()
        )
        d["coupled_dims"] = [
            prepare_pdim_info(cpdim, name=name, keyseq=keyseq)
            for (name, cpdim), keyseq in cpdim_iter
        ]

        return d

    def get_info_str(self) -> str:
        """Returns a string that gives information about shape and size of
        this ParamSpace.
        """
        # Gather lines in a list
        l = ["ParamSpace Information"]
        l += ["======================"]
        l += [""]

        # General information about the Parameter Space
        l += [f"  Dimensions:  {self.num_dims}"]
        l += [f"  Coupled:     {self.num_coupled_dims}"]
        l += [f"  Shape:       {self.shape}"]
        l += [f"  Volume:      {self.volume}"]
        l += [""]

        # ParamDim information
        l += ["Parameter Dimensions"]
        l += ["--------------------"]
        l += [
            "  (Dimensions further up in the list are iterated over less "
            "frequently)"
        ]
        l += [""]

        for name, pdim in self.dims.items():
            l += [f"  - {name}"]
            l += [f"      {pdim.values}"]

            if pdim.mask is True:
                l += [f"      fully masked -> using default:  {pdim.default}"]

            if pdim.order < np.inf:
                l += [f"      order: {pdim.order}"]

            l += [""]

        # CoupledParamDim information
        if self.num_coupled_dims:
            l += [""]
            l += ["Coupled Parameter Dimensions"]
            l += ["----------------------------"]
            l += ["  (Move alongside the state of the coupled ParamDim)"]
            l += [""]

            for name, cpdim in self.coupled_dims.items():
                l += [f"  - {name}"]
                l += [f"      Coupled to:  {cpdim.target_name}"]

                # Add resolved target name, if it differs
                for pdim_name, pdim in self.dims.items():
                    if pdim is cpdim.target_pdim:
                        # Found the coupling target object; get the full name
                        resolved_target_name = pdim_name
                        break

                if resolved_target_name != cpdim.target_name:
                    l[-1] += f"  [resolves to: {resolved_target_name}]"

                l += [f"      Values:      {cpdim.values}"]
                l += [""]

        return "\n".join(l)

    def _parse_dims(
        self,
        *,
        mode: str = "names",
        join_str: str = " -> ",
        prefix: str = "  * ",
    ) -> str:
        """Returns a multi-line string of dimension names or locations.

        This function is intended mostly for internal representation, thus
        defaulting to the longer join strings.
        """
        if mode in ["names"]:
            lines = [n for n in self.dims.keys()]

        elif mode in ["locs"]:
            lines = [
                join_str.join([str(s) for s in p])
                for p in self.dims_by_loc.keys()
            ]

        elif mode in ["both"]:
            max_name_len = max([len(n) for n in self.dims])
            lines = [
                "{name:>{w:d}} :  {path:}".format(
                    name=name,
                    w=max_name_len,
                    path=join_str.join([str(s) for s in path]),
                )
                for name, path in zip(
                    self.dims.keys(), self.dims_by_loc.keys()
                )
            ]

        else:
            raise ValueError(f"Invalid mode: {mode}")

        # Create the multi-line string
        return "\n" + prefix + ("\n" + prefix).join(lines)

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

    # Dict access .............................................................
    # This is a restricted interface for accessing dictionary items
    # It ensures that the ParamSpace remains in a valid state: items are only
    # returned by copy or, if popping them, it is ensured that the item was not
    # a parameter dimension.

    def get(self, key, default=None):
        """Returns a _copy_ of the item in the underlying dict"""
        return copy.deepcopy(self._dict.get(key, default))

    def pop(self, key, default=None):
        """Pops an item from the underlying dict, if it is not a ParamDim"""
        item = self._dict.get(key, None)
        if item in self.dims.values() or item in self.coupled_dims.values():
            raise KeyError(
                f"Cannot remove item with key '{key}' as it is part of a "
                f"parameter dimension."
            )

        return self._dict.pop(key, default)

    # Iterator functionality ..................................................

    def __iter__(self) -> dict:
        """Move to the next valid point in parameter space and return the
        corresponding dictionary.

        Returns:
            The current value of the iteration

        Raises:
            StopIteration: When the iteration has finished
        """
        if self._iter is None:
            # Associate with the iterate function
            self._iter = self.iterator

        # Let generator yield and given the return value, check how to proceed
        return self._iter()
        # NOTE the generator will also raise StopIteration once it ended

    def iterator(
        self,
        *,
        with_info: Union[str, Tuple[str]] = None,
        omit_pt: bool = False,
    ) -> Generator[dict, None, None]:
        """Returns an iterator (more precisely: a generator) yielding all
        unmasked points of the parameter space.

        To control which information is returned at each point, the `with_info`
        and `omit_pt` arguments can be used. By default, the generator will
        return a single dictionary.

        Note that an iteration is also possible for zero-volume parameter
        spaces, i.e. where no parameter dimensions were defined.

        Args:
            with_info (Union[str, Tuple[str]], optional): Can pass strings
                here that are to be returned as the second value. Possible
                values are: 'state_no', 'state_vector', 'state_no_str', and
                'current_coords'.
                To get multiple, add them to a tuple.
            omit_pt (bool, optional): If true, the current value is omitted and
                only the information is returned.

        Returns:
            Generator[dict, None, None]: yields point after point of the
                ParamSpace and the corresponding information
        """

        # Parse the with_info argument, making sure it is a tuple
        if isinstance(with_info, str):
            with_info = (with_info,)

        if self.volume > 0:
            log.debug(
                "Starting iteration over %d points in ParamSpace ...",
                self.volume,
            )
        else:
            log.debug(
                "Starting iteration over zero-volume ParamSpace, i.e.: "
                "will return only the current state of the dict."
            )

        # Prepare parameter dimensions: set them to state 0
        for pdim in self.dims.values():
            pdim.enter_iteration()

        # Yield the first state
        yield self._gen_iter_rv(
            self.current_point if not omit_pt else None, with_info=with_info
        )

        # Now yield all the other states, while available.
        while self._next_state():
            yield self._gen_iter_rv(
                (self.current_point if not omit_pt else None),
                with_info=with_info,
            )

        else:
            log.debug("Iteration finished.")
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

        # Iterate at least one parameter dimensions' state.
        # Do this in reverse such that the last dimensions are iterated over
        # most frequently.
        for pdim in reversed(self.dims.values()):
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
            #    iterate method anyway.
            self.reset()
            return False

        # If this point is reached: broke out of loop
        # -> The next state was reached and we are not at the end yet.
        #    Communicate that by returning True.
        return True

    def _gen_iter_rv(self, pt, *, with_info: Sequence[str]) -> tuple:
        """Is used during iteration to generate the iteration return value,
        adding additional information if specified.

        Note that pt can also be None if iterate is a dry_run
        """
        if not with_info:
            return pt

        # Parse the tuple and add information
        info_tup = tuple()
        for info in with_info:
            if info in ["state_no"]:
                info_tup += (self.state_no,)

            elif info in ["state_no_str", "padded_state_no"]:
                info_tup += (
                    "{sno:0{digs:d}d}"
                    "".format(
                        sno=self.state_no, digs=len(str(self.max_state_no))
                    ),
                )

            elif info in ["state_vector", "state_vec"]:
                info_tup += (self.state_vector,)

            elif info in ["current_coords", "coords"]:
                info_tup += (self.current_coords,)

            else:
                raise ValueError(
                    f"No such information '{info}' available. Check the "
                    f"`with_info` argument!"
                )

        # Return depending on whether a point was given or not
        if pt is not None:
            # Concatenate and return
            return (pt,) + info_tup

        elif len(info_tup) == 1:
            # Return only the single info entry
            return info_tup[0]

        # else: return as tuple
        return info_tup

    # Mapping .................................................................

    @property
    def state_map(self) -> xr.DataArray:
        """Returns an inverse mapping, i.e. an n-dimensional array where the
        indices along the dimensions relate to the states of the parameter
        dimensions and the content of the array relates to the state numbers.

        Returns:
            xr.DataArray: A mapping of indices and coordinates to the state
                number. Note that it is not ensured that the coordinates are
                unique, so it _might_ not be possible to use location-based
                indexing.

        Raises:
            RuntimeError: If -- for an unknown reason -- the iteration did not
                cover all of the state mapping. Should not occur.
        """
        # Check if the cached result can be returned
        if self._smap is not None:
            log.debug("Returning previously cached inverse mapping ...")
            return self._smap

        # else: need to calculate the inverse mapping

        # Create empty n-dimensional array which will hold state numbers
        smap = np.ndarray(self.states_shape, dtype=int)
        smap.fill(-1)  # i.e., not set yet

        # As .iterator does not allow iterating over default states, iterate
        # over the multi-index of the smap, which is equivalent to a valid
        # state vector, and get the corresponding state number
        for midx in np.ndindex(smap.shape):
            # Resolve the corresponding state number from the multi-index
            # (which is equivalent to a state vector) and store at this midx
            smap[tuple(midx)] = self._calc_state_no(midx)

        # Convert to DataArray
        smap = xr.DataArray(
            smap,
            dims=self.pure_coords.keys(),
            coords=self.pure_coords.values(),
        )

        # Cache and make it read-only before returning
        log.debug(
            "Finished creating inverse mapping. Caching it and making "
            "the cache read-only ..."
        )
        self._smap = smap
        self._smap.data.flags.writeable = False

        return self._smap

    @property
    def active_state_map(self) -> xr.DataArray:
        """Returns a subset of the state map, where masked coordinates are
        removed and only the active coordinates are present.

        Note that this array has to be re-calculated every time, as the mask
        status of the ParamDim objects is not controlled by the ParamSpace and
        can change without notice.

        Also: the indices will no longer match the states of the dimensions!
        Values of the DataArray should only be accessed via the coordinates!

        Returns:
            xr.DataArray: A reduced state map which only includes active, i.e.:
                unmasked coordinates.
        """
        # Work on a copy of the state map
        amap = self.state_map.copy()

        # Create a dict of (dimension names, indices to keep)
        indcs = {
            dim: [
                i
                for i, coord in enumerate(coords)
                if not isinstance(coord, Masked)
            ]
            for dim, coords in self.coords.items()
        }

        # Apply the selection and return
        return amap.isel(indcs)

    def get_state_vector(self, *, state_no: int) -> Tuple[int]:
        """Returns the state vector that corresponds to a state number

        Args:
            state_no (int): The state number to look for in the inverse mapping

        Returns:
            Tuple[int]: the state vector corresponding to the state number
        """
        try:
            # Get it from the state map data ...
            vec = np.argwhere(self.state_map.data == state_no)[0]

            # Convert entries to integers, as they might be np.int64 ...
            return tuple([int(idx) for idx in vec])

        except IndexError as err:
            raise ValueError(
                f"Did not find state number {state_no} in inverse mapping! "
                f"Make sure it is an integer in the closed interval "
                f"[0, {reduce(lambda x, y: x * y, self.states_shape) - 1}]."
            )

    def get_dim_values(
        self, *, state_no: int = None, state_vector: Tuple[int] = None
    ) -> OrderedDict:
        """Returns the current parameter dimension values or those of a
        certain state number or state vector.
        """
        if state_no is None and state_vector is None:
            # Return the current value
            return OrderedDict(
                [
                    (name, pdim.current_value)
                    for name, pdim in self.dims.items()
                ]
            )

        # Check that only one of the arguments was given
        if state_no is not None and state_vector is not None:
            raise TypeError(
                "Expected only one of the arguments `state_no` "
                "and `state_vector`, got both!"
            )

        elif state_no is not None:
            state_vector = self.get_state_vector(state_no=state_no)

        # Can now assume that state_vector variable (not the property!) is set
        return OrderedDict(
            [
                (name, pdim.coords[s])
                for (name, pdim), s in zip(self.dims.items(), state_vector)
            ]
        )

    def _calc_state_no(self, state_vector: Tuple[int]) -> int:
        log.debug("Calculating state number from state vector ...")

        # Use the given state vector
        log.debug("  state vector: %s", state_vector)

        # Now need the full shape of the parameter space, i.e. ignoring masked
        # values but including the default values
        states_shape = self.states_shape
        log.debug(
            "  states shape: %s  (volume: %s)",
            states_shape,
            reduce(lambda x, y: x * y, states_shape) if states_shape else 0,
        )

        # The lengths will now be used to calculate the multipliers, where the
        # _last_ dimension will have the multiplier 1.
        # For example, given lengths [   5,  20, 10, 10], the corresponding
        # multipliers are:           [2000, 100, 10,  1]
        mults = [
            reduce(lambda x, y: x * y, states_shape[i + 1 :], 1)
            for i in range(self.num_dims)
        ]
        log.debug("  multipliers:  %s", mults)

        # Now, calculate the state number
        state_no = sum([(s * m) for s, m in zip(state_vector, mults)])
        log.debug("  state no:     %s", state_no)

        return state_no

    # Masking .................................................................

    def set_mask(
        self,
        name: Union[str, Tuple[str]],
        mask: Union[bool, Tuple[bool]],
        invert: bool = False,
    ) -> None:
        """Set the mask value of the parameter dimension with the given name.

        Args:
            name (Union[str, Tuple[str]]): the name of the dim, which can be a
                tuple of strings or a string.
                If name is a string, it will be converted to a tuple, regarding
                the '/' character as splitting string.
                The tuple is compared to the paths of the dimensions, starting
                from the back; thus, not the whole path needs to be given, it
                just needs to be enough to resolve the dimension names
                unambiguously.
                For names at the root level that could be ambiguous, a leading
                "/" in the string argument or an empty string in the tuple-form
                of the argument needs to be set to symbolise the dimension
                being at root level.
                Also, the ParamDim's custom name attribute can be used to
                identify it.
            mask (Union[bool, Tuple[bool]]): The new mask values. Can also be
                a slice, the result of which defines the True values of the
                mask.
            invert (bool, optional): If set, the mask will be inverted _after_
                application.
        """
        # Resolve the parameter dimension
        pdim = self._get_dim(name)

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

    # TODO consider using the xarray interface here? i.e.: sel and isel
    def activate_subspace(
        self,
        *,
        allow_default: bool = False,
        reset_all_others: bool = True,
        **selector,
    ) -> None:
        """Selects a subspace of the parameter space and makes only that part
        active for iteration.

        This is a wrapper around set_mask, implementing more arguments and also
        checking if any dimension is reduced to a default value, which might
        cause problems elsewhere.

        Args:
            allow_default (bool, optional): If True, a ValueError is raised
                when any of the dimensions is completely masked or when the
                index 0 is used during selecting of a mask.
            reset_all_others (bool, optional): If True, resets all masks before
                activating the subspace. If False, the previously applied masks
                are untouched.
            **selector: A dict specifying the *active* states. A key of the
                key-value pairs should be the name of the dimension, the
                value should be a dict with one of the following keys:

                    - idx: to select by index
                    - loc: to select by coordinate values
                    - ``**tol_kwargs``: passed on to ``np.isclose`` when
                        comparing coordinate values.

                Non-sequence values will be put into lists. Alternatively,
                slices can be specified, which are applied on the list of all
                available indices or coordinates, respectively.
                As a shorthand, not specifying a dict but directly a list or a
                slice defaults to ``loc``-behaviour.

        Raises:
            ValueError: Description
        """

        def calc_mask(name, *, idx=None, loc=None, **tol_kwargs) -> List[bool]:
            """Calculates the mask to use such that the given indices or
            locations are _un_masked.

            The ``tol_kwargs`` are passed on to ``np.isclose`` for cases where
            a coordinate is selected by ``loc``.
            """

            def contains_close(a, seq, **tol_kwargs) -> bool:
                """Whether ``a`` is contained in ``seq`` when comparing a
                numeric-typed ``a`` via ``np.isclose`` rather than ``==``.

                For non-numeric types, the regular ``__contains__`` is used.

                NOTE: The decision is made via the type of ``a``
                """
                if isinstance(a, (float, int)):
                    try:
                        return any(
                            [np.isclose(a, v, **tol_kwargs) for v in seq]
                        )
                    except TypeError as err:
                        raise TypeError(
                            f"Could not ascertain whether {a} is contained in "
                            f"{seq}! This is probably due to values of "
                            f"numeric type being mixed with non-numeric ones. "
                            f"Check the definition of your parameter "
                            f"dimensions."
                        ) from err
                return a in seq

            if idx is not None and loc is not None:
                raise ValueError(
                    "Only accepting _either_ of the arguments "
                    "`idx` and `loc`, but got both!"
                )

            pdim = self._get_dim(name)

            # Distinguish idx and loc
            if idx is not None:
                if isinstance(idx, slice):
                    # Apply it to the list of possible indices
                    idcs = list(range(1, 1 + pdim.num_values))[idx]

                    # Done.

                else:
                    # Indices explicitly given.
                    # Only need to check for invalid values
                    if not isinstance(idx, (list, tuple)):
                        idx = [idx]

                    if 0 in idx:
                        raise IndexError(
                            "Encountered index 0 in list of "
                            "indices to be selected! This is an "
                            "invalid value when selecting a "
                            "subspace, as that index corresponds "
                            "to the default state of a parameter "
                            "dimension; indices for iteration "
                            "values start at 1!"
                        )

                    elif max(idx) > pdim.num_values:
                        raise IndexError(
                            f"Given indices {idx} contained a value that "
                            f"exceeds the highest index, {pdim.num_values}!"
                        )

                    elif len(set(idx)) != len(idx):
                        raise ValueError(
                            f"Given indices {idx} contained at least "
                            f"one duplicate element!"
                        )

                    # Everything ok.
                    idcs = idx

            elif loc is not None:
                # Get the coordinates (without the default, thus +1s below)
                coords = pdim.pure_coords[1:]

                if isinstance(loc, slice):
                    # From the slice, extract start, stop and step
                    start = loc.start if loc.start is not None else -np.inf
                    stop = loc.stop if loc.stop is not None else +np.inf

                    # Filter out those that are not within start, stop
                    idcs = [
                        (idx + 1)
                        for idx, val in enumerate(coords)
                        if start <= val < stop
                    ]

                    # If a step was given, apply it in a second step
                    if loc.step is not None:
                        idcs = idcs[slice(None, None, loc.step)]

                    # Done.

                else:
                    # Got a list of explicit coordinates to use.
                    # Only need to make a few checks.
                    if not isinstance(loc, (list, tuple)):
                        loc = [loc]

                    if any([not contains_close(val, coords) for val in loc]):
                        raise KeyError(
                            f"At least one of the labels in {loc} is not "
                            f"available as coordinate of this parameter "
                            f"dimension, {coords}!"
                        )

                    elif len(set(loc)) != len(loc):
                        raise ValueError(
                            f"Given labels {loc} contained at least "
                            f"one duplicate item!"
                        )

                    # Everything ok. Get the indices. Iterate over coordinates
                    # rather than loc in order to ascertain the correct order
                    # and have the indices available. The checks above make
                    # sure that this is no issue.
                    idcs = [
                        (idx + 1)
                        for idx, val in enumerate(coords)
                        if contains_close(val, loc)
                    ]

            else:
                raise ValueError(
                    "Missing one of the required keyword "
                    "arguments `idx` or `loc`!"
                )

            # Given the indices, create and return the mask
            return [bool(i not in idcs) for i in range(1, 1 + pdim.num_values)]

        # Determine whether to reset all masks
        if reset_all_others:
            for dim_name in self.dims.keys():
                self.set_mask(dim_name, False)

        # Calculate all the masks
        masks = {
            k: calc_mask(k, **v)
            if isinstance(v, dict)
            else calc_mask(k, loc=v)
            for k, v in selector.items()
        }
        log.debug("Calculated masks: %s", masks)

        # Apply the masks, checking if it would result in defaulting
        for dim_name, mask in masks.items():
            if not allow_default and all(mask):
                raise ValueError(
                    f"With the given selector, parameter "
                    f"dimension '{dim_name}' would be totally masked, "
                    f"thus resulting in shifting to its default "
                    f"state in iteration. If you want to permit "
                    f"this, set the allow_default argument.\n"
                    f"Selector:\n{selector}"
                )

            # Everything ok, set the mask now.
            self.set_mask(dim_name, mask)

        log.debug(
            "Selected subspace. New volume: %d,  shape: %s.",
            self.volume,
            self.shape,
        )
