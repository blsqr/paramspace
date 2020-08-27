"""The ParamDim classes define parameter dimensions along which discrete values
can be assumed. While they provide iteration abilities on their own, they make
sense mostly to use as objects in a dict that is converted to a ParamSpace.
"""
import abc
import copy
import logging
import warnings
from typing import Hashable, Iterable, List, Sequence, Tuple, Union

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
        return f"<{self.value}>"

    def __repr__(self) -> str:
        return f"<Masked object, value: {repr(self.value)}>"

    def __eq__(self, other) -> bool:
        return self.value == other

    # YAML representation .....................................................

    @classmethod
    def to_yaml(cls, representer, node: "Masked"):
        """
        Args:
            representer (ruamel.yaml.representer): The representer module
            node (Masked): The node, i.e. an instance of this class

        Returns:
            the scalar value that this object masks
        """
        return representer.represent_data(node._val)


class MaskedValueError(ValueError):
    """Raised when trying to set the state of a ParamDim to a masked value"""

    pass


# -----------------------------------------------------------------------------


class ParamDimBase(metaclass=abc.ABCMeta):
    """The ParamDim base class."""

    # Which __dict__ content to omit when checking for equivalence
    _OMIT_ATTR_IN_EQ = tuple()

    # Which additional attributes to use in __repr__
    _REPR_ATTRS = tuple()

    # Keyword parameters used to set the values
    _VKWARGS = (
        "values",
        "range",
        "linspace",
        "logspace",
    )

    # .........................................................................

    def __init__(
        self,
        *,
        default,
        values: Iterable = None,
        order: float = None,
        name: str = None,
        as_type: str = None,
        assert_unique: bool = True,
        **kwargs,
    ) -> None:
        """Initialise a parameter dimension object.

        Args:
            default: default value of this parameter dimension
            values (Iterable, optional): Which discrete values this parameter
                dimension can take. This argument takes precedence over any
                constructors given in the kwargs (like range, linspace, …).
            order (float, optional): If given, this allows to specify an order
                within a ParamSpace that includes this ParamDim object. If not,
                will use np.inf instead.
            name (str, optional): If given, this is an *additional* name of
                this ParamDim object, and can be used by the ParamSpace to
                access this object.
            as_type (str, optional): If given, casts the individual created
                values to a certain python type. The following string values
                are possible: str, int, bool, float
            assert_unique (bool, optional): Whether to assert uniqueness of
                the values among them.
            **kwargs: Constructors for the `values` argument, valid keys are
                `range`, `linspace`, and `logspace`; corresponding values are
                expected to be iterables and are passed to `range(*args)`,
                `np.linspace(*args)`, or `np.logspace(*args)`, respectively.

        Raises:
            TypeError: For invalid arguments
        """
        # Initialize attributes that are managed by properties or methods
        self._state = 0  # corresponding to the default state

        # Set attributes that need no further checks
        self._name = name
        self._order = order if order is not None else np.inf

        # Package values into kwargs, for easier handling
        if values is not None:
            kwargs["values"] = values

        # Gather the initialization kwargs for use with yaml representer
        init_kwargs = dict(
            default=default,
            order=order,
            name=name,
            as_type=as_type,
            assert_unique=assert_unique,
            **kwargs,
        )

        # TODO Make this more elegant!
        # As the base class __init__ is used by derived classes that might
        # already have set private members, check if the `default` argument
        # should even be used...
        if not hasattr(self, "_default"):
            self._default = self._parse_value(default, as_type=as_type)

        else:
            # Was already set; don't store the passed value
            del init_kwargs["default"]

        # Same for value-setting arguments
        if not hasattr(self, "_vals"):
            # Now let the helper function take care of the rest
            self._init_vals(
                as_type=as_type, assert_unique=assert_unique, **kwargs
            )

        # Store the initialization kwargs
        self._init_kwargs = init_kwargs
        # Derived classes should add the necessary kwargs in their __init__s

        # Done.

    def _init_vals(self, *, as_type: str, assert_unique: bool, **kwargs):
        """Parses the arguments and invokes ``_set_vals``"""

        # Now check for unexpected ones and set the valid ones
        if any([k not in self._VKWARGS for k in kwargs.keys()]):
            raise TypeError(
                "Received invalid keyword argument(s) for {}: {}. "
                "Allowed arguments: {}"
                "".format(
                    self.__class__.__name__,
                    ", ".join([k for k in kwargs if k not in self._VKWARGS]),
                    ", ".join(self._VKWARGS),
                )
            )

        elif len(kwargs) > 1:
            _vkws = ", ".join(self._VKWARGS)
            raise TypeError(
                f"Received too many keyword arguments. Need _one_ of:  {_vkws}"
            )

        elif "values" in kwargs:
            self._set_values(
                kwargs["values"], assert_unique=assert_unique, as_type=as_type
            )

        elif "range" in kwargs:
            self._set_values(
                range(*kwargs["range"]),
                assert_unique=assert_unique,
                as_type=as_type,
            )

        elif "linspace" in kwargs:
            self._set_values(
                np.linspace(*kwargs["linspace"]),
                assert_unique=assert_unique,
                as_type="float" if as_type is None else as_type,
            )

        elif "logspace" in kwargs:
            self._set_values(
                np.logspace(*kwargs["logspace"]),
                assert_unique=assert_unique,
                as_type="float" if as_type is None else as_type,
            )

        else:
            _vkws = ", ".join(self._VKWARGS)
            raise TypeError(
                f"Missing one of the following required keyword arguments to "
                f"set the values of {self.__class__.__name__}: {_vkws}."
            )

    # Properties ..............................................................

    @property
    def name(self):
        """The name value."""
        return self._name

    @property
    def order(self):
        """The order value."""
        return self._order

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
    def coords(self) -> tuple:
        """Returns the coordinates of this parameter dimension, i.e., the
        combined default value and the sequence of iteration values.

        Returns:
            tuple: coordinates associated with the indices of this dimension
        """
        return (self.default,) + self._vals

    @property
    def pure_coords(self) -> tuple:
        """Returns the pure coordinates of this parameter dimension, i.e., the
        combined default value and the sequence of iteration values, but with
        masked values resolved.

        Returns:
            tuple: coordinates associated with the indices of this dimension
        """
        return tuple(
            [c if not isinstance(c, Masked) else c.value for c in self.coords]
        )

    @property
    def num_values(self) -> int:
        """The number of values available.

        Returns:
            int: The number of available values
        """
        return len(self.values)

    @property
    def num_states(self) -> int:
        """The number of possible states, i.e., including the default state

        Returns:
            int: The number of possible states
        """
        return self.num_values + 1

    @property
    def state(self) -> int:
        """The current iterator state

        Returns:
            Union[int, None]: The state of the iterator; if it is None, the
                ParamDim is not inside an iteration.
        """
        return self._state

    @property
    def current_value(self):
        """If in an iteration, returns the value according to the current
        state. Otherwise, returns the default value.
        """
        if self.state == 0:
            return self.default
        return self.values[self.state - 1]

    # Magic methods ...........................................................

    def __eq__(self, other) -> bool:
        """Check for equality between self and other

        Args:
            other: the object to compare to

        Returns:
            bool: Whether the two objects are equivalent
        """
        if not isinstance(other, type(self)):
            return False

        # Check equality of the objects' __dict__s, leaving out _mask_cache
        return all(
            [
                self.__dict__[k] == other.__dict__[k]
                for k in self.__dict__.keys()
                if k not in ("_init_kwargs",) + self._OMIT_ATTR_IN_EQ
            ]
        )

    @abc.abstractmethod
    def __len__(self) -> int:
        """Returns the effective length of the parameter dimension, i.e. the
        number of values that will be iterated over

        Returns:
            int: The number of values to be iterated over
        """

    def __str__(self) -> str:
        """
        Returns:
            str: Returns the string representation of the ParamDimBase-derived
                object
        """
        return repr(self)

    def __repr__(self) -> str:
        """
        Returns:
            str: Returns the string representation of the ParamDimBase-derived
                object
        """
        return (
            f"<paramspace.paramdim.{self.__class__.__name__} object at "
            f"{id(self)} with {repr(self._parse_repr_attrs())}>"
        )

    def _parse_repr_attrs(self) -> dict:
        """For the __repr__ method, collects some attributes into a dict"""
        d = dict(
            default=self.default,
            order=self.order,
            values=self.values,
            name=self.name,
        )

        for attr_name in self._REPR_ATTRS:
            d[attr_name] = getattr(self, attr_name)

        return d

    # Iterator functionality ..................................................

    def __iter__(self):
        """Iterate over available values"""
        return self

    def __next__(self):
        """Move to the next valid state and return the corresponding parameter
        value.

        Returns:
            The current value (inside an iteration)
        """
        self.iterate_state()
        return self.current_value

    # Public API ..............................................................
    # These are needed by the ParamSpace class to have more control over the
    # iteration.

    @abc.abstractmethod
    def enter_iteration(self) -> None:
        """Sets the state to the first possible one, symbolising that an
        iteration has started.

        Returns:
            None

        Raises:
            StopIteration: If no iteration is possible
        """

    @abc.abstractmethod
    def iterate_state(self) -> None:
        """Iterates the state of the parameter dimension.

        Returns:
            None

        Raises:
            StopIteration: Upon end of iteration
        """

    @abc.abstractmethod
    def reset(self) -> None:
        """Called after the end of an iteration and should reset the object to
        a state where it is possible to start another iteration over it.

        Returns:
            None
        """

    # Non-public API ..........................................................

    def _parse_value(self, val, *, as_type: str = None):
        """Parses a single value and ensures it is of correct type."""

        # Map of available type conversions
        type_convs = dict(
            str=str,
            int=int,
            float=float,
            bool=bool,
            tuple=self._rec_tuple_conv,
        )

        # Apply type conversion
        if as_type is not None:
            val = type_convs[as_type](val)

        return val

    def _set_values(
        self, values: Iterable, *, assert_unique: bool, as_type: str = None
    ):
        """This function sets the values attribute; it is needed for the
        values setter function that is overwritten when changing the property
        in a derived class.

        Args:
            values (Iterable): The iterable to set the values with
            assert_unique (bool): Whether to assert uniqueness of the values
            as_type (str, optional): The following values are possible:
                str, int, bool, float. If not given, will leave the values
                as they are.

        Raises:
            AttributeError: If the attribute is already set
            ValueError: If the iterator is invalid

        Deleted Parameters:
            as_float (bool, optional): If given, makes sure that values are
                of type float; this is needed for the numpy initializers
        """
        # Check the values
        if hasattr(self, "_vals"):
            # Was already set
            raise AttributeError("Values already set; cannot be set again!")

        elif len(values) < 1:
            raise ValueError(
                f"{self.__class__.__name__} values need be a container of "
                f"length >= 1, was {values}"
            )

        # Parse each individual value, changing type if configured to do so
        values = [self._parse_value(v, as_type=as_type) for v in values]

        # Convert it to a tuple
        values = tuple(values)

        # Assert that values are unique
        if assert_unique and any([values.count(v) > 1 for v in values]):
            raise ValueError(
                f"Values need to be unique, but there were "
                f"duplicates: {values}"
            )

        # Now store it as attribute
        self._vals = values

    def _rec_tuple_conv(self, obj: list):
        """Recursively converts a list-like object into a tuple, replacing
        all occurences of lists with tuples.
        """
        # Recursive branch
        if isinstance(obj, list):
            return tuple(
                [
                    o if not isinstance(o, list) else self._rec_tuple_conv(o)
                    for o in obj
                ]
            )

        # End of recursion
        return obj

    # YAML representation .....................................................
    # NOTE The `yaml_tag` class variable needs be set in the derived classes

    # Define some settings needed for saving to yaml
    # Which entries to update and with which attribute
    _YAML_UPDATE = dict()

    # Which entries to remove if they have a certain value
    _YAML_REMOVE_IF = dict(name=(None,), order=(None,))

    @classmethod
    def to_yaml(cls, representer, node):
        """
        Args:
            representer (ruamel.yaml.representer): The representer module
            node (type(self)): The node, i.e. an instance of this class

        Returns:
            a yaml mapping that is able to recreate this object
        """
        # Get the init_kwargs and use them as basis for the mapping
        d = copy.deepcopy(node._init_kwargs)

        # Depending on the class variables, update some entries
        for k, attr_name in cls._YAML_UPDATE.items():
            # Resolve the attribute and, if possible, call it
            attr = getattr(node, attr_name)
            new_val = attr if not callable(attr) else attr()

            d[k] = new_val

        # ... and remove some if they match the given value
        d = {
            k: v
            for k, v in d.items()
            if k not in cls._YAML_REMOVE_IF or v not in cls._YAML_REMOVE_IF[k]
        }

        # Can now call the representer
        return representer.represent_mapping(cls.yaml_tag, d)

    @classmethod
    def from_yaml(cls, constructor, node):
        """The default constructor for ParamDim-derived objects"""
        return cls(**constructor.construct_mapping(node, deep=True))


# -----------------------------------------------------------------------------


class ParamDim(ParamDimBase):
    """The ParamDim class."""

    # Which __dict__ content to omit when checking for equivalence
    _OMIT_ATTR_IN_EQ = (
        "_mask_cache",
        "_inside_iter",
        "_target_of",
    )

    # Define the additional attribute names that are to be added to __repr__
    _REPR_ATTRS = ("mask",)

    # Define the yaml tag to use
    yaml_tag = "!pdim"

    # And the other yaml representer settings
    _YAML_UPDATE = dict(
        mask="mask",
    )
    _YAML_REMOVE_IF = dict(
        name=(None,),
        order=(None,),
        mask=(None, False),
    )

    # .........................................................................

    def __init__(self, *, mask: Union[bool, Tuple[bool]] = False, **kwargs):
        """Initialize a regular parameter dimension.

        Args:
            mask (Union[bool, Tuple[bool]], optional): Which values of the
                dimension to mask, i.e., skip in iteration. Note that masked
                values still count to the length of the parameter dimension!
            **kwargs: Passed to ``ParamDimBase.__init__``.
                Possible arguments:

                - default: default value of this parameter dimension
                - values (Iterable, optional): Which discrete values this
                    parameter dimension can take. This argument takes
                    precedence over any constructors given in the kwargs
                    (like range, linspace, …).
                - order (float, optional): If given, this allows to specify an
                    order within a ParamSpace that includes this ParamDim. If
                    not given, np.inf will be used, i.e., dimension is last.
                - name (str, optional): If given, this is an *additional* name
                    of this ParamDim object, and can be used by the ParamSpace
                    to access this object.
                - ``**kwargs``: Constructors for the ``values`` argument, valid
                    keys are ``range``, ``linspace``, and ``logspace``;
                    corresponding values are expected to be iterables and are
                    passed to ``range(*args)``, ``np.linspace(*args)``, or
                    ``np.logspace(*args)``, respectively.
        """
        super().__init__(**kwargs)

        # Additional attributes, needed for coupling, masking, iteration
        self._target_of = []

        self._inside_iter = False

        self._mask_cache = None
        self.mask = mask

        # Add further initialization kwargs, needed for yaml
        self._init_kwargs["mask"] = mask

        log.debug("ParamDim initialised.")

    # Additional properties ...................................................

    @property
    def target_of(self):
        """Returns the list that holds all the CoupledParamDim objects that
        point to this instance of ParamDim.
        """
        return self._target_of

    @property
    def state(self) -> int:
        """The current iterator state

        Returns:
            Union[int, None]: The state of the iterator; if it is None, the
                ParamDim is not inside an iteration.
        """
        return super().state

    @state.setter
    def state(self, new_state: int):
        """Sets the current iterator state."""
        # Perform type and value checks
        if not isinstance(new_state, int):
            raise TypeError(
                f"New state can only be of type int, was {type(new_state)}!"
            )

        elif new_state < 0:
            raise ValueError(f"New state needs to be >= 0, was {new_state}.")

        elif new_state > self.num_values:
            raise ValueError(
                f"New state needs to be <= {self.num_values}, was {new_state}."
            )

        elif new_state > 0 and self.mask_tuple[new_state - 1] is True:
            raise MaskedValueError(
                f"Value at index {new_state} is masked: "
                f"{self.values[new_state - 1]}. Cannot set the state to this "
                "index!"
            )

        # Everything ok. Can set the state
        self._state = new_state

    @property
    def mask_tuple(self) -> Tuple[bool]:
        """Returns a tuple representation of the current mask"""
        if self._mask_cache is None:
            self._mask_cache = tuple(
                [isinstance(v, Masked) for v in self.values]
            )
        return self._mask_cache

    @property
    def mask(self) -> Union[bool, Tuple[bool]]:
        """Returns False if no value is masked or a tuple of booleans that
        represents the mask
        """
        m = self.mask_tuple  # uses a cached value, if available

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
            mask = [mask] * self.num_values

        elif isinstance(mask, slice):
            # Apply the slice to a list of indices in order to know which ones
            # to set to True in the mask
            idcs = list(range(self.num_values))[mask]
            mask = [(i in idcs) for i in range(self.num_values)]

        # Should be a container now. Assert correct length.
        if len(mask) != self.num_values:
            raise ValueError(
                f"Given mask needs to be a boolean, a slice, or a container "
                f"of same length as the values container ({self.num_values}), "
                f"was:  {mask}"
            )

        # Mark the mask cache as invalid, such that it is re-calculated when
        # the mask getter is accessed the next time
        self._mask_cache = None

        # Now build a new values container and store as attribute
        self._vals = tuple([set_val(m, v) for m, v in zip(mask, self.values)])

        # Mask the default value, if all other values are masked
        self._default = set_val(not all(mask), self.default)

    @property
    def num_masked(self) -> int:
        """Returns the number of unmasked values"""
        return sum(self.mask_tuple)

    # Magic Methods ...........................................................

    def __len__(self) -> int:
        """Returns the effective length of the parameter dimension, i.e. the
        number of values that will be iterated over.

        Returns:
            int: The number of values to be iterated over
        """
        if self.mask is True:
            # Will only return the default value, thus effective length is 1
            return 1
        return len(self._vals) - self.num_masked

    # Public API ..............................................................

    def enter_iteration(self) -> None:
        """Sets the state to the first possible one, symbolising that an
        iteration has started.

        Raises:
            StopIteration: If no iteration is possible because all values are
                masked.
        """
        # Need to distinguish mask states
        if self.mask is False:
            # Trivial case, start with 1, the first iteration value state
            self.state = 1

        elif self.mask is True:
            # There is only the default state to go to; go there, but then
            # communicate that the iteration is over
            self.state = 0

        else:
            # Find the first unmasked state
            self.state = self.mask.index(False) + 1  # +1 accounts for default

        # Set the flag to signify that inside iteration
        self._inside_iter = True

    def iterate_state(self) -> None:
        """Iterates the state of the parameter dimension.

        Raises:
            StopIteration: Upon end of iteration
        """
        # Set to zero or increment, depending on whether inside or outside of
        # an iteration
        if not self._inside_iter:
            self.enter_iteration()
            return

        # Else: within iteration
        # Look for further possible states in the remainder of the mask tuple
        sub_mask = self.mask_tuple[self.state :]

        if False in sub_mask:
            # There is another possible state, find it via index
            self.state += sub_mask.index(False) + 1

        else:
            # No more possible state values
            # Reset the state, allowing to reuse the object (unlike with
            # other Python iterators). Then communicate: iteration should stop.
            self.reset()
            raise StopIteration

    def reset(self) -> None:
        """Called after the end of an iteration and should reset the object to
        a state where it is possible to start another iteration over it.

        Returns:
            None
        """
        self.state = 0  # the state corresponding to the default value
        self._inside_iter = False


# -----------------------------------------------------------------------------


class CoupledParamDim(ParamDimBase):
    """A CoupledParamDim object is recognized by the ParamSpace and its state
    moves alongside with another ParamDim's state.
    """

    # Which __dict__ content to omit when checking for equivalence
    _OMIT_ATTR_IN_EQ = ()

    # Define the additional attribute names that are to be added to __repr__
    _REPR_ATTRS = (
        "target_pdim",
        "target_name",
        "_use_coupled_default",
        "_use_coupled_values",
    )

    # Define the yaml tag to use
    yaml_tag = "!coupled-pdim"

    # And the other yaml representer settings
    _YAML_UPDATE = dict(
        target_name="_target_name_as_list",
    )
    _YAML_REMOVE_IF = dict(
        name=(None,),
        order=(None,),
        assert_unique=(True, False),
        default=(None,),
        values=(None, [None]),
        use_coupled_default=(None,),
        use_coupled_values=(None,),
        target_name=(None,),
        target_pdim=(None,),
    )

    # .........................................................................

    def __init__(
        self,
        *,
        default=None,
        target_pdim: ParamDim = None,
        target_name: Union[str, Sequence[str]] = None,
        use_coupled_default: bool = None,
        use_coupled_values: bool = None,
        **kwargs,
    ):
        """Initialize a coupled parameter dimension.

        If the `default` or any values-setting argument is set, those will be
        used. If that is not the case, the respective parts from the coupled
        dimension will be used.

        Args:
            default (None, optional): The default value. If not given, will
                use the one from the coupled object.
            target_pdim (ParamDim, optional): The ParamDim object to couple to
            target_name (Union[str, Sequence[str]], optional): The *name* of
                the ParamDim object to couple to; needs to be within the same
                ParamSpace and the ParamSpace needs to be able to resolve it
                using this name.
            use_coupled_default (bool, optional): DEPRECATED
            use_coupled_values (bool, optional): DEPRECATED
            **kwargs: Passed to ParamDimBase.__init__

        Raises:
            TypeError: If neither target_pdim nor target_name were given or
                or both were given
        """
        # TODO Make this __init__ more elegant!

        # Deprecation warnings for old parameters
        if use_coupled_default is not None or use_coupled_values is not None:
            warnings.warn(
                "The CoupledParamDim.__init__ parameters "
                "`use_coupled_default` and `use_coupled_values` are "
                "deprecated and will soon be removed. Whether the "
                "counterpart from the coupled parameter dimension "
                "is to be used is determined by whether the "
                "`default` or any value-setting argument was given.",
                DeprecationWarning,
            )

        # Disallow mask argument
        if "mask" in kwargs:
            raise TypeError(
                "Received invalid keyword-argument `mask` for "
                "CoupledParamDim!"
            )

        # Set attributes
        self._target_pdim = None  # the object that is coupled to
        self._target_name = None  # the name of it in a ParamSpace

        # Determine whether coupled values or given values are to be used
        self._use_coupled_default = default is None
        self._use_coupled_values = not any(
            [k in self._VKWARGS for k in kwargs]
        )

        # In order to not invoke the default- and value-setters in the parent
        # classes' initializer, set them here already. Setting the attributes
        # (regardless of value) already makes the base class __init__ jump that
        # part of the initialization
        if self._use_coupled_default:
            self._default = None  # Value does not matter, is never used

        if self._use_coupled_values:
            self._vals = [None]  # Value does not matter, is never used

        # Initialise via parent
        super().__init__(default=default, assert_unique=False, **kwargs)

        # Check and set the target-related attributes
        if target_pdim is not None and target_name is not None:
            raise TypeError(
                "Got both `target_pdim` and `target_name` "
                "arguments, but only accepting one of them at the "
                "same time!"
            )

        elif target_name:
            # Save only the name of object to couple to. Resolved by ParamSpace
            self.target_name = target_name

        elif target_pdim:
            # Directly save the object to couple to
            self.target_pdim = target_pdim

        else:
            raise TypeError(
                "Expected either argument `target_pdim` or "
                "`target_name`, got neither."
            )

        # Add further initialization kwargs, needed for yaml representation
        self._init_kwargs["target_name"] = target_name
        self._init_kwargs["target_pdim"] = target_pdim
        # NOTE The other kwargs are already stored in the base class __init__

        # Done now.
        log.debug("CoupledParamDim initialised.")

    # Magic methods ...........................................................

    def __len__(self) -> int:
        """Returns the effective length of the parameter dimension, i.e. the
        number of values that will be iterated over; corresponds to that of
        the target ParamDim

        Returns:
            int: The number of values to be iterated over
        """
        return len(self.target_pdim)

    # Public API ..............................................................
    # These are needed by the ParamSpace class to have more control over the
    # iteration. Here, the parent class' behaviour is overwritten as the
    # CoupledParamDim's state and iteration should depend completely on that of
    # the target ParamDim...
    # TODO these should allow standalone iteration as well!

    def enter_iteration(self) -> None:
        """Does nothing, as state has no effect for CoupledParamDim"""

    def iterate_state(self) -> None:
        """Does nothing, as state has no effect for CoupledParamDim"""

    def reset(self) -> None:
        """Does nothing, as state has no effect for CoupledParamDim"""

    # Properties that only the CoupledParamDim has ............................

    @property
    def target_name(self) -> Union[str, Sequence[str]]:
        """The ParamDim object this CoupledParamDim couples to."""
        return self._target_name

    @target_name.setter
    def target_name(self, target_name: Union[str, Sequence[str]]):
        """Sets the target name, ensuring it to be a valid string or key
        sequence.
        """
        # Check if it is even reasonable to set it
        if self._target_name is not None:
            raise ValueError("Target name cannot be changed!")

        elif self._target_pdim is not None:
            raise ValueError(
                "Target name cannot be changed after the target "
                "object is already set!"
            )

        # Make sure it is of valid type
        elif not isinstance(target_name, (tuple, list, str)):
            raise TypeError(
                f"Argument `target_name` should be a tuple or list (i.e., a "
                f"key sequence) or a string! "
                f"Was {type(target_name)}: {target_name}"
            )

        elif isinstance(target_name, list):
            target_name = tuple(target_name)

        self._target_name = target_name

    @property
    def _target_name_as_list(self) -> Union[str, List[str]]:
        """For the safe yaml representer, the target_name cannot be a tuple.

        This property returns it as str or list of strings.
        """
        if self.target_name is None or isinstance(self.target_name, str):
            return self.target_name
        return list(self.target_name)

    @property
    def target_pdim(self) -> ParamDim:
        """The ParamDim object this CoupledParamDim couples to."""
        if self._target_pdim is None:
            raise ValueError(
                "The coupling target has not been set! Either "
                "set the `target_pdim` to a ParamDim object or "
                "incorporate this CoupledParamDim into a "
                "ParamSpace to resolve its coupling target using "
                "the given `target_name` attribute."
            )

        return self._target_pdim

    @target_pdim.setter
    def target_pdim(self, pdim: ParamDim):
        """Set the target ParamDim"""
        if not isinstance(pdim, ParamDim):
            raise TypeError(
                f"Target of CoupledParamDim needs to be of type "
                f"ParamDim, was {type(pdim)}!"
            )

        elif (
            not self._use_coupled_values and self.num_values != pdim.num_values
        ):
            raise ValueError(
                f"The lengths of the value sequences of target ParamDim and "
                f"this CoupledParamDim need to match, were: "
                f"{pdim.num_values} and {self.num_values}, respectively."
            )

        self._target_pdim = pdim
        log.debug("Set CoupledParamDim target.")

    # Properties that need to relay to the coupled ParamDim ...................

    @property
    def default(self):
        """The default value.

        Returns:
            the default value this parameter dimension can take.

        Raises:
            RuntimeError: If no ParamDim was associated yet
        """
        if self._use_coupled_default:
            return self.target_pdim.default

        return self._default

    @property
    def values(self) -> tuple:
        """The values that are iterated over.

        If self._use_coupled_values is set, will be those of the coupled pdim.

        Returns:
            tuple: The values of this CoupledParamDim or the target ParamDim
        """
        if self._use_coupled_values:
            return self.target_pdim.values

        return self._vals

    @property
    def state(self) -> int:
        """The current iterator state of the target ParamDim

        Returns:
            Union[int, None]: The state of the iterator; if it is None, the
                ParamDim is not inside an iteration.
        """
        return self.target_pdim.state

    @property
    def current_value(self):
        """If in an iteration, returns the value according to the current
        state. Otherwise, returns the default value.
        """
        if self.state == 0:
            return self.default
        return self.values[self.state - 1]

    @property
    def mask(self) -> Union[bool, Tuple[bool]]:
        """Return the coupled object's mask value"""
        return self.target_pdim.mask
