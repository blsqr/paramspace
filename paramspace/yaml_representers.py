"""This module implements custom YAML representer functions"""
import logging
from typing import Iterable, Union

import ruamel.yaml

# Get logger
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


def _slice_representer(representer, node: slice):
    """Represents a Python slice object using the ``!slice`` YAML tag.

    Args:
        representer (ruamel.yaml.representer): The representer module
        node (slice): The node, i.e. a slice instance

    Returns:
        a yaml sequence that is able to recreate a slice
    """
    slice_args = [node.start, node.stop, node.step]
    return representer.represent_sequence("!slice", slice_args)


def _range_representer(representer, node: range):
    """Represents a Python range object using the ``!range`` YAML tag.

    Args:
        representer (ruamel.yaml.representer): The representer module
        node (range): The node, i.e. a range instance

    Returns:
        a yaml sequence that is able to recreate a range
    """
    range_args = [node.start, node.stop, node.step]
    return representer.represent_sequence("!range", range_args)
