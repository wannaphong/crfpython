"""
CRFPython - Python implementation of CRFsuite

A Python implementation of Conditional Random Fields (CRF) for labeling sequential data.
This is a port of the CRFsuite library (https://github.com/chokkan/crfsuite) to pure Python.

Copyright (c) 2024 Wannaphong Phatthiyaphaibun
Licensed under the Apache License, Version 2.0
"""

__version__ = "0.1.0"
__author__ = "Wannaphong Phatthiyaphaibun"
__license__ = "Apache-2.0"

from .attribute import Attribute
from .item import Item, ItemSequence
from .trainer import Trainer
from .tagger import Tagger

__all__ = [
    'Attribute',
    'Item',
    'ItemSequence',
    'Trainer',
    'Tagger',
]
