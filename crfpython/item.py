"""
Item and ItemSequence classes for CRFsuite.

An Item represents a single token/element in a sequence with its features (attributes).
An ItemSequence represents a complete sequence of Items.
"""

from typing import List, Union
from .attribute import Attribute


class Item:
    """
    An item in a sequence.
    
    An item consists of a list of attributes (features). Each item typically 
    represents a token or element in a sequential labeling task.
    
    Examples:
        >>> item = Item()
        >>> item.append(Attribute("word=hello"))
        >>> item.append(Attribute("pos=NOUN"))
        >>> len(item)
        2
    """
    
    def __init__(self):
        """Initialize an empty Item."""
        self.attributes = []
    
    def append(self, attr):
        """
        Append an attribute to this item.
        
        Args:
            attr: An Attribute object or tuple (name, value) or string name
        """
        if isinstance(attr, Attribute):
            self.attributes.append(attr)
        elif isinstance(attr, tuple) and len(attr) == 2:
            self.attributes.append(Attribute(attr[0], attr[1]))
        elif isinstance(attr, str):
            self.attributes.append(Attribute(attr))
        else:
            raise TypeError("Attribute must be an Attribute object, tuple (name, value), or string")
    
    def __len__(self):
        """Return the number of attributes in this item."""
        return len(self.attributes)
    
    def __iter__(self):
        """Iterate over attributes in this item."""
        return iter(self.attributes)
    
    def __getitem__(self, index):
        """Get attribute at specified index."""
        return self.attributes[index]
    
    def __repr__(self):
        """String representation of the Item."""
        return f"Item({len(self.attributes)} attributes)"
    
    def clear(self):
        """Remove all attributes from this item."""
        self.attributes.clear()


class ItemSequence:
    """
    A sequence of items.
    
    An ItemSequence represents a complete sequence for training or tagging,
    such as a sentence in POS tagging or NER tasks.
    
    Examples:
        >>> seq = ItemSequence()
        >>> item = Item()
        >>> item.append(Attribute("word=hello"))
        >>> seq.append(item)
        >>> len(seq)
        1
    """
    
    def __init__(self):
        """Initialize an empty ItemSequence."""
        self.items = []
    
    def append(self, item):
        """
        Append an item to this sequence.
        
        Args:
            item: An Item object
        """
        if not isinstance(item, Item):
            raise TypeError("Only Item objects can be appended to ItemSequence")
        self.items.append(item)
    
    def __len__(self):
        """Return the number of items in this sequence."""
        return len(self.items)
    
    def __iter__(self):
        """Iterate over items in this sequence."""
        return iter(self.items)
    
    def __getitem__(self, index):
        """Get item at specified index."""
        return self.items[index]
    
    def __repr__(self):
        """String representation of the ItemSequence."""
        return f"ItemSequence({len(self.items)} items)"
    
    def clear(self):
        """Remove all items from this sequence."""
        self.items.clear()
