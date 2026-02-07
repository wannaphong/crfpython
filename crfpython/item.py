"""
Item and ItemSequence classes for CRFsuite.

An Item represents a single token/element in a sequence with its features (attributes).
An ItemSequence represents a complete sequence of Items.
"""

from typing import List, Union, Dict, Any
from .attribute import Attribute


def _process_item_features(x, prefix=''):
    """
    Process item features in various formats and convert to Attribute list.
    
    Supports:
    - {"string_key": float_value} dicts
    - {"string_key": bool} dicts (True -> 1.0, False -> 0.0)
    - {"string_key": "string_value"} dicts (converts to "string_key:string_value")
    - ["string_key1", "string_key2"] lists
    - {"string_prefix": {...}} nested dicts
    - {"string_prefix": [...]} nested lists
    - {"string_prefix": set([...])} nested sets
    
    Args:
        x: Feature data in one of the supported formats
        prefix: Prefix to prepend to feature names (for nested structures)
        
    Returns:
        List of Attribute objects
    """
    attributes = []
    
    if isinstance(x, dict):
        for key, value in x.items():
            full_key = f"{prefix}:{key}" if prefix else key
            
            if isinstance(value, (dict, list, set)):
                # Nested structure - recurse
                attributes.extend(_process_item_features(value, full_key))
            elif isinstance(value, bool):
                # Boolean value
                attributes.append(Attribute(full_key, 1.0 if value else 0.0))
            elif isinstance(value, str):
                # String value - create "key:value" feature
                attributes.append(Attribute(f"{full_key}:{value}", 1.0))
            elif isinstance(value, (int, float)):
                # Numeric value
                attributes.append(Attribute(full_key, float(value)))
            else:
                # Default: treat as string
                attributes.append(Attribute(f"{full_key}:{str(value)}", 1.0))
                
    elif isinstance(x, (list, set)):
        for key in x:
            if isinstance(key, str):
                full_key = f"{prefix}:{key}" if prefix else key
                attributes.append(Attribute(full_key, 1.0))
            else:
                # If list contains non-strings, convert to string
                full_key = f"{prefix}:{str(key)}" if prefix else str(key)
                attributes.append(Attribute(full_key, 1.0))
    else:
        # Single string or other type
        key = str(x)
        full_key = f"{prefix}:{key}" if prefix else key
        attributes.append(Attribute(full_key, 1.0))
    
    return attributes


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
    
    This class is compatible with python-crfsuite API and supports various
    input formats:
    
    Examples:
        # Traditional API with Item objects
        >>> seq = ItemSequence()
        >>> item = Item()
        >>> item.append(Attribute("word=hello"))
        >>> seq.append(item)
        
        # python-crfsuite compatible API with dicts
        >>> seq = ItemSequence([
        ...     {"word": "hello", "pos": "INTJ"},
        ...     {"word": "world", "pos": "NOUN"}
        ... ])
        
        # Mixed dict formats
        >>> seq = ItemSequence([
        ...     {"word": "hello", "len": 5, "is_upper": False},
        ...     {"prefix": ["pre1", "pre2"], "suffix": "ing"}
        ... ])
    """
    
    def __init__(self, pyseq=None):
        """
        Initialize an ItemSequence.
        
        Args:
            pyseq: Optional sequence data. Can be:
                - None (empty sequence)
                - List of Item objects
                - List of dicts with features
                - List of lists of feature strings
                - Another ItemSequence
        """
        self.items = []
        
        if pyseq is not None:
            if isinstance(pyseq, ItemSequence):
                # Copy from another ItemSequence
                self.items = list(pyseq.items)
            else:
                # Process each element
                for x in pyseq:
                    if isinstance(x, Item):
                        self.items.append(x)
                    else:
                        # Convert to Item using python-crfsuite style processing
                        item = Item()
                        for attr in _process_item_features(x):
                            item.append(attr)
                        self.items.append(item)
    
    def append(self, item):
        """
        Append an item to this sequence.
        
        Args:
            item: An Item object, dict, or list of features
        """
        if isinstance(item, Item):
            self.items.append(item)
        else:
            # Convert dict/list to Item
            new_item = Item()
            for attr in _process_item_features(item):
                new_item.append(attr)
            self.items.append(new_item)
    
    def items_as_dicts(self):
        """
        Return items as list of dicts (python-crfsuite compatible format).
        
        Returns:
            List of dicts with {feature_name: feature_value}
        """
        result = []
        for item in self.items:
            item_dict = {}
            for attr in item:
                item_dict[attr.name] = attr.value
            result.append(item_dict)
        return result
    
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
        return f"<ItemSequence of size {len(self.items)}>"
    
    def clear(self):
        """Remove all items from this sequence."""
        self.items.clear()
