"""
Attribute class for CRFsuite.

An attribute represents a feature with an optional weight.
"""


class Attribute:
    """
    An attribute (feature) with an optional weight.
    
    Attributes consist of a name (string) and an optional numerical value (weight).
    If no value is provided, it defaults to 1.0.
    
    Args:
        name (str): The name/identifier of the attribute
        value (float, optional): The weight/value of the attribute. Defaults to 1.0
        
    Examples:
        >>> attr1 = Attribute("word=hello")
        >>> attr1.name
        'word=hello'
        >>> attr1.value
        1.0
        >>> attr2 = Attribute("length", 5.0)
        >>> attr2.value
        5.0
    """
    
    def __init__(self, name, value=1.0):
        """
        Initialize an Attribute.
        
        Args:
            name (str): The name/identifier of the attribute
            value (float, optional): The weight/value of the attribute. Defaults to 1.0
        """
        if not isinstance(name, str):
            raise TypeError("Attribute name must be a string")
        if not isinstance(value, (int, float)):
            raise TypeError("Attribute value must be a number")
            
        self.name = name
        self.value = float(value)
    
    def __repr__(self):
        """String representation of the Attribute."""
        if self.value == 1.0:
            return f"Attribute('{self.name}')"
        return f"Attribute('{self.name}', {self.value})"
    
    def __str__(self):
        """String conversion of the Attribute."""
        if self.value == 1.0:
            return self.name
        return f"{self.name}:{self.value}"
    
    def __eq__(self, other):
        """Check equality with another Attribute."""
        if not isinstance(other, Attribute):
            return False
        return self.name == other.name and self.value == other.value
    
    def __hash__(self):
        """Hash function for Attribute."""
        return hash((self.name, self.value))
