"""
Utility functions for working with CRFPython.

This module provides helper functions for feature extraction, data reading,
and other common tasks when working with CRF models.
"""

import sys
from typing import List, Dict, Tuple, Callable


def apply_templates(X, templates):
    """
    Generate features for an item sequence by applying feature templates.
    
    A feature template consists of a tuple of (name, offset) pairs,
    where name and offset specify a field name and offset from which
    the template extracts a feature value. Generated features are stored
    in the 'F' field of each item in the sequence.
    
    Args:
        X: List of dictionaries representing items in a sequence
        templates: List of feature templates (tuples of (field, offset) pairs)
    
    Examples:
        >>> X = [{'w': 'hello', 'F': []}, {'w': 'world', 'F': []}]
        >>> templates = [(('w', 0),), (('w', -1),), (('w', 1),)]
        >>> apply_templates(X, templates)
    """
    for template in templates:
        name = '|'.join(['%s[%d]' % (f, o) for f, o in template])
        X_len = len(X)
        for t in range(X_len):
            values = []
            for field, offset in template:
                p = t + offset
                if p < 0 or p >= X_len:
                    values = []
                    break
                values.append(X[p][field])
            if values:
                X[t]['F'].append('%s=%s' % (name, '|'.join(values)))


def readiter(fi, names, sep=' '):
    """
    Return an iterator for item sequences read from a file object.
    
    This function reads sequences from a file object, where each line
    represents an item and blank lines separate sequences. Each line
    is split by the separator character and values are stored in a
    dictionary with keys from the names parameter.
    
    Args:
        fi: File object to read from
        names: Tuple of field names for each column
        sep: Separator character (default: space)
        
    Yields:
        List of dictionaries representing a sequence
        
    Examples:
        >>> with open('data.txt') as f:
        ...     for seq in readiter(f, ('w', 'pos', 'y')):
        ...         print(len(seq))
    """
    X = []
    for line in fi:
        line = line.strip('\n')
        if not line:
            if X:
                yield X
                X = []
        else:
            fields = line.split(sep)
            if len(fields) < len(names):
                raise ValueError(
                    'Too few fields (%d) for %r\n%s' % (len(fields), names, line))
            item = {'F': []}  # 'F' is reserved for features
            for i in range(len(names)):
                item[names[i]] = fields[i]
            X.append(item)
    
    # Don't forget the last sequence if file doesn't end with blank line
    if X:
        yield X


def escape(src):
    """
    Escape colon characters from feature names.
    
    Args:
        src: A feature name string
        
    Returns:
        The feature name with colons escaped
    """
    return src.replace(':', '__COLON__')


def output_features(fo, X, field=''):
    """
    Output features (and reference labels) of a sequence in CRFsuite format.
    
    For each item in the sequence, this function writes a reference label
    (if field is provided) and features to the output file.
    
    Args:
        fo: File object to write to
        X: List of items (dictionaries)
        field: Field name for reference labels (empty string to omit labels)
    """
    for t in range(len(X)):
        if field:
            fo.write('%s' % X[t][field])
        for a in X[t]['F']:
            if isinstance(a, str):
                fo.write('\t%s' % escape(a))
            else:
                fo.write('\t%s:%f' % (escape(a[0]), a[1]))
        fo.write('\n')
    fo.write('\n')


def to_crfpython(X):
    """
    Convert an item sequence into CRFPython objects.
    
    Args:
        X: List of items (dictionaries) with 'F' field containing features
        
    Returns:
        ItemSequence object compatible with CRFPython
    """
    import crfpython
    
    xseq = crfpython.ItemSequence()
    for x in X:
        item = crfpython.Item()
        for f in x['F']:
            if isinstance(f, str):
                item.append(crfpython.Attribute(escape(f)))
            else:
                item.append(crfpython.Attribute(escape(f[0]), f[1]))
        xseq.append(item)
    return xseq


def main(feature_extractor, fields='w pos y', sep=' '):
    """
    Main function for command-line usage.
    
    This function provides a standard interface for processing data with
    CRFPython. It reads data from stdin, applies feature extraction,
    and either outputs features or performs tagging.
    
    Args:
        feature_extractor: Function to extract features from a sequence
        fields: Space-separated field names (default: 'w pos y')
        sep: Field separator in input (default: space)
    """
    import optparse
    
    fi = sys.stdin
    fo = sys.stdout
    
    # Parse command-line arguments
    parser = optparse.OptionParser(usage="""usage: %prog [options]
This utility reads a data set from STDIN, and outputs attributes to STDOUT.
Each line of a data set must consist of field values separated by SEPARATOR
characters. The names and order of field values can be specified by -f option.
The separator character can be specified with -s option. Instead of outputting
attributes, this utility tags the input data when a model file is specified by
-t option.""")
    
    parser.add_option(
        '-t', dest='model',
        help='tag the input using the model'
    )
    parser.add_option(
        '-f', dest='fields', default=fields,
        help='specify field names of input data [default: "%default"]'
    )
    parser.add_option(
        '-s', dest='separator', default=sep,
        help='specify the separator of columns of input data [default: "%default"]'
    )
    
    (options, args) = parser.parse_args()
    
    # The fields of input
    F = options.fields.split(' ')
    
    if not options.model:
        # Output features mode
        for X in readiter(fi, F, options.separator):
            feature_extractor(X)
            output_features(fo, X, 'y')
    else:
        # Tagging mode
        import crfpython
        tagger = crfpython.Tagger()
        tagger.open(options.model)
        
        for X in readiter(fi, F, options.separator):
            # Extract features
            feature_extractor(X)
            xseq = to_crfpython(X)
            yseq = tagger.tag(xseq)
            
            # Output results
            for t in range(len(X)):
                v = X[t]
                fo.write('\t'.join([v[f] for f in F]))
                fo.write('\t%s\n' % yseq[t])
            fo.write('\n')
