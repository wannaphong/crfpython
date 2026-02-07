# CRFPython

[![Tests](https://github.com/wannaphong/crfpython/actions/workflows/tests.yml/badge.svg)](https://github.com/wannaphong/crfpython/actions/workflows/tests.yml)

A Python implementation of CRFsuite - Conditional Random Fields for labeling sequential data.

## Overview

CRFPython is a pure Python port of [CRFsuite](https://github.com/chokkan/crfsuite), a fast implementation of Conditional Random Fields (CRFs) for labeling sequential data. This library provides a Python-native implementation that is easy to install, understand, and modify.

## Features

- **Pure Python implementation**: No C dependencies required
- **Compatible API**: Designed to be similar to the original CRFsuite Python bindings
- **Multiple training algorithms**: Support for LBFGS, L2SGD, Averaged Perceptron, and more
- **Sequence labeling**: Perfect for NER, POS tagging, chunking, and other sequence labeling tasks
- **Easy to use**: Simple, intuitive API for training and tagging

## Installation

```bash
pip install -e .
```

Or for development:

```bash
pip install -e ".[dev]"
```

## Quick Start

Here's a simple example of training and using a CRF model:

```python
import crfpython

# Create training data
trainer = crfpython.Trainer(algorithm='lbfgs', verbose=True)

# Create a sequence (e.g., a sentence with POS tags)
xseq = crfpython.ItemSequence()

# Add items (tokens) with features
item1 = crfpython.Item()
item1.append(crfpython.Attribute("word=Hello"))
item1.append(crfpython.Attribute("pos=INTJ"))
xseq.append(item1)

item2 = crfpython.Item()
item2.append(crfpython.Attribute("word=world"))
item2.append(crfpython.Attribute("pos=NOUN"))
xseq.append(item2)

# Labels for the sequence
yseq = ['GREETING', 'OBJECT']

# Add to trainer
trainer.append(xseq, yseq)

# Train model
trainer.train('model.crfsuite')

# Load model and tag new sequences
tagger = crfpython.Tagger()
tagger.open('model.crfsuite')
labels = tagger.tag(xseq)
print(labels)  # ['GREETING', 'OBJECT']
```

### python-crfsuite Compatible API

CRFPython also supports the [python-crfsuite](https://github.com/scrapinghub/python-crfsuite) API for easy migration:

```python
import crfpython

# Dict-based feature format (python-crfsuite style)
train_data = [
    ([{"word": "Paris", "pos": "NNP", "is_cap": True},
      {"word": "is", "pos": "VBZ", "is_cap": False}],
     ["B-LOC", "O"]),
]

# Initialize trainer
trainer = crfpython.Trainer(algorithm='lbfgs')
trainer.set_params({'c1': 0.1, 'c2': 0.1, 'max_iterations': 100})

# Add training data (accepts dicts directly)
for xseq, yseq in train_data:
    trainer.append(xseq, yseq)

# Train model
trainer.train('model.crfsuite')

# Tag with context manager (python-crfsuite style)
tagger = crfpython.Tagger()
with tagger.open('model.crfsuite'):
    # Set sequence and tag
    test_seq = [{"word": "London", "pos": "NNP", "is_cap": True}]
    tagger.set(test_seq)
    labels = tagger.tag()  # No argument needed after set()
    
    # Get probabilities
    prob = tagger.probability(labels)
    marginal = tagger.marginal("B-LOC", 0)
    
    # Inspect model
    info = tagger.info()
    print(f"Model has {info['num_labels']} labels")
```

**Supported python-crfsuite features:**
- Dict-based feature input with nested dicts: `{"prefix": ["p1", "p2"]}`
- Boolean/string values: `{"is_cap": True}`, `{"pos": "NOUN"}`
- `Trainer.select()`, `params()`, `set()`, `get()`, `help()`
- `Tagger.set()`, `probability()`, `marginal()`, `dump()`, `info()`
- Context manager support: `with tagger.open()`

## API Reference

### Core Classes

#### `Attribute(name, value=1.0)`
Represents a feature with an optional weight.

- `name`: Feature name (string)
- `value`: Feature weight (float, default 1.0)

#### `Item()`
Represents a single item (token) in a sequence with its features.

- `append(attr)`: Add an attribute to this item

#### `ItemSequence(pyseq=None)`
Represents a sequence of items.

Can be initialized with:
- Empty: `ItemSequence()`
- Item objects: `ItemSequence([item1, item2])`
- Dicts (python-crfsuite style): `ItemSequence([{"word": "hello", "pos": "NOUN"}])`
- Lists: `ItemSequence([["feature1", "feature2"]])`

Methods:
- `append(item)`: Add an item (accepts Item, dict, or list)
- `items_as_dicts()`: Return items as list of dicts

#### `Trainer(algorithm=None, params=None, verbose=True)`
Used for training CRF models.

**Training methods:**
- `append(xseq, yseq, group=0)`: Add a training instance (accepts ItemSequence, dicts, or lists)
- `train(model, holdout=-1)`: Train and save the model
- `select(algorithm, type='crf1d')`: Initialize training algorithm

**Parameter methods (python-crfsuite compatible):**
- `params()`: Get list of available parameter names
- `set_params(params)`: Set multiple parameters
- `get_params()`: Get all current parameters
- `set(name, value)`: Set individual parameter
- `get(name)`: Get individual parameter
- `help(name)`: Get parameter help text

Supported algorithms:
- `'lbfgs'`: Limited-memory BFGS
- `'l2sgd'`: L2-regularized SGD
- `'ap'`: Averaged Perceptron
- `'pa'`: Passive Aggressive
- `'arow'`: Adaptive Regularization of Weights

#### `Tagger()`
Used for tagging sequences with a trained model.

**Tagging methods:**
- `open(name)`: Load a trained model (returns context manager)
- `close()`: Close the model
- `set(xseq)`: Set current sequence (accepts ItemSequence, dicts, or lists)
- `tag(xseq=None)`: Tag a sequence (uses current if xseq is None)

**Probability methods (python-crfsuite compatible):**
- `labels()`: Get all possible labels
- `probability(yseq)`: Compute P(yseq|xseq) for current sequence
- `marginal(label, pos)`: Compute marginal P(label at position)

**Inspection methods:**
- `info()`: Get model information (dict)
- `dump(filename=None)`: Dump model in text format

## Examples

See the `examples/` directory for more detailed examples:

- `simple_example.py`: Basic usage demonstration
- `pycrfsuite_api_example.py`: python-crfsuite compatible API example
- `crfutils.py`: Utility functions for feature extraction and data processing

To run examples:

```bash
cd examples
python simple_example.py
python pycrfsuite_api_example.py
```

## Training Algorithms

CRFPython supports several training algorithms:

### L-BFGS (Default)
Limited-memory BFGS with L1/L2 regularization.

```python
trainer = crfpython.Trainer(algorithm='lbfgs')
trainer.set_params({
    'c1': 0.0,  # L1 regularization
    'c2': 1.0,  # L2 regularization
    'max_iterations': 100,
})
```

### L2-SGD
Stochastic Gradient Descent with L2 regularization.

```python
trainer = crfpython.Trainer(algorithm='l2sgd')
trainer.set_params({
    'c2': 1.0,
    'max_iterations': 100,
})
```

### Averaged Perceptron
Fast online learning algorithm.

```python
trainer = crfpython.Trainer(algorithm='ap')
```

## Use Cases

CRFPython is suitable for various sequence labeling tasks:

- **Named Entity Recognition (NER)**: Identify and classify named entities in text
- **Part-of-Speech (POS) Tagging**: Assign grammatical tags to words
- **Chunking**: Identify phrases and syntactic units
- **Text Segmentation**: Break text into meaningful segments
- **Information Extraction**: Extract structured information from unstructured text

## Acknowledgments

This project is a Python port of [CRFsuite](https://github.com/chokkan/crfsuite) by Naoaki Okazaki. The original CRFsuite is a fast, efficient C implementation of CRFs. This Python port aims to provide similar functionality with the benefits of pure Python code.

## Testing

CRFPython includes comprehensive test coverage with both pytest and unittest frameworks.

### Running Tests

Run all tests with pytest:

```bash
pytest tests/ -v
```

Run with coverage report:

```bash
pytest tests/ --cov=crfpython --cov-report=term --cov-report=html
```

Run with unittest:

```bash
python -m unittest discover tests/
```

Run specific test file:

```bash
pytest tests/test_crfpython.py -v
python -m unittest tests.test_unittest
```

### Test Structure

- `tests/test_crfpython.py` - Original pytest-style tests (19 tests)
- `tests/test_unittest.py` - Unittest-style tests (30 tests)
- Total: 49 tests with ~84% code coverage

### Continuous Integration

Tests are automatically run on GitHub Actions for:
- Python versions: 3.8, 3.9, 3.10, 3.11, 3.12
- Operating systems: Ubuntu, Windows, macOS
- Linting with flake8 and black
- Example scripts validation

See `.github/workflows/tests.yml` for the full CI configuration.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

The original CRFsuite is licensed under the BSD license. See the original repository for details.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## Citation

If you use CRFPython in your research, please cite the original CRFsuite:

```
Naoaki Okazaki, CRFsuite: a fast implementation of Conditional Random Fields (CRFs)
http://www.chokkan.org/software/crfsuite/
```