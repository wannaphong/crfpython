# CRFPython

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

## API Reference

### Core Classes

#### `Attribute(name, value=1.0)`
Represents a feature with an optional weight.

- `name`: Feature name (string)
- `value`: Feature weight (float, default 1.0)

#### `Item()`
Represents a single item (token) in a sequence with its features.

- `append(attr)`: Add an attribute to this item

#### `ItemSequence()`
Represents a sequence of items.

- `append(item)`: Add an item to this sequence

#### `Trainer(algorithm='lbfgs', verbose=False)`
Used for training CRF models.

- `append(xseq, yseq, group=0)`: Add a training instance
- `train(model_filename)`: Train and save the model
- `set_params(params)`: Set training parameters
- `get_params()`: Get current parameters

Supported algorithms:
- `'lbfgs'`: Limited-memory BFGS
- `'l2sgd'`: L2-regularized SGD
- `'ap'`: Averaged Perceptron
- `'pa'`: Passive Aggressive
- `'arow'`: Adaptive Regularization of Weights

#### `Tagger()`
Used for tagging sequences with a trained model.

- `open(filename)`: Load a trained model
- `tag(xseq)`: Tag a sequence and return predicted labels
- `labels()`: Get all possible labels
- `marginal(label, pos)`: Get marginal probability
- `probability(yseq)`: Get probability of a label sequence

## Examples

See the `examples/` directory for more detailed examples:

- `simple_example.py`: Basic usage demonstration
- `crfutils.py`: Utility functions for feature extraction and data processing

To run the simple example:

```bash
cd examples
python simple_example.py
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