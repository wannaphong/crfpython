#!/usr/bin/env python
"""
Example demonstrating the open_inmemory() functionality.

This example shows how to:
1. Train a CRF model
2. Load the model from file into memory
3. Use open_inmemory() to load the model from bytes
"""

import crfpython
import tempfile
import os


def main():
    print("=" * 60)
    print("CRFPython open_inmemory() Example")
    print("=" * 60)
    print()
    
    # Step 1: Create and train a model
    print("Step 1: Training a simple model...")
    trainer = crfpython.Trainer(verbose=False)
    
    # Add some training data
    training_examples = [
        ([{"word": "Paris", "pos": "NNP", "is_cap": True},
          {"word": "is", "pos": "VBZ", "is_cap": False},
          {"word": "beautiful", "pos": "JJ", "is_cap": False}],
         ["B-LOC", "O", "O"]),
        
        ([{"word": "London", "pos": "NNP", "is_cap": True},
          {"word": "is", "pos": "VBZ", "is_cap": False},
          {"word": "nice", "pos": "JJ", "is_cap": False}],
         ["B-LOC", "O", "O"]),
        
        ([{"word": "I", "pos": "PRP", "is_cap": True},
          {"word": "love", "pos": "VBP", "is_cap": False},
          {"word": "Python", "pos": "NNP", "is_cap": True}],
         ["O", "O", "B-MISC"]),
    ]
    
    for xseq, yseq in training_examples:
        trainer.append(xseq, yseq)
    
    # Train and save model
    with tempfile.NamedTemporaryFile(delete=False, suffix='.crfsuite') as f:
        model_file = f.name
    
    trainer.train(model_file)
    print(f"Model trained and saved to: {model_file}")
    print()
    
    # Step 2: Load model from file into memory
    print("Step 2: Loading model file into memory...")
    with open(model_file, 'rb') as f:
        model_bytes = f.read()
    
    print(f"Model loaded into memory: {len(model_bytes)} bytes")
    print()
    
    # Step 3: Use open_inmemory() to load the model
    print("Step 3: Using open_inmemory() to load model from bytes...")
    tagger = crfpython.Tagger()
    tagger.open_inmemory(model_bytes)
    print("Model loaded successfully from memory!")
    print()
    
    # Step 4: Use the model for tagging
    print("Step 4: Using the model to tag a new sequence...")
    test_seq = [
        {"word": "Tokyo", "pos": "NNP", "is_cap": True},
        {"word": "is", "pos": "VBZ", "is_cap": False},
        {"word": "great", "pos": "JJ", "is_cap": False}
    ]
    
    labels = tagger.tag(test_seq)
    print("Test sequence:")
    for token, label in zip(test_seq, labels):
        print(f"  {token['word']:15} -> {label}")
    print()
    
    # Step 5: Show model info
    print("Step 5: Model information:")
    info = tagger.info()
    print(f"  Number of labels: {info['num_labels']}")
    print(f"  Number of features: {info['num_features']}")
    print(f"  Labels: {', '.join(info['labels'])}")
    print()
    
    # Clean up
    tagger.close()
    os.unlink(model_file)
    
    print("=" * 60)
    print("Example completed successfully!")
    print()
    print("Key benefits of open_inmemory():")
    print("  - Load models from bytes instead of files")
    print("  - Useful for distributed systems or cloud storage")
    print("  - Can load models from network, databases, etc.")
    print("  - Compatible with python-crfsuite API")
    print("=" * 60)


if __name__ == "__main__":
    main()
