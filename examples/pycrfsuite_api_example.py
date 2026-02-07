#!/usr/bin/env python
"""
Example demonstrating python-crfsuite compatible API.

This example shows the dict-based feature input and enhanced API
that is compatible with python-crfsuite (scrapinghub).
"""

import crfpython


def main():
    print("=" * 70)
    print("CRFPython - python-crfsuite Compatible API Example")
    print("=" * 70)
    
    # 1. Create training data using dict format (python-crfsuite style)
    print("\n1. Creating training data with dict-based features...")
    
    train_data = [
        # Each sequence is a list of dicts with features
        ([
            {"word": "Melbourne", "pos": "NNP", "is_capitalized": True},
            {"word": "is", "pos": "VBZ", "is_capitalized": False},
            {"word": "in", "pos": "IN", "is_capitalized": False},
            {"word": "Australia", "pos": "NNP", "is_capitalized": True},
        ], ["B-LOC", "O", "O", "B-LOC"]),
        
        ([
            {"word": "Paris", "pos": "NNP", "is_capitalized": True},
            {"word": "is", "pos": "VBZ", "is_capitalized": False},
            {"word": "beautiful", "pos": "JJ", "is_capitalized": False},
        ], ["B-LOC", "O", "O"]),
        
        ([
            {"word": "John", "pos": "NNP", "is_capitalized": True},
            {"word": "works", "pos": "VBZ", "is_capitalized": False},
            {"word": "in", "pos": "IN", "is_capitalized": False},
            {"word": "London", "pos": "NNP", "is_capitalized": True},
        ], ["B-PER", "O", "O", "B-LOC"]),
    ]
    
    print(f"   Created {len(train_data)} training sequences")
    
    # 2. Initialize trainer with algorithm and parameters
    print("\n2. Initializing trainer with python-crfsuite API...")
    
    trainer = crfpython.Trainer(algorithm='lbfgs', verbose=False)
    
    # Set parameters using python-crfsuite style
    trainer.set_params({
        'c1': 0.1,
        'c2': 0.1,
        'max_iterations': 50,
    })
    
    print(f"   Algorithm: {trainer._algorithm}")
    print(f"   Parameters: c1={trainer.get('c1')}, c2={trainer.get('c2')}")
    
    # 3. Add training data (accepts dicts directly)
    print("\n3. Adding training data...")
    
    for xseq, yseq in train_data:
        trainer.append(xseq, yseq)  # Direct dict input
    
    print(f"   Added {len(trainer.data)} instances")
    
    # 4. Train model
    print("\n4. Training model...")
    
    import tempfile
    import os
    model_file = os.path.join(tempfile.gettempdir(), 'ner_model.crfsuite')
    trainer.train(model_file)
    
    print(f"   Model saved to: {model_file}")
    
    # 5. Use the tagger with python-crfsuite API
    print("\n5. Using trained model...")
    
    tagger = crfpython.Tagger()
    
    # Use context manager (python-crfsuite style)
    with tagger.open(model_file):
        # Test sequence (dict format)
        test_seq = [
            {"word": "Berlin", "pos": "NNP", "is_capitalized": True},
            {"word": "is", "pos": "VBZ", "is_capitalized": False},
            {"word": "great", "pos": "JJ", "is_capitalized": False},
        ]
        
        # Set sequence and tag
        tagger.set(test_seq)
        labels = tagger.tag()  # No argument needed after set()
        
        print(f"   Input:  Berlin is great")
        print(f"   Labels: {' '.join(labels)}")
        
        # Get probability of predicted sequence
        prob = tagger.probability(labels)
        print(f"   P(sequence): {prob:.6e}")
        
        # Get marginal probability
        marginal = tagger.marginal("B-LOC", 0)
        print(f"   P(B-LOC at pos 0): {marginal:.4f}")
        
        # Show model info
        info = tagger.info()
        print(f"\n   Model info:")
        print(f"     - Algorithm: {info['algorithm']}")
        print(f"     - Labels: {info['labels']}")
        print(f"     - Features: {info['num_features']}")
    
    # Clean up
    os.unlink(model_file)
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
