#!/usr/bin/env python
"""
Simple example demonstrating CRFPython usage.

This example shows how to:
1. Create training data
2. Train a CRF model
3. Use the model to tag new sequences
"""

import crfpython


def create_sample_data():
    """Create some sample training data for demonstration."""
    # Training data: sequences of items with labels
    # Format: (ItemSequence, labels)
    
    training_data = []
    
    # Example 1: Simple sentence "I love Python"
    xseq1 = crfpython.ItemSequence()
    
    item1 = crfpython.Item()
    item1.append(crfpython.Attribute("word=I"))
    item1.append(crfpython.Attribute("pos=PRON"))
    xseq1.append(item1)
    
    item2 = crfpython.Item()
    item2.append(crfpython.Attribute("word=love"))
    item2.append(crfpython.Attribute("pos=VERB"))
    xseq1.append(item2)
    
    item3 = crfpython.Item()
    item3.append(crfpython.Attribute("word=Python"))
    item3.append(crfpython.Attribute("pos=NOUN"))
    item3.append(crfpython.Attribute("capitalized"))
    xseq1.append(item3)
    
    yseq1 = ['PRON', 'VERB', 'NOUN']
    training_data.append((xseq1, yseq1))
    
    # Example 2: Another sentence "She writes code"
    xseq2 = crfpython.ItemSequence()
    
    item1 = crfpython.Item()
    item1.append(crfpython.Attribute("word=She"))
    item1.append(crfpython.Attribute("pos=PRON"))
    item1.append(crfpython.Attribute("capitalized"))
    xseq2.append(item1)
    
    item2 = crfpython.Item()
    item2.append(crfpython.Attribute("word=writes"))
    item2.append(crfpython.Attribute("pos=VERB"))
    xseq2.append(item2)
    
    item3 = crfpython.Item()
    item3.append(crfpython.Attribute("word=code"))
    item3.append(crfpython.Attribute("pos=NOUN"))
    xseq2.append(item3)
    
    yseq2 = ['PRON', 'VERB', 'NOUN']
    training_data.append((xseq2, yseq2))
    
    return training_data


def main():
    import tempfile
    import os
    
    print("CRFPython Example")
    print("=" * 50)
    
    # Create sample training data
    print("\n1. Creating sample training data...")
    training_data = create_sample_data()
    print(f"   Created {len(training_data)} training sequences")
    
    # Create and configure trainer
    print("\n2. Training CRF model...")
    trainer = crfpython.Trainer(algorithm='lbfgs', verbose=True)
    
    # Add training instances
    for xseq, yseq in training_data:
        trainer.append(xseq, yseq)
    
    # Train model
    model_file = os.path.join(tempfile.gettempdir(), 'example_model.crfsuite')
    stats = trainer.train(model_file)
    print(f"   Training complete!")
    print(f"   Model saved to: {model_file}")
    
    # Create a tagger and load the model
    print("\n3. Loading model for tagging...")
    tagger = crfpython.Tagger()
    tagger.open(model_file)
    print("   Model loaded successfully")
    
    # Create a test sequence
    print("\n4. Tagging a test sequence...")
    test_seq = crfpython.ItemSequence()
    
    item1 = crfpython.Item()
    item1.append(crfpython.Attribute("word=I"))
    item1.append(crfpython.Attribute("pos=PRON"))
    test_seq.append(item1)
    
    item2 = crfpython.Item()
    item2.append(crfpython.Attribute("word=write"))
    item2.append(crfpython.Attribute("pos=VERB"))
    test_seq.append(item2)
    
    item3 = crfpython.Item()
    item3.append(crfpython.Attribute("word=Python"))
    item3.append(crfpython.Attribute("pos=NOUN"))
    item3.append(crfpython.Attribute("capitalized"))
    test_seq.append(item3)
    
    # Tag the sequence
    predicted_labels = tagger.tag(test_seq)
    
    print("   Test sequence: I write Python")
    print(f"   Predicted labels: {predicted_labels}")
    
    # Show available labels
    print("\n5. Model information:")
    labels = tagger.labels()
    print(f"   Available labels: {labels}")
    
    print("\n" + "=" * 50)
    print("Example complete!")


if __name__ == '__main__':
    main()
