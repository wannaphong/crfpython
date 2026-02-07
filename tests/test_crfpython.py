"""
Tests for CRFPython core functionality.
"""

import pytest
import tempfile
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import crfpython


class TestAttribute:
    """Tests for the Attribute class."""
    
    def test_create_simple_attribute(self):
        """Test creating a simple attribute."""
        attr = crfpython.Attribute("word=hello")
        assert attr.name == "word=hello"
        assert attr.value == 1.0
    
    def test_create_weighted_attribute(self):
        """Test creating an attribute with a weight."""
        attr = crfpython.Attribute("length", 5.0)
        assert attr.name == "length"
        assert attr.value == 5.0
    
    def test_attribute_equality(self):
        """Test attribute equality comparison."""
        attr1 = crfpython.Attribute("word=hello", 1.0)
        attr2 = crfpython.Attribute("word=hello", 1.0)
        attr3 = crfpython.Attribute("word=world", 1.0)
        
        assert attr1 == attr2
        assert attr1 != attr3
    
    def test_attribute_string_representation(self):
        """Test string representation of attributes."""
        attr1 = crfpython.Attribute("feature")
        attr2 = crfpython.Attribute("feature", 2.5)
        
        assert str(attr1) == "feature"
        assert "2.5" in str(attr2)


class TestItem:
    """Tests for the Item class."""
    
    def test_create_empty_item(self):
        """Test creating an empty item."""
        item = crfpython.Item()
        assert len(item) == 0
    
    def test_append_attribute(self):
        """Test appending attributes to an item."""
        item = crfpython.Item()
        item.append(crfpython.Attribute("word=hello"))
        item.append(crfpython.Attribute("pos=NOUN"))
        
        assert len(item) == 2
    
    def test_append_string(self):
        """Test appending a string as an attribute."""
        item = crfpython.Item()
        item.append("word=hello")
        
        assert len(item) == 1
        assert item[0].name == "word=hello"
    
    def test_append_tuple(self):
        """Test appending a tuple as an attribute."""
        item = crfpython.Item()
        item.append(("length", 5.0))
        
        assert len(item) == 1
        assert item[0].name == "length"
        assert item[0].value == 5.0
    
    def test_iterate_over_item(self):
        """Test iterating over attributes in an item."""
        item = crfpython.Item()
        item.append("feature1")
        item.append("feature2")
        
        features = [attr.name for attr in item]
        assert features == ["feature1", "feature2"]


class TestItemSequence:
    """Tests for the ItemSequence class."""
    
    def test_create_empty_sequence(self):
        """Test creating an empty sequence."""
        seq = crfpython.ItemSequence()
        assert len(seq) == 0
    
    def test_append_items(self):
        """Test appending items to a sequence."""
        seq = crfpython.ItemSequence()
        
        item1 = crfpython.Item()
        item1.append("word=hello")
        seq.append(item1)
        
        item2 = crfpython.Item()
        item2.append("word=world")
        seq.append(item2)
        
        assert len(seq) == 2
    
    def test_iterate_over_sequence(self):
        """Test iterating over items in a sequence."""
        seq = crfpython.ItemSequence()
        
        for _ in range(3):
            item = crfpython.Item()
            item.append("feature")
            seq.append(item)
        
        count = 0
        for item in seq:
            count += 1
        
        assert count == 3


class TestTrainer:
    """Tests for the Trainer class."""
    
    def test_create_trainer(self):
        """Test creating a trainer."""
        trainer = crfpython.Trainer()
        assert trainer.algorithm == 'lbfgs'
    
    def test_create_trainer_with_algorithm(self):
        """Test creating a trainer with a specific algorithm."""
        trainer = crfpython.Trainer(algorithm='l2sgd')
        assert trainer.algorithm == 'l2sgd'
    
    def test_append_training_data(self):
        """Test appending training data."""
        trainer = crfpython.Trainer()
        
        xseq = crfpython.ItemSequence()
        item = crfpython.Item()
        item.append("feature")
        xseq.append(item)
        
        yseq = ['LABEL']
        
        trainer.append(xseq, yseq)
        assert len(trainer.data) == 1
    
    def test_set_and_get_params(self):
        """Test setting and getting parameters."""
        trainer = crfpython.Trainer()
        
        params = {'c1': 0.1, 'c2': 0.5}
        trainer.set_params(params)
        
        retrieved_params = trainer.get_params()
        assert retrieved_params['c1'] == 0.1
        assert retrieved_params['c2'] == 0.5
    
    def test_training_simple_model(self):
        """Test training a simple model."""
        trainer = crfpython.Trainer(verbose=False)
        
        # Create simple training data
        for _ in range(3):
            xseq = crfpython.ItemSequence()
            
            item1 = crfpython.Item()
            item1.append("word=hello")
            xseq.append(item1)
            
            item2 = crfpython.Item()
            item2.append("word=world")
            xseq.append(item2)
            
            yseq = ['GREETING', 'NOUN']
            trainer.append(xseq, yseq)
        
        # Train model
        with tempfile.NamedTemporaryFile(delete=False, suffix='.crfsuite') as f:
            model_file = f.name
        
        try:
            trainer.train(model_file)  # train() returns None now
            assert os.path.exists(model_file)
            assert trainer.algorithm in ['lbfgs', 'l2sgd', 'ap', 'pa', 'arow']
        finally:
            if os.path.exists(model_file):
                os.unlink(model_file)


class TestTagger:
    """Tests for the Tagger class."""
    
    def test_create_tagger(self):
        """Test creating a tagger."""
        tagger = crfpython.Tagger()
        assert tagger._model is None
    
    def test_train_and_tag(self):
        """Test training a model and using it for tagging."""
        # Train a model
        trainer = crfpython.Trainer(verbose=False)
        
        # Add multiple training instances for better model
        for _ in range(5):
            xseq = crfpython.ItemSequence()
            
            item1 = crfpython.Item()
            item1.append("word=hello")
            item1.append("pos=INTJ")
            xseq.append(item1)
            
            item2 = crfpython.Item()
            item2.append("word=world")
            item2.append("pos=NOUN")
            xseq.append(item2)
            
            yseq = ['GREETING', 'NOUN']
            trainer.append(xseq, yseq)
        
        # Train and save model
        with tempfile.NamedTemporaryFile(delete=False, suffix='.crfsuite') as f:
            model_file = f.name
        
        try:
            trainer.train(model_file)
            
            # Load model
            tagger = crfpython.Tagger()
            tagger.open(model_file)
            
            # Create test sequence
            test_seq = crfpython.ItemSequence()
            
            item1 = crfpython.Item()
            item1.append("word=hello")
            item1.append("pos=INTJ")
            test_seq.append(item1)
            
            item2 = crfpython.Item()
            item2.append("word=world")
            item2.append("pos=NOUN")
            test_seq.append(item2)
            
            # Tag sequence
            labels = tagger.tag(test_seq)
            
            assert len(labels) == 2
            assert all(isinstance(label, str) for label in labels)
            
            # Check available labels
            all_labels = tagger.labels()
            assert len(all_labels) > 0
            assert 'GREETING' in all_labels or 'NOUN' in all_labels
            
        finally:
            if os.path.exists(model_file):
                os.unlink(model_file)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
