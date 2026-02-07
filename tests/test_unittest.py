"""
Additional unittest-style tests for CRFPython.

This file provides unittest.TestCase-based tests for compatibility
with unittest test runners.
"""

import unittest
import tempfile
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import crfpython


class TestAttributeUnittest(unittest.TestCase):
    """Unittest-style tests for the Attribute class."""
    
    def test_create_attribute_with_defaults(self):
        """Test creating an attribute with default value."""
        attr = crfpython.Attribute("feature")
        self.assertEqual(attr.name, "feature")
        self.assertEqual(attr.value, 1.0)
    
    def test_create_attribute_with_custom_value(self):
        """Test creating an attribute with a custom value."""
        attr = crfpython.Attribute("weight", 0.5)
        self.assertEqual(attr.name, "weight")
        self.assertEqual(attr.value, 0.5)
    
    def test_attribute_type_validation(self):
        """Test that attribute validates input types."""
        with self.assertRaises(TypeError):
            crfpython.Attribute(123)  # Name must be string
        
        with self.assertRaises(TypeError):
            crfpython.Attribute("feature", "not_a_number")  # Value must be numeric
    
    def test_attribute_hash(self):
        """Test that attributes can be hashed."""
        attr1 = crfpython.Attribute("feature", 1.0)
        attr2 = crfpython.Attribute("feature", 1.0)
        
        # Should be able to use in sets
        attr_set = {attr1, attr2}
        self.assertEqual(len(attr_set), 1)  # Same attributes


class TestItemUnittest(unittest.TestCase):
    """Unittest-style tests for the Item class."""
    
    def test_empty_item(self):
        """Test creating an empty item."""
        item = crfpython.Item()
        self.assertEqual(len(item), 0)
        self.assertEqual(list(item), [])
    
    def test_append_multiple_types(self):
        """Test appending different attribute types."""
        item = crfpython.Item()
        
        # String
        item.append("feature1")
        self.assertEqual(len(item), 1)
        
        # Tuple
        item.append(("feature2", 2.0))
        self.assertEqual(len(item), 2)
        
        # Attribute object
        item.append(crfpython.Attribute("feature3", 3.0))
        self.assertEqual(len(item), 3)
    
    def test_item_indexing(self):
        """Test item indexing."""
        item = crfpython.Item()
        item.append("f1")
        item.append("f2")
        
        self.assertEqual(item[0].name, "f1")
        self.assertEqual(item[1].name, "f2")
    
    def test_item_clear(self):
        """Test clearing items."""
        item = crfpython.Item()
        item.append("feature")
        self.assertEqual(len(item), 1)
        
        item.clear()
        self.assertEqual(len(item), 0)


class TestItemSequenceUnittest(unittest.TestCase):
    """Unittest-style tests for the ItemSequence class."""
    
    def test_create_from_dicts(self):
        """Test creating ItemSequence from dicts (python-crfsuite style)."""
        data = [
            {"word": "hello", "pos": "INTJ"},
            {"word": "world", "pos": "NOUN"}
        ]
        seq = crfpython.ItemSequence(data)
        
        self.assertEqual(len(seq), 2)
        self.assertGreater(len(seq[0]), 0)
    
    def test_create_from_lists(self):
        """Test creating ItemSequence from lists."""
        data = [
            ["feature1", "feature2"],
            ["feature3", "feature4"]
        ]
        seq = crfpython.ItemSequence(data)
        
        self.assertEqual(len(seq), 2)
    
    def test_nested_dict_features(self):
        """Test nested dictionary features."""
        data = [{"prefix": {"p1": True, "p2": False}}]
        seq = crfpython.ItemSequence(data)
        
        # Should have features with colon separator
        items_dict = seq.items_as_dicts()
        self.assertTrue(any("prefix:" in key for key in items_dict[0].keys()))
    
    def test_items_as_dicts(self):
        """Test items_as_dicts method."""
        item = crfpython.Item()
        item.append(("feature1", 1.5))
        item.append(("feature2", 2.5))
        
        seq = crfpython.ItemSequence()
        seq.append(item)
        
        dicts = seq.items_as_dicts()
        self.assertEqual(len(dicts), 1)
        self.assertEqual(dicts[0]["feature1"], 1.5)
        self.assertEqual(dicts[0]["feature2"], 2.5)


class TestTrainerUnittest(unittest.TestCase):
    """Unittest-style tests for the Trainer class."""
    
    def test_trainer_initialization(self):
        """Test trainer initialization with different algorithms."""
        # Test algorithm initialization
        trainer1 = crfpython.Trainer(algorithm='lbfgs')
        self.assertEqual(trainer1.algorithm, 'lbfgs')
        
        trainer2 = crfpython.Trainer(algorithm='l2sgd')
        self.assertEqual(trainer2.algorithm, 'l2sgd')
        
        # Test that aliases work
        trainer3 = crfpython.Trainer(algorithm='ap')
        self.assertEqual(trainer3.algorithm, 'averaged-perceptron')
        
        trainer4 = crfpython.Trainer(algorithm='pa')
        self.assertEqual(trainer4.algorithm, 'passive-aggressive')
    
    def test_trainer_select_method(self):
        """Test selecting algorithms."""
        trainer = crfpython.Trainer()
        
        trainer.select('l2sgd')
        self.assertEqual(trainer.algorithm, 'l2sgd')
        
        trainer.select('ap')
        self.assertEqual(trainer.algorithm, 'averaged-perceptron')
    
    def test_parameter_management(self):
        """Test parameter get/set methods."""
        trainer = crfpython.Trainer()
        
        # Test set and get
        trainer.set('c1', 0.5)
        self.assertEqual(trainer.get('c1'), 0.5)
        
        # Test set_params
        trainer.set_params({'c2': 0.8, 'max_iterations': 50})
        self.assertEqual(trainer.get('c2'), 0.8)
        self.assertEqual(trainer.get('max_iterations'), 50)
        
        # Test get_params
        params = trainer.get_params()
        self.assertIn('c1', params)
        self.assertIn('c2', params)
    
    def test_params_list(self):
        """Test getting parameter list."""
        trainer = crfpython.Trainer()
        params = trainer.params()
        
        self.assertIsInstance(params, list)
        self.assertGreater(len(params), 0)
        self.assertIn('c1', params)
    
    def test_help_method(self):
        """Test parameter help method."""
        trainer = crfpython.Trainer()
        help_text = trainer.help('c1')
        
        self.assertIsInstance(help_text, str)
        self.assertGreater(len(help_text), 0)
    
    def test_append_with_dicts(self):
        """Test appending training data with dicts."""
        trainer = crfpython.Trainer()
        
        xseq = [{"word": "hello", "pos": "INTJ"}]
        yseq = ["GREETING"]
        
        trainer.append(xseq, yseq)
        self.assertEqual(len(trainer.data), 1)
    
    def test_training_with_minimal_data(self):
        """Test training with minimal data."""
        trainer = crfpython.Trainer(verbose=False)
        
        # Add minimal training data
        xseq = [{"word": "test"}]
        yseq = ["LABEL"]
        trainer.append(xseq, yseq)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.model') as f:
            model_file = f.name
        
        try:
            trainer.train(model_file)
            self.assertTrue(os.path.exists(model_file))
        finally:
            if os.path.exists(model_file):
                os.unlink(model_file)
    
    def test_clear_method(self):
        """Test clearing training data."""
        trainer = crfpython.Trainer()
        trainer.append([{"word": "test"}], ["LABEL"])
        
        self.assertEqual(len(trainer.data), 1)
        trainer.clear()
        self.assertEqual(len(trainer.data), 0)


class TestTaggerUnittest(unittest.TestCase):
    """Unittest-style tests for the Tagger class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Train a simple model for testing
        self.trainer = crfpython.Trainer(verbose=False)
        
        for _ in range(3):
            xseq = [
                {"word": "hello", "pos": "INTJ"},
                {"word": "world", "pos": "NOUN"}
            ]
            yseq = ["GREETING", "NOUN"]
            self.trainer.append(xseq, yseq)
        
        self.model_file = tempfile.NamedTemporaryFile(delete=False, suffix='.model').name
        self.trainer.train(self.model_file)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.model_file):
            os.unlink(self.model_file)
    
    def test_tagger_open_and_close(self):
        """Test opening and closing a model."""
        tagger = crfpython.Tagger()
        
        with tagger.open(self.model_file):
            labels = tagger.labels()
            self.assertIsInstance(labels, list)
            self.assertGreater(len(labels), 0)
        
        # After closing, model should be None
        self.assertIsNone(tagger._model)
    
    def test_tagger_set_and_tag(self):
        """Test set() and tag() without arguments."""
        tagger = crfpython.Tagger()
        tagger.open(self.model_file)
        
        test_seq = [{"word": "hello", "pos": "INTJ"}]
        tagger.set(test_seq)
        labels = tagger.tag()  # Should use the set sequence
        
        self.assertIsInstance(labels, list)
        self.assertEqual(len(labels), 1)
    
    def test_tagger_tag_with_arg(self):
        """Test tag() with sequence argument."""
        tagger = crfpython.Tagger()
        tagger.open(self.model_file)
        
        test_seq = [{"word": "world", "pos": "NOUN"}]
        labels = tagger.tag(test_seq)
        
        self.assertIsInstance(labels, list)
        self.assertEqual(len(labels), 1)
    
    def test_labels_method(self):
        """Test getting all labels."""
        tagger = crfpython.Tagger()
        tagger.open(self.model_file)
        
        labels = tagger.labels()
        self.assertIsInstance(labels, list)
        self.assertIn("GREETING", labels)
        self.assertIn("NOUN", labels)
    
    def test_info_method(self):
        """Test getting model info."""
        tagger = crfpython.Tagger()
        tagger.open(self.model_file)
        
        info = tagger.info()
        self.assertIsInstance(info, dict)
        self.assertIn('num_labels', info)
        self.assertIn('num_features', info)
        self.assertIn('algorithm', info)
    
    def test_probability_method(self):
        """Test probability computation."""
        tagger = crfpython.Tagger()
        tagger.open(self.model_file)
        
        test_seq = [{"word": "hello", "pos": "INTJ"}]
        tagger.set(test_seq)
        
        labels = tagger.tag()
        prob = tagger.probability(labels)
        
        self.assertIsInstance(prob, float)
        self.assertGreaterEqual(prob, 0.0)
    
    def test_marginal_method(self):
        """Test marginal probability computation."""
        tagger = crfpython.Tagger()
        tagger.open(self.model_file)
        
        test_seq = [{"word": "hello", "pos": "INTJ"}]
        tagger.set(test_seq)
        
        marginal = tagger.marginal("GREETING", 0)
        
        self.assertIsInstance(marginal, float)
        self.assertGreaterEqual(marginal, 0.0)
        self.assertLessEqual(marginal, 1.0)
    
    def test_dump_method(self):
        """Test model dumping."""
        tagger = crfpython.Tagger()
        tagger.open(self.model_file)
        
        dump_file = tempfile.NamedTemporaryFile(delete=False, suffix='.txt').name
        
        try:
            tagger.dump(dump_file)
            self.assertTrue(os.path.exists(dump_file))
            
            # Check that file has content
            with open(dump_file, 'r') as f:
                content = f.read()
                self.assertGreater(len(content), 0)
                self.assertIn("CRF Model Dump", content)
        finally:
            if os.path.exists(dump_file):
                os.unlink(dump_file)


class TestEndToEndUnittest(unittest.TestCase):
    """End-to-end integration tests."""
    
    def test_complete_workflow(self):
        """Test complete train and tag workflow."""
        # Create training data
        train_data = [
            ([{"word": "Paris", "pos": "NNP", "is_cap": True}], ["B-LOC"]),
            ([{"word": "London", "pos": "NNP", "is_cap": True}], ["B-LOC"]),
            ([{"word": "John", "pos": "NNP", "is_cap": True}], ["B-PER"]),
        ]
        
        # Train
        trainer = crfpython.Trainer(algorithm='lbfgs', verbose=False)
        trainer.set_params({'c1': 0.1, 'c2': 0.1, 'max_iterations': 20})
        
        for xseq, yseq in train_data:
            trainer.append(xseq, yseq)
        
        model_file = tempfile.NamedTemporaryFile(delete=False, suffix='.model').name
        
        try:
            trainer.train(model_file)
            self.assertTrue(os.path.exists(model_file))
            
            # Tag
            tagger = crfpython.Tagger()
            with tagger.open(model_file):
                test_seq = [{"word": "Berlin", "pos": "NNP", "is_cap": True}]
                labels = tagger.tag(test_seq)
                
                self.assertIsInstance(labels, list)
                self.assertEqual(len(labels), 1)
                self.assertIn(labels[0], ["B-LOC", "B-PER"])
        finally:
            if os.path.exists(model_file):
                os.unlink(model_file)
    
    def test_multiple_sequences(self):
        """Test with multiple sequences."""
        trainer = crfpython.Trainer(verbose=False)
        
        # Add multiple sequences
        for i in range(5):
            xseq = [
                {"word": f"word{i}", "idx": i},
                {"word": f"word{i+1}", "idx": i+1}
            ]
            yseq = ["LABEL1", "LABEL2"]
            trainer.append(xseq, yseq)
        
        self.assertEqual(len(trainer.data), 5)
        
        model_file = tempfile.NamedTemporaryFile(delete=False, suffix='.model').name
        
        try:
            trainer.train(model_file)
            
            tagger = crfpython.Tagger()
            tagger.open(model_file)
            
            test_seq = [{"word": "test", "idx": 99}]
            labels = tagger.tag(test_seq)
            
            self.assertIsInstance(labels, list)
            self.assertEqual(len(labels), 1)
        finally:
            if os.path.exists(model_file):
                os.unlink(model_file)


if __name__ == '__main__':
    unittest.main()
