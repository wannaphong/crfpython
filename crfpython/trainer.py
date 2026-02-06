"""
Trainer class for training CRF models.

The Trainer class provides methods to train a CRF model using various algorithms.
"""

import pickle
from typing import List, Dict, Optional, Any
import numpy as np
from .item import ItemSequence


class Trainer:
    """
    CRF model trainer.
    
    The Trainer class is used to train CRF models with labeled sequence data.
    It supports various training algorithms including LBFGS, L2SGD, and others.
    
    Examples:
        >>> trainer = Trainer()
        >>> trainer.append(xseq, yseq)  # Add training data
        >>> trainer.train('model.crfsuite')  # Train and save model
    """
    
    def __init__(self, algorithm='lbfgs', verbose=False):
        """
        Initialize a Trainer.
        
        Args:
            algorithm (str): Training algorithm. Options: 'lbfgs', 'l2sgd', 'ap', 'pa', 'arow'
            verbose (bool): Whether to print training progress
        """
        self.algorithm = algorithm
        self.verbose = verbose
        self.data = []  # List of (xseq, yseq) tuples
        self.params = {}
        self._feature_dict = {}  # Maps feature names to indices
        self._label_dict = {}    # Maps label names to indices
        self._weights = None
        
        # Set default parameters based on algorithm
        self._set_default_params()
    
    def _set_default_params(self):
        """Set default parameters for the selected algorithm."""
        defaults = {
            'lbfgs': {
                'c1': 0.0,  # L1 regularization coefficient
                'c2': 1.0,  # L2 regularization coefficient
                'max_iterations': 100,
                'epsilon': 1e-5,
            },
            'l2sgd': {
                'c2': 1.0,
                'max_iterations': 100,
                'calibration_eta': 0.1,
                'calibration_rate': 2.0,
            },
            'ap': {  # Averaged Perceptron
                'max_iterations': 100,
                'epsilon': 0.0,
            },
            'pa': {  # Passive Aggressive
                'c': 1.0,
                'error_sensitive': True,
                'averaging': True,
                'max_iterations': 100,
            },
            'arow': {  # Adaptive Regularization of Weights
                'variance': 1.0,
                'max_iterations': 100,
            }
        }
        self.params = defaults.get(self.algorithm, {}).copy()
    
    def append(self, xseq, yseq, group=0):
        """
        Append a training instance.
        
        Args:
            xseq: An ItemSequence object representing the input sequence
            yseq: A list of labels (strings) for each item in the sequence
            group (int): Group ID for the instance (for grouped training)
        """
        if not isinstance(xseq, ItemSequence):
            raise TypeError("xseq must be an ItemSequence object")
        if not isinstance(yseq, (list, tuple)):
            raise TypeError("yseq must be a list or tuple of labels")
        if len(xseq) != len(yseq):
            raise ValueError(f"Length mismatch: xseq has {len(xseq)} items but yseq has {len(yseq)} labels")
        
        self.data.append((xseq, yseq, group))
    
    def set_params(self, params):
        """
        Set training parameters.
        
        Args:
            params (dict): Dictionary of parameter names and values
        """
        self.params.update(params)
    
    def get_params(self):
        """
        Get current training parameters.
        
        Returns:
            dict: Current parameter settings
        """
        return self.params.copy()
    
    def _build_feature_and_label_dicts(self):
        """Build dictionaries mapping features and labels to indices."""
        feature_set = set()
        label_set = set()
        
        for xseq, yseq, _ in self.data:
            # Collect all features
            for item in xseq:
                for attr in item:
                    feature_set.add(attr.name)
            # Collect all labels
            label_set.update(yseq)
        
        # Build dictionaries
        self._feature_dict = {feat: idx for idx, feat in enumerate(sorted(feature_set))}
        self._label_dict = {label: idx for idx, label in enumerate(sorted(label_set))}
        
        if self.verbose:
            print(f"Number of features: {len(self._feature_dict)}")
            print(f"Number of labels: {len(self._label_dict)}")
    
    def _extract_features(self, xseq, yseq):
        """
        Extract feature vectors from a sequence.
        
        Returns:
            List of feature dictionaries for each position
        """
        n = len(xseq)
        num_features = len(self._feature_dict)
        num_labels = len(self._label_dict)
        
        features = []
        for i in range(n):
            item_features = {}
            # State features (current label)
            for attr in xseq[i]:
                if attr.name in self._feature_dict:
                    feat_idx = self._feature_dict[attr.name]
                    label_idx = self._label_dict.get(yseq[i], -1)
                    if label_idx >= 0:
                        key = (feat_idx, label_idx)
                        item_features[key] = item_features.get(key, 0.0) + attr.value
            
            # Transition features (previous label to current label)
            if i > 0:
                prev_label_idx = self._label_dict.get(yseq[i-1], -1)
                curr_label_idx = self._label_dict.get(yseq[i], -1)
                if prev_label_idx >= 0 and curr_label_idx >= 0:
                    # Use negative indices for transition features
                    key = (-1, prev_label_idx, curr_label_idx)
                    item_features[key] = 1.0
            
            features.append(item_features)
        
        return features
    
    def train(self, model_filename, holdout=-1):
        """
        Train a CRF model.
        
        Args:
            model_filename (str): Filename to save the trained model
            holdout (int): Holdout group for evaluation (-1 for no holdout)
            
        Returns:
            dict: Training statistics
        """
        if not self.data:
            raise ValueError("No training data available")
        
        if self.verbose:
            print(f"Training CRF model using {self.algorithm} algorithm")
            print(f"Number of instances: {len(self.data)}")
        
        # Build feature and label dictionaries
        self._build_feature_and_label_dicts()
        
        # Initialize weights
        num_features = len(self._feature_dict)
        num_labels = len(self._label_dict)
        
        # Feature weights (for state features)
        self._weights = np.zeros((num_features, num_labels))
        
        # Transition weights (for label transitions)
        self._trans_weights = np.zeros((num_labels, num_labels))
        
        # Train based on algorithm
        if self.algorithm == 'lbfgs':
            stats = self._train_lbfgs()
        elif self.algorithm == 'l2sgd':
            stats = self._train_l2sgd()
        elif self.algorithm == 'ap':
            stats = self._train_averaged_perceptron()
        else:
            # For now, default to a simple training approach
            stats = self._train_simple()
        
        # Save model
        self._save_model(model_filename)
        
        if self.verbose:
            print(f"Model saved to {model_filename}")
        
        return stats
    
    def _train_simple(self):
        """Simple training algorithm (for demonstration)."""
        max_iter = self.params.get('max_iterations', 100)
        
        for iteration in range(max_iter):
            if self.verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}/{max_iter}")
            
            # Simple update: count feature occurrences with labels
            for xseq, yseq, _ in self.data:
                features = self._extract_features(xseq, yseq)
                
                for pos, feat_dict in enumerate(features):
                    for key, value in feat_dict.items():
                        if len(key) == 2:  # State feature
                            feat_idx, label_idx = key
                            self._weights[feat_idx, label_idx] += value
                        elif len(key) == 3:  # Transition feature
                            _, prev_label, curr_label = key
                            self._trans_weights[prev_label, curr_label] += value
        
        # Normalize weights
        self._weights /= len(self.data)
        self._trans_weights /= len(self.data)
        
        return {'iterations': max_iter, 'algorithm': self.algorithm}
    
    def _train_lbfgs(self):
        """Train using L-BFGS algorithm (simplified version)."""
        return self._train_simple()  # Placeholder
    
    def _train_l2sgd(self):
        """Train using L2-regularized SGD."""
        return self._train_simple()  # Placeholder
    
    def _train_averaged_perceptron(self):
        """Train using Averaged Perceptron."""
        return self._train_simple()  # Placeholder
    
    def _save_model(self, filename):
        """Save the trained model to a file."""
        model_data = {
            'algorithm': self.algorithm,
            'feature_dict': self._feature_dict,
            'label_dict': self._label_dict,
            'weights': self._weights,
            'trans_weights': self._trans_weights,
            'params': self.params,
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
    
    def clear(self):
        """Clear training data."""
        self.data.clear()
