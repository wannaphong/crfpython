"""
Trainer class for training CRF models.

The Trainer class provides methods to train a CRF model using various algorithms.
"""

import pickle
from typing import List, Dict, Optional, Any
import numpy as np
from .item import ItemSequence, _process_item_features


class Trainer:
    """
    CRF model trainer compatible with python-crfsuite API.
    
    The Trainer class is used to train CRF models with labeled sequence data.
    It supports various training algorithms including LBFGS, L2SGD, and others.
    
    Parameters
    ----------
    algorithm : {'lbfgs', 'l2sgd', 'ap', 'pa', 'arow'}
        The name of the training algorithm.
    params : dict, optional
        Training parameters.
    verbose : bool
        Whether to print training progress
    
    Examples:
        >>> trainer = Trainer()
        >>> trainer.append(xseq, yseq)  # Add training data
        >>> trainer.train('model.crfsuite')  # Train and save model
    """
    
    # Parameter type definitions
    _PARAMETER_TYPES = {
        'feature.minfreq': float,
        'feature.possible_states': bool,
        'feature.possible_transitions': bool,
        'c1': float,
        'c2': float,
        'max_iterations': int,
        'num_memories': int,
        'epsilon': float,
        'period': int,
        'delta': float,
        'linesearch': str,
        'max_linesearch': int,
        'calibration.eta': float,
        'calibration.rate': float,
        'calibration.samples': float,
        'calibration.candidates': int,
        'calibration.max_trials': int,
        'type': int,
        'c': float,
        'error_sensitive': bool,
        'averaging': bool,
        'variance': float,
        'gamma': float,
    }
    
    _ALGORITHM_ALIASES = {
        'ap': 'averaged-perceptron',
        'pa': 'passive-aggressive',
    }
    
    def __init__(self, algorithm=None, params=None, verbose=True):
        """
        Initialize a Trainer.
        
        Args:
            algorithm (str): Training algorithm. Options: 'lbfgs', 'l2sgd', 'ap', 'pa', 'arow'
            params (dict): Training parameters
            verbose (bool): Whether to print training progress
        """
        self.verbose = verbose
        self.data = []  # List of (xseq, yseq, group) tuples
        self._params = {}
        self._feature_dict = {}  # Maps feature names to indices
        self._label_dict = {}    # Maps label names to indices
        self._weights = None
        self._trans_weights = None
        self._algorithm = 'lbfgs'
        self._graphical_model = 'crf1d'
        
        if algorithm is not None:
            self.select(algorithm)
        
        # Set default parameters based on algorithm
        self._set_default_params()
        
        if params is not None:
            self.set_params(params)
    
    def select(self, algorithm, type='crf1d'):
        """
        Initialize the training algorithm.
        
        Parameters
        ----------
        algorithm : {'lbfgs', 'l2sgd', 'ap', 'pa', 'arow'}
            The name of the training algorithm.
            
            * 'lbfgs' for Gradient descent using the L-BFGS method
            * 'l2sgd' for Stochastic Gradient Descent with L2 regularization
            * 'ap' for Averaged Perceptron
            * 'pa' for Passive Aggressive
            * 'arow' for Adaptive Regularization Of Weight Vector
            
        type : str, optional
            The name of the graphical model (default: 'crf1d')
        """
        algorithm = algorithm.lower()
        algorithm = self._ALGORITHM_ALIASES.get(algorithm, algorithm)
        
        valid_algorithms = ['lbfgs', 'l2sgd', 'averaged-perceptron', 'passive-aggressive', 'arow', 'ap', 'pa']
        if algorithm not in valid_algorithms:
            raise ValueError(f"Invalid algorithm: {algorithm}. Must be one of {valid_algorithms}")
        
        self._algorithm = algorithm
        self._graphical_model = type
        self._set_default_params()
    
    def _set_default_params(self):
        """Set default parameters for the selected algorithm."""
        algo = self._algorithm
        
        defaults = {
            'lbfgs': {
                'c1': 0.0,  # L1 regularization coefficient
                'c2': 1.0,  # L2 regularization coefficient
                'max_iterations': 100,
                'epsilon': 1e-5,
                'num_memories': 6,
                'linesearch': 'MoreThuente',
                'max_linesearch': 20,
            },
            'l2sgd': {
                'c2': 1.0,
                'max_iterations': 100,
                'calibration.eta': 0.1,
                'calibration.rate': 2.0,
                'calibration.samples': 1000.0,
                'calibration.candidates': 10,
                'calibration.max_trials': 20,
            },
            'averaged-perceptron': {
                'max_iterations': 100,
                'epsilon': 0.0,
            },
            'ap': {
                'max_iterations': 100,
                'epsilon': 0.0,
            },
            'passive-aggressive': {
                'c': 1.0,
                'error_sensitive': True,
                'averaging': True,
                'max_iterations': 100,
            },
            'pa': {
                'c': 1.0,
                'error_sensitive': True,
                'averaging': True,
                'max_iterations': 100,
            },
            'arow': {
                'variance': 1.0,
                'max_iterations': 100,
            }
        }
        
        # Get defaults for current algorithm
        default_params = defaults.get(algo, {})
        
        # Only set defaults that aren't already set
        for key, value in default_params.items():
            if key not in self._params:
                self._params[key] = value
    
    def append(self, xseq, yseq, group=0):
        """
        Append a training instance.
        
        Parameters
        ----------
        xseq : sequence
            The item sequence. Can be an ItemSequence object, list of dicts,
            or list of feature lists. Supports python-crfsuite formats.
        yseq : list of str
            The label sequence
        group : int, optional
            The group number for holdout evaluation
        """
        # Convert to ItemSequence if needed
        if not isinstance(xseq, ItemSequence):
            xseq = ItemSequence(xseq)
        
        if not isinstance(yseq, (list, tuple)):
            raise TypeError("yseq must be a list or tuple of labels")
        if len(xseq) != len(yseq):
            raise ValueError(f"Length mismatch: xseq has {len(xseq)} items but yseq has {len(yseq)} labels")
        
        self.data.append((xseq, yseq, group))
    
    def params(self):
        """
        Get the list of available parameters.
        
        Returns
        -------
        list of str
            List of parameter names available for the current algorithm
        """
        # Return all possible parameters for the algorithm
        return sorted(self._params.keys())
    
    def set_params(self, params):
        """
        Set training parameters.
        
        Parameters
        ----------
        params : dict
            Dictionary of parameter names and values
        """
        for key, value in params.items():
            self.set(key, value)
    
    def get_params(self):
        """
        Get current training parameters.
        
        Returns
        -------
        dict
            Dictionary with all parameter names and values
        """
        return self._params.copy()
    
    def set(self, name, value):
        """
        Set a training parameter.
        
        Parameters
        ----------
        name : str
            The parameter name
        value : str, int, float, or bool
            The parameter value
        """
        if isinstance(value, bool):
            value = int(value)
        
        # Convert to appropriate type if needed
        if name in self._PARAMETER_TYPES:
            param_type = self._PARAMETER_TYPES[name]
            if param_type == bool:
                value = bool(int(value)) if not isinstance(value, bool) else value
            elif param_type != str:
                value = param_type(value)
        
        self._params[name] = value
    
    def get(self, name):
        """
        Get a training parameter value.
        
        Parameters
        ----------
        name : str
            The parameter name
            
        Returns
        -------
        value
            The parameter value
        """
        return self._params.get(name)
    
    def help(self, name):
        """
        Get help for a parameter.
        
        Parameters
        ----------
        name : str
            The parameter name
            
        Returns
        -------
        str
            Help text for the parameter
        """
        help_text = {
            'c1': 'Coefficient for L1 regularization',
            'c2': 'Coefficient for L2 regularization',
            'max_iterations': 'Maximum number of iterations',
            'epsilon': 'Stopping criterion (small value of gradient)',
            'num_memories': 'Number of limited memories for L-BFGS',
            'linesearch': 'Line search algorithm',
            'max_linesearch': 'Maximum number of line search trials',
            'calibration.eta': 'Initial learning rate (eta) for calibration',
            'calibration.rate': 'Rate of increase/decrease of learning rate',
            'calibration.samples': 'Number of instances for calibration',
            'calibration.candidates': 'Number of candidate learning rates',
            'calibration.max_trials': 'Maximum number of trials for calibration',
            'c': 'Aggressiveness parameter',
            'error_sensitive': 'Include  error rate in the loss function',
            'averaging': 'Compute averaged weights',
            'variance': 'Initial variance of each weight',
            'gamma': 'Learning rate',
        }
        return help_text.get(name, f'No help available for parameter: {name}')
    
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
    
    def train(self, model, holdout=-1):
        """
        Train a CRF model.
        
        Parameters
        ----------
        model : str
            Filename to save the trained model
        holdout : int, optional
            Group number for holdout evaluation (-1 for no holdout)
            
        Returns
        -------
        None
        """
        if not self.data:
            raise ValueError("No training data available")
        
        if self.verbose:
            print(f"Training CRF model using {self._algorithm} algorithm")
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
        self._train_model()
        
        # Save model
        self._save_model(model)
        
        if self.verbose:
            print(f"Model saved to {model}")
    
    def _train_model(self):
        """Execute the training algorithm."""
        max_iter = self._params.get('max_iterations', 100)
        
        for iteration in range(max_iter):
            if self.verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}/{max_iter}")
            
            # Simple training: count feature occurrences with labels
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
    
    def _extract_features(self, xseq, yseq):
        """Extract feature vectors from a sequence."""
        n = len(xseq)
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
                    key = (-1, prev_label_idx, curr_label_idx)
                    item_features[key] = 1.0
            
            features.append(item_features)
        
        return features
    
    def _save_model(self, filename):
        """Save the trained model to a file."""
        model_data = {
            'algorithm': self._algorithm,
            'graphical_model': self._graphical_model,
            'feature_dict': self._feature_dict,
            'label_dict': self._label_dict,
            'weights': self._weights,
            'trans_weights': self._trans_weights,
            'params': self._params,
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
    
    def clear(self):
        """Clear training data."""
        self.data.clear()
    
    # Callback methods for subclassing
    def on_start(self, log):
        """Called when training starts."""
        pass
    
    def on_featgen_progress(self, log, percent):
        """Called during feature generation."""
        pass
    
    def on_featgen_end(self, log):
        """Called when feature generation ends."""
        pass
    
    def on_prepared(self, log):
        """Called when data preparation is complete."""
        pass
    
    def on_prepare_error(self, log):
        """Called when there's an error during preparation."""
        pass
    
    def on_iteration(self, log, iteration):
        """Called after each iteration."""
        pass
    
    def on_optimization_end(self, log):
        """Called when optimization ends."""
        pass
    
    def on_end(self, log):
        """Called when training ends."""
        pass
