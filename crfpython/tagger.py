"""
Tagger class for tagging sequences with a trained CRF model.

The Tagger class loads a trained model and predicts labels for new sequences.
"""

import pickle
import numpy as np
from typing import List, Optional
from .item import ItemSequence


class Tagger:
    """
    CRF model tagger for sequence labeling.
    
    The Tagger class is used to label sequences using a trained CRF model.
    It implements the Viterbi algorithm for finding the most likely label sequence.
    
    Examples:
        >>> tagger = Tagger()
        >>> tagger.open('model.crfsuite')
        >>> labels = tagger.tag(xseq)
    """
    
    def __init__(self):
        """Initialize a Tagger."""
        self._model = None
        self._feature_dict = None
        self._label_dict = None
        self._weights = None
        self._trans_weights = None
        self._labels = None  # Inverse mapping: index -> label
    
    def open(self, filename):
        """
        Open a model file.
        
        Args:
            filename (str): Path to the trained model file
            
        Returns:
            bool: True if successful
        """
        try:
            with open(filename, 'rb') as f:
                self._model = pickle.load(f)
            
            self._feature_dict = self._model['feature_dict']
            self._label_dict = self._model['label_dict']
            self._weights = self._model['weights']
            self._trans_weights = self._model['trans_weights']
            
            # Create inverse label mapping
            self._labels = {idx: label for label, idx in self._label_dict.items()}
            
            return True
        except Exception as e:
            raise IOError(f"Failed to load model from {filename}: {e}")
    
    def tag(self, xseq):
        """
        Tag a sequence.
        
        Args:
            xseq: An ItemSequence object
            
        Returns:
            list: A list of predicted labels
        """
        if self._model is None:
            raise RuntimeError("No model loaded. Call open() first.")
        
        if not isinstance(xseq, ItemSequence):
            raise TypeError("xseq must be an ItemSequence object")
        
        n = len(xseq)
        if n == 0:
            return []
        
        # Use Viterbi algorithm to find the best label sequence
        return self._viterbi(xseq)
    
    def _compute_scores(self, xseq):
        """
        Compute emission scores for each position and label.
        
        Returns:
            numpy array of shape (n, num_labels) with emission scores
        """
        n = len(xseq)
        num_labels = len(self._label_dict)
        scores = np.zeros((n, num_labels))
        
        for i, item in enumerate(xseq):
            for attr in item:
                if attr.name in self._feature_dict:
                    feat_idx = self._feature_dict[attr.name]
                    # Add weighted feature contribution for each label
                    scores[i, :] += self._weights[feat_idx, :] * attr.value
        
        return scores
    
    def _viterbi(self, xseq):
        """
        Viterbi algorithm for finding the best label sequence.
        
        Args:
            xseq: An ItemSequence object
            
        Returns:
            list: The most likely label sequence
        """
        n = len(xseq)
        num_labels = len(self._label_dict)
        
        # Compute emission scores
        emission_scores = self._compute_scores(xseq)
        
        # Initialize Viterbi tables
        # delta[t][j] = max probability of paths ending at label j at position t
        delta = np.zeros((n, num_labels))
        # psi[t][j] = argmax of previous label for best path to label j at position t
        psi = np.zeros((n, num_labels), dtype=int)
        
        # Initialization (t=0)
        delta[0, :] = emission_scores[0, :]
        
        # Recursion (t=1..n-1)
        for t in range(1, n):
            for j in range(num_labels):
                # Compute scores from all previous labels
                trans_scores = delta[t-1, :] + self._trans_weights[:, j]
                max_idx = np.argmax(trans_scores)
                delta[t, j] = trans_scores[max_idx] + emission_scores[t, j]
                psi[t, j] = max_idx
        
        # Termination: find best final label
        best_path = np.zeros(n, dtype=int)
        best_path[-1] = np.argmax(delta[-1, :])
        
        # Backtracking
        for t in range(n-2, -1, -1):
            best_path[t] = psi[t+1, best_path[t+1]]
        
        # Convert label indices to label names
        return [self._labels[idx] for idx in best_path]
    
    def set(self, xseq):
        """
        Set an instance for tagging (alternative interface).
        
        Args:
            xseq: An ItemSequence object
        """
        self._current_xseq = xseq
    
    def labels(self):
        """
        Get all possible labels.
        
        Returns:
            list: List of all label names
        """
        if self._labels is None:
            return []
        return sorted(self._labels.values())
    
    def marginal(self, label, pos):
        """
        Compute the marginal probability of a label at a position.
        
        Args:
            label (str): The label
            pos (int): Position in the sequence
            
        Returns:
            float: Marginal probability (simplified implementation)
        """
        # This is a simplified implementation
        # A full implementation would use forward-backward algorithm
        if self._model is None:
            raise RuntimeError("No model loaded")
        
        # For now, return a placeholder
        return 0.0
    
    def probability(self, yseq):
        """
        Compute the probability of a label sequence.
        
        Args:
            yseq: A list of labels
            
        Returns:
            float: Log probability of the sequence
        """
        # Simplified implementation
        return 0.0
