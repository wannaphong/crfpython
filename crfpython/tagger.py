"""
Tagger class for tagging sequences with a trained CRF model.

The Tagger class loads a trained model and predicts labels for new sequences.
"""

import pickle
import numpy as np
from typing import List, Optional
import contextlib
from .item import ItemSequence


class Tagger:
    """
    CRF model tagger for sequence labeling (python-crfsuite compatible API).
    
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
        self._current_xseq = None
        self._alpha = None  # Forward probabilities
        self._beta = None   # Backward probabilities
        self._scale = None  # Scaling factors
    
    def open(self, name):
        """
        Open a model file.
        
        Parameters
        ----------
        name : str
            The file name of the model file
            
        Returns
        -------
        contextlib.closing
            Context manager that closes the model
        """
        try:
            with open(name, 'rb') as f:
                self._model = pickle.load(f)
            
            self._feature_dict = self._model['feature_dict']
            self._label_dict = self._model['label_dict']
            self._weights = self._model['weights']
            self._trans_weights = self._model['trans_weights']
            
            # Create inverse label mapping
            self._labels = {idx: label for label, idx in self._label_dict.items()}
            
            return contextlib.closing(self)
            
        except Exception as e:
            raise IOError(f"Failed to load model from {name}: {e}")
    
    def close(self):
        """Close the model."""
        self._model = None
        self._feature_dict = None
        self._label_dict = None
        self._weights = None
        self._trans_weights = None
        self._labels = None
        self._current_xseq = None
    
    def tag(self, xseq=None):
        """
        Predict the label sequence for the item sequence.
        
        Parameters
        ----------
        xseq : sequence, optional
            The item sequence. If omitted, uses the current sequence
            (set via set() or a previous tag() call). Can be ItemSequence,
            list of dicts, or other python-crfsuite compatible formats.
            
        Returns
        -------
        list of str
            The predicted label sequence
        """
        if self._model is None:
            raise RuntimeError("No model loaded. Call open() first.")
        
        if xseq is not None:
            self.set(xseq)
        
        if self._current_xseq is None:
            raise RuntimeError("No sequence set. Pass xseq or call set() first.")
        
        n = len(self._current_xseq)
        if n == 0:
            return []
        
        # Use Viterbi algorithm to find the best label sequence
        return self._viterbi(self._current_xseq)
    
    def set(self, xseq):
        """
        Set an instance (item sequence) for future operations.
        
        Parameters
        ----------
        xseq : sequence
            The item sequence. Can be ItemSequence, list of dicts,
            or other python-crfsuite compatible formats.
        """
        if not isinstance(xseq, ItemSequence):
            xseq = ItemSequence(xseq)
        self._current_xseq = xseq
        # Reset cached probabilities
        self._alpha = None
        self._beta = None
        self._scale = None
    
    def labels(self):
        """
        Obtain the list of labels.
        
        Returns
        -------
        list of str
            The list of labels in the model
        """
        if self._labels is None:
            return []
        return sorted(self._labels.values())
    
    def probability(self, yseq):
        """
        Compute the probability of a label sequence.
        
        Computes P(yseq|xseq) for the current input sequence
        (set via set() or tag() call).
        
        **Note:** This is a simplified implementation that computes
        unnormalized scores. The returned value may not be a proper
        probability (i.e., may not sum to 1 across all sequences).
        For ranking sequences or comparing alternatives, the relative
        values are meaningful, but absolute values should be interpreted
        with caution.
        
        Parameters
        ----------
        yseq : list of str
            The label sequence
            
        Returns
        -------
        float
            The unnormalized probability/score for the sequence
        """
        if self._model is None:
            raise RuntimeError("No model loaded")
        if self._current_xseq is None:
            raise RuntimeError("No sequence set")
        
        # Compute forward-backward if not cached
        if self._alpha is None:
            self._forward_backward(self._current_xseq)
        
        n = len(yseq)
        if n != len(self._current_xseq):
            return 0.0
        
        # Compute unnormalized log probability
        log_prob = 0.0
        emission_scores = self._compute_scores(self._current_xseq)
        
        for t in range(n):
            label_idx = self._label_dict.get(yseq[t])
            if label_idx is None:
                return 0.0
            
            # Add emission score
            log_prob += emission_scores[t, label_idx]
            
            # Add transition score
            if t > 0:
                prev_label_idx = self._label_dict.get(yseq[t-1])
                if prev_label_idx is None:
                    return 0.0
                log_prob += self._trans_weights[prev_label_idx, label_idx]
        
        # Normalize using partition function
        # Z = sum over all possible sequences
        # For now, return exp of unnormalized score (simplified)
        return np.exp(log_prob)
    
    def marginal(self, y, pos):
        """
        Compute the marginal probability of a label at a position.
        
        Computes P(y_t = y | xseq) for the current input sequence.
        
        Parameters
        ----------
        y : str
            The label
        pos : int
            The position in the sequence
            
        Returns
        -------
        float
            The marginal probability
        """
        if self._model is None:
            raise RuntimeError("No model loaded")
        if self._current_xseq is None:
            raise RuntimeError("No sequence set")
        
        if pos < 0 or pos >= len(self._current_xseq):
            return 0.0
        
        label_idx = self._label_dict.get(y)
        if label_idx is None:
            return 0.0
        
        # Compute forward-backward if not cached
        if self._alpha is None:
            self._forward_backward(self._current_xseq)
        
        # Marginal probability = alpha[t, y] * beta[t, y] / Z
        # For simplified implementation, return a normalized value
        if self._alpha is not None and self._beta is not None:
            marginal = self._alpha[pos, label_idx] * self._beta[pos, label_idx]
            normalization = np.sum(self._alpha[pos, :] * self._beta[pos, :])
            if normalization > 0:
                return marginal / normalization
        
        return 0.0
    
    def dump(self, filename=None):
        """
        Dump the CRF model in plain-text format.
        
        Parameters
        ----------
        filename : str, optional
            File name to dump the model to. If None, prints to stdout.
        """
        if self._model is None:
            raise RuntimeError("No model loaded")
        
        import sys
        f = open(filename, 'w') if filename else sys.stdout
        
        try:
            f.write("CRF Model Dump\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Algorithm: {self._model.get('algorithm', 'unknown')}\n")
            f.write(f"Graphical Model: {self._model.get('graphical_model', 'unknown')}\n\n")
            
            f.write(f"Number of labels: {len(self._label_dict)}\n")
            f.write(f"Number of features: {len(self._feature_dict)}\n\n")
            
            f.write("Labels:\n")
            for label in sorted(self._label_dict.keys()):
                f.write(f"  {label}\n")
            
            f.write("\nFeature weights (top 20):\n")
            # Get top weighted features
            feature_weights = []
            for feat_name, feat_idx in self._feature_dict.items():
                for label_name, label_idx in self._label_dict.items():
                    weight = self._weights[feat_idx, label_idx]
                    if abs(weight) > 0.01:
                        feature_weights.append((abs(weight), feat_name, label_name, weight))
            
            feature_weights.sort(reverse=True)
            for _, feat_name, label_name, weight in feature_weights[:20]:
                f.write(f"  {feat_name} -> {label_name}: {weight:.4f}\n")
            
            f.write("\nTransition weights:\n")
            for prev_label, prev_idx in sorted(self._label_dict.items()):
                for curr_label, curr_idx in sorted(self._label_dict.items()):
                    weight = self._trans_weights[prev_idx, curr_idx]
                    if abs(weight) > 0.01:
                        f.write(f"  {prev_label} -> {curr_label}: {weight:.4f}\n")
        
        finally:
            if filename:
                f.close()
    
    def info(self):
        """
        Return model internal information.
        
        Returns
        -------
        dict
            Dictionary with model information including labels,
            features, and weights
        """
        if self._model is None:
            raise RuntimeError("No model loaded")
        
        return {
            'algorithm': self._model.get('algorithm'),
            'graphical_model': self._model.get('graphical_model'),
            'num_labels': len(self._label_dict),
            'num_features': len(self._feature_dict),
            'labels': sorted(self._label_dict.keys()),
            'state_features': len(self._feature_dict) * len(self._label_dict),
            'transition_features': len(self._label_dict) * len(self._label_dict),
        }
    
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
    
    def _forward_backward(self, xseq):
        """
        Compute forward and backward probabilities.
        
        Used for computing marginals and probabilities.
        """
        n = len(xseq)
        num_labels = len(self._label_dict)
        
        # Compute emission scores
        emission_scores = self._compute_scores(xseq)
        
        # Forward pass
        self._alpha = np.zeros((n, num_labels))
        self._scale = np.zeros(n)
        
        # Initialize
        self._alpha[0, :] = np.exp(emission_scores[0, :])
        self._scale[0] = np.sum(self._alpha[0, :])
        if self._scale[0] > 0:
            self._alpha[0, :] /= self._scale[0]
        
        # Forward recursion
        for t in range(1, n):
            for j in range(num_labels):
                # Sum over all previous states
                trans_probs = self._alpha[t-1, :] * np.exp(self._trans_weights[:, j])
                self._alpha[t, j] = np.sum(trans_probs) * np.exp(emission_scores[t, j])
            
            # Scale to prevent underflow
            self._scale[t] = np.sum(self._alpha[t, :])
            if self._scale[t] > 0:
                self._alpha[t, :] /= self._scale[t]
        
        # Backward pass
        self._beta = np.zeros((n, num_labels))
        
        # Initialize
        self._beta[-1, :] = 1.0
        
        # Backward recursion
        for t in range(n-2, -1, -1):
            for i in range(num_labels):
                # Sum over all next states
                trans_probs = np.exp(self._trans_weights[i, :]) * np.exp(emission_scores[t+1, :])
                self._beta[t, i] = np.sum(trans_probs * self._beta[t+1, :])
            
            # Scale
            if self._scale[t+1] > 0:
                self._beta[t, :] /= self._scale[t+1]
    
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
        delta = np.zeros((n, num_labels))
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
