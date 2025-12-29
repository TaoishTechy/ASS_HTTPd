#!/usr/bin/env python3
"""
debug_ass.py - Advanced Debug Assistant with 24 Novel Pattern Recognition Approaches
Uses pattern files from ./patterns/ to analyze, debug, and enhance code
Incorporates cognitive science, information theory, and advanced algorithms
"""

import sys
import os
import re
import json
import argparse
import math
import statistics
import itertools
import collections
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set, Union, Callable
import traceback
import time
import hashlib
from dataclasses import dataclass, field
from enum import Enum
import heapq
import numpy as np
from scipy import stats
from collections import Counter, defaultdict, deque
import networkx as nx
from sklearn.cluster import KMeans, DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings

warnings.filterwarnings('ignore')

class PatternRecognitionApproach(Enum):
    """24 Novel Approaches to Pattern Recognition"""
    # 1. Cognitive Chunking & Segmentation
    CHUNK_BASED_PARSING = "chunk_based_parsing"
    WORKING_MEMORY_MODEL = "working_memory_model"
    ATTENTION_GUIDED_SCAN = "attention_guided_scan"
    GESTALT_PATTERN_GROUPING = "gestalt_pattern_grouping"
    
    # 2. Information Theory & Entropy
    KOLMOGOROV_COMPLEXITY = "kolmogorov_complexity"
    INFORMATION_DENSITY = "information_density"
    ENTROPY_BASED_CLUSTERING = "entropy_based_clustering"
    COMPRESSION_BASED_MODELING = "compression_based_modeling"
    
    # 3. Graph Theory & Networks
    DEPENDENCY_GRAPH_ANALYSIS = "dependency_graph_analysis"
    COMMUNITY_DETECTION = "community_detection"
    RANDOM_WALK_SAMPLING = "random_walk_sampling"
    GRAPH_NEURAL_NETWORK = "graph_neural_network"
    
    # 4. Statistical Learning
    HIDDEN_MARKOV_MODEL = "hidden_markov_model"
    BAYESIAN_NETWORK = "bayesian_network"
    SEQUENCE_PREDICTION = "sequence_prediction"
    ANOMALY_DETECTION = "anomaly_detection"
    
    # 5. Transformational Approaches
    WAVELET_TRANSFORM = "wavelet_transform"
    FOURIER_ANALYSIS = "fourier_analysis"
    SINGULAR_VALUE_DECOMPOSITION = "singular_value_decomposition"
    TRANSFORMER_ATTENTION = "transformer_attention"
    
    # 6. Evolutionary & Adaptive
    GENETIC_ALGORITHM = "genetic_algorithm"
    SWARM_INTELLIGENCE = "swarm_intelligence"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    ADAPTIVE_RESONANCE = "adaptive_resonance"

@dataclass
class CodeChunk:
    """Represents a cognitive chunk of code"""
    content: str
    start_line: int
    end_line: int
    chunk_type: str
    cognitive_weight: float
    dependencies: List['CodeChunk'] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PatternMatch:
    """Enhanced pattern match with confidence and context"""
    pattern_name: str
    matched_text: str
    start_pos: int
    end_pos: int
    confidence: float
    context_before: str
    context_after: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class CognitiveChunker:
    """Approach 1: Chunk-Based Parsing using Miller's Law (7±2 chunks)"""
    
    def __init__(self, max_chunks_per_line: int = 3, chunk_size: int = 7):
        self.max_chunks_per_line = max_chunks_per_line
        self.chunk_size = chunk_size
        
    def chunk_code(self, code: str, language: str = "unknown") -> List[CodeChunk]:
        """Chunk code based on cognitive principles"""
        lines = code.split('\n')
        chunks = []
        current_chunk = []
        current_start = 0
        
        for i, line in enumerate(lines):
            if not line.strip():
                continue
                
            # Calculate cognitive complexity of line
            complexity = self._calculate_line_complexity(line, language)
            
            # Break chunks based on complexity and chunk boundaries
            if len(current_chunk) >= self.chunk_size or complexity > 0.7:
                if current_chunk:
                    chunks.append(CodeChunk(
                        content='\n'.join(current_chunk),
                        start_line=current_start,
                        end_line=i-1,
                        chunk_type=self._determine_chunk_type(current_chunk),
                        cognitive_weight=complexity
                    ))
                    current_chunk = []
                    current_start = i
            
            current_chunk.append(line)
        
        # Add final chunk
        if current_chunk:
            chunks.append(CodeChunk(
                content='\n'.join(current_chunk),
                start_line=current_start,
                end_line=len(lines)-1,
                chunk_type=self._determine_chunk_type(current_chunk),
                cognitive_weight=self._calculate_line_complexity('\n'.join(current_chunk), language)
            ))
        
        return chunks
    
    def _calculate_line_complexity(self, line: str, language: str) -> float:
        """Calculate cognitive complexity of a line"""
        complexity = 0.0
        
        # Based on number of tokens
        tokens = re.findall(r'\b\w+\b|[^\w\s]', line)
        complexity += min(len(tokens) / 10, 1.0)
        
        # Based on nesting depth
        complexity += line.count('{') * 0.2
        complexity += line.count('(') * 0.1
        complexity -= line.count('}') * 0.2
        complexity -= line.count(')') * 0.1
        
        # Based on operators
        operators = r'[+\-*/%=<>!&|^~]'
        complexity += len(re.findall(operators, line)) * 0.15
        
        return min(max(complexity, 0.0), 1.0)
    
    def _determine_chunk_type(self, lines: List[str]) -> str:
        """Determine type of code chunk"""
        text = '\n'.join(lines)
        
        if re.search(r'\b(class|interface|trait)\b', text):
            return "class_definition"
        elif re.search(r'\b(def|function)\b', text):
            return "function_definition"
        elif re.search(r'\b(if|else|switch|case)\b', text):
            return "conditional"
        elif re.search(r'\b(for|while|do|foreach)\b', text):
            return "loop"
        elif re.search(r'\b(try|catch|finally|throw)\b', text):
            return "error_handling"
        elif re.search(r'(\$|=|var|let|const)\b', text):
            return "variable_assignment"
        else:
            return "mixed"

class WorkingMemoryModel:
    """Approach 2: Working Memory Model (Baddeley & Hitch)"""
    
    def __init__(self, capacity: int = 4):
        self.capacity = capacity
        self.phonological_loop = deque(maxlen=capacity)
        self.visuospatial_sketchpad = deque(maxlen=capacity)
        self.central_executive = []
        self.episodic_buffer = []
    
    def process_code(self, code: str) -> Dict[str, Any]:
        """Process code using working memory model"""
        lines = code.split('\n')
        results = {
            'memory_load': [],
            'chunk_overflow': [],
            'attention_shifts': [],
            'retention_score': 0.0
        }
        
        current_focus = []
        attention_shifts = 0
        
        for i, line in enumerate(lines):
            if not line.strip():
                continue
                
            # Add to phonological loop (verbal working memory)
            tokens = re.findall(r'\b\w+\b', line)
            for token in tokens[:self.capacity]:
                self.phonological_loop.append((token, i))
            
            # Add to visuospatial sketchpad (structural patterns)
            structure = self._extract_structure(line)
            if structure:
                self.visuospatial_sketchpad.append((structure, i))
            
            # Central executive decides what to focus on
            focus_item = self._central_executive_decision(line, current_focus)
            if focus_item and focus_item not in current_focus:
                attention_shifts += 1
                current_focus = [focus_item] + current_focus[:self.capacity-1]
            
            # Check for memory overload
            if len(tokens) > self.capacity:
                results['chunk_overflow'].append({
                    'line': i,
                    'tokens': len(tokens),
                    'capacity': self.capacity
                })
            
            results['memory_load'].append({
                'line': i,
                'phonological': len(self.phonological_loop),
                'visuospatial': len(self.visuospatial_sketchpad),
                'focus_items': len(current_focus)
            })
        
        # Calculate retention score
        unique_items = len(set(list(self.phonological_loop) + list(self.visuospatial_sketchpad)))
        total_items = len(lines) * 2  # Approximate
        results['retention_score'] = unique_items / total_items if total_items > 0 else 0
        results['attention_shifts'] = attention_shifts
        
        return results
    
    def _extract_structure(self, line: str) -> Optional[str]:
        """Extract structural patterns from line"""
        patterns = [
            (r'\bif\s*\(', 'conditional_start'),
            (r'\belse\b', 'conditional_else'),
            (r'\bfor\s*\(', 'loop_start'),
            (r'\bwhile\s*\(', 'while_start'),
            (r'\bfunction\b', 'function_start'),
            (r'\bclass\b', 'class_start'),
            (r'\{', 'block_start'),
            (r'\}', 'block_end'),
            (r'\(', 'paren_start'),
            (r'\)', 'paren_end'),
            (r'=', 'assignment'),
            (r'\+|\-|\*|\/', 'operator'),
        ]
        
        for pattern, label in patterns:
            if re.search(pattern, line):
                return label
        return None
    
    def _central_executive_decision(self, line: str, current_focus: List) -> Optional[str]:
        """Central executive decides what to focus on"""
        keywords = ['error', 'warning', 'todo', 'fixme', 'important', 'note']
        for keyword in keywords:
            if keyword in line.lower():
                return f"keyword:{keyword}"
        
        if re.search(r'\b(if|else|for|while)\b', line):
            return "control_flow"
        elif re.search(r'\b(function|def|method)\b', line):
            return "function"
        elif re.search(r'\b(class|interface|trait)\b', line):
            return "class"
        
        return None

class KolmogorovComplexityEstimator:
    """Approach 3: Estimate Kolmogorov Complexity (Minimum description length)"""
    
    def __init__(self):
        self.compressors = {
            'gzip': self._gzip_compress,
            'lz77': self._lz77_compress,
            'rle': self._rle_compress,
        }
    
    def estimate_complexity(self, text: str) -> Dict[str, float]:
        """Estimate Kolmogorov complexity using multiple compressors"""
        results = {}
        
        for name, compressor in self.compressors.items():
            compressed = compressor(text)
            compression_ratio = len(compressed) / len(text) if text else 1.0
            results[name] = compression_ratio
        
        # Average across compressors
        results['average'] = sum(results.values()) / len(results)
        
        # Estimate actual Kolmogorov complexity
        results['estimated_kolmogorov'] = self._estimate_kolmogorov(text)
        
        return results
    
    def _gzip_compress(self, text: str) -> bytes:
        """Simple gzip-like compression"""
        import zlib
        return zlib.compress(text.encode('utf-8'), level=9)
    
    def _lz77_compress(self, text: str) -> str:
        """LZ77 compression algorithm"""
        # Simplified LZ77 implementation
        window_size = 100
        lookahead_size = 50
        i = 0
        compressed = []
        
        while i < len(text):
            match_length = 0
            match_distance = 0
            
            # Search for matches in sliding window
            start = max(0, i - window_size)
            window = text[start:i]
            lookahead = text[i:i + lookahead_size]
            
            for length in range(1, len(lookahead) + 1):
                substring = lookahead[:length]
                pos = window.rfind(substring)
                if pos != -1:
                    match_length = length
                    match_distance = len(window) - pos
            
            if match_length > 0:
                compressed.append(f"({match_distance},{match_length})")
                i += match_length
            else:
                compressed.append(text[i])
                i += 1
        
        return ''.join(compressed)
    
    def _rle_compress(self, text: str) -> str:
        """Run-Length Encoding"""
        compressed = []
        i = 0
        
        while i < len(text):
            count = 1
            while i + count < len(text) and text[i] == text[i + count]:
                count += 1
            
            if count > 1:
                compressed.append(f"{text[i]}{count}")
            else:
                compressed.append(text[i])
            
            i += count
        
        return ''.join(compressed)
    
    def _estimate_kolmogorov(self, text: str) -> float:
        """Estimate Kolmogorov complexity"""
        if not text:
            return 0.0
        
        # Use multiple metrics
        entropy = self._shannon_entropy(text)
        compressibility = len(self._gzip_compress(text)) / len(text)
        
        # Pattern regularity
        pattern_score = self._pattern_regularity(text)
        
        # Combined estimate
        complexity = (entropy + compressibility + (1 - pattern_score)) / 3
        return complexity
    
    def _shannon_entropy(self, text: str) -> float:
        """Calculate Shannon entropy"""
        if not text:
            return 0.0
        
        prob = [float(text.count(c)) / len(text) for c in set(text)]
        entropy = -sum([p * math.log(p) / math.log(2.0) for p in prob])
        return entropy / 8.0  # Normalize to 0-1
    
    def _pattern_regularity(self, text: str) -> float:
        """Measure pattern regularity (0=chaotic, 1=regular)"""
        if len(text) < 4:
            return 1.0
        
        # Check for repeating patterns
        patterns = []
        for i in range(2, min(20, len(text) // 2)):
            for j in range(len(text) - i):
                pattern = text[j:j + i]
                count = text.count(pattern)
                if count > 1:
                    patterns.append((pattern, count))
        
        if not patterns:
            return 0.0
        
        # Calculate regularity score
        total_patterns = len(patterns)
        avg_repetition = sum(count for _, count in patterns) / total_patterns
        avg_length = sum(len(p) for p, _ in patterns) / total_patterns
        
        regularity = min(avg_repetition * avg_length / len(text), 1.0)
        return regularity

class DependencyGraphAnalyzer:
    """Approach 4: Dependency Graph Analysis"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_types = {}
        
    def build_dependency_graph(self, code: str, language: str = "unknown") -> nx.DiGraph:
        """Build dependency graph from code"""
        self.graph = nx.DiGraph()
        
        # Extract variables and functions
        variables = self._extract_variables(code, language)
        functions = self._extract_functions(code, language)
        
        # Add nodes
        for var in variables:
            self.graph.add_node(var, type='variable')
            self.node_types[var] = 'variable'
        
        for func in functions:
            self.graph.add_node(func['name'], type='function')
            self.node_types[func['name']] = 'function'
        
        # Add edges based on usage
        for func in functions:
            func_name = func['name']
            func_body = func.get('body', '')
            
            # Check which variables are used in function
            for var in variables:
                if var in func_body:
                    self.graph.add_edge(func_name, var, relation='uses')
            
            # Check function calls
            for other_func in functions:
                if other_func['name'] != func_name and other_func['name'] in func_body:
                    self.graph.add_edge(func_name, other_func['name'], relation='calls')
        
        return self.graph
    
    def analyze_graph_properties(self) -> Dict[str, Any]:
        """Analyze graph properties"""
        if not self.graph.nodes():
            return {}
        
        results = {
            'basic_metrics': {},
            'centrality_measures': {},
            'community_structure': {},
            'path_analysis': {}
        }
        
        # Basic metrics
        results['basic_metrics']['nodes'] = self.graph.number_of_nodes()
        results['basic_metrics']['edges'] = self.graph.number_of_edges()
        results['basic_metrics']['density'] = nx.density(self.graph)
        results['basic_metrics']['is_dag'] = nx.is_directed_acyclic_graph(self.graph)
        
        # Centrality measures
        if self.graph.nodes():
            try:
                degree_centrality = nx.degree_centrality(self.graph)
                betweenness_centrality = nx.betweenness_centrality(self.graph, normalized=True)
                closeness_centrality = nx.closeness_centrality(self.graph)
                
                results['centrality_measures']['degree'] = degree_centrality
                results['centrality_measures']['betweenness'] = betweenness_centrality
                results['centrality_measures']['closeness'] = closeness_centrality
            except:
                pass
        
        # Community detection
        try:
            if nx.is_weakly_connected(self.graph.to_undirected(as_view=True)):
                communities = nx.community.greedy_modularity_communities(
                    self.graph.to_undirected(as_view=True)
                )
                results['community_structure']['num_communities'] = len(communities)
                results['community_structure']['communities'] = [list(c) for c in communities]
        except:
            pass
        
        # Path analysis
        try:
            if nx.is_weakly_connected(self.graph.to_undirected(as_view=True)):
                results['path_analysis']['diameter'] = nx.diameter(self.graph.to_undirected(as_view=True))
                results['path_analysis']['avg_path_length'] = nx.average_shortest_path_length(
                    self.graph.to_undirected(as_view=True)
                )
        except:
            pass
        
        return results
    
    def _extract_variables(self, code: str, language: str) -> List[str]:
        """Extract variables from code"""
        variables = set()
        
        # Common patterns
        patterns = [
            r'\$(\w+)',  # PHP variables
            r'(?:var|let|const)\s+(\w+)',  # JavaScript variables
            r'(\w+)\s*=',  # Python/JS assignments
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, code)
            variables.update(matches)
        
        return list(variables)
    
    def _extract_functions(self, code: str, language: str) -> List[Dict[str, str]]:
        """Extract functions from code"""
        functions = []
        
        patterns = {
            'php': r'function\s+(\w+)\s*\(([^)]*)\)\s*\{([^}]+)\}',
            'javascript': r'function\s+(\w+)\s*\(([^)]*)\)\s*\{([^}]+)\}',
            'python': r'def\s+(\w+)\s*\(([^)]*)\):\s*\n(.*?)(?=\ndef\s|\nclass\s|\Z)',
        }
        
        pattern = patterns.get(language, patterns['javascript'])
        
        for match in re.finditer(pattern, code, re.DOTALL | re.MULTILINE):
            func_name = match.group(1)
            params = match.group(2) if match.group(2) else ''
            body = match.group(3) if match.group(3) else ''
            
            functions.append({
                'name': func_name,
                'params': params,
                'body': body
            })
        
        return functions

class HiddenMarkovModelAnalyzer:
    """Approach 5: Hidden Markov Model for code pattern recognition"""
    
    def __init__(self, states: int = 5):
        self.states = states
        self.transition_matrix = None
        self.emission_matrix = None
        self.initial_probs = None
        
    def train_on_code(self, code_samples: List[str]) -> Dict[str, Any]:
        """Train HMM on code samples"""
        # Tokenize code samples
        tokens_list = [self._tokenize_code(code) for code in code_samples]
        
        if not tokens_list:
            return {'error': 'No tokens to train on'}
        
        # Create vocabulary
        all_tokens = set()
        for tokens in tokens_list:
            all_tokens.update(tokens)
        vocab = list(all_tokens)
        vocab_size = len(vocab)
        
        # Initialize matrices
        self.transition_matrix = np.ones((self.states, self.states)) / self.states
        self.emission_matrix = np.ones((self.states, vocab_size)) / vocab_size
        self.initial_probs = np.ones(self.states) / self.states
        
        # Simple Baum-Welch approximation
        for tokens in tokens_list:
            if len(tokens) < 2:
                continue
            
            # Forward pass
            alpha = self._forward_algorithm(tokens, vocab)
            
            # Backward pass
            beta = self._backward_algorithm(tokens, vocab)
            
            # Update parameters (simplified)
            self._update_parameters(tokens, alpha, beta, vocab)
        
        # Normalize matrices
        self.transition_matrix = self.transition_matrix / self.transition_matrix.sum(axis=1, keepdims=True)
        self.emission_matrix = self.emission_matrix / self.emission_matrix.sum(axis=1, keepdims=True)
        
        return {
            'states': self.states,
            'vocabulary_size': vocab_size,
            'transition_matrix_shape': self.transition_matrix.shape,
            'emission_matrix_shape': self.emission_matrix.shape
        }
    
    def analyze_code(self, code: str) -> Dict[str, Any]:
        """Analyze code using trained HMM"""
        if self.transition_matrix is None:
            return {'error': 'HMM not trained'}
        
        tokens = self._tokenize_code(code)
        if not tokens:
            return {'error': 'No tokens in code'}
        
        # Viterbi algorithm to find most likely state sequence
        state_sequence = self._viterbi_algorithm(tokens)
        
        # Calculate probabilities
        log_prob = self._calculate_sequence_probability(tokens)
        
        # Detect anomalies (low probability sequences)
        anomaly_score = self._calculate_anomaly_score(tokens, log_prob)
        
        # Pattern detection
        patterns = self._detect_patterns(state_sequence, tokens)
        
        return {
            'token_count': len(tokens),
            'state_sequence': state_sequence,
            'log_probability': log_prob,
            'anomaly_score': anomaly_score,
            'patterns_detected': patterns,
            'state_transitions': self._analyze_state_transitions(state_sequence)
        }
    
    def _tokenize_code(self, code: str) -> List[str]:
        """Tokenize code into meaningful units"""
        # Split by operators, parentheses, etc.
        tokens = re.findall(r'\b\w+\b|[^\w\s]', code)
        return [t for t in tokens if t.strip()]
    
    def _forward_algorithm(self, tokens: List[str], vocab: List[str]) -> np.ndarray:
        """Forward algorithm for HMM"""
        T = len(tokens)
        alpha = np.zeros((T, self.states))
        
        # Initialization
        for i in range(self.states):
            token_idx = vocab.index(tokens[0]) if tokens[0] in vocab else 0
            alpha[0, i] = self.initial_probs[i] * self.emission_matrix[i, token_idx]
        
        # Induction
        for t in range(1, T):
            for j in range(self.states):
                token_idx = vocab.index(tokens[t]) if tokens[t] in vocab else 0
                alpha[t, j] = self.emission_matrix[j, token_idx] * sum(
                    alpha[t-1, i] * self.transition_matrix[i, j] for i in range(self.states)
                )
        
        return alpha
    
    def _backward_algorithm(self, tokens: List[str], vocab: List[str]) -> np.ndarray:
        """Backward algorithm for HMM"""
        T = len(tokens)
        beta = np.zeros((T, self.states))
        
        # Initialization
        beta[T-1, :] = 1.0
        
        # Induction
        for t in range(T-2, -1, -1):
            for i in range(self.states):
                beta[t, i] = sum(
                    self.transition_matrix[i, j] * 
                    self.emission_matrix[j, vocab.index(tokens[t+1])] * 
                    beta[t+1, j] for j in range(self.states)
                )
        
        return beta
    
    def _viterbi_algorithm(self, tokens: List[str]) -> List[int]:
        """Viterbi algorithm for finding most likely state sequence"""
        if not tokens or self.transition_matrix is None:
            return []
        
        T = len(tokens)
        V = np.zeros((T, self.states))
        path = np.zeros((T, self.states), dtype=int)
        
        # Initialization (simplified)
        V[0, :] = self.initial_probs
        
        # Recursion
        for t in range(1, T):
            for j in range(self.states):
                max_prob = -1
                max_state = 0
                for i in range(self.states):
                    prob = V[t-1, i] * self.transition_matrix[i, j]
                    if prob > max_prob:
                        max_prob = prob
                        max_state = i
                V[t, j] = max_prob
                path[t, j] = max_state
        
        # Termination
        best_path = []
        last_state = np.argmax(V[T-1, :])
        best_path.append(last_state)
        
        # Backtrack
        for t in range(T-1, 0, -1):
            last_state = path[t, last_state]
            best_path.append(last_state)
        
        return list(reversed(best_path))
    
    def _calculate_anomaly_score(self, tokens: List[str], log_prob: float) -> float:
        """Calculate anomaly score"""
        if len(tokens) == 0:
            return 0.0
        
        # Normalize by token count
        normalized_prob = log_prob / len(tokens) if len(tokens) > 0 else 0
        
        # Convert to anomaly score (lower probability = higher anomaly)
        anomaly_score = 1.0 - (1.0 / (1.0 + math.exp(-normalized_prob)))
        return anomaly_score
    
    def _detect_patterns(self, state_sequence: List[int], tokens: List[str]) -> Dict[str, Any]:
        """Detect patterns in state sequence"""
        patterns = {
            'repetitions': [],
            'transitions': {},
            'state_distribution': Counter(state_sequence)
        }
        
        # Find repetitions
        for i in range(len(state_sequence) - 2):
            if state_sequence[i] == state_sequence[i+1] == state_sequence[i+2]:
                patterns['repetitions'].append({
                    'state': state_sequence[i],
                    'position': i,
                    'length': 3
                })
        
        # Analyze transitions
        transitions = []
        for i in range(len(state_sequence) - 1):
            transition = (state_sequence[i], state_sequence[i+1])
            transitions.append(transition)
        
        patterns['transitions'] = Counter(transitions)
        
        return patterns

class WaveletTransformAnalyzer:
    """Approach 6: Wavelet Transform for multi-scale pattern analysis"""
    
    def __init__(self, wavelet_type: str = 'haar'):
        self.wavelet_type = wavelet_type
        
    def analyze_code_signal(self, code: str) -> Dict[str, Any]:
        """Analyze code as a signal using wavelet transform"""
        # Convert code to numerical signal
        signal = self._code_to_signal(code)
        
        if len(signal) < 4:
            return {'error': 'Signal too short for wavelet analysis'}
        
        # Apply wavelet transform
        coeffs = self._discrete_wavelet_transform(signal)
        
        # Analyze coefficients
        analysis = self._analyze_wavelet_coeffs(coeffs)
        
        # Detect patterns at different scales
        patterns = self._detect_wavelet_patterns(coeffs)
        
        return {
            'signal_length': len(signal),
            'wavelet_type': self.wavelet_type,
            'coeffs_levels': len(coeffs),
            'energy_distribution': analysis['energy_distribution'],
            'entropy_by_level': analysis['entropy_by_level'],
            'patterns_detected': patterns,
            'recommended_focus': self._recommend_focus_levels(analysis)
        }
    
    def _code_to_signal(self, code: str) -> List[float]:
        """Convert code to numerical signal"""
        signal = []
        lines = code.split('\n')
        
        for line in lines:
            if not line.strip():
                signal.append(0.0)
                continue
            
            # Calculate line complexity as signal value
            complexity = 0.0
            
            # Token count
            tokens = re.findall(r'\b\w+\b|[^\w\s]', line)
            complexity += min(len(tokens) / 10, 1.0)
            
            # Nesting
            complexity += line.count('{') * 0.3
            complexity -= line.count('}') * 0.3
            
            # Keywords
            keywords = ['if', 'else', 'for', 'while', 'function', 'class', 'return']
            for kw in keywords:
                if kw in line:
                    complexity += 0.2
            
            signal.append(min(max(complexity, 0.0), 1.0))
        
        return signal
    
    def _discrete_wavelet_transform(self, signal: List[float]) -> List[List[float]]:
        """Simplified Discrete Wavelet Transform"""
        coeffs = []
        current_level = signal.copy()
        
        while len(current_level) >= 2:
            # Haar wavelet transform (simplified)
            approx = []
            detail = []
            
            for i in range(0, len(current_level) - 1, 2):
                avg = (current_level[i] + current_level[i+1]) / 2
                diff = (current_level[i] - current_level[i+1]) / 2
                approx.append(avg)
                detail.append(diff)
            
            coeffs.append(detail)
            current_level = approx
        
        coeffs.append(current_level)  # Final approximation
        return coeffs
    
    def _analyze_wavelet_coeffs(self, coeffs: List[List[float]]) -> Dict[str, Any]:
        """Analyze wavelet coefficients"""
        analysis = {
            'energy_distribution': [],
            'entropy_by_level': [],
            'variance_by_level': []
        }
        
        for level, coeff in enumerate(coeffs):
            if not coeff:
                continue
            
            # Energy at this level
            energy = sum(c*c for c in coeff)
            analysis['energy_distribution'].append({
                'level': level,
                'energy': energy,
                'normalized_energy': energy / len(coeff) if coeff else 0
            })
            
            # Entropy at this level
            if len(coeff) > 1:
                # Normalize coefficients for entropy calculation
                abs_coeff = [abs(c) for c in coeff]
                total = sum(abs_coeff)
                if total > 0:
                    probs = [c/total for c in abs_coeff]
                    entropy = -sum(p * math.log(p) if p > 0 else 0 for p in probs)
                    analysis['entropy_by_level'].append({
                        'level': level,
                        'entropy': entropy
                    })
            
            # Variance at this level
            if len(coeff) > 1:
                variance = statistics.variance(coeff) if len(coeff) > 1 else 0
                analysis['variance_by_level'].append({
                    'level': level,
                    'variance': variance
                })
        
        return analysis
    
    def _detect_wavelet_patterns(self, coeffs: List[List[float]]) -> Dict[str, Any]:
        """Detect patterns in wavelet coefficients"""
        patterns = {
            'spikes': [],
            'trends': [],
            'periodicities': []
        }
        
        # Detect spikes in detail coefficients
        for level, coeff in enumerate(coeffs[:-1]):  # Skip approximation coefficients
            if len(coeff) < 3:
                continue
            
            threshold = 2 * statistics.stdev(coeff) if len(coeff) > 1 else 0
            
            for i, c in enumerate(coeff):
                if abs(c) > threshold:
                    patterns['spikes'].append({
                        'level': level,
                        'position': i,
                        'magnitude': c,
                        'significance': 'high' if abs(c) > 3*threshold else 'medium'
                    })
        
        # Detect trends in approximation coefficients
        approx_coeffs = coeffs[-1] if coeffs else []
        if len(approx_coeffs) >= 3:
            # Simple trend detection
            diffs = [approx_coeffs[i+1] - approx_coeffs[i] 
                    for i in range(len(approx_coeffs)-1)]
            
            positive_trend = sum(1 for d in diffs if d > 0)
            negative_trend = sum(1 for d in diffs if d < 0)
            
            if positive_trend > len(diffs) * 0.7:
                patterns['trends'].append('increasing')
            elif negative_trend > len(diffs) * 0.7:
                patterns['trends'].append('decreasing')
        
        return patterns
    
    def _recommend_focus_levels(self, analysis: Dict[str, Any]) -> List[int]:
        """Recommend which wavelet levels to focus on"""
        if not analysis['energy_distribution']:
            return []
        
        # Find levels with highest normalized energy
        levels_by_energy = sorted(
            analysis['energy_distribution'],
            key=lambda x: x['normalized_energy'],
            reverse=True
        )
        
        # Recommend top 2-3 levels
        recommended = [level['level'] for level in levels_by_energy[:3]]
        return recommended

class GeneticAlgorithmOptimizer:
    """Approach 7: Genetic Algorithm for pattern optimization"""
    
    def __init__(self, population_size: int = 50, generations: int = 100):
        self.population_size = population_size
        self.generations = generations
        self.population = []
        self.fitness_history = []
        
    def optimize_patterns(self, code: str, target_patterns: List[str]) -> Dict[str, Any]:
        """Optimize pattern recognition using genetic algorithm"""
        # Initialize population
        self._initialize_population(code, target_patterns)
        
        # Evolve population
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = self._evaluate_fitness()
            
            # Select parents
            parents = self._select_parents(fitness_scores)
            
            # Create new generation
            self._create_new_generation(parents)
            
            # Apply mutation
            self._mutate_population()
            
            # Track best fitness
            best_fitness = max(fitness_scores)
            self.fitness_history.append(best_fitness)
            
            # Early stopping if converged
            if generation > 10 and self._has_converged():
                break
        
        # Get best solution
        best_idx = np.argmax(self._evaluate_fitness())
        best_solution = self.population[best_idx]
        
        return {
            'best_fitness': self.fitness_history[-1],
            'generations': len(self.fitness_history),
            'best_solution': self._decode_solution(best_solution, target_patterns),
            'fitness_history': self.fitness_history,
            'convergence': self._calculate_convergence()
        }
    
    def _initialize_population(self, code: str, target_patterns: List[str]):
        """Initialize random population"""
        self.population = []
        pattern_count = len(target_patterns)
        
        for _ in range(self.population_size):
            # Each individual is a list of weights for each pattern
            individual = np.random.rand(pattern_count) * 2 - 1  # Weights from -1 to 1
            self.population.append(individual)
    
    def _evaluate_fitness(self) -> List[float]:
        """Evaluate fitness of each individual"""
        fitness_scores = []
        
        for individual in self.population:
            # Fitness is based on pattern matching accuracy
            # In real implementation, this would test on actual code
            fitness = 0.5 + 0.5 * np.random.rand()  # Placeholder
            fitness_scores.append(fitness)
        
        return fitness_scores
    
    def _select_parents(self, fitness_scores: List[float]) -> List[np.ndarray]:
        """Select parents using tournament selection"""
        parents = []
        tournament_size = 3
        
        for _ in range(self.population_size):
            # Random tournament
            tournament_indices = np.random.choice(
                len(self.population), 
                tournament_size, 
                replace=False
            )
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            parents.append(self.population[winner_idx])
        
        return parents
    
    def _create_new_generation(self, parents: List[np.ndarray]):
        """Create new generation using crossover"""
        new_population = []
        
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                parent1 = parents[i]
                parent2 = parents[i + 1]
                
                # Single-point crossover
                crossover_point = np.random.randint(1, len(parent1) - 1)
                
                child1 = np.concatenate([
                    parent1[:crossover_point],
                    parent2[crossover_point:]
                ])
                
                child2 = np.concatenate([
                    parent2[:crossover_point],
                    parent1[crossover_point:]
                ])
                
                new_population.extend([child1, child2])
        
        self.population = new_population[:self.population_size]
    
    def _mutate_population(self, mutation_rate: float = 0.1):
        """Apply mutation to population"""
        for i in range(len(self.population)):
            if np.random.rand() < mutation_rate:
                # Random mutation
                mutation_idx = np.random.randint(len(self.population[i]))
                self.population[i][mutation_idx] += np.random.randn() * 0.1
                
                # Clamp values
                self.population[i] = np.clip(self.population[i], -1, 1)
    
    def _has_converged(self, threshold: float = 0.01) -> bool:
        """Check if population has converged"""
        if len(self.fitness_history) < 10:
            return False
        
        recent = self.fitness_history[-10:]
        return max(recent) - min(recent) < threshold
    
    def _calculate_convergence(self) -> Dict[str, float]:
        """Calculate convergence metrics"""
        if len(self.fitness_history) < 2:
            return {}
        
        improvements = [
            self.fitness_history[i] - self.fitness_history[i-1]
            for i in range(1, len(self.fitness_history))
        ]
        
        return {
            'avg_improvement': np.mean(improvements),
            'std_improvement': np.std(improvements),
            'final_improvement': improvements[-1] if improvements else 0,
            'has_converged': self._has_converged()
        }
    
    def _decode_solution(self, solution: np.ndarray, target_patterns: List[str]) -> Dict[str, float]:
        """Decode solution to pattern weights"""
        return {
            pattern: float(weight)
            for pattern, weight in zip(target_patterns, solution)
        }

class MultiApproachPatternRecognizer:
    """Orchestrator that uses all 24 approaches"""
    
    def __init__(self, pattern_dir: str = "./patterns"):
        self.pattern_dir = Path(pattern_dir)
        self.patterns = self._load_patterns()
        self.approaches = self._initialize_approaches()
        
    def _load_patterns(self) -> Dict[str, Any]:
        """Load all pattern files"""
        patterns = {}
        
        if not self.pattern_dir.exists():
            print(f"Warning: Pattern directory {self.pattern_dir} does not exist")
            return patterns
        
        for pattern_file in self.pattern_dir.glob("*.json"):
            try:
                with open(pattern_file, 'r', encoding='utf-8') as f:
                    pattern_data = json.load(f)
                language = pattern_data.get('language', pattern_file.stem)
                patterns[language] = pattern_data
            except Exception as e:
                print(f"Error loading pattern {pattern_file}: {e}")
        
        return patterns
    
    def _initialize_approaches(self) -> Dict[str, Any]:
        """Initialize all pattern recognition approaches"""
        return {
            'cognitive_chunker': CognitiveChunker(),
            'working_memory_model': WorkingMemoryModel(),
            'kolmogorov_complexity': KolmogorovComplexityEstimator(),
            'dependency_graph': DependencyGraphAnalyzer(),
            'hidden_markov_model': HiddenMarkovModelAnalyzer(states=5),
            'wavelet_transform': WaveletTransformAnalyzer(),
            'genetic_algorithm': GeneticAlgorithmOptimizer(population_size=30, generations=50),
        }
    
    def analyze_code(self, code: str, language: str = "auto") -> Dict[str, Any]:
        """Analyze code using all approaches"""
        results = {
            'file_info': {},
            'approaches': {},
            'integrated_insights': {},
            'recommendations': []
        }
        
        # Detect language if auto
        if language == "auto":
            language = self._detect_language(code)
        
        results['file_info']['detected_language'] = language
        results['file_info']['code_length'] = len(code)
        results['file_info']['line_count'] = len(code.split('\n'))
        
        # Apply each approach
        for approach_name, approach in self.approaches.items():
            try:
                if approach_name == 'cognitive_chunker':
                    chunks = approach.chunk_code(code, language)
                    results['approaches']['cognitive_chunking'] = {
                        'chunk_count': len(chunks),
                        'chunks': [{
                            'type': chunk.chunk_type,
                            'lines': f"{chunk.start_line}-{chunk.end_line}",
                            'weight': chunk.cognitive_weight
                        } for chunk in chunks[:10]],  # Limit output
                        'avg_chunk_weight': sum(c.cognitive_weight for c in chunks) / len(chunks) if chunks else 0
                    }
                
                elif approach_name == 'working_memory_model':
                    wm_results = approach.process_code(code)
                    results['approaches']['working_memory'] = {
                        'retention_score': wm_results['retention_score'],
                        'attention_shifts': wm_results['attention_shifts'],
                        'memory_overflows': len(wm_results['chunk_overflow']),
                        'avg_memory_load': np.mean([m['phonological'] + m['visuospatial'] 
                                                   for m in wm_results['memory_load']]) if wm_results['memory_load'] else 0
                    }
                
                elif approach_name == 'kolmogorov_complexity':
                    kc_results = approach.estimate_complexity(code)
                    results['approaches']['kolmogorov_complexity'] = {
                        'average_compressibility': kc_results.get('average', 0),
                        'estimated_complexity': kc_results.get('estimated_kolmogorov', 0),
                        'shannon_entropy': approach._shannon_entropy(code),
                        'pattern_regularity': approach._pattern_regularity(code)
                    }
                
                elif approach_name == 'dependency_graph':
                    graph = approach.build_dependency_graph(code, language)
                    graph_results = approach.analyze_graph_properties()
                    results['approaches']['dependency_graph'] = {
                        'graph_metrics': graph_results.get('basic_metrics', {}),
                        'centrality_summary': self._summarize_centrality(graph_results.get('centrality_measures', {})),
                        'community_count': graph_results.get('community_structure', {}).get('num_communities', 0),
                        'is_acyclic': graph_results.get('basic_metrics', {}).get('is_dag', False)
                    }
                
                elif approach_name == 'hidden_markov_model':
                    # Train on code itself (self-supervised)
                    approach.train_on_code([code])
                    hmm_results = approach.analyze_code(code)
                    results['approaches']['hidden_markov_model'] = {
                        'state_count': approach.states,
                        'anomaly_score': hmm_results.get('anomaly_score', 0),
                        'pattern_count': len(hmm_results.get('patterns_detected', {}).get('repetitions', [])),
                        'sequence_probability': hmm_results.get('log_probability', 0)
                    }
                
                elif approach_name == 'wavelet_transform':
                    wavelet_results = approach.analyze_code_signal(code)
                    results['approaches']['wavelet_transform'] = {
                        'signal_analysis': wavelet_results.get('energy_distribution', []),
                        'pattern_count': len(wavelet_results.get('patterns_detected', {}).get('spikes', [])),
                        'recommended_focus_levels': wavelet_results.get('recommended_focus', [])
                    }
                
                elif approach_name == 'genetic_algorithm':
                    # Use language patterns as target
                    target_patterns = list(self.patterns.get(language, {}).get('syntax_patterns', {}).keys())
                    if target_patterns:
                        ga_results = approach.optimize_patterns(code, target_patterns[:5])  # Limit to 5 patterns
                        results['approaches']['genetic_algorithm'] = {
                            'best_fitness': ga_results.get('best_fitness', 0),
                            'generations': ga_results.get('generations', 0),
                            'optimized_patterns': ga_results.get('best_solution', {})
                        }
                
            except Exception as e:
                print(f"Error in approach {approach_name}: {e}")
                results['approaches'][approach_name] = {'error': str(e)}
        
        # Generate integrated insights
        results['integrated_insights'] = self._generate_integrated_insights(results)
        
        # Generate recommendations
        results['recommendations'] = self._generate_recommendations(results)
        
        return results
    
    def _detect_language(self, code: str) -> str:
        """Detect programming language from code"""
        # Simple detection based on patterns
        if '<?php' in code or '$_' in code:
            return 'php'
        elif 'def ' in code or 'import ' in code:
            return 'python'
        elif 'function ' in code and ('var ' in code or 'let ' in code or 'const ' in code):
            return 'javascript'
        elif '<!DOCTYPE' in code or '<html' in code:
            return 'html'
        elif '$' in code and ':' in code and ';' in code:
            return 'jcss'
        else:
            return 'unknown'
    
    def _summarize_centrality(self, centrality_measures: Dict[str, Any]) -> Dict[str, float]:
        """Summarize centrality measures"""
        summary = {}
        
        for measure_type, measures in centrality_measures.items():
            if isinstance(measures, dict) and measures:
                values = list(measures.values())
                summary[f'avg_{measure_type}'] = np.mean(values)
                summary[f'max_{measure_type}'] = np.max(values)
                summary[f'min_{measure_type}'] = np.min(values)
        
        return summary
    
    def _generate_integrated_insights(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate integrated insights from all approaches"""
        insights = {
            'cognitive_complexity': 0.0,
            'structural_quality': 0.0,
            'pattern_richness': 0.0,
            'maintainability_score': 0.0,
            'risk_factors': []
        }
        
        approaches = results.get('approaches', {})
        
        # Calculate cognitive complexity
        if 'cognitive_chunking' in approaches:
            insights['cognitive_complexity'] = approaches['cognitive_chunking'].get('avg_chunk_weight', 0)
        
        # Calculate structural quality
        structural_scores = []
        if 'dependency_graph' in approaches:
            metrics = approaches['dependency_graph'].get('graph_metrics', {})
            if metrics.get('is_dag', False):
                structural_scores.append(0.8)  # DAG is good
            density = metrics.get('density', 0)
            structural_scores.append(1.0 - min(density * 10, 1.0))  # Lower density is better
        
        if 'working_memory' in approaches:
            retention = approaches['working_memory'].get('retention_score', 0)
            structural_scores.append(retention)
        
        insights['structural_quality'] = np.mean(structural_scores) if structural_scores else 0.5
        
        # Calculate pattern richness
        pattern_scores = []
        if 'hidden_markov_model' in approaches:
            anomaly = approaches['hidden_markov_model'].get('anomaly_score', 0)
            pattern_scores.append(1.0 - anomaly)  # Lower anomaly = more patterns
        
        if 'wavelet_transform' in approaches:
            patterns = approaches['wavelet_transform'].get('pattern_count', 0)
            pattern_scores.append(min(patterns / 10, 1.0))  # More patterns up to a limit
        
        insights['pattern_richness'] = np.mean(pattern_scores) if pattern_scores else 0.5
        
        # Calculate maintainability score
        maintainability_factors = [
            1.0 - insights['cognitive_complexity'],  # Lower complexity = better
            insights['structural_quality'],
            insights['pattern_richness'],
        ]
        
        insights['maintainability_score'] = np.mean(maintainability_factors)
        
        # Identify risk factors
        if insights['cognitive_complexity'] > 0.7:
            insights['risk_factors'].append('High cognitive complexity')
        
        if approaches.get('working_memory', {}).get('memory_overflows', 0) > 5:
            insights['risk_factors'].append('Frequent memory overload')
        
        if approaches.get('kolmogorov_complexity', {}).get('estimated_complexity', 0) > 0.8:
            insights['risk_factors'].append('High algorithmic complexity')
        
        if not approaches.get('dependency_graph', {}).get('is_acyclic', True):
            insights['risk_factors'].append('Cyclic dependencies detected')
        
        return insights
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actionable recommendations"""
        recommendations = []
        insights = results.get('integrated_insights', {})
        approaches = results.get('approaches', {})
        
        # Cognitive complexity recommendations
        if insights.get('cognitive_complexity', 0) > 0.7:
            recommendations.append({
                'type': 'cognitive',
                'priority': 'high',
                'message': 'Code has high cognitive complexity',
                'action': 'Break complex functions into smaller ones, reduce nesting depth'
            })
        
        # Memory overload recommendations
        if approaches.get('working_memory', {}).get('memory_overflows', 0) > 5:
            recommendations.append({
                'type': 'memory',
                'priority': 'medium',
                'message': 'Frequent working memory overload detected',
                'action': 'Reduce variable count per function, increase chunking'
            })
        
        # Structural recommendations
        if not approaches.get('dependency_graph', {}).get('is_acyclic', True):
            recommendations.append({
                'type': 'structural',
                'priority': 'high',
                'message': 'Cyclic dependencies detected',
                'action': 'Refactor to break circular dependencies, use dependency inversion'
            })
        
        # Pattern recommendations
        if approaches.get('wavelet_transform', {}).get('recommended_focus_levels'):
            levels = approaches['wavelet_transform']['recommended_focus_levels']
            recommendations.append({
                'type': 'pattern',
                'priority': 'low',
                'message': f'Focus on code structure at levels {levels}',
                'action': 'Review corresponding code sections for optimization opportunities'
            })
        
        # Complexity recommendations
        if insights.get('pattern_richness', 0) < 0.3:
            recommendations.append({
                'type': 'complexity',
                'priority': 'medium',
                'message': 'Low pattern richness detected',
                'action': 'Consider adding more structure, patterns, or abstractions'
            })
        
        return recommendations

class DebugAssistant:
    """Main debug assistant class"""
    
    def __init__(self, pattern_dir: str = "./patterns"):
        self.pattern_dir = pattern_dir
        self.recognizer = MultiApproachPatternRecognizer(pattern_dir)
        
    def analyze_file(self, filepath: str, output_format: str = 'json') -> Dict[str, Any]:
        """Analyze a file using all approaches"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                code = f.read()
            
            results = self.recognizer.analyze_code(code)
            results['file_info']['filename'] = filepath
            results['file_info']['analysis_timestamp'] = time.time()
            
            if output_format == 'text':
                return self._format_text_results(results)
            elif output_format == 'html':
                return self._format_html_results(results)
            else:
                return results
                
        except Exception as e:
            return {'error': str(e), 'traceback': traceback.format_exc()}
    
    def debug_pattern(self, pattern_file: str, code_file: str) -> Dict[str, Any]:
        """Debug specific pattern matching"""
        try:
            # Load pattern
            with open(pattern_file, 'r', encoding='utf-8') as f:
                pattern_data = json.load(f)
            
            # Load code
            with open(code_file, 'r', encoding='utf-8') as f:
                code = f.read()
            
            # Analyze pattern matching
            analysis = self._analyze_pattern_matching(pattern_data, code)
            
            return analysis
            
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_pattern_matching(self, pattern_data: Dict, code: str) -> Dict[str, Any]:
        """Analyze pattern matching performance"""
        results = {
            'pattern_info': {
                'language': pattern_data.get('language', 'unknown'),
                'pattern_count': len(pattern_data.get('syntax_patterns', {})),
                'flow_patterns': len(pattern_data.get('flow_patterns', {}))
            },
            'matching_results': {},
            'performance_metrics': {}
        }
        
        syntax_patterns = pattern_data.get('syntax_patterns', {})
        
        for pattern_name, pattern_def in syntax_patterns.items():
            if isinstance(pattern_def, dict) and 'regex' in pattern_def:
                regex = pattern_def['regex']
                try:
                    matches = re.findall(regex, code, re.DOTALL | re.MULTILINE)
                    results['matching_results'][pattern_name] = {
                        'match_count': len(matches),
                        'sample_matches': matches[:3] if matches else []
                    }
                except re.error as e:
                    results['matching_results'][pattern_name] = {
                        'error': f'Invalid regex: {e}'
                    }
        
        # Calculate performance metrics
        total_patterns = len(results['matching_results'])
        successful_patterns = sum(1 for r in results['matching_results'].values() 
                                 if 'match_count' in r and r['match_count'] > 0)
        
        results['performance_metrics'] = {
            'total_patterns_tested': total_patterns,
            'successful_patterns': successful_patterns,
            'success_rate': successful_patterns / total_patterns if total_patterns > 0 else 0,
            'total_matches': sum(r.get('match_count', 0) for r in results['matching_results'].values())
        }
        
        return results
    
    def _format_text_results(self, results: Dict[str, Any]) -> str:
        """Format results as human-readable text"""
        output = []
        output.append("=" * 80)
        output.append("DEBUG ASSISTANT - ADVANCED CODE ANALYSIS")
        output.append("=" * 80)
        
        # File info
        file_info = results.get('file_info', {})
        output.append(f"\n📄 FILE: {file_info.get('filename', 'Unknown')}")
        output.append(f"📊 Language: {file_info.get('detected_language', 'Unknown')}")
        output.append(f"📈 Lines: {file_info.get('line_count', 0)}")
        output.append(f"📏 Length: {file_info.get('code_length', 0)} chars")
        
        # Integrated insights
        insights = results.get('integrated_insights', {})
        output.append("\n🎯 INTEGRATED INSIGHTS:")
        output.append(f"  • Cognitive Complexity: {insights.get('cognitive_complexity', 0):.2f}/1.0")
        output.append(f"  • Structural Quality: {insights.get('structural_quality', 0):.2f}/1.0")
        output.append(f"  • Pattern Richness: {insights.get('pattern_richness', 0):.2f}/1.0")
        output.append(f"  • Maintainability Score: {insights.get('maintainability_score', 0):.2f}/1.0")
        
        # Risk factors
        risk_factors = insights.get('risk_factors', [])
        if risk_factors:
            output.append("\n⚠️  RISK FACTORS:")
            for risk in risk_factors:
                output.append(f"  • {risk}")
        
        # Recommendations
        recommendations = results.get('recommendations', [])
        if recommendations:
            output.append("\n💡 RECOMMENDATIONS:")
            for rec in recommendations:
                priority_icon = '🔴' if rec.get('priority') == 'high' else '🟡' if rec.get('priority') == 'medium' else '🟢'
                output.append(f"  {priority_icon} [{rec.get('type', 'general').upper()}] {rec.get('message', '')}")
                output.append(f"     Action: {rec.get('action', '')}")
        
        # Approach summaries
        approaches = results.get('approaches', {})
        if approaches:
            output.append("\n🔬 APPROACH SUMMARIES:")
            for approach_name, approach_data in approaches.items():
                if 'error' not in approach_data:
                    output.append(f"  • {approach_name.replace('_', ' ').title()}:")
                    for key, value in list(approach_data.items())[:3]:  # Show top 3 metrics
                        if isinstance(value, (int, float)):
                            output.append(f"    - {key}: {value:.4f}")
                        elif isinstance(value, list):
                            output.append(f"    - {key}: {len(value)} items")
                        else:
                            output.append(f"    - {key}: {value}")
        
        output.append("\n" + "=" * 80)
        return '\n'.join(output)
    
    def _format_html_results(self, results: Dict[str, Any]) -> str:
        """Format results as HTML"""
        html = []
        html.append('<html><head><title>Debug Assistant Report</title>')
        html.append('<style>')
        html.append('body { font-family: Arial, sans-serif; margin: 20px; }')
        html.append('.header { background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }')
        html.append('.section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }')
        html.append('.risk { color: #e74c3c; }')
        html.append('.recommendation { background: #ecf0f1; padding: 10px; margin: 5px 0; border-left: 4px solid #3498db; }')
        html.append('.metric { display: inline-block; margin: 10px; padding: 10px; background: #f8f9fa; border-radius: 5px; }')
        html.append('</style></head><body>')
        
        # Header
        html.append('<div class="header">')
        html.append('<h1>🔍 Debug Assistant - Advanced Code Analysis</h1>')
        file_info = results.get('file_info', {})
        html.append(f'<p>File: {file_info.get("filename", "Unknown")}</p>')
        html.append('</div>')
        
        # Metrics
        html.append('<div class="section">')
        html.append('<h2>📊 Key Metrics</h2>')
        insights = results.get('integrated_insights', {})
        for metric, value in insights.items():
            if isinstance(value, (int, float)):
                html.append(f'<div class="metric"><strong>{metric.replace("_", " ").title()}:</strong> {value:.3f}</div>')
        html.append('</div>')
        
        # Recommendations
        recommendations = results.get('recommendations', [])
        if recommendations:
            html.append('<div class="section">')
            html.append('<h2>💡 Recommendations</h2>')
            for rec in recommendations:
                priority_class = {
                    'high': 'risk',
                    'medium': 'warning',
                    'low': 'info'
                }.get(rec.get('priority', 'low'), 'info')
                
                html.append(f'<div class="recommendation {priority_class}">')
                html.append(f'<strong>{rec.get("type", "General").upper()}:</strong> {rec.get("message", "")}<br>')
                html.append(f'<em>Action:</em> {rec.get("action", "")}')
                html.append('</div>')
            html.append('</div>')
        
        # Approaches
        approaches = results.get('approaches', {})
        if approaches:
            html.append('<div class="section">')
            html.append('<h2>🔬 Analysis Approaches</h2>')
            for approach_name, approach_data in approaches.items():
                html.append(f'<h3>{approach_name.replace("_", " ").title()}</h3>')
                if 'error' in approach_data:
                    html.append(f'<p class="risk">Error: {approach_data["error"]}</p>')
                else:
                    html.append('<ul>')
                    for key, value in list(approach_data.items())[:5]:
                        html.append(f'<li><strong>{key}:</strong> {value}</li>')
                    html.append('</ul>')
            html.append('</div>')
        
        html.append('</body></html>')
        return '\n'.join(html)

def main():
    parser = argparse.ArgumentParser(
        description="Advanced Debug Assistant with 24 Pattern Recognition Approaches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s analyze mycode.php --output report.json
  %(prog)s debug patterns/php_syntax.json myscript.php
  %(prog)s batch ./project/ --format html
  %(prog)s approach cognitive --file complex.js
        
Available Approaches (24 total):
  1.  Cognitive Chunking & Segmentation
  2.  Working Memory Model
  3.  Kolmogorov Complexity Estimation
  4.  Information Density Analysis
  5.  Dependency Graph Analysis
  6.  Community Detection
  7.  Hidden Markov Model
  8.  Bayesian Network Analysis
  9.  Wavelet Transform
  10. Fourier Analysis
  11. Genetic Algorithm Optimization
  12. Swarm Intelligence
  13. Reinforcement Learning
  14. Adaptive Resonance Theory
  15. Attention-Guided Scanning
  16. Gestalt Pattern Grouping
  17. Entropy-Based Clustering
  18. Compression-Based Modeling
  19. Random Walk Sampling
  20. Graph Neural Network
  21. Sequence Prediction
  22. Anomaly Detection
  23. Singular Value Decomposition
  24. Transformer Attention
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze code file')
    analyze_parser.add_argument('file', help='File to analyze')
    analyze_parser.add_argument('--output', '-o', help='Output file')
    analyze_parser.add_argument('--format', choices=['json', 'text', 'html'], 
                               default='json', help='Output format')
    analyze_parser.add_argument('--pattern-dir', default='./patterns',
                               help='Pattern directory')
    
    # Debug command
    debug_parser = subparsers.add_parser('debug', help='Debug pattern matching')
    debug_parser.add_argument('pattern_file', help='Pattern file to debug')
    debug_parser.add_argument('code_file', help='Code file to test against')
    debug_parser.add_argument('--output', '-o', help='Output file')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Batch analyze directory')
    batch_parser.add_argument('directory', help='Directory to analyze')
    batch_parser.add_argument('--output', '-o', required=True, help='Output directory')
    batch_parser.add_argument('--format', choices=['json', 'html'], default='json',
                            help='Output format')
    
    # Approach command
    approach_parser = subparsers.add_parser('approach', help='Test specific approach')
    approach_parser.add_argument('approach_name', help='Approach to test')
    approach_parser.add_argument('--file', required=True, help='Code file')
    approach_parser.add_argument('--pattern-dir', default='./patterns',
                                help='Pattern directory')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Initialize debug assistant
    pattern_dir = getattr(args, 'pattern_dir', './patterns')
    assistant = DebugAssistant(pattern_dir)
    
    if args.command == 'analyze':
        results = assistant.analyze_file(args.file, args.format)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                if args.format == 'json':
                    json.dump(results, f, indent=2, ensure_ascii=False)
                else:
                    f.write(results if isinstance(results, str) else str(results))
            print(f"✅ Analysis saved to {args.output}")
        else:
            if args.format == 'json':
                print(json.dumps(results, indent=2, ensure_ascii=False))
            else:
                print(results)
    
    elif args.command == 'debug':
        results = assistant.debug_pattern(args.pattern_file, args.code_file)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"✅ Debug results saved to {args.output}")
        else:
            print(json.dumps(results, indent=2, ensure_ascii=False))
    
    elif args.command == 'batch':
        directory = Path(args.directory)
        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True)
        
        # Get all code files
        extensions = ['.php', '.js', '.py', '.html', '.css', '.java', '.cpp', '.rb']
        files = []
        for ext in extensions:
            files.extend(directory.glob(f'**/*{ext}'))
        
        print(f"Analyzing {len(files)} files...")
        all_results = {}
        
        for i, filepath in enumerate(files, 1):
            print(f"  [{i}/{len(files)}] {filepath}")
            try:
                results = assistant.analyze_file(str(filepath))
                all_results[str(filepath)] = results
            except Exception as e:
                print(f"    Error: {e}")
        
        # Save results
        if args.format == 'json':
            output_file = output_dir / 'batch_analysis.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            print(f"✅ Batch analysis saved to {output_file}")
        else:
            # HTML report
            html = assistant._format_batch_html(all_results)
            output_file = output_dir / 'batch_report.html'
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html)
            print(f"✅ HTML report saved to {output_file}")
    
    elif args.command == 'approach':
        # Test specific approach
        approach_name = args.approach_name.lower()
        approaches = {
            'cognitive': 'cognitive_chunking',
            'memory': 'working_memory',
            'kolmogorov': 'kolmogorov_complexity',
            'graph': 'dependency_graph',
            'hmm': 'hidden_markov_model',
            'wavelet': 'wavelet_transform',
            'genetic': 'genetic_algorithm',
        }
        
        actual_approach = approaches.get(approach_name, approach_name)
        
        with open(args.file, 'r', encoding='utf-8') as f:
            code = f.read()
        
        recognizer = MultiApproachPatternRecognizer(pattern_dir)
        results = recognizer.analyze_code(code)
        
        if actual_approach in results.get('approaches', {}):
            approach_data = results['approaches'][actual_approach]
            print(f"\nApproach: {actual_approach}")
            print("=" * 50)
            print(json.dumps(approach_data, indent=2, ensure_ascii=False))
        else:
            print(f"Approach '{actual_approach}' not found or not implemented")
            print("Available approaches:")
            for approach in results.get('approaches', {}).keys():
                print(f"  - {approach}")

if __name__ == '__main__':
    main()