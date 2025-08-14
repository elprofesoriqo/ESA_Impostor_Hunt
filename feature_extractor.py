import re
import logging
from typing import List
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, entropy
import textstat


class FeatureExtractor:
    """Advanced feature extraction with NaN handling for fake text detection."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        
    def safe_division(self, a: float, b: float, default: float = 0.0) -> float:
        """Safe division with NaN handling."""
        try:
            if b == 0 or np.isnan(a) or np.isnan(b):
                return default
            result = a / b
            return result if not np.isnan(result) else default
        except:
            return default
            
    def safe_stat(self, values: List[float], stat_func, default: float = 0.0) -> float:
        """Safe statistical calculation with NaN handling."""
        try:
            if not values or len(values) == 0:
                return default
            clean_values = [v for v in values if not np.isnan(v) and np.isfinite(v)]
            if len(clean_values) == 0:
                return default
            result = stat_func(clean_values)
            return result if not np.isnan(result) and np.isfinite(result) else default
        except:
            return default
            
    def preprocess_text(self, text: str) -> str:
        """Advanced preprocessing with robust normalization."""
        if not text or pd.isna(text):
            return ""
            
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Scientific notation normalization
        text = re.sub(r'\b\d+(?:\.\d+)?[eE][+-]?\d+\b', 'SCIENTIFIC_NUM', text)
        
        # Measurement normalization
        text = re.sub(r'\b\d+(?:\.\d+)?\s*(?:kilometers?|km)\b', 'DIST_KM', text, flags=re.IGNORECASE)
        text = re.sub(r'\b\d+(?:\.\d+)?\s*(?:meters?|m)\b', 'DIST_M', text, flags=re.IGNORECASE)
        text = re.sub(r'\b\d+(?:\.\d+)?\s*(?:centimeters?|cm)\b', 'DIST_CM', text, flags=re.IGNORECASE)
        text = re.sub(r'\b\d+(?:\.\d+)?\s*(?:kilograms?|kg)\b', 'MASS_KG', text, flags=re.IGNORECASE)
        text = re.sub(r'\b\d+(?:\.\d+)?\s*(?:grams?|g)\b', 'MASS_G', text, flags=re.IGNORECASE)
        text = re.sub(r'\b\d+(?:\.\d+)?\s*(?:celsius|fahrenheit|°[CF]|kelvin|K)\b', 'TEMP', text, flags=re.IGNORECASE)
        text = re.sub(r'\b\d+(?:\.\d+)?\s*(?:hertz|Hz|MHz|GHz|kHz)\b', 'FREQ', text, flags=re.IGNORECASE)
        text = re.sub(r'\b\d+(?:\.\d+)?\s*(?:watts?|W|kW|MW)\b', 'POWER', text, flags=re.IGNORECASE)
        
        # Temporal normalization
        text = re.sub(r'\b(?:19|20)\d{2}\b', 'YEAR', text)
        text = re.sub(r'\b\d{1,2}:\d{2}(?::\d{2})?\b', 'TIME', text)
        
        # Number precision normalization
        text = re.sub(r'\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b', 'LARGE_NUM', text)
        text = re.sub(r'\b\d+\.\d{3,}\b', 'PRECISION_NUM', text)
        text = re.sub(r'\b\d+\.\d{1,2}\b', 'DECIMAL_NUM', text)
        text = re.sub(r'\b\d+\b', 'INTEGER', text)
        
        # Technical patterns
        text = re.sub(r'\b[A-Z]{2,}[-_]\w+\b', 'TECH_ID', text)
        text = re.sub(r'\b[A-Z]{3,}\b', 'ACRONYM', text)
        text = re.sub(r'http[s]?://\S+', 'URL', text)
        text = re.sub(r'\[[^\]]*\d+[^\]]*\]', 'CITATION', text)
        
        return text.lower()
        
    def extract_linguistic_features(self, text1: str, text2: str) -> List[float]:
        """Extract linguistic features comparing two texts."""
        features = []
        
        if not text1: text1 = ""
        if not text2: text2 = ""
        
        # Character-level analysis
        features.extend([
            len(text1) - len(text2),
            self.safe_division(len(text1), len(text2), 1.0),
            self.safe_division(abs(len(text1) - len(text2)), len(text1) + len(text2), 0.0),
            len(set(text1)) - len(set(text2)),
        ])
        
        # Word-level analysis
        words1 = text1.split() if text1 else []
        words2 = text2.split() if text2 else []
        
        features.extend([
            len(words1) - len(words2),
            self.safe_division(len(words1), len(words2), 1.0),
            self.safe_division(abs(len(words1) - len(words2)), len(words1) + len(words2), 0.0),
        ])
        
        # Word length analysis
        if words1 and words2:
            word_lens1 = [len(w) for w in words1]
            word_lens2 = [len(w) for w in words2]
            
            features.extend([
                self.safe_stat(word_lens1, np.mean) - self.safe_stat(word_lens2, np.mean),
                self.safe_stat(word_lens1, np.std) - self.safe_stat(word_lens2, np.std),
                self.safe_stat(word_lens1, np.median) - self.safe_stat(word_lens2, np.median),
                max(word_lens1) - max(word_lens2),
                min(word_lens1) - min(word_lens2),
            ])
        else:
            features.extend([0.0] * 5)
            
        # Sentence analysis
        sents1 = [s.strip() for s in re.split(r'[.!?]+', text1) if s.strip()]
        sents2 = [s.strip() for s in re.split(r'[.!?]+', text2) if s.strip()]
        
        features.extend([
            len(sents1) - len(sents2),
            self.safe_division(len(sents1), len(sents2), 1.0),
        ])
        
        if sents1 and sents2:
            sent_lens1 = [len(s.split()) for s in sents1]
            sent_lens2 = [len(s.split()) for s in sents2]
            
            features.extend([
                self.safe_stat(sent_lens1, np.mean) - self.safe_stat(sent_lens2, np.mean),
                self.safe_stat(sent_lens1, np.std) - self.safe_stat(sent_lens2, np.std),
            ])
        else:
            features.extend([0.0, 0.0])
            
        # Punctuation analysis
        punct_patterns = [r'\.', r',', r';', r':', r'!', r'\?', r'["\']', r'[()]', r'[\[\]]', r'[-–—]']
        
        for pattern in punct_patterns:
            count1 = len(re.findall(pattern, text1))
            count2 = len(re.findall(pattern, text2))
            features.append(count1 - count2)
            
        # Capital letters
        caps1 = sum(1 for c in text1 if c.isupper())
        caps2 = sum(1 for c in text2 if c.isupper())
        features.extend([
            caps1 - caps2,
            self.safe_division(caps1, len(text1), 0.0) - self.safe_division(caps2, len(text2), 0.0),
        ])
        
        return features
        
    def extract_vocabulary_features(self, text1: str, text2: str) -> List[float]:
        """Extract vocabulary-based features."""
        features = []
        
        words1 = text1.split() if text1 else []
        words2 = text2.split() if text2 else []
        
        vocab1 = set(w.lower() for w in words1)
        vocab2 = set(w.lower() for w in words2)
        
        # Vocabulary size and diversity
        features.extend([
            len(vocab1) - len(vocab2),
            self.safe_division(len(vocab1), len(vocab2), 1.0),
            self.safe_division(len(vocab1), len(words1), 0.0) - self.safe_division(len(vocab2), len(words2), 0.0),
        ])
        
        # Vocabulary overlap
        if vocab1 or vocab2:
            intersection = len(vocab1 & vocab2)
            union = len(vocab1 | vocab2)
            
            features.extend([
                self.safe_division(intersection, union, 0.0),
                self.safe_division(intersection, len(vocab1), 0.0),
                self.safe_division(intersection, len(vocab2), 0.0),
                len(vocab1 - vocab2),
                len(vocab2 - vocab1),
            ])
        else:
            features.extend([0.0] * 5)
            
        # Word frequency analysis
        freq1 = Counter(w.lower() for w in words1)
        freq2 = Counter(w.lower() for w in words2)
        
        # Entropy calculation
        if freq1.values():
            probs1 = np.array(list(freq1.values())) / sum(freq1.values())
            entropy1 = self.safe_stat(probs1, entropy, 0.0)
        else:
            entropy1 = 0.0
            
        if freq2.values():
            probs2 = np.array(list(freq2.values())) / sum(freq2.values())
            entropy2 = self.safe_stat(probs2, entropy, 0.0)
        else:
            entropy2 = 0.0
            
        features.append(entropy1 - entropy2)
        
        # Hapax legomena
        hapax1 = sum(1 for c in freq1.values() if c == 1)
        hapax2 = sum(1 for c in freq2.values() if c == 1)
        features.append(hapax1 - hapax2)
        
        return features
        
    def extract_statistical_features(self, text1: str, text2: str) -> List[float]:
        """Extract statistical features."""
        features = []
        
        # Character frequency analysis
        chars1 = [c.lower() for c in text1 if c.isalpha()]
        chars2 = [c.lower() for c in text2 if c.isalpha()]
        
        if len(chars1) > 10 and len(chars2) > 10:
            freq1 = Counter(chars1)
            freq2 = Counter(chars2)
            
            # Common characters correlation
            common_chars = sorted(set(freq1.keys()) & set(freq2.keys()))
            if len(common_chars) > 5:
                freqs1 = [freq1[c] for c in common_chars]
                freqs2 = [freq2[c] for c in common_chars]
                
                try:
                    corr, _ = pearsonr(freqs1, freqs2)
                    features.append(corr if not np.isnan(corr) else 0.0)
                except:
                    features.append(0.0)
            else:
                features.append(0.0)
        else:
            features.append(0.0)
            
        # Function word analysis
        function_categories = {
            'articles': ['the', 'a', 'an'],
            'prepositions': ['in', 'on', 'at', 'by', 'for', 'with', 'from', 'to', 'of'],
            'pronouns': ['i', 'you', 'he', 'she', 'it', 'we', 'they'],
            'conjunctions': ['and', 'but', 'or', 'so'],
            'auxiliary': ['is', 'are', 'was', 'were', 'have', 'has', 'had', 'do', 'does', 'did'],
        }
        
        words1_lower = [w.lower() for w in text1.split()]
        words2_lower = [w.lower() for w in text2.split()]
        
        for category, word_list in function_categories.items():
            count1 = sum(1 for w in words1_lower if w in word_list)
            count2 = sum(1 for w in words2_lower if w in word_list)
            
            if words1_lower and words2_lower:
                density1 = self.safe_division(count1, len(words1_lower), 0.0)
                density2 = self.safe_division(count2, len(words2_lower), 0.0)
                features.append(density1 - density2)
            else:
                features.append(0.0)
                
        # N-gram overlap
        def get_ngrams(words, n):
            if len(words) < n:
                return []
            return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
            
        for n in [2, 3, 4]:
            ngrams1 = set(get_ngrams(words1_lower, n))
            ngrams2 = set(get_ngrams(words2_lower, n))
            
            if ngrams1 or ngrams2:
                overlap = self.safe_division(len(ngrams1 & ngrams2), len(ngrams1 | ngrams2), 0.0)
                features.append(overlap)
            else:
                features.append(0.0)
                
        return features
        
    def extract_domain_features(self, text1: str, text2: str) -> List[float]:
        """Extract domain-specific features for space/science texts."""
        features = []
        
        # Space/science vocabularies
        domain_vocabularies = {
            'space': ['satellite', 'orbit', 'mission', 'spacecraft', 'probe', 'rover', 'astronaut'],
            'astronomy': ['galaxy', 'planet', 'solar', 'lunar', 'mars', 'earth', 'telescope', 'star'],
            'research': ['research', 'experiment', 'study', 'analysis', 'data', 'observation', 'measurement'],
            'instruments': ['spectrometer', 'camera', 'radar', 'sensor', 'detector', 'antenna'],
            'physics': ['temperature', 'pressure', 'radiation', 'magnetic', 'frequency', 'energy'],
            'organizations': ['nasa', 'esa', 'spacex', 'roscosmos']
        }
        
        words1_lower = set(w.lower() for w in text1.split())
        words2_lower = set(w.lower() for w in text2.split())
        
        for category, terms in domain_vocabularies.items():
            count1 = len(words1_lower & set(terms))
            count2 = len(words2_lower & set(terms))
            features.append(count1 - count2)
            
        # Technical patterns
        patterns = [
            (r'\b\d+(?:\.\d+)?\s*(?:km|m|cm|kg|g|°[CF]|Hz)\b', 'measurements'),
            (r'\b(?:19|20)\d{2}\b', 'years'),
            (r'\b\d+(?:\.\d+)?%\b', 'percentages'),
            (r'\[[^\]]*\]', 'citations'),
            (r'\b[A-Z]{2,}[-_]\w+\b', 'tech_ids'),
        ]
        
        for pattern, name in patterns:
            count1 = len(re.findall(pattern, text1, re.IGNORECASE))
            count2 = len(re.findall(pattern, text2, re.IGNORECASE))
            features.append(count1 - count2)
            
        return features
        
    def extract_readability_features(self, text1: str, text2: str) -> List[float]:
        """Extract readability features."""
        features = []
        
        if not text1 or not text2:
            return [0.0] * 6
            
        try:
            # Basic readability metrics
            metrics = [
                textstat.flesch_reading_ease,
                textstat.flesch_kincaid_grade,
                textstat.gunning_fog,
                textstat.automated_readability_index,
            ]
            
            for metric_func in metrics:
                try:
                    score1 = metric_func(text1)
                    score2 = metric_func(text2)
                    
                    if np.isnan(score1) or np.isinf(score1): score1 = 0.0
                    if np.isnan(score2) or np.isinf(score2): score2 = 0.0
                        
                    features.append(score1 - score2)
                except:
                    features.append(0.0)
                    
            # Additional complexity metrics
            try:
                avg_sent_len1 = textstat.avg_sentence_length(text1)
                avg_sent_len2 = textstat.avg_sentence_length(text2)
                
                if np.isnan(avg_sent_len1) or np.isinf(avg_sent_len1): avg_sent_len1 = 0.0
                if np.isnan(avg_sent_len2) or np.isinf(avg_sent_len2): avg_sent_len2 = 0.0
                    
                features.append(avg_sent_len1 - avg_sent_len2)
                
                syllables1 = textstat.avg_syllables_per_word(text1)
                syllables2 = textstat.avg_syllables_per_word(text2)
                
                if np.isnan(syllables1) or np.isinf(syllables1): syllables1 = 0.0
                if np.isnan(syllables2) or np.isinf(syllables2): syllables2 = 0.0
                    
                features.append(syllables1 - syllables2)
            except:
                features.extend([0.0, 0.0])
                
        except:
            features = [0.0] * 6
            
        return features
        
    def extract_all_features(self, text1: str, text2: str) -> List[float]:
        """Extract all features with NaN protection."""
        features = []
        
        try:
            features.extend(self.extract_linguistic_features(text1, text2))
            features.extend(self.extract_vocabulary_features(text1, text2))
            features.extend(self.extract_statistical_features(text1, text2))
            features.extend(self.extract_domain_features(text1, text2))
            features.extend(self.extract_readability_features(text1, text2))
        except Exception as e:
            logging.warning(f"Feature extraction error: {e}")
            features = [0.0] * 100  # Approximate expected feature count
            
        # Final NaN check and replacement
        clean_features = []
        for f in features:
            if np.isnan(f) or np.isinf(f):
                clean_features.append(0.0)
            else:
                clean_features.append(float(f))
                
        return clean_features