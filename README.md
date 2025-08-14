# Fake Text Detection Competition - ESA DataX AI Security
https://www.kaggle.com/competitions/fake-or-real-the-impostor-hunt
## 🏆 Competition Overview

This repository contains my solution for the **ESA DataX AI Security Fake Text Detection Competition**. The challenge addresses critical AI security threats in space operations, specifically data poisoning and overreliance on LLM outputs.

### 📋 Task Description

**Objective**: Distinguish between real and fake texts in paired documents from The Messenger journal (space research, scientific devices, research results).

**Key Challenge**: 
- Each sample contains two texts (one real, one fake)
- Both texts have been significantly modified using LLMs
- Order of real/fake is random
- Subtle differences require advanced detection techniques

## 🔧 Solution Architecture

### Core Components

1. **Advanced Feature Engineering** (`feature_extractor.py:FeatureExtractor`)
   - **TF-IDF Features**: Multi-level n-gram analysis (1-2, 2-3 grams, character 3-6 grams)
   - **Linguistic Features**: Length ratios, vocabulary diversity, sentence complexity
   - **Statistical Features**: Character frequency correlations, entropy analysis
   - **Domain Features**: Space/science terminology detection
   - **Readability Features**: Flesch-Kincaid, Gunning Fog, complexity metrics

2. **Robust Ensemble Model** (`detector.py:FakeTextDetector`)
   - **5 Diverse Algorithms**:
     - Logistic Regression (balanced)
     - Random Forest (400 trees)
     - Extra Trees (300 trees)
     - Gradient Boosting (200 estimators)
     - Support Vector Machine (RBF kernel)

3. **Advanced Optimization**
   - **Calibrated Probabilities**: CalibratedClassifierCV for better uncertainty
   - **Threshold Optimization**: Differential evolution for optimal decision boundary
   - **Cross-Validation**: 7-fold × 3 repeats = 21 training runs
   - **Robust NaN Handling**: Comprehensive error protection

### Key Technical Features

```python
# Advanced text preprocessing with domain-specific normalization
text = re.sub(r'\b\d+(?:\.\d+)?[eE][+-]?\d+\b', 'SCIENTIFIC_NUM', text)
text = re.sub(r'\b\d+(?:\.\d+)?\s*(?:kilometers?|km)\b', 'DIST_KM', text)

# Multi-vectorizer TF-IDF approach
vectorizers = {
    'tfidf_1_2': TfidfVectorizer(max_features=20000, ngram_range=(1, 2)),
    'tfidf_2_3': TfidfVectorizer(max_features=15000, ngram_range=(2, 3)),
    'tfidf_chars': TfidfVectorizer(max_features=12000, analyzer='char', ngram_range=(3, 6))
}

# Comprehensive feature differences
X_diff = X_a - X_b  # Direct difference
X_abs_diff = abs(X_diff)  # Magnitude difference
X_product = X_a.multiply(X_b)  # Element-wise product
X_cosine = cosine_similarity(X_a, X_b)  # Semantic similarity
```

## 📂 Repository Structure

```
impostor_hunt/
├── Task_Description/
│   └── description.txt          # Competition description
├── train/                       # Training data (95 article pairs)
├── test/                        # Test data (400+ article pairs)
├── train.csv                    # Ground truth labels
├── main.ipynb                   # Training notebook with analysis
├── feature_extractor.py         # Advanced feature extraction class
├── detector.py                  # Main fake text detector class
├── utils.py                     # Helper functions and data structures
├── model.py                     # Legacy monolithic model (deprecated)
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## 🎯 Key Innovation Points

### 1. **Multi-Level Text Analysis**
- Character-level patterns (scientific notation, measurements)
- Word-level features (vocabulary diversity, function words)
- Sentence-level structure (complexity, readability)
- Document-level statistics (entropy, correlations)

### 2. **Domain-Aware Feature Engineering**
```python
domain_vocabularies = {
    'space': ['satellite', 'orbit', 'mission', 'spacecraft'],
    'astronomy': ['galaxy', 'planet', 'solar', 'telescope'],
    'research': ['experiment', 'study', 'analysis', 'data'],
    'physics': ['temperature', 'pressure', 'radiation', 'magnetic']
}
```

### 3. **Robust Statistical Approach**
- Comprehensive NaN handling throughout pipeline
- Multiple validation strategies
- Calibrated probability estimation
- Automated threshold optimization

### 4. **Advanced Ensemble Design**
- Diverse algorithm selection for complementary strengths
- Soft voting for probability-based decisions
- Class balancing for optimal performance
- Cross-validation integration

## 📚 References

The solution builds upon research in adversarial AI and text authenticity:

1. **ESA DataX Strategy**: Advanced data analytics for space operations
2. **AI Security Research**: Resistance against manipulative AI techniques
3. **Text Authentication**: Statistical and linguistic authenticity markers
4. **Ensemble Methods**: Robust classification through model diversity

---

*Developed for ESA DataX AI Security Competition - Addressing critical challenges in space domain AI applications.*