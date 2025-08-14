import random
import logging
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                            ExtraTreesClassifier, VotingClassifier)
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import RobustScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from scipy import sparse
from scipy.optimize import differential_evolution

from feature_extractor import FeatureExtractor
from utils import DatasetPaths, read_text_pair, list_test_article_ids


class FakeTextDetector:
    """Main detector class for fake text detection."""
    
    def __init__(self, n_splits: int = 8, n_repeats: int = 3, seed: int = 42):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.seed = seed
        self.feature_extractor = FeatureExtractor(seed)
        self.best_threshold = 0.5
        
        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)
        
    def build_tfidf_features(self, train_pairs: List[Tuple[str, str]], test_pairs: List[Tuple[str, str]]) -> Tuple[np.ndarray, np.ndarray]:
        """Build TF-IDF features with comprehensive coverage."""
        
        # Preprocess texts
        train_texts_a = [self.feature_extractor.preprocess_text(pair[0]) for pair in train_pairs]
        train_texts_b = [self.feature_extractor.preprocess_text(pair[1]) for pair in train_pairs]
        test_texts_a = [self.feature_extractor.preprocess_text(pair[0]) for pair in test_pairs]
        test_texts_b = [self.feature_extractor.preprocess_text(pair[1]) for pair in test_pairs]
        
        all_texts = train_texts_a + train_texts_b + test_texts_a + test_texts_b
        
        # TF-IDF vectorizers
        vectorizers = {
            'tfidf_1_2': TfidfVectorizer(
                max_features=20000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                stop_words='english',
                sublinear_tf=True,
                norm='l2'
            ),
            'tfidf_2_3': TfidfVectorizer(
                max_features=15000,
                ngram_range=(2, 3),
                min_df=2,
                max_df=0.90,
                stop_words='english',
                sublinear_tf=True,
                norm='l2'
            ),
            'tfidf_chars': TfidfVectorizer(
                max_features=12000,
                analyzer='char',
                ngram_range=(3, 6),
                min_df=3,
                max_df=0.95,
                sublinear_tf=True
            ),
        }
        
        # Fit vectorizers
        for name, vectorizer in vectorizers.items():
            vectorizer.fit(all_texts)
            
        # Build difference features
        def build_differences(texts_a, texts_b):
            all_features = []
            
            for name, vectorizer in vectorizers.items():
                X_a = vectorizer.transform(texts_a)
                X_b = vectorizer.transform(texts_b)
                
                # Difference operations
                X_diff = X_a - X_b
                X_abs_diff = sparse.csr_matrix(np.abs(X_diff.toarray()))
                X_product = X_a.multiply(X_b)
                
                # Safe cosine similarity
                X_a_dense = X_a.toarray()
                X_b_dense = X_b.toarray()
                
                norms_a = np.linalg.norm(X_a_dense, axis=1, keepdims=True)
                norms_b = np.linalg.norm(X_b_dense, axis=1, keepdims=True)
                
                norms_a[norms_a == 0] = 1e-8
                norms_b[norms_b == 0] = 1e-8
                
                X_a_norm = X_a_dense / norms_a
                X_b_norm = X_b_dense / norms_b
                
                cosine_sim = np.sum(X_a_norm * X_b_norm, axis=1, keepdims=True)
                cosine_sim = np.where(np.isnan(cosine_sim), 0.0, cosine_sim)
                X_cosine = sparse.csr_matrix(cosine_sim)
                
                X_combined = sparse.hstack([X_diff, X_abs_diff, X_product, X_cosine])
                all_features.append(X_combined)
                
            return sparse.hstack(all_features)
            
        X_train_tfidf = build_differences(train_texts_a, train_texts_b)
        X_test_tfidf = build_differences(test_texts_a, test_texts_b)
        
        return X_train_tfidf, X_test_tfidf
        
    def build_ensemble(self) -> VotingClassifier:
        """Build ensemble optimized for performance."""
        
        models = [
            ('lr_balanced', LogisticRegression(
                C=2.0, 
                solver='liblinear',
                max_iter=1000,
                random_state=self.seed,
                class_weight='balanced'
            )),
            ('rf_robust', RandomForestClassifier(
                n_estimators=400,
                max_depth=20,
                min_samples_split=3,
                min_samples_leaf=2,
                random_state=self.seed,
                class_weight='balanced',
                n_jobs=-1
            )),
            ('et_robust', ExtraTreesClassifier(
                n_estimators=300,
                max_depth=18,
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=self.seed,
                class_weight='balanced',
                n_jobs=-1
            )),
            ('gbm_robust', GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.08,
                max_depth=7,
                min_samples_split=4,
                min_samples_leaf=3,
                random_state=self.seed
            )),
            ('svm_robust', SVC(
                C=2.5,
                kernel='rbf',
                gamma='scale',
                probability=True,
                random_state=self.seed,
                class_weight='balanced'
            ))
        ]
        
        ensemble = VotingClassifier(
            estimators=models,
            voting='soft',
            n_jobs=-1
        )
        
        return ensemble
        
    def optimize_threshold(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """Threshold optimization with multiple objectives."""
        
        def fitness_function(threshold):
            if threshold <= 0 or threshold >= 1:
                return 1e6
                
            y_pred = (y_proba >= threshold).astype(int)
            
            try:
                f1 = f1_score(y_true, y_pred, average='weighted')
                acc = accuracy_score(y_true, y_pred)
                pred_ratio = np.mean(y_pred)
                balance_score = 1.0 - abs(pred_ratio - 0.5) * 2
                
                combined = 0.6 * f1 + 0.3 * acc + 0.1 * balance_score
                return -combined
            except:
                return 1e6
                
        result = differential_evolution(
            fitness_function,
            bounds=[(0.25, 0.75)],
            seed=self.seed,
            maxiter=30,
            popsize=10
        )
        
        return result.x[0]
        
    def run_cross_validation(self, paths: DatasetPaths) -> Tuple[List[int], np.ndarray, Dict[str, float]]:
        """Run cross-validation and return predictions and scores."""
        
        # Load data
        df = pd.read_csv(paths.train_csv_path)
        y = (df["real_text_id"].values == 1).astype(int)
        
        train_pairs = []
        for aid in df["id"].tolist():
            text1, text2 = read_text_pair(paths.train_dir, int(aid))
            train_pairs.append((text1, text2))
            
        # Load test data
        test_ids = list_test_article_ids(paths.test_dir)
        test_pairs = []
        for aid in test_ids:
            text1, text2 = read_text_pair(paths.test_dir, aid)
            test_pairs.append((text1, text2))
            
        # Build features
        X_train_tfidf, X_test_tfidf = self.build_tfidf_features(train_pairs, test_pairs)
        
        # Engineered features
        def extract_features_batch(pairs):
            features = []
            for text1, text2 in pairs:
                pair_features = self.feature_extractor.extract_all_features(text1, text2)
                features.append(pair_features)
            return np.array(features)
            
        X_train_eng = extract_features_batch(train_pairs)
        X_test_eng = extract_features_batch(test_pairs)
        
        # Handle NaN and scaling
        imputer = SimpleImputer(strategy='median')
        X_train_eng = imputer.fit_transform(X_train_eng)
        X_test_eng = imputer.transform(X_test_eng)
        
        scaler = RobustScaler()
        X_train_eng = scaler.fit_transform(X_train_eng)
        X_test_eng = scaler.transform(X_test_eng)
        
        # Combine features
        X_train = sparse.hstack([X_train_tfidf, X_train_eng]).tocsr()
        X_test = sparse.hstack([X_test_tfidf, X_test_eng]).tocsr()
        
        # Cross-validation
        rskf = RepeatedStratifiedKFold(
            n_splits=self.n_splits, 
            n_repeats=self.n_repeats, 
            random_state=self.seed
        )
        
        test_predictions = np.zeros(len(test_pairs))
        oof_predictions = np.zeros(len(train_pairs))
        cv_scores = {'accuracy': [], 'f1': [], 'precision': [], 'recall': [], 'auc': []}
        thresholds = []
        
        fold_count = 0
        total_folds = self.n_splits * self.n_repeats
        
        for train_idx, val_idx in rskf.split(X_train, y):
            fold_count += 1
            
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]
            
            # Train ensemble
            ensemble = self.build_ensemble()
            
            try:
                calibrated_ensemble = CalibratedClassifierCV(ensemble, cv=3)
                calibrated_ensemble.fit(X_fold_train, y_fold_train)
                
                # Predictions
                val_proba = calibrated_ensemble.predict_proba(X_fold_val)[:, 1]
                test_proba = calibrated_ensemble.predict_proba(X_test)[:, 1]
                
                oof_predictions[val_idx] += val_proba / self.n_repeats
                test_predictions += test_proba / total_folds
                
                # Optimize threshold
                fold_threshold = self.optimize_threshold(y_fold_val, val_proba)
                thresholds.append(fold_threshold)
                
                val_preds = (val_proba >= fold_threshold).astype(int)
                
                # Calculate metrics
                fold_acc = accuracy_score(y_fold_val, val_preds)
                fold_f1 = f1_score(y_fold_val, val_preds, average='weighted')
                fold_precision = precision_score(y_fold_val, val_preds, average='weighted', zero_division=0)
                fold_recall = recall_score(y_fold_val, val_preds, average='weighted')
                fold_auc = roc_auc_score(y_fold_val, val_proba)
                
                cv_scores['accuracy'].append(fold_acc)
                cv_scores['f1'].append(fold_f1)
                cv_scores['precision'].append(fold_precision)
                cv_scores['recall'].append(fold_recall)
                cv_scores['auc'].append(fold_auc)
                
            except Exception as e:
                logging.warning(f"Fold {fold_count} failed: {e}")
                continue
                
        # Global threshold optimization
        if thresholds:
            global_threshold = self.optimize_threshold(y, oof_predictions)
            self.best_threshold = global_threshold
        else:
            self.best_threshold = 0.5
            
        # Calculate final metrics
        mean_scores = {metric: np.mean(scores) for metric, scores in cv_scores.items()}
        
        return test_ids, test_predictions, mean_scores
        
    def train_and_predict(self, paths: DatasetPaths) -> pd.DataFrame:
        """Main training and prediction pipeline."""
        
        test_ids, test_probs, cv_scores = self.run_cross_validation(paths)
        predictions = np.where(test_probs >= self.best_threshold, 1, 2)
        return pd.DataFrame({"id": test_ids, "real_text_id": predictions}), cv_scores