from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
import joblib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelTrainer:
    
    def __init__(
        self,
        random_state: int = 1,
        test_size: float = 0.25,
        cv_folds: int = 5,
        use_class_weight: bool = True
    ):
        self.random_state = random_state
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.use_class_weight = use_class_weight
        
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None
        
        self.model: Optional[LogisticRegression] = None
        self.pipeline: Optional[Pipeline] = None
        self.cv_scores: Optional[Dict[str, Any]] = None

    def split_data(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        logger.info(f"Splitting data with test_size={self.test_size}, random_state={self.random_state}")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        logger.info(f"Training set size: {len(self.X_train)}")
        logger.info(f"Test set size: {len(self.X_test)}")
        logger.info(f"Training target distribution: {self.y_train.value_counts().to_dict()}")
        logger.info(f"Test target distribution: {self.y_test.value_counts().to_dict()}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test

    def create_model(
        self,
        solver: str = 'lbfgs',
        max_iter: int = 1000,
        C: float = 1.0,
        penalty: str = 'l2'
    ) -> LogisticRegression:
        class_weight = 'balanced' if self.use_class_weight else None
        
        self.model = LogisticRegression(
            solver=solver,
            max_iter=max_iter,
            C=C,
            penalty=penalty,
            class_weight=class_weight,
            random_state=self.random_state
        )
        
        logger.info(f"Created LogisticRegression model with class_weight={class_weight}")
        
        return self.model

    def build_pipeline(self, feature_pipeline: Pipeline) -> Pipeline:
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        pipeline_steps = feature_pipeline.steps + [('classifier', self.model)]
        self.pipeline = Pipeline(pipeline_steps)
        
        logger.info("Full pipeline built successfully")
        
        return self.pipeline

    def train(self, pipeline: Optional[Pipeline] = None) -> Pipeline:
        if pipeline is not None:
            self.pipeline = pipeline
        
        if self.pipeline is None:
            raise ValueError("Pipeline not set. Call build_pipeline() first.")
        
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data not split. Call split_data() first.")
        
        logger.info("Training model...")
        self.pipeline.fit(self.X_train, self.y_train)
        logger.info("Model training completed")
        
        return self.pipeline

    def cross_validate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        if self.pipeline is None:
            raise ValueError("Pipeline not built. Call build_pipeline() first.")
        
        logger.info(f"Performing {self.cv_folds}-fold cross-validation...")
        
        skf = StratifiedKFold(
            n_splits=self.cv_folds, 
            shuffle=True, 
            random_state=self.random_state
        )
        
        scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        self.cv_scores = {}
        
        for metric in scoring_metrics:
            scores = cross_val_score(
                self.pipeline, X, y,
                cv=skf,
                scoring=metric,
                n_jobs=-1
            )
            self.cv_scores[metric] = {
                'mean': float(scores.mean()),
                'std': float(scores.std()),
                'scores': scores.tolist()
            }
            logger.info(f"{metric}: {scores.mean():.4f} (+/- {scores.std():.4f})")
        
        return self.cv_scores

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        if self.pipeline is None:
            return None
        
        classifier = None
        for name, step in self.pipeline.steps:
            if name == 'classifier':
                classifier = step
                break
        
        if classifier is None or not hasattr(classifier, 'coef_'):
            return None
        
        feature_importance = {
            'coefficients': classifier.coef_.tolist(),
            'intercept': float(classifier.intercept_[0])
        }
        
        return feature_importance

    def save_model(self, filepath: str) -> None:
        if self.pipeline is None:
            raise ValueError("Pipeline not trained. Call train() first.")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'pipeline': self.pipeline,
            'random_state': self.random_state,
            'test_size': self.test_size,
            'cv_scores': self.cv_scores
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> Pipeline:
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        self.pipeline = model_data['pipeline']
        self.cv_scores = model_data.get('cv_scores')
        
        logger.info(f"Model loaded from {filepath}")
        
        return self.pipeline

    def get_training_summary(self) -> Dict[str, Any]:
        summary = {
            'random_state': self.random_state,
            'test_size': self.test_size,
            'cv_folds': self.cv_folds,
            'use_class_weight': self.use_class_weight,
            'train_size': len(self.X_train) if self.X_train is not None else None,
            'test_size_actual': len(self.X_test) if self.X_test is not None else None,
            'cv_scores': self.cv_scores
        }
        
        return summary
