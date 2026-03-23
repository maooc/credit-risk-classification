from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    RobustScaler, StandardScaler, KBinsDiscretizer,
    PolynomialFeatures
)
from sklearn.feature_selection import SelectKBest, f_classif
from typing import Dict, List, Tuple, Optional, Any
import joblib
from pathlib import Path
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureEngineeringPipeline:
    
    def __init__(
        self,
        numerical_features: List[str],
        discrete_features: List[str],
        use_polynomial: bool = True,
        use_discretizer: bool = True,
        n_bins: int = 5,
        k_best: int = 10
    ):
        self.numerical_features = numerical_features
        self.discrete_features = discrete_features
        self.use_polynomial = use_polynomial
        self.use_discretizer = use_discretizer
        self.n_bins = n_bins
        self.k_best = k_best
        
        self.preprocessor: Optional[ColumnTransformer] = None
        self.feature_selector: Optional[SelectKBest] = None
        self.full_pipeline: Optional[Pipeline] = None
        
        self.polynomial_features = ['interest_rate', 'borrower_income', 'total_debt']
        self.robust_scaler_features = ['loan_size']
        self.standard_scaler_features = ['debt_to_income']
        
        self._build_preprocessor()

    def _build_preprocessor(self) -> None:
        transformers = []
        
        polynomial_features_available = [
            f for f in self.polynomial_features 
            if f in self.numerical_features
        ]
        
        if self.use_polynomial and polynomial_features_available:
            polynomial_pipeline = Pipeline([
                ('scaler', RobustScaler()),
                ('polynomial', PolynomialFeatures(
                    degree=2,
                    interaction_only=True,
                    include_bias=False
                ))
            ])
            transformers.append(
                ('polynomial_interaction', polynomial_pipeline, polynomial_features_available)
            )
            logger.info(f"Applied PolynomialFeatures (interaction_only=True) to: {polynomial_features_available}")
        
        robust_features = [
            f for f in self.robust_scaler_features 
            if f in self.numerical_features and f not in polynomial_features_available
        ]
        if robust_features:
            transformers.append(
                ('robust_scaler', RobustScaler(), robust_features)
            )
            logger.info(f"Applied RobustScaler to: {robust_features}")
        
        standard_features = [
            f for f in self.standard_scaler_features 
            if f in self.numerical_features and f not in polynomial_features_available
        ]
        if standard_features:
            transformers.append(
                ('standard_scaler', StandardScaler(), standard_features)
            )
            logger.info(f"Applied StandardScaler to: {standard_features}")
        
        if self.use_discretizer and self.discrete_features:
            discretizer = KBinsDiscretizer(
                n_bins=self.n_bins,
                encode='onehot-dense',
                strategy='uniform'
            )
            transformers.append(
                ('discretizer', discretizer, self.discrete_features)
            )
            logger.info(f"Applied KBinsDiscretizer to: {self.discrete_features}")
        
        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='passthrough',
            sparse_threshold=0
        )
        
        logger.info("Preprocessor built successfully with refined feature groups")

    def build_full_pipeline(self, classifier: Optional[Any] = None) -> Pipeline:
        pipeline_steps = [('preprocessor', self.preprocessor)]
        
        pipeline_steps.append(('feature_selector', SelectKBest(
            score_func=f_classif,
            k=self.k_best
        )))
        logger.info(f"Added SelectKBest with k={self.k_best}")
        
        if classifier is not None:
            pipeline_steps.append(('classifier', classifier))
            logger.info("Added classifier to pipeline")
        
        self.full_pipeline = Pipeline(pipeline_steps)
        
        return self.full_pipeline

    def fit_transform_features(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_test: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if self.full_pipeline is None:
            raise ValueError("Pipeline not built. Call build_full_pipeline() first.")
        
        feature_pipeline = Pipeline(self.full_pipeline.steps[:-1]) if hasattr(self.full_pipeline, 'steps') else None
        
        if feature_pipeline is None:
            raise ValueError("Pipeline steps not properly initialized")
        
        X_train_transformed = feature_pipeline.fit_transform(X_train, y_train)
        
        X_test_transformed = None
        if X_test is not None:
            X_test_transformed = feature_pipeline.transform(X_test)
        
        logger.info(f"Feature transformation completed. Train shape: {X_train_transformed.shape}")
        
        return X_train_transformed, X_test_transformed

    def get_feature_names(self) -> List[str]:
        if self.preprocessor is None:
            raise ValueError("Preprocessor not built yet")
        
        feature_names = []
        
        for name, transformer, columns in self.preprocessor.transformers_:
            if name == 'remainder' and transformer == 'passthrough':
                continue
            
            if hasattr(transformer, 'get_feature_names_out'):
                if isinstance(columns, list):
                    names = transformer.get_feature_names_out(columns)
                else:
                    names = transformer.get_feature_names_out()
                feature_names.extend(names)
            else:
                if isinstance(columns, list):
                    feature_names.extend(columns)
                else:
                    feature_names.append(columns)
        
        return feature_names

    def save_pipeline(self, filepath: str) -> None:
        if self.full_pipeline is None:
            raise ValueError("Pipeline not built. Call build_full_pipeline() first.")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.full_pipeline, filepath)
        logger.info(f"Pipeline saved to {filepath}")

    def load_pipeline(self, filepath: str) -> Pipeline:
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Pipeline file not found: {filepath}")
        
        self.full_pipeline = joblib.load(filepath)
        logger.info(f"Pipeline loaded from {filepath}")
        
        return self.full_pipeline

    def get_selected_features_info(self) -> Dict[str, Any]:
        if self.full_pipeline is None:
            return {}
        
        feature_selector = None
        for name, step in self.full_pipeline.steps:
            if name == 'feature_selector':
                feature_selector = step
                break
        
        if feature_selector is None:
            return {}
        
        scores = feature_selector.scores_
        selected_indices = feature_selector.get_support(indices=True)
        
        return {
            'feature_scores': scores.tolist() if scores is not None else [],
            'selected_indices': selected_indices.tolist(),
            'n_features_selected': len(selected_indices)
        }

    def get_pipeline_summary(self) -> Dict[str, Any]:
        return {
            'polynomial_features': self.polynomial_features,
            'robust_scaler_features': self.robust_scaler_features,
            'standard_scaler_features': self.standard_scaler_features,
            'discrete_features': self.discrete_features,
            'use_polynomial': self.use_polynomial,
            'use_discretizer': self.use_discretizer,
            'n_bins': self.n_bins,
            'k_best': self.k_best
        }
