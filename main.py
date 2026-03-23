import argparse
import logging
from pathlib import Path
from datetime import datetime

from src.data_processor import DataProcessor
from src.feature_engineering import FeatureEngineeringPipeline
from src.train import ModelTrainer
from src.evaluate import ModelEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main(
    data_path: str,
    output_dir: str = './outputs',
    test_size: float = 0.25,
    random_state: int = 1,
    cv_folds: int = 5,
    use_polynomial: bool = True,
    use_discretizer: bool = True,
    n_bins: int = 5,
    k_best: int = 15,
    save_model: bool = True
):
    logger.info("="*60)
    logger.info("Credit Risk Classification Pipeline")
    logger.info("="*60)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("\n[Step 1] Data Processing")
    logger.info("-"*40)
    processor = DataProcessor(data_path=data_path)
    X, y = processor.process()
    
    feature_info = processor.get_feature_info()
    logger.info(f"Numerical features: {feature_info['numerical_features']}")
    logger.info(f"Discrete features: {feature_info['discrete_features']}")
    
    logger.info("\n[Step 2] Feature Engineering Pipeline Setup")
    logger.info("-"*40)
    feature_pipeline = FeatureEngineeringPipeline(
        numerical_features=feature_info['numerical_features'],
        discrete_features=feature_info['discrete_features'],
        use_polynomial=use_polynomial,
        use_discretizer=use_discretizer,
        n_bins=n_bins,
        k_best=k_best
    )
    
    logger.info("\n[Step 3] Model Training Setup")
    logger.info("-"*40)
    trainer = ModelTrainer(
        random_state=random_state,
        test_size=test_size,
        cv_folds=cv_folds,
        use_class_weight=True
    )
    
    X_train, X_test, y_train, y_test = trainer.split_data(X, y)
    
    model = trainer.create_model(
        solver='lbfgs',
        max_iter=1000,
        C=1.0,
        penalty='l2'
    )
    
    preprocessor_pipeline = feature_pipeline.build_full_pipeline(classifier=None)
    
    full_pipeline = trainer.build_pipeline(preprocessor_pipeline)
    
    logger.info("\n[Step 4] Model Training")
    logger.info("-"*40)
    trainer.train()
    
    logger.info("\n[Step 5] Cross-Validation")
    logger.info("-"*40)
    cv_scores = trainer.cross_validate(X, y)
    
    logger.info("\n[Step 6] Model Evaluation")
    logger.info("-"*40)
    evaluator = ModelEvaluator(output_dir=output_dir)
    
    y_pred = trainer.pipeline.predict(X_test)
    y_pred_proba = trainer.pipeline.predict_proba(X_test)
    
    metrics = evaluator.evaluate(y_test, y_pred, y_pred_proba)
    
    logger.info("\n[Step 7] Generating Reports and Visualizations")
    logger.info("-"*40)
    report_files = evaluator.generate_full_report()
    
    logger.info("Generated files:")
    for name, filepath in report_files.items():
        logger.info(f"  - {name}: {filepath}")
    
    if save_model:
        logger.info("\n[Step 8] Saving Model")
        logger.info("-"*40)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = output_path / f'model_{timestamp}.joblib'
        trainer.save_model(str(model_path))
        logger.info(f"Model saved to: {model_path}")
    
    logger.info("\n" + "="*60)
    logger.info("Pipeline Completed Successfully!")
    logger.info("="*60)
    
    results = {
        'metrics': metrics,
        'cv_scores': cv_scores,
        'report_files': report_files,
        'feature_info': feature_info
    }
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Credit Risk Classification Pipeline'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        default='./Credit_Risk/Resources/lending_data.csv',
        help='Path to the lending data CSV file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./outputs',
        help='Directory to save outputs'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.25,
        help='Proportion of test set'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=1,
        help='Random state for reproducibility'
    )
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Number of cross-validation folds'
    )
    parser.add_argument(
        '--use-polynomial',
        action='store_true',
        default=True,
        help='Use polynomial features'
    )
    parser.add_argument(
        '--no-polynomial',
        action='store_false',
        dest='use_polynomial',
        help='Disable polynomial features'
    )
    parser.add_argument(
        '--use-discretizer',
        action='store_true',
        default=True,
        help='Use KBinsDiscretizer for discrete features'
    )
    parser.add_argument(
        '--no-discretizer',
        action='store_false',
        dest='use_discretizer',
        help='Disable discretizer'
    )
    parser.add_argument(
        '--n-bins',
        type=int,
        default=5,
        help='Number of bins for discretizer'
    )
    parser.add_argument(
        '--k-best',
        type=int,
        default=15,
        help='Number of best features to select'
    )
    parser.add_argument(
        '--save-model',
        action='store_true',
        default=True,
        help='Save trained model'
    )
    parser.add_argument(
        '--no-save-model',
        action='store_false',
        dest='save_model',
        help='Do not save model'
    )
    
    args = parser.parse_args()
    
    results = main(
        data_path=args.data_path,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        cv_folds=args.cv_folds,
        use_polynomial=args.use_polynomial,
        use_discretizer=args.use_discretizer,
        n_bins=args.n_bins,
        k_best=args.k_best,
        save_model=args.save_model
    )
