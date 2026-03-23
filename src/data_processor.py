import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataProcessor:
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.df: Optional[pd.DataFrame] = None
        self.X: Optional[pd.DataFrame] = None
        self.y: Optional[pd.Series] = None
        self.target_column: str = 'loan_status'
        
        self.numerical_features = [
            'loan_size', 'interest_rate', 'borrower_income', 
            'debt_to_income', 'total_debt'
        ]
        self.discrete_features = ['num_of_accounts', 'derogatory_marks']
        self.all_features = self.numerical_features + self.discrete_features

    def load_data(self) -> pd.DataFrame:
        logger.info(f"Loading data from {self.data_path}")
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        self.df = pd.read_csv(self.data_path)
        logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
        
        return self.df

    def check_data_quality(self) -> Dict[str, Any]:
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        quality_report = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'missing_percentage': (self.df.isnull().sum() / len(self.df) * 100).to_dict(),
            'duplicates': self.df.duplicated().sum(),
            'statistics': self.df.describe().to_dict()
        }
        
        logger.info("Data quality check completed")
        logger.info(f"Missing values: {quality_report['missing_values']}")
        logger.info(f"Duplicates: {quality_report['duplicates']}")
        
        return quality_report

    def check_class_distribution(self) -> Dict[str, Any]:
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        distribution = self.df[self.target_column].value_counts().to_dict()
        distribution_pct = (self.df[self.target_column].value_counts(normalize=True) * 100).to_dict()
        
        class_info = {
            'counts': distribution,
            'percentages': distribution_pct,
            'imbalance_ratio': distribution.get(0, 1) / distribution.get(1, 1)
        }
        
        logger.info(f"Class distribution: {distribution}")
        logger.info(f"Class percentages: {distribution_pct}")
        logger.info(f"Imbalance ratio (0:1): {class_info['imbalance_ratio']:.2f}")
        
        return class_info

    def clean_data(self) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        initial_shape = self.df.shape
        
        self.df = self.df.drop_duplicates()
        
        for col in self.df.columns:
            if self.df[col].isnull().sum() > 0:
                if self.df[col].dtype in ['int64', 'float64']:
                    median_val = self.df[col].median()
                    self.df[col].fillna(median_val, inplace=True)
                    logger.info(f"Filled missing values in {col} with median: {median_val}")
                else:
                    mode_val = self.df[col].mode()[0]
                    self.df[col].fillna(mode_val, inplace=True)
                    logger.info(f"Filled missing values in {col} with mode: {mode_val}")
        
        self.df = self._remove_outliers_iqr(self.df, self.numerical_features)
        
        logger.info(f"Data cleaning completed. Shape: {initial_shape} -> {self.df.shape}")
        
        return self.df

    def _remove_outliers_iqr(self, df: pd.DataFrame, columns: list, multiplier: float = 1.5) -> pd.DataFrame:
        df_clean = df.copy()
        
        for col in columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
            outliers_removed = (~mask).sum()
            
            if outliers_removed > 0:
                logger.info(f"Removing {outliers_removed} outliers from {col}")
                df_clean = df_clean[mask]
        
        return df_clean

    def separate_features_target(self) -> Tuple[pd.DataFrame, pd.Series]:
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        self.y = self.df[self.target_column]
        self.X = self.df.drop(columns=[self.target_column])
        
        logger.info(f"Features shape: {self.X.shape}, Target shape: {self.y.shape}")
        
        return self.X, self.y

    def get_feature_info(self) -> Dict[str, list]:
        return {
            'numerical_features': self.numerical_features,
            'discrete_features': self.discrete_features,
            'all_features': self.all_features,
            'target_column': self.target_column
        }

    def process(self) -> Tuple[pd.DataFrame, pd.Series]:
        self.load_data()
        self.check_data_quality()
        self.check_class_distribution()
        self.clean_data()
        X, y = self.separate_features_target()
        
        return X, y
