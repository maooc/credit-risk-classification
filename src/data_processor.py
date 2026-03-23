import pandas as pd
import numpy as np
from pathlib import Path

class DataProcessor:
    """数据处理器：负责数据读取、检查与清理"""
    
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        
    def load_data(self, data_path=None):
        """加载CSV数据"""
        if data_path:
            self.data_path = data_path
            
        if not self.data_path:
            raise ValueError("请提供数据路径")
            
        self.df = pd.read_csv(self.data_path)
        return self.df
    
    def check_data_quality(self):
        """检查数据质量"""
        if self.df is None:
            raise ValueError("请先加载数据")
            
        quality_report = {
            "total_rows": len(self.df),
            "total_columns": len(self.df.columns),
            "missing_values": self.df.isnull().sum().to_dict(),
            "duplicate_count": self.df.duplicated().sum(),
            "data_types": self.df.dtypes.astype(str).to_dict(),
            "class_distribution": self.df["loan_status"].value_counts().to_dict() if "loan_status" in self.df.columns else None
        }
        
        return quality_report
    
    def clean_data(self):
        """数据清理：去除重复值、处理缺失值"""
        if self.df is None:
            raise ValueError("请先加载数据")
            
        # 去除重复行
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates()
        duplicate_removed = initial_rows - len(self.df)
        
        # 处理缺失值 - 使用中位数填充数值列
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if self.df[col].isnull().any():
                self.df[col] = self.df[col].fillna(self.df[col].median())
        
        return {
            "duplicate_removed": duplicate_removed,
            "missing_values_handled": self.df.isnull().sum().sum()
        }
    
    def split_features_labels(self, target_column="loan_status"):
        """分离特征和标签"""
        if self.df is None:
            raise ValueError("请先加载数据")
            
        if target_column not in self.df.columns:
            raise ValueError(f"目标列 {target_column} 不存在于数据中")
            
        self.y = self.df[target_column]
        self.X = self.df.drop(columns=[target_column])
        
        return self.X, self.y
    
    def get_feature_names(self):
        """获取特征名称"""
        if self.X is None:
            raise ValueError("请先分离特征和标签")
        return self.X.columns.tolist()
