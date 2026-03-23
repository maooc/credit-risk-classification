import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataProcessor:
    """
    数据处理器：负责数据的读取、检查与清理
    """
    
    def __init__(self, data_path: str):
        """
        初始化数据处理器
        
        Args:
            data_path: 数据文件路径
        """
        self.data_path = Path(data_path)
        self.df = None
        self.target_col = 'loan_status'
        
    def load_data(self) -> pd.DataFrame:
        """
        加载数据文件
        
        Returns:
            pd.DataFrame: 加载的数据
        """
        logger.info(f"正在加载数据: {self.data_path}")
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {self.data_path}")
        
        self.df = pd.read_csv(self.data_path)
        logger.info(f"数据加载完成，形状: {self.df.shape}")
        
        return self.df
    
    def check_data_quality(self) -> dict:
        """
        检查数据质量，返回数据质量报告
        
        Returns:
            dict: 数据质量报告
        """
        if self.df is None:
            raise ValueError("请先调用 load_data() 加载数据")
        
        report = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'missing_values': self.df.isnull().sum().to_dict(),
            'missing_percentage': (self.df.isnull().sum() / len(self.df) * 100).to_dict(),
            'duplicate_rows': self.df.duplicated().sum(),
            'data_types': self.df.dtypes.to_dict()
        }
        
        logger.info("数据质量检查完成")
        logger.info(f"总行数: {report['total_rows']}")
        logger.info(f"缺失值: {report['missing_values']}")
        logger.info(f"重复行数: {report['duplicate_rows']}")
        
        return report
    
    def clean_data(self) -> pd.DataFrame:
        """
        清理数据：处理缺失值和重复值
        
        Returns:
            pd.DataFrame: 清理后的数据
        """
        if self.df is None:
            raise ValueError("请先调用 load_data() 加载数据")
        
        original_shape = self.df.shape
        
        # 删除重复行
        self.df = self.df.drop_duplicates()
        
        # 处理缺失值（如果有的话）
        if self.df.isnull().sum().sum() > 0:
            # 数值型列用中位数填充
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if self.df[col].isnull().sum() > 0:
                    median_val = self.df[col].median()
                    self.df[col].fillna(median_val, inplace=True)
                    logger.info(f"列 '{col}' 用中位数 {median_val} 填充缺失值")
        
        logger.info(f"数据清理完成: {original_shape} -> {self.df.shape}")
        
        return self.df
    
    def get_feature_columns(self) -> list:
        """
        获取特征列名（排除目标列）
        
        Returns:
            list: 特征列名列表
        """
        if self.df is None:
            raise ValueError("请先调用 load_data() 加载数据")
        
        return [col for col in self.df.columns if col != self.target_col]
    
    def split_features_target(self) -> tuple:
        """
        分离特征和目标变量
        
        Returns:
            tuple: (X, y) 特征矩阵和目标向量
        """
        if self.df is None:
            raise ValueError("请先调用 load_data() 加载数据")
        
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        
        logger.info(f"特征矩阵形状: {X.shape}, 目标向量形状: {y.shape}")
        
        return X, y
    
    def get_column_types(self) -> dict:
        """
        获取列类型分类（数值型、类别型）
        
        Returns:
            dict: 包含数值列和类别列的字典
        """
        if self.df is None:
            raise ValueError("请先调用 load_data() 加载数据")
        
        feature_cols = self.get_feature_columns()
        
        # 数值型列
        numeric_cols = self.df[feature_cols].select_dtypes(
            include=[np.number]
        ).columns.tolist()
        
        # 类别型列
        categorical_cols = self.df[feature_cols].select_dtypes(
            include=['object', 'category']
        ).columns.tolist()
        
        return {
            'numeric': numeric_cols,
            'categorical': categorical_cols
        }
    
    def get_data_summary(self) -> dict:
        """
        获取数据摘要统计
        
        Returns:
            dict: 数据摘要
        """
        if self.df is None:
            raise ValueError("请先调用 load_data() 加载数据")
        
        summary = {
            'target_distribution': self.df[self.target_col].value_counts().to_dict(),
            'target_proportion': self.df[self.target_col].value_counts(normalize=True).to_dict(),
            'numeric_summary': self.df[self.get_feature_columns()].describe().to_dict()
        }
        
        logger.info(f"目标变量分布: {summary['target_distribution']}")
        
        return summary


def main():
    """测试数据处理器"""
    # 使用原始数据路径
    data_path = "Credit_Risk/Resources/lending_data.csv"
    
    processor = DataProcessor(data_path)
    
    # 加载数据
    df = processor.load_data()
    print(f"\n数据前5行:\n{df.head()}")
    
    # 数据质量检查
    quality_report = processor.check_data_quality()
    print(f"\n数据质量报告:\n{quality_report}")
    
    # 清理数据
    df_clean = processor.clean_data()
    
    # 获取列类型
    column_types = processor.get_column_types()
    print(f"\n数值型列: {column_types['numeric']}")
    print(f"类别型列: {column_types['categorical']}")
    
    # 数据摘要
    summary = processor.get_data_summary()
    print(f"\n目标变量分布: {summary['target_distribution']}")
    
    # 分离特征和目标
    X, y = processor.split_features_target()
    print(f"\n特征矩阵形状: {X.shape}")
    print(f"目标向量形状: {y.shape}")


if __name__ == "__main__":
    main()
