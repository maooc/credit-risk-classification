import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler, 
    RobustScaler, 
    PolynomialFeatures,
    KBinsDiscretizer,
    OneHotEncoder
)
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureEngineering:
    """
    特征工程类：构建和保存特征工程 Pipeline
    
    管道结构：
    1. ColumnTransformer 分组处理不同特征：
       - 广域数值特征 (loan_size, borrower_income, total_debt): RobustScaler
       - 交互特征 (interest_rate, borrower_income, total_debt): RobustScaler -> PolynomialFeatures
       - 离散特征 (num_of_accounts, derogatory_marks): KBinsDiscretizer
       - 其他数值特征 (debt_to_income): StandardScaler
    2. SelectKBest 特征选择
    
    注意：模型训练时会将 LogisticRegression 添加为最后一步，形成完整闭环
    """
    
    # 列分类定义
    # 广域数值特征 - 使用 RobustScaler
    WIDE_RANGE_NUMERIC_COLS = ['loan_size', 'borrower_income', 'total_debt']
    
    # 需要特征交互的列 - interest_rate, borrower_income, total_debt
    # 注意：borrower_income 和 total_debt 同时出现在广域特征中，
    # 但在这里它们会单独走交互通道
    INTERACTION_COLS = ['interest_rate', 'borrower_income', 'total_debt']
    
    # 离散特征 - 使用 KBinsDiscretizer
    DISCRETE_COLS = ['num_of_accounts', 'derogatory_marks']
    
    # 其他数值特征 - 使用 StandardScaler
    # debt_to_income 不在其他组中，所以单独处理
    OTHER_NUMERIC_COLS = ['debt_to_income']
    
    def __init__(self, k_best: int = 15):
        """
        初始化特征工程器
        
        Args:
            k_best: SelectKBest 选择的特征数量
        """
        self.k_best = k_best
        self.preprocessor = None
        self.feature_pipeline = None
        self.full_pipeline = None  # 包含模型的完整管道
        
    def build_feature_pipeline(self) -> Pipeline:
        """
        构建特征工程 Pipeline（不包含模型）
        
        结构：
        ColumnTransformer(
            - wide_range: RobustScaler (loan_size)
            - interaction: RobustScaler + PolynomialFeatures (interest_rate, borrower_income, total_debt)
            - discrete: KBinsDiscretizer (num_of_accounts, derogatory_marks)
            - other: StandardScaler (debt_to_income)
        ) -> SelectKBest
        
        Returns:
            Pipeline: 特征工程管道
        """
        logger.info("开始构建特征工程 Pipeline...")
        
        # 1. 广域数值特征 - 只包含 loan_size（因为 borrower_income 和 total_debt 走交互通道）
        wide_range_cols = ['loan_size']
        wide_range_transformer = Pipeline(steps=[
            ('robust_scaler', RobustScaler())
        ])
        
        # 2. 交互特征通道 - interest_rate, borrower_income, total_debt
        # 先标准化，再生成多项式交互特征
        interaction_transformer = Pipeline(steps=[
            ('robust_scaler', RobustScaler()),
            ('poly', PolynomialFeatures(
                degree=2, 
                interaction_only=True, 
                include_bias=False
            ))
        ])
        
        # 3. 离散特征 - 分箱离散化
        discrete_transformer = Pipeline(steps=[
            ('discretizer', KBinsDiscretizer(
                n_bins=5, 
                encode='onehot-dense', 
                strategy='quantile',
                subsample=None
            ))
        ])
        
        # 4. 其他数值特征 - StandardScaler
        other_transformer = Pipeline(steps=[
            ('standard_scaler', StandardScaler())
        ])
        
        # 组合所有特征处理
        preprocessor = ColumnTransformer(
            transformers=[
                ('wide_range', wide_range_transformer, wide_range_cols),
                ('interaction', interaction_transformer, self.INTERACTION_COLS),
                ('discrete', discrete_transformer, self.DISCRETE_COLS),
                ('other', other_transformer, self.OTHER_NUMERIC_COLS)
            ],
            remainder='drop'  # 丢弃未指定的列
        )
        
        self.preprocessor = preprocessor
        
        # 构建完整特征工程管道
        self.feature_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('feature_selection', SelectKBest(
                score_func=f_classif, 
                k=self.k_best
            ))
        ])
        
        logger.info("特征工程 Pipeline 构建完成")
        logger.info(f"  - 广域特征: {wide_range_cols}")
        logger.info(f"  - 交互特征: {self.INTERACTION_COLS}")
        logger.info(f"  - 离散特征: {self.DISCRETE_COLS}")
        logger.info(f"  - 其他特征: {self.OTHER_NUMERIC_COLS}")
        logger.info(f"  - 选择特征数: {self.k_best}")
        
        return self.feature_pipeline
    
    def build_full_pipeline(self, model) -> Pipeline:
        """
        构建完整 Pipeline（特征工程 + 模型）
        
        Args:
            model: 机器学习模型（如 LogisticRegression）
            
        Returns:
            Pipeline: 完整管道
        """
        if self.feature_pipeline is None:
            self.build_feature_pipeline()
        
        self.full_pipeline = Pipeline(steps=[
            ('features', self.feature_pipeline),
            ('model', model)
        ])
        
        logger.info("完整 Pipeline（特征工程 + 模型）构建完成")
        
        return self.full_pipeline
    
    def get_expected_columns(self) -> list:
        """
        获取期望的列顺序
        
        Returns:
            list: 列名列表
        """
        # 注意：loan_size 单独处理，但 borrower_income 和 total_debt 在交互组中
        # 我们需要确保所有列都被覆盖
        return (
            ['loan_size'] +  # wide_range
            self.INTERACTION_COLS +  # interaction (包含 borrower_income, total_debt)
            self.DISCRETE_COLS +
            self.OTHER_NUMERIC_COLS
        )
    
    def prepare_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        准备数据，确保列顺序正确
        
        Args:
            X: 输入特征
            
        Returns:
            pd.DataFrame: 准备好的数据
        """
        expected_cols = self.get_expected_columns()
        
        # 检查是否包含所有需要的列
        missing_cols = set(expected_cols) - set(X.columns)
        if missing_cols:
            raise ValueError(f"缺少必要的列: {missing_cols}")
        
        # 按期望顺序选择列
        return X[expected_cols]
    
    def fit_transform(self, X: pd.DataFrame, y: np.ndarray = None) -> np.ndarray:
        """
        拟合并转换数据（仅特征工程，不包含模型）
        
        注意：此方法仅用于非交叉验证场景。交叉验证时应使用完整 Pipeline。
        
        Args:
            X: 特征矩阵
            y: 目标向量（用于特征选择）
            
        Returns:
            np.ndarray: 转换后的特征矩阵
        """
        if self.feature_pipeline is None:
            self.build_feature_pipeline()
        
        logger.info(f"开始拟合和转换数据，输入形状: {X.shape}")
        
        X_prepared = self.prepare_data(X)
        X_transformed = self.feature_pipeline.fit_transform(X_prepared, y)
        
        logger.info(f"数据转换完成，输出形状: {X_transformed.shape}")
        
        return X_transformed
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        转换数据（使用已拟合的 Pipeline）
        
        Args:
            X: 特征矩阵
            
        Returns:
            np.ndarray: 转换后的特征矩阵
        """
        if self.feature_pipeline is None:
            raise ValueError("请先调用 fit_transform() 或 build_feature_pipeline()")
        
        X_prepared = self.prepare_data(X)
        return self.feature_pipeline.transform(X_prepared)
    
    def save_pipeline(self, filepath: str):
        """
        保存特征工程 Pipeline
        
        Args:
            filepath: 保存路径
        """
        if self.feature_pipeline is None:
            raise ValueError("Pipeline 尚未构建")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.feature_pipeline, filepath)
        logger.info(f"特征工程 Pipeline 已保存至: {filepath}")
    
    def load_pipeline(self, filepath: str):
        """
        加载特征工程 Pipeline
        
        Args:
            filepath: 加载路径
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Pipeline 文件不存在: {filepath}")
        
        self.feature_pipeline = joblib.load(filepath)
        logger.info(f"特征工程 Pipeline 已从 {filepath} 加载")
        
        return self.feature_pipeline
    
    def get_column_groups(self) -> dict:
        """
        获取列分组信息
        
        Returns:
            dict: 列分组字典
        """
        return {
            'wide_range_numeric': ['loan_size'],
            'interaction': self.INTERACTION_COLS,
            'discrete': self.DISCRETE_COLS,
            'other_numeric': self.OTHER_NUMERIC_COLS
        }


def main():
    """测试特征工程"""
    import sys
    sys.path.append(str(Path(__file__).parent))
    from data_processor import DataProcessor
    
    # 加载数据
    data_path = "Credit_Risk/Resources/lending_data.csv"
    processor = DataProcessor(data_path)
    processor.load_data()
    processor.clean_data()
    X, y = processor.split_features_target()
    
    # 构建特征工程 Pipeline
    fe = FeatureEngineering(k_best=15)
    pipeline = fe.build_feature_pipeline()
    
    print(f"\n期望的列: {fe.get_expected_columns()}")
    print(f"列分组: {fe.get_column_groups()}")
    
    # 拟合和转换
    X_transformed = fe.fit_transform(X, y)
    
    print(f"\n原始特征形状: {X.shape}")
    print(f"转换后特征形状: {X_transformed.shape}")
    
    # 保存 Pipeline
    fe.save_pipeline("models/feature_pipeline.joblib")
    
    # 测试加载
    fe2 = FeatureEngineering()
    fe2.load_pipeline("models/feature_pipeline.joblib")
    X_test = fe2.transform(X)
    print(f"\n加载 Pipeline 后转换形状: {X_test.shape}")


if __name__ == "__main__":
    main()
