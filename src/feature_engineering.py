from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    RobustScaler,
    StandardScaler,
    KBinsDiscretizer,
    PolynomialFeatures,
    FunctionTransformer
)
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel
from sklearn.linear_model import LogisticRegression
import numpy as np
import joblib

class FeatureEngineer:
    """特征工程：使用Pipeline和ColumnTransformer组织特征处理流程"""
    
    def __init__(self):
        self.feature_pipeline = None
        self.preprocessor = None
        
        # 定义不同类型的特征
        self.wide_numeric_features = [
            "loan_size", 
            "borrower_income", 
            "total_debt"
        ]  # 广域连续数值特征 - 使用RobustScaler
        
        self.normal_numeric_features = [
            "interest_rate", 
            "debt_to_income"
        ]  # 常规数值特征 - 使用StandardScaler
        
        self.discrete_features = [
            "num_of_accounts", 
            "derogatory_marks"
        ]  # 离散统计特征 - 分箱处理
        
        self.interaction_features = [
            "interest_rate", 
            "borrower_income", 
            "total_debt"
        ]  # 用于交互特征的列
        
    def build_preprocessor(self):
        """构建特征预处理管道"""
        
        # 广域数值特征处理：防异常值标准化
        wide_numeric_transformer = Pipeline(steps=[
            ("scaler", RobustScaler())
        ])
        
        # 常规数值特征处理：标准化
        normal_numeric_transformer = Pipeline(steps=[
            ("scaler", StandardScaler())
        ])
        
        # 离散特征处理：分箱离散化
        discrete_transformer = Pipeline(steps=[
            ("binning", KBinsDiscretizer(n_bins=5, encode="onehot-dense", strategy="quantile"))
        ])
        
        # 组合所有预处理器
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("wide_numeric", wide_numeric_transformer, self.wide_numeric_features),
                ("normal_numeric", normal_numeric_transformer, self.normal_numeric_features),
                ("discrete", discrete_transformer, self.discrete_features)
            ],
            remainder="passthrough"  # 保留其他未指定的特征
        )
        
        return self.preprocessor
    
    def build_feature_selector(self, method="kbest", k=10):
        """
        构建特征选择器
        method: "kbest" 或 "l1" 
        """
        if method == "kbest":
            # 使用SelectKBest基于方差分析选择特征
            return SelectKBest(score_func=f_classif, k=k)
        elif method == "l1":
            # 使用L1正则化进行特征选择
            lr = LogisticRegression(penalty="l1", solver="liblinear", random_state=42)
            return SelectFromModel(lr)
        else:
            raise ValueError("不支持的特征选择方法")
    
    def build_interaction_features_only(self):
        """
        仅对指定特征构建交互特征
        仅对 interest_rate, borrower_income, total_debt 这三个特征生成度为2的交互项
        """
        # 对指定特征进行标准化后再生成交互项
        interaction_transformer = Pipeline(steps=[
            ("scaler", StandardScaler()),  # 交互特征前先标准化
            ("polynomial", PolynomialFeatures(
                degree=2, 
                interaction_only=True, 
                include_bias=False
            ))
        ])
        
        # 只对指定的三个特征应用交互变换
        return ColumnTransformer(
            transformers=[
                ("interaction", interaction_transformer, self.interaction_features)
            ],
            remainder="drop"  # 只保留交互特征
        )
    
    def build_full_pipeline(self, feature_selection_method="kbest", k_features=15):
        """
        构建完整的特征工程管道
        分为两个分支：
        1. 基础预处理特征（所有特征经过预处理）
        2. 特定特征的交互项（仅对利息、收入、负债三个特征）
        然后将两部分特征合并
        """
        
        # 分支1：基础预处理特征
        base_features = self.build_preprocessor()
        
        # 分支2：特定特征的交互项（仅对指定的三个特征）
        interaction_features = self.build_interaction_features_only()
        
        # 使用FeatureUnion合并两个分支的结果
        combined_features = FeatureUnion(
            transformer_list=[
                ("base_features", base_features),
                ("interaction_features", interaction_features)
            ]
        )
        
        # 特征选择
        selector = self.build_feature_selector(method=feature_selection_method, k=k_features)
        
        # 完整管道：特征合并 -> 特征选择
        self.feature_pipeline = Pipeline(steps=[
            ("feature_combination", combined_features),
            ("feature_selection", selector)
        ])
        
        return self.feature_pipeline
    
    def fit_transform(self, X, y=None):
        """拟合并转换数据"""
        if self.feature_pipeline is None:
            self.build_full_pipeline()
            
        return self.feature_pipeline.fit_transform(X, y)
    
    def transform(self, X):
        """转换数据"""
        if self.feature_pipeline is None:
            raise ValueError("请先拟合特征工程管道")
            
        return self.feature_pipeline.transform(X)
    
    def save_pipeline(self, filepath="feature_pipeline.joblib"):
        """保存特征工程管道"""
        if self.feature_pipeline is None:
            raise ValueError("没有可保存的管道")
            
        joblib.dump(self.feature_pipeline, filepath)
    
    def load_pipeline(self, filepath="feature_pipeline.joblib"):
        """加载特征工程管道"""
        self.feature_pipeline = joblib.load(filepath)
    
    def get_selected_features_info(self):
        """获取被选择的特征信息"""
        if self.feature_pipeline is None:
            raise ValueError("请先拟合特征工程管道")
            
        selector = self.feature_pipeline.named_steps["feature_selection"]
        
        if hasattr(selector, "get_support"):
            return {
                "selected_indices": np.where(selector.get_support())[0].tolist(),
                "feature_scores": selector.scores_.tolist() if hasattr(selector, "scores_") else None
            }
        return {}
