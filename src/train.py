import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    模型训练器：负责模型训练、交叉验证
    
    关键设计：使用完整 Pipeline（特征工程 + 模型）进行交叉验证，
    确保在每个 fold 内独立进行特征工程，防止数据泄露。
    
    使用 LogisticRegression 并设置 class_weight='balanced' 处理类别不平衡
    """
    
    def __init__(
        self, 
        random_state: int = 1,
        test_size: float = 0.2,
        cv_folds: int = 5,
        max_iter: int = 1000
    ):
        """
        初始化模型训练器
        
        Args:
            random_state: 随机种子
            test_size: 测试集比例
            cv_folds: 交叉验证折数
            max_iter: 最大迭代次数
        """
        self.random_state = random_state
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.max_iter = max_iter
        
        self.model = None
        self.full_pipeline = None  # 完整管道：特征工程 + 模型
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.training_history = {}
        
    def split_data(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        划分训练集和测试集
        
        注意：在特征工程之前划分，避免数据泄露
        
        Args:
            X: 特征矩阵（原始未转换的特征）
            y: 目标向量
            
        Returns:
            Tuple: (X_train, X_test, y_train, y_test)
        """
        logger.info(f"划分数据集，测试集比例: {self.test_size}")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y  # 保持类别比例
        )
        
        logger.info(f"训练集大小: {self.X_train.shape[0]}")
        logger.info(f"测试集大小: {self.X_test.shape[0]}")
        logger.info(f"训练集类别分布:\n{self.y_train.value_counts()}")
        logger.info(f"测试集类别分布:\n{self.y_test.value_counts()}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def build_model(self) -> LogisticRegression:
        """
        构建逻辑回归模型
        
        使用 class_weight='balanced' 处理类别不平衡
        
        Returns:
            LogisticRegression: 逻辑回归模型
        """
        self.model = LogisticRegression(
            solver='lbfgs',
            random_state=self.random_state,
            max_iter=self.max_iter,
            class_weight='balanced'  # 处理类别不平衡
        )
        
        logger.info("构建 LogisticRegression 模型")
        logger.info(f"  - solver: lbfgs")
        logger.info(f"  - class_weight: balanced")
        logger.info(f"  - max_iter: {self.max_iter}")
        
        return self.model
    
    def build_full_pipeline(self, feature_pipeline) -> Pipeline:
        """
        构建完整 Pipeline（特征工程 + 模型）
        
        Args:
            feature_pipeline: 特征工程管道
            
        Returns:
            Pipeline: 完整管道
        """
        if self.model is None:
            self.build_model()
        
        self.full_pipeline = Pipeline(steps=[
            ('features', feature_pipeline),
            ('model', self.model)
        ])
        
        logger.info("完整 Pipeline（特征工程 + 模型）构建完成")
        
        return self.full_pipeline
    
    def cross_validate(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        scoring: str = 'f1_weighted'
    ) -> Dict[str, Any]:
        """
        交叉验证 - 使用完整 Pipeline 防止数据泄露
        
        关键：直接传入原始特征 X_train 和完整 Pipeline，
        让交叉验证在每个 fold 内独立进行特征工程和模型训练
        
        Args:
            X_train: 原始训练特征（未转换）
            y_train: 训练目标
            scoring: 评估指标
            
        Returns:
            Dict: 交叉验证结果
        """
        if self.full_pipeline is None:
            raise ValueError("请先调用 build_full_pipeline() 构建完整 Pipeline")
        
        logger.info(f"开始 {self.cv_folds} 折交叉验证（使用完整 Pipeline，防止数据泄露）...")
        
        cv = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state
        )
        
        # 关键：直接传入原始特征和完整 Pipeline
        # 这样在每个 fold 内，特征工程会基于训练集重新拟合
        cv_scores = cross_val_score(
            self.full_pipeline, 
            X_train, 
            y_train,
            cv=cv,
            scoring=scoring
        )
        
        cv_results = {
            'scoring': scoring,
            'cv_scores': cv_scores.tolist(),
            'mean_score': float(cv_scores.mean()),
            'std_score': float(cv_scores.std()),
            'folds': self.cv_folds
        }
        
        logger.info(f"交叉验证结果 ({scoring}):")
        logger.info(f"  - 各折分数: {cv_scores}")
        logger.info(f"  - 平均分: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        self.training_history['cross_validation'] = cv_results
        
        return cv_results
    
    def train(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series
    ) -> Pipeline:
        """
        训练完整 Pipeline
        
        Args:
            X_train: 原始训练特征（未转换）
            y_train: 训练目标
            
        Returns:
            Pipeline: 训练好的完整 Pipeline
        """
        if self.full_pipeline is None:
            raise ValueError("请先调用 build_full_pipeline() 构建完整 Pipeline")
        
        logger.info("开始训练完整 Pipeline...")
        
        # 直接传入原始特征，Pipeline 内部会处理特征工程
        self.full_pipeline.fit(X_train, y_train)
        
        logger.info("Pipeline 训练完成")
        
        # 记录训练信息
        self.training_history['training_time'] = datetime.now().isoformat()
        self.training_history['n_samples'] = len(y_train)
        
        return self.full_pipeline
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        使用完整 Pipeline 进行预测
        
        Args:
            X: 原始特征（未转换）
            
        Returns:
            np.ndarray: 预测结果
        """
        if self.full_pipeline is None:
            raise ValueError("Pipeline 尚未训练")
        
        return self.full_pipeline.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        使用完整 Pipeline 进行概率预测
        
        Args:
            X: 原始特征（未转换）
            
        Returns:
            np.ndarray: 预测概率
        """
        if self.full_pipeline is None:
            raise ValueError("Pipeline 尚未训练")
        
        return self.full_pipeline.predict_proba(X)
    
    def save_model(self, filepath: str):
        """
        保存完整 Pipeline（包含特征工程和模型）
        
        Args:
            filepath: 保存路径
        """
        if self.full_pipeline is None:
            raise ValueError("Pipeline 尚未训练")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存完整 Pipeline
        joblib.dump(self.full_pipeline, filepath)
        logger.info(f"完整 Pipeline 已保存至: {filepath}")
        
        # 保存训练历史
        history_path = filepath.parent / f"{filepath.stem}_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        logger.info(f"训练历史已保存至: {history_path}")
    
    def load_model(self, filepath: str):
        """
        加载完整 Pipeline
        
        Args:
            filepath: 加载路径
            
        Returns:
            Pipeline: 加载的 Pipeline
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Pipeline 文件不存在: {filepath}")
        
        self.full_pipeline = joblib.load(filepath)
        logger.info(f"完整 Pipeline 已从 {filepath} 加载")
        
        # 尝试加载训练历史
        history_path = filepath.parent / f"{filepath.stem}_history.json"
        if history_path.exists():
            with open(history_path, 'r') as f:
                self.training_history = json.load(f)
            logger.info(f"训练历史已从 {history_path} 加载")
        
        return self.full_pipeline
    
    def get_model_params(self) -> Dict[str, Any]:
        """
        获取模型参数
        
        Returns:
            Dict: 模型参数
        """
        if self.full_pipeline is None:
            raise ValueError("Pipeline 尚未构建")
        
        return self.full_pipeline.named_steps['model'].get_params()
    
    def get_feature_importance(self, feature_names: list = None) -> pd.DataFrame:
        """
        获取特征重要性（逻辑回归的系数）
        
        Args:
            feature_names: 特征名称列表
            
        Returns:
            pd.DataFrame: 特征重要性
        """
        if self.full_pipeline is None:
            raise ValueError("Pipeline 尚未训练")
        
        model = self.full_pipeline.named_steps['model']
        coefficients = model.coef_[0]
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(coefficients))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        }).sort_values('abs_coefficient', ascending=False)
        
        return importance_df


def main():
    """测试模型训练"""
    import sys
    sys.path.append(str(Path(__file__).parent))
    from data_processor import DataProcessor
    from feature_engineering import FeatureEngineering
    
    # 加载数据
    data_path = "Credit_Risk/Resources/lending_data.csv"
    processor = DataProcessor(data_path)
    processor.load_data()
    processor.clean_data()
    X, y = processor.split_features_target()
    
    # 划分数据集（在特征工程之前，避免数据泄露）
    trainer = ModelTrainer(random_state=1, test_size=0.2)
    X_train, X_test, y_train, y_test = trainer.split_data(X, y)
    
    # 构建特征工程 Pipeline
    fe = FeatureEngineering(k_best=15)
    feature_pipeline = fe.build_feature_pipeline()
    
    # 构建完整 Pipeline（特征工程 + 模型）
    trainer.build_model()
    trainer.build_full_pipeline(feature_pipeline)
    
    # 交叉验证 - 使用完整 Pipeline，传入原始特征
    cv_results = trainer.cross_validate(X_train, y_train, scoring='f1_weighted')
    
    # 训练模型 - 使用完整 Pipeline，传入原始特征
    trainer.train(X_train, y_train)
    
    # 预测 - 使用完整 Pipeline，传入原始特征
    y_pred = trainer.predict(X_test)
    y_pred_proba = trainer.predict_proba(X_test)
    
    # 评估
    from evaluate import ModelEvaluator
    evaluator = ModelEvaluator(model_name="LogisticRegression")
    results = evaluator.evaluate(y_test.values, y_pred, y_pred_proba)
    
    # 保存模型
    trainer.save_model("models/full_pipeline.joblib")
    
    print("\n训练完成！")


if __name__ == "__main__":
    main()
