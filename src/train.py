from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, f1_score
import joblib
import numpy as np

class ModelTrainer:
    """模型训练器：负责模型训练、交叉验证，防止数据泄露"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.full_pipeline = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def split_data(self, X, y, test_size=0.25, stratify=True):
        """
        划分训练集和测试集
        注意：必须在特征工程之前进行划分，防止数据泄露
        """
        split_kwargs = {
            "test_size": test_size,
            "random_state": self.random_state,
            "shuffle": True
        }
        
        if stratify:
            split_kwargs["stratify"] = y
            
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, **split_kwargs
        )
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def build_model(self, class_weight="balanced", max_iter=1000):
        """构建逻辑回归模型，处理类别不平衡"""
        self.model = LogisticRegression(
            class_weight=class_weight,
            max_iter=max_iter,
            random_state=self.random_state,
            solver="lbfgs"
        )
        return self.model
    
    def build_full_training_pipeline(self, feature_pipeline, model=None):
        """
        构建完整的训练管道：特征工程 + 模型
        这是防止数据泄露的关键：将特征工程和模型组合成单一管道
        """
        if model is None:
            if self.model is None:
                self.build_model()
            model = self.model
            
        self.full_pipeline = Pipeline(steps=[
            ("feature_engineering", feature_pipeline),
            ("classifier", model)
        ])
        
        return self.full_pipeline
    
    def train(self, X=None, y=None, full_pipeline=None):
        """
        训练模型
        注意：只使用训练集数据进行训练
        """
        if full_pipeline is not None:
            self.full_pipeline = full_pipeline
            
        if self.full_pipeline is None:
            raise ValueError("请先构建训练管道")
            
        # 如果提供了X和y，使用这些数据训练，否则使用已划分的训练集
        if X is not None and y is not None:
            self.full_pipeline.fit(X, y)
        elif self.X_train is not None and self.y_train is not None:
            self.full_pipeline.fit(self.X_train, self.y_train)
        else:
            raise ValueError("请提供训练数据或先划分数据集")
            
        return self.full_pipeline
    
    def cross_validate(self, X, y, cv=5, scoring="f1_weighted"):
        """
        交叉验证评估模型性能
        使用完整管道确保特征工程不会泄露数据
        """
        if self.full_pipeline is None:
            raise ValueError("请先构建训练管道")
            
        # 使用StratifiedKFold保持类别分布
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        # 执行交叉验证
        scores = cross_val_score(
            self.full_pipeline, 
            X, 
            y, 
            cv=skf, 
            scoring=scoring,
            n_jobs=-1
        )
        
        return {
            "mean_score": np.mean(scores),
            "std_score": np.std(scores),
            "all_scores": scores.tolist()
        }
    
    def predict(self, X):
        """使用训练好的管道进行预测"""
        if self.full_pipeline is None:
            raise ValueError("请先训练模型")
            
        return self.full_pipeline.predict(X)
    
    def predict_proba(self, X):
        """预测概率"""
        if self.full_pipeline is None:
            raise ValueError("请先训练模型")
            
        return self.full_pipeline.predict_proba(X)
    
    def save_model(self, filepath="model_pipeline.joblib"):
        """保存完整的模型管道（包括特征工程）"""
        if self.full_pipeline is None:
            raise ValueError("没有可保存的模型管道")
            
        joblib.dump(self.full_pipeline, filepath)
    
    def load_model(self, filepath="model_pipeline.joblib"):
        """加载模型管道"""
        self.full_pipeline = joblib.load(filepath)
    
    def get_train_test_data(self):
        """获取训练和测试数据"""
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def get_classifier(self):
        """获取分类器"""
        if self.full_pipeline is not None:
            return self.full_pipeline.named_steps["classifier"]
        return self.model
