import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve
)
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import warnings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    模型评估器：抽离模型评估逻辑
    
    生成详细的测试集评测报告，并持久化为 JSON 文件
    """
    
    def __init__(self, model_name: str = "LogisticRegression"):
        """
        初始化模型评估器
        
        Args:
            model_name: 模型名称
        """
        self.model_name = model_name
        self.evaluation_results = {}
        self.y_true = None
        self.y_pred = None
        self.y_pred_proba = None
        
    def evaluate(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        评估模型性能
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_pred_proba: 预测概率（可选）
            
        Returns:
            Dict: 评估结果字典
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
        
        logger.info("开始评估模型性能...")
        
        # 基础指标
        results = {
            'model_name': self.model_name,
            'evaluation_time': datetime.now().isoformat(),
            'sample_size': len(y_true),
            'metrics': self._calculate_metrics(y_true, y_pred, y_pred_proba),
            'confusion_matrix': self._calculate_confusion_matrix(y_true, y_pred),
            'classification_report': self._get_classification_report(y_true, y_pred)
        }
        
        # 如果有预测概率，计算 ROC 和 PR 曲线
        if y_pred_proba is not None:
            results['roc_auc'] = self._calculate_roc_auc(y_true, y_pred_proba)
        
        self.evaluation_results = results
        
        # 打印评估结果
        self._print_results(results)
        
        return results
    
    def _calculate_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        计算基础评估指标
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_pred_proba: 预测概率
            
        Returns:
            Dict: 评估指标
        """
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision_macro': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
            'precision_weighted': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            'recall_macro': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
            'recall_weighted': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
            'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
            'f1_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        }
        
        # 每个类别的指标
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        metrics['per_class'] = {
            f'class_{i}': {
                'precision': float(p),
                'recall': float(r),
                'f1': float(f)
            }
            for i, (p, r, f) in enumerate(zip(precision_per_class, recall_per_class, f1_per_class))
        }
        
        return metrics
    
    def _calculate_confusion_matrix(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """
        计算混淆矩阵
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            
        Returns:
            Dict: 混淆矩阵信息
        """
        cm = confusion_matrix(y_true, y_pred)
        
        # 对于二分类问题，计算特定指标
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        else:
            tn = fp = fn = tp = None
            specificity = sensitivity = None
        
        return {
            'matrix': cm.tolist(),
            'true_negatives': int(tn) if tn is not None else None,
            'false_positives': int(fp) if fp is not None else None,
            'false_negatives': int(fn) if fn is not None else None,
            'true_positives': int(tp) if tp is not None else None,
            'specificity': float(specificity) if specificity is not None else None,
            'sensitivity': float(sensitivity) if sensitivity is not None else None
        }
    
    def _get_classification_report(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> str:
        """
        获取分类报告
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            
        Returns:
            str: 分类报告字符串
        """
        return classification_report(y_true, y_pred, zero_division=0)
    
    def _calculate_roc_auc(
        self, 
        y_true: np.ndarray, 
        y_pred_proba: np.ndarray
    ) -> Dict[str, Any]:
        """
        计算 ROC AUC
        
        Args:
            y_true: 真实标签
            y_pred_proba: 预测概率
            
        Returns:
            Dict: ROC AUC 信息
        """
        # 对于二分类问题
        if y_pred_proba.ndim == 1 or y_pred_proba.shape[1] == 2:
            if y_pred_proba.ndim > 1:
                y_proba_positive = y_pred_proba[:, 1]
            else:
                y_proba_positive = y_pred_proba
            
            try:
                auc = roc_auc_score(y_true, y_proba_positive)
                fpr, tpr, thresholds = roc_curve(y_true, y_proba_positive)
                
                return {
                    'auc_score': float(auc),
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'thresholds': thresholds.tolist()
                }
            except Exception as e:
                logger.warning(f"计算 ROC AUC 时出错: {e}")
                return {'auc_score': None, 'error': str(e)}
        else:
            # 多分类问题
            try:
                auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
                return {'auc_score': float(auc)}
            except Exception as e:
                logger.warning(f"计算多分类 ROC AUC 时出错: {e}")
                return {'auc_score': None, 'error': str(e)}
    
    def _print_results(self, results: Dict[str, Any]):
        """
        打印评估结果
        
        Args:
            results: 评估结果字典
        """
        print("\n" + "="*60)
        print(f"模型评估报告 - {results['model_name']}")
        print("="*60)
        
        print(f"\n样本数量: {results['sample_size']}")
        print(f"评估时间: {results['evaluation_time']}")
        
        print("\n【基础指标】")
        metrics = results['metrics']
        print(f"  准确率 (Accuracy):     {metrics['accuracy']:.4f}")
        print(f"  精确率 (Precision):    Macro={metrics['precision_macro']:.4f}, Weighted={metrics['precision_weighted']:.4f}")
        print(f"  召回率 (Recall):       Macro={metrics['recall_macro']:.4f}, Weighted={metrics['recall_weighted']:.4f}")
        print(f"  F1 分数:               Macro={metrics['f1_macro']:.4f}, Weighted={metrics['f1_weighted']:.4f}")
        
        if 'roc_auc' in results and results['roc_auc'].get('auc_score') is not None:
            print(f"  ROC AUC:               {results['roc_auc']['auc_score']:.4f}")
        
        print("\n【混淆矩阵】")
        cm = results['confusion_matrix']
        print(f"  矩阵: {cm['matrix']}")
        if cm['true_negatives'] is not None:
            print(f"  TN={cm['true_negatives']}, FP={cm['false_positives']}")
            print(f"  FN={cm['false_negatives']}, TP={cm['true_positives']}")
            print(f"  特异性 (Specificity): {cm['specificity']:.4f}")
            print(f"  敏感性 (Sensitivity): {cm['sensitivity']:.4f}")
        
        print("\n【分类报告】")
        print(results['classification_report'])
        
        print("="*60)
    
    def save_results(self, filepath: str):
        """
        保存评估结果到 JSON 文件
        
        Args:
            filepath: 保存路径
        """
        if not self.evaluation_results:
            raise ValueError("请先调用 evaluate() 进行评估")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.evaluation_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"评估结果已保存至: {filepath}")
    
    def save_report_txt(self, filepath: str):
        """
        保存评估报告到文本文件
        
        Args:
            filepath: 保存路径
        """
        if not self.evaluation_results:
            raise ValueError("请先调用 evaluate() 进行评估")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"模型评估报告 - {self.evaluation_results['model_name']}\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"样本数量: {self.evaluation_results['sample_size']}\n")
            f.write(f"评估时间: {self.evaluation_results['evaluation_time']}\n\n")
            
            f.write("【基础指标】\n")
            metrics = self.evaluation_results['metrics']
            f.write(f"  准确率 (Accuracy):     {metrics['accuracy']:.4f}\n")
            f.write(f"  精确率 (Precision):    Macro={metrics['precision_macro']:.4f}, Weighted={metrics['precision_weighted']:.4f}\n")
            f.write(f"  召回率 (Recall):       Macro={metrics['recall_macro']:.4f}, Weighted={metrics['recall_weighted']:.4f}\n")
            f.write(f"  F1 分数:               Macro={metrics['f1_macro']:.4f}, Weighted={metrics['f1_weighted']:.4f}\n")
            
            if 'roc_auc' in self.evaluation_results and self.evaluation_results['roc_auc'].get('auc_score') is not None:
                f.write(f"  ROC AUC:               {self.evaluation_results['roc_auc']['auc_score']:.4f}\n")
            
            f.write("\n【混淆矩阵】\n")
            cm = self.evaluation_results['confusion_matrix']
            f.write(f"  矩阵: {cm['matrix']}\n")
            if cm['true_negatives'] is not None:
                f.write(f"  TN={cm['true_negatives']}, FP={cm['false_positives']}\n")
                f.write(f"  FN={cm['false_negatives']}, TP={cm['true_positives']}\n")
                f.write(f"  特异性 (Specificity): {cm['specificity']:.4f}\n")
                f.write(f"  敏感性 (Sensitivity): {cm['sensitivity']:.4f}\n")
            
            f.write("\n【分类报告】\n")
            f.write(self.evaluation_results['classification_report'])
        
        logger.info(f"评估报告已保存至: {filepath}")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        获取评估摘要
        
        Returns:
            Dict: 评估摘要
        """
        if not self.evaluation_results:
            raise ValueError("请先调用 evaluate() 进行评估")
        
        metrics = self.evaluation_results['metrics']
        
        return {
            'model_name': self.evaluation_results['model_name'],
            'accuracy': metrics['accuracy'],
            'f1_weighted': metrics['f1_weighted'],
            'precision_weighted': metrics['precision_weighted'],
            'recall_weighted': metrics['recall_weighted']
        }


def main():
    """测试模型评估"""
    import sys
    sys.path.append(str(Path(__file__).parent))
    from data_processor import DataProcessor
    from feature_engineering import FeatureEngineering
    from train import ModelTrainer
    from sklearn.linear_model import LogisticRegression
    import joblib
    
    # 加载数据
    data_path = "Credit_Risk/Resources/lending_data.csv"
    processor = DataProcessor(data_path)
    processor.load_data()
    processor.clean_data()
    X, y = processor.split_features_target()
    
    # 划分数据集
    trainer = ModelTrainer(random_state=1, test_size=0.2)
    X_train, X_test, y_train, y_test = trainer.split_data(X, y)
    
    # 特征工程
    fe = FeatureEngineering(k_best=15)
    fe.build_pipeline()
    X_train_transformed = fe.fit_transform(X_train, y_train)
    X_test_transformed = fe.transform(X_test)
    
    # 训练模型
    model = LogisticRegression(solver='lbfgs', random_state=1, max_iter=1000, class_weight='balanced')
    model.fit(X_train_transformed, y_train)
    
    # 预测
    y_pred = model.predict(X_test_transformed)
    y_pred_proba = model.predict_proba(X_test_transformed)
    
    # 评估
    evaluator = ModelEvaluator(model_name="LogisticRegression")
    results = evaluator.evaluate(y_test.values, y_pred, y_pred_proba)
    
    # 保存结果
    evaluator.save_results("logs/evaluation_results.json")
    evaluator.save_report_txt("logs/evaluation_report.txt")
    
    print("\n评估完成！")


if __name__ == "__main__":
    main()
