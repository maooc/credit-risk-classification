from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve
)
import json
import numpy as np
from datetime import datetime

class ModelEvaluator:
    """模型评估器：生成详细的评估报告并持久化结果"""
    
    def __init__(self):
        self.y_true = None
        self.y_pred = None
        self.y_pred_proba = None
        self.evaluation_results = {}
        
    def set_predictions(self, y_true, y_pred, y_pred_proba=None):
        """设置真实值和预测值"""
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
        
    def generate_basic_metrics(self):
        """生成基础评估指标"""
        if self.y_true is None or self.y_pred is None:
            raise ValueError("请先设置真实值和预测值")
            
        metrics = {
            "accuracy": float(accuracy_score(self.y_true, self.y_pred)),
            "precision_weighted": float(precision_score(self.y_true, self.y_pred, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(self.y_true, self.y_pred, average="weighted", zero_division=0)),
            "f1_weighted": float(f1_score(self.y_true, self.y_pred, average="weighted", zero_division=0)),
            "precision_macro": float(precision_score(self.y_true, self.y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(self.y_true, self.y_pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(self.y_true, self.y_pred, average="macro", zero_division=0))
        }
        
        # 按类别计算指标
        labels = np.unique(self.y_true)
        for label in labels:
            label_str = str(label)
            y_true_bin = (self.y_true == label).astype(int)
            y_pred_bin = (self.y_pred == label).astype(int)
            
            metrics[f"class_{label_str}"] = {
                "precision": float(precision_score(y_true_bin, y_pred_bin, zero_division=0)),
                "recall": float(recall_score(y_true_bin, y_pred_bin, zero_division=0)),
                "f1": float(f1_score(y_true_bin, y_pred_bin, zero_division=0)),
                "support": int(np.sum(self.y_true == label))
            }
        
        # AUC（如果有概率预测）
        if self.y_pred_proba is not None and self.y_pred_proba.shape[1] >= 2:
            try:
                metrics["roc_auc_ovr"] = float(roc_auc_score(
                    self.y_true, self.y_pred_proba, multi_class="ovr"
                ))
                metrics["roc_auc_ovo"] = float(roc_auc_score(
                    self.y_true, self.y_pred_proba, multi_class="ovo"
                ))
            except:
                # 二分类情况
                try:
                    metrics["roc_auc"] = float(roc_auc_score(
                        self.y_true, self.y_pred_proba[:, 1]
                    ))
                except:
                    pass
        
        self.evaluation_results["basic_metrics"] = metrics
        return metrics
    
    def generate_confusion_matrix(self):
        """生成混淆矩阵"""
        if self.y_true is None or self.y_pred is None:
            raise ValueError("请先设置真实值和预测值")
            
        cm = confusion_matrix(self.y_true, self.y_pred)
        labels = np.unique(np.concatenate([self.y_true, self.y_pred])).tolist()
        
        cm_result = {
            "matrix": cm.tolist(),
            "labels": [str(label) for label in labels],
            "true_positives": int(cm[1, 1]) if len(cm) > 1 else int(cm[0, 0]),
            "false_positives": int(cm[0, 1]) if len(cm) > 1 else 0,
            "true_negatives": int(cm[0, 0]) if len(cm) > 1 else 0,
            "false_negatives": int(cm[1, 0]) if len(cm) > 1 else 0
        }
        
        self.evaluation_results["confusion_matrix"] = cm_result
        return cm_result
    
    def generate_classification_report(self):
        """生成分类报告"""
        if self.y_true is None or self.y_pred is None:
            raise ValueError("请先设置真实值和预测值")
            
        report = classification_report(
            self.y_true, 
            self.y_pred, 
            output_dict=True,
            zero_division=0
        )
        
        self.evaluation_results["classification_report"] = report
        return report
    
    def generate_full_report(self):
        """生成完整的评估报告"""
        self.generate_basic_metrics()
        self.generate_confusion_matrix()
        self.generate_classification_report()
        
        # 添加元数据
        self.evaluation_results["metadata"] = {
            "evaluation_time": datetime.now().isoformat(),
            "total_samples": len(self.y_true),
            "unique_classes": len(np.unique(self.y_true))
        }
        
        return self.evaluation_results
    
    def save_results_to_json(self, filepath="evaluation_results.json"):
        """将评估结果保存为JSON文件"""
        if not self.evaluation_results:
            self.generate_full_report()
            
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.evaluation_results, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def print_evaluation_summary(self):
        """打印评估摘要到控制台"""
        if not self.evaluation_results:
            self.generate_full_report()
            
        print("=" * 60)
        print("模型评估摘要")
        print("=" * 60)
        
        metrics = self.evaluation_results.get("basic_metrics", {})
        print(f"\n准确率: {metrics.get('accuracy', 0):.4f}")
        print(f"加权F1分数: {metrics.get('f1_weighted', 0):.4f}")
        print(f"宏F1分数: {metrics.get('f1_macro', 0):.4f}")
        
        if "roc_auc" in metrics:
            print(f"ROC AUC: {metrics.get('roc_auc', 0):.4f}")
        
        print("\n类别详细指标:")
        for key, value in metrics.items():
            if key.startswith("class_"):
                class_label = key.split("_")[1]
                print(f"  类别 {class_label}:")
                print(f"    精确率: {value.get('precision', 0):.4f}")
                print(f"    召回率: {value.get('recall', 0):.4f}")
                print(f"    F1: {value.get('f1', 0):.4f}")
                print(f"    样本数: {value.get('support', 0)}")
        
        cm = self.evaluation_results.get("confusion_matrix", {})
        if "matrix" in cm:
            print("\n混淆矩阵:")
            for row in cm["matrix"]:
                print(f"  {row}")
        
        print("=" * 60)
    
    def get_results(self):
        """获取评估结果"""
        if not self.evaluation_results:
            self.generate_full_report()
        return self.evaluation_results
