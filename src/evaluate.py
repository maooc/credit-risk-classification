from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import json
from pathlib import Path
from datetime import datetime
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelEvaluator:
    
    def __init__(self, output_dir: str = './outputs'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.y_true: Optional[np.ndarray] = None
        self.y_pred: Optional[np.ndarray] = None
        self.y_pred_proba: Optional[np.ndarray] = None
        
        self.metrics: Dict[str, Any] = {}
        self.confusion_matrix_result: Optional[np.ndarray] = None

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_pred_proba = np.array(y_pred_proba) if y_pred_proba is not None else None
        
        self.metrics = self._calculate_metrics()
        self.confusion_matrix_result = confusion_matrix(self.y_true, self.y_pred)
        
        logger.info("Evaluation completed")
        self._print_results()
        
        return self.metrics

    def _calculate_metrics(self) -> Dict[str, Any]:
        metrics = {
            'accuracy': float(accuracy_score(self.y_true, self.y_pred)),
            'precision_macro': float(precision_score(self.y_true, self.y_pred, average='macro')),
            'precision_weighted': float(precision_score(self.y_true, self.y_pred, average='weighted')),
            'recall_macro': float(recall_score(self.y_true, self.y_pred, average='macro')),
            'recall_weighted': float(recall_score(self.y_true, self.y_pred, average='weighted')),
            'f1_macro': float(f1_score(self.y_true, self.y_pred, average='macro')),
            'f1_weighted': float(f1_score(self.y_true, self.y_pred, average='weighted')),
        }
        
        classes = np.unique(self.y_true)
        per_class_metrics = {}
        
        for cls in classes:
            cls_str = str(cls)
            y_true_binary = (self.y_true == cls).astype(int)
            y_pred_binary = (self.y_pred == cls).astype(int)
            
            per_class_metrics[cls_str] = {
                'precision': float(precision_score(y_true_binary, y_pred_binary)),
                'recall': float(recall_score(y_true_binary, y_pred_binary)),
                'f1': float(f1_score(y_true_binary, y_pred_binary)),
                'support': int(np.sum(y_true_binary))
            }
        
        metrics['per_class_metrics'] = per_class_metrics
        
        if self.y_pred_proba is not None:
            if len(self.y_pred_proba.shape) > 1 and self.y_pred_proba.shape[1] > 1:
                y_score = self.y_pred_proba[:, 1]
            else:
                y_score = self.y_pred_proba
            
            metrics['roc_auc'] = float(roc_auc_score(self.y_true, y_score))
            metrics['average_precision'] = float(average_precision_score(self.y_true, y_score))
        
        return metrics

    def _print_results(self) -> None:
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        
        print("\n--- Confusion Matrix ---")
        print(self.confusion_matrix_result)
        
        print("\n--- Classification Report ---")
        print(classification_report(self.y_true, self.y_pred))
        
        print("\n--- Key Metrics ---")
        print(f"Accuracy: {self.metrics['accuracy']:.4f}")
        print(f"Precision (Macro): {self.metrics['precision_macro']:.4f}")
        print(f"Recall (Macro): {self.metrics['recall_macro']:.4f}")
        print(f"F1 Score (Macro): {self.metrics['f1_macro']:.4f}")
        
        if 'roc_auc' in self.metrics:
            print(f"ROC AUC: {self.metrics['roc_auc']:.4f}")
            print(f"Average Precision: {self.metrics['average_precision']:.4f}")
        
        print("="*60 + "\n")

    def save_results(self, filename: Optional[str] = None) -> str:
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'evaluation_results_{timestamp}.json'
        
        filepath = self.output_dir / filename
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'metrics': self.metrics,
            'confusion_matrix': self.confusion_matrix_result.tolist(),
            'classification_report': classification_report(self.y_true, self.y_pred, output_dict=True)
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Evaluation results saved to {filepath}")
        
        return str(filepath)

    def plot_confusion_matrix(
        self,
        labels: Optional[list] = None,
        filename: Optional[str] = None
    ) -> str:
        if self.confusion_matrix_result is None:
            raise ValueError("No evaluation results. Call evaluate() first.")
        
        if labels is None:
            labels = ['Healthy Loan (0)', 'High-Risk Loan (1)']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        cm = self.confusion_matrix_result
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=labels,
               yticklabels=labels,
               title='Confusion Matrix',
               ylabel='True Label',
               xlabel='Predicted Label')
        
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        fig.tight_layout()
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'confusion_matrix_{timestamp}.png'
        
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix plot saved to {filepath}")
        
        return str(filepath)

    def plot_roc_curve(self, filename: Optional[str] = None) -> str:
        if self.y_pred_proba is None:
            raise ValueError("Prediction probabilities not available.")
        
        if len(self.y_pred_proba.shape) > 1 and self.y_pred_proba.shape[1] > 1:
            y_score = self.y_pred_proba[:, 1]
        else:
            y_score = self.y_pred_proba
        
        fpr, tpr, thresholds = roc_curve(self.y_true, y_score)
        roc_auc = self.metrics.get('roc_auc', roc_auc_score(self.y_true, y_score))
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'roc_curve_{timestamp}.png'
        
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ROC curve plot saved to {filepath}")
        
        return str(filepath)

    def plot_precision_recall_curve(self, filename: Optional[str] = None) -> str:
        if self.y_pred_proba is None:
            raise ValueError("Prediction probabilities not available.")
        
        if len(self.y_pred_proba.shape) > 1 and self.y_pred_proba.shape[1] > 1:
            y_score = self.y_pred_proba[:, 1]
        else:
            y_score = self.y_pred_proba
        
        precision, recall, thresholds = precision_recall_curve(self.y_true, y_score)
        avg_precision = self.metrics.get('average_precision', average_precision_score(self.y_true, y_score))
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, color='blue', lw=2, 
                label=f'PR curve (AP = {avg_precision:.4f})')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'precision_recall_curve_{timestamp}.png'
        
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Precision-Recall curve plot saved to {filepath}")
        
        return str(filepath)

    def generate_full_report(self) -> Dict[str, str]:
        results = {}
        
        results['evaluation_json'] = self.save_results()
        results['confusion_matrix_plot'] = self.plot_confusion_matrix()
        
        if self.y_pred_proba is not None:
            results['roc_curve_plot'] = self.plot_roc_curve()
            results['precision_recall_plot'] = self.plot_precision_recall_curve()
        
        logger.info("Full evaluation report generated")
        
        return results

    def get_summary(self) -> Dict[str, Any]:
        return {
            'metrics': self.metrics,
            'confusion_matrix': self.confusion_matrix_result.tolist() if self.confusion_matrix_result is not None else None
        }
