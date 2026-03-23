#!/usr/bin/env python3
"""
信用风险分类项目 - 主入口程序

一键运行全阶段：数据加载 -> 特征工程 -> 模型训练 -> 模型评估

使用示例:
    python main.py --data-path Credit_Risk/Resources/lending_data.csv
    python main.py --k-best 20 --cv-folds 5

关键设计：
1. 使用完整 Pipeline（特征工程 + 模型）进行交叉验证，防止数据泄露
2. 特征交互仅应用于 interest_rate, borrower_income, total_debt 三列
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd

# 添加 src 到路径
sys.path.append(str(Path(__file__).parent / "src"))

from data_processor import DataProcessor
from feature_engineering import FeatureEngineering
from train import ModelTrainer
from evaluate import ModelEvaluator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/training.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='信用风险分类 - 机器学习全流程（防数据泄露设计）',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 数据路径
    parser.add_argument(
        '--data-path',
        type=str,
        default='Credit_Risk/Resources/lending_data.csv',
        help='数据文件路径'
    )
    
    # 特征工程参数
    parser.add_argument(
        '--k-best',
        type=int,
        default=15,
        help='SelectKBest 选择的特征数量'
    )
    
    # 训练参数
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='测试集比例'
    )
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='交叉验证折数'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=1,
        help='随机种子'
    )
    
    # 模型保存路径
    parser.add_argument(
        '--model-path',
        type=str,
        default='models/full_pipeline.joblib',
        help='完整 Pipeline 保存路径'
    )
    
    # 评估结果保存路径
    parser.add_argument(
        '--eval-json-path',
        type=str,
        default='logs/evaluation_results.json',
        help='评估结果 JSON 保存路径'
    )
    parser.add_argument(
        '--eval-txt-path',
        type=str,
        default='logs/evaluation_report.txt',
        help='评估报告文本保存路径'
    )
    
    # 跳过步骤选项
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='跳过训练，直接加载已有模型进行评估'
    )
    
    return parser.parse_args()


def setup_directories():
    """创建必要的目录"""
    dirs = ['logs', 'models', 'data']
    for d in dirs:
        Path(d).mkdir(exist_ok=True)
    logger.info("目录结构检查完成")


def run_data_loading(data_path: str) -> tuple:
    """
    阶段 1: 数据加载与预处理
    
    Args:
        data_path: 数据文件路径
        
    Returns:
        tuple: (processor, X, y)
    """
    logger.info("="*60)
    logger.info("阶段 1: 数据加载与预处理")
    logger.info("="*60)
    
    # 初始化数据处理器
    processor = DataProcessor(data_path)
    
    # 加载数据
    processor.load_data()
    
    # 数据质量检查
    quality_report = processor.check_data_quality()
    
    # 清理数据
    processor.clean_data()
    
    # 获取数据摘要
    summary = processor.get_data_summary()
    
    # 分离特征和目标
    X, y = processor.split_features_target()
    
    logger.info("数据加载与预处理完成\n")
    
    return processor, X, y


def run_training_with_full_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    k_best: int,
    cv_folds: int,
    random_state: int,
    model_path: str
) -> ModelTrainer:
    """
    阶段 2 & 3: 构建完整 Pipeline 并进行交叉验证和训练
    
    关键设计：
    1. 构建特征工程 Pipeline（仅对 interest_rate, borrower_income, total_debt 进行多项式交叉）
    2. 将模型封装到 Pipeline 中形成闭环
    3. 交叉验证时直接传入原始特征和完整 Pipeline，防止数据泄露
    
    Args:
        X_train: 原始训练特征（未转换）
        y_train: 训练目标
        X_test: 原始测试特征（未转换）
        y_test: 测试目标
        k_best: 选择的特征数量
        cv_folds: 交叉验证折数
        random_state: 随机种子
        model_path: 模型保存路径
        
    Returns:
        ModelTrainer: 训练器对象
    """
    logger.info("="*60)
    logger.info("阶段 2 & 3: 特征工程与模型训练（防数据泄露设计）")
    logger.info("="*60)
    
    # 初始化训练器
    trainer = ModelTrainer(
        random_state=random_state,
        cv_folds=cv_folds
    )
    
    # 构建特征工程 Pipeline
    logger.info("\n【构建特征工程 Pipeline】")
    fe = FeatureEngineering(k_best=k_best)
    feature_pipeline = fe.build_feature_pipeline()
    
    # 显示特征分组
    column_groups = fe.get_column_groups()
    logger.info(f"特征分组:")
    for group, cols in column_groups.items():
        logger.info(f"  - {group}: {cols}")
    
    # 构建完整 Pipeline（特征工程 + 模型）
    logger.info("\n【构建完整 Pipeline（特征工程 + 模型）】")
    trainer.build_model()
    trainer.build_full_pipeline(feature_pipeline)
    
    # 交叉验证 - 关键：使用完整 Pipeline，传入原始特征
    logger.info("\n【交叉验证（使用完整 Pipeline，防止数据泄露）】")
    cv_results = trainer.cross_validate(
        X_train, 
        y_train, 
        scoring='f1_weighted'
    )
    
    # 训练完整 Pipeline - 传入原始特征
    logger.info("\n【训练完整 Pipeline】")
    trainer.train(X_train, y_train)
    
    # 保存完整 Pipeline
    trainer.save_model(model_path)
    
    logger.info("训练完成\n")
    
    return trainer


def run_model_evaluation(
    trainer: ModelTrainer,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    eval_json_path: str,
    eval_txt_path: str
) -> dict:
    """
    阶段 4: 模型评估
    
    Args:
        trainer: 训练器对象（包含训练好的完整 Pipeline）
        X_test: 原始测试特征（未转换）
        y_test: 测试目标
        eval_json_path: JSON 评估结果保存路径
        eval_txt_path: 文本评估报告保存路径
        
    Returns:
        dict: 评估结果
    """
    logger.info("="*60)
    logger.info("阶段 4: 模型评估")
    logger.info("="*60)
    
    # 使用完整 Pipeline 进行预测 - 传入原始特征
    logger.info("使用完整 Pipeline 进行预测...")
    y_pred = trainer.predict(X_test)
    y_pred_proba = trainer.predict_proba(X_test)
    
    # 初始化评估器
    evaluator = ModelEvaluator(model_name="LogisticRegression")
    
    # 评估
    results = evaluator.evaluate(
        y_test.values if hasattr(y_test, 'values') else y_test,
        y_pred,
        y_pred_proba
    )
    
    # 保存评估结果
    evaluator.save_results(eval_json_path)
    evaluator.save_report_txt(eval_txt_path)
    
    logger.info("模型评估完成\n")
    
    return results


def main():
    """主函数：运行全流程"""
    # 解析参数
    args = parse_arguments()
    
    # 设置目录
    setup_directories()
    
    logger.info("="*60)
    logger.info("信用风险分类 - 机器学习全流程开始")
    logger.info("="*60)
    logger.info(f"数据路径: {args.data_path}")
    logger.info(f"K-Best 特征数: {args.k_best}")
    logger.info(f"测试集比例: {args.test_size}")
    logger.info(f"交叉验证折数: {args.cv_folds}")
    logger.info(f"随机种子: {args.random_state}")
    logger.info("关键设计: 使用完整 Pipeline 防止数据泄露")
    logger.info("特征交互: 仅应用于 interest_rate, borrower_income, total_debt")
    logger.info("")
    
    try:
        # ========== 阶段 1: 数据加载 ==========
        processor, X, y = run_data_loading(args.data_path)
        
        # ========== 划分数据集（在特征工程之前，避免数据泄露）==========
        logger.info("="*60)
        logger.info("数据划分（在特征工程之前，避免数据泄露）")
        logger.info("="*60)
        
        trainer = ModelTrainer(
            random_state=args.random_state,
            test_size=args.test_size
        )
        X_train, X_test, y_train, y_test = trainer.split_data(X, y)
        
        # 保存测试集标签（用于后续评估）
        test_data = pd.DataFrame({'y_true': y_test})
        test_data.to_csv('data/test_labels.csv', index=False)
        logger.info("测试集标签已保存至: data/test_labels.csv\n")
        
        if args.skip_training:
            # ========== 跳过训练，加载已有模型 ==========
            logger.info("="*60)
            logger.info("跳过训练，加载已有模型")
            logger.info("="*60)
            
            trainer.load_model(args.model_path)
            
        else:
            # ========== 阶段 2 & 3: 特征工程与模型训练 ==========
            trainer = run_training_with_full_pipeline(
                X_train, y_train,
                X_test, y_test,
                args.k_best,
                args.cv_folds,
                args.random_state,
                args.model_path
            )
        
        # ========== 阶段 4: 模型评估 ==========
        results = run_model_evaluation(
            trainer,
            X_test, y_test,
            args.eval_json_path,
            args.eval_txt_path
        )
        
        # ========== 完成 ==========
        logger.info("="*60)
        logger.info("信用风险分类 - 机器学习全流程完成")
        logger.info("="*60)
        logger.info(f"完整 Pipeline 保存路径: {args.model_path}")
        logger.info(f"评估结果 JSON: {args.eval_json_path}")
        logger.info(f"评估报告文本: {args.eval_txt_path}")
        
        # 打印关键指标
        summary = {
            'accuracy': results['metrics']['accuracy'],
            'f1_weighted': results['metrics']['f1_weighted'],
            'precision_weighted': results['metrics']['precision_weighted'],
            'recall_weighted': results['metrics']['recall_weighted']
        }
        if 'roc_auc' in results and results['roc_auc'].get('auc_score'):
            summary['roc_auc'] = results['roc_auc']['auc_score']
        
        logger.info("\n【关键指标摘要】")
        for metric, value in summary.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return 0
        
    except Exception as e:
        logger.error(f"运行过程中出现错误: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
