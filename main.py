#!/usr/bin/env python3
"""
信用风险分类模型 - 主入口
一键运行完整流程：数据处理 -> 特征工程 -> 模型训练 -> 评估
"""

import os
import sys

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.data_processor import DataProcessor
from src.feature_engineering import FeatureEngineer
from src.train import ModelTrainer
from src.evaluate import ModelEvaluator

def main():
    print("=" * 60)
    print("信用风险分类模型 - 完整训练流程")
    print("=" * 60)
    
    # 配置参数
    DATA_PATH = os.path.join(
        os.path.dirname(__file__), 
        "Credit_Risk", 
        "Resources", 
        "lending_data.csv"
    )
    MODEL_OUTPUT_PATH = "credit_risk_model.joblib"
    EVALUATION_OUTPUT_PATH = "evaluation_results.json"
    FEATURE_PIPELINE_PATH = "feature_pipeline.joblib"
    
    # 步骤1: 数据加载和预处理
    print("\n[1/5] 数据加载和预处理...")
    processor = DataProcessor(DATA_PATH)
    
    try:
        df = processor.load_data()
        print(f"数据加载完成，共 {len(df)} 条记录")
        
        # 数据质量检查
        quality_report = processor.check_data_quality()
        print(f"数据质量检查完成，重复记录: {quality_report['duplicate_count']} 条")
        print(f"类别分布: {quality_report['class_distribution']}")
        
        # 数据清理
        clean_result = processor.clean_data()
        print(f"数据清理完成，移除重复: {clean_result['duplicate_removed']} 条")
        
        # 分离特征和标签
        X, y = processor.split_features_labels()
        print(f"特征数: {X.shape[1]}, 标签数: {len(y.unique())}")
        
    except Exception as e:
        print(f"数据处理失败: {str(e)}")
        return 1
    
    # 步骤2: 构建特征工程管道
    print("\n[2/5] 构建特征工程管道...")
    try:
        engineer = FeatureEngineer()
        feature_pipeline = engineer.build_full_pipeline(
            feature_selection_method="l1",  # 使用L1正则进行特征选择
            k_features=15
        )
        print("特征工程管道构建完成")
        print(f"  - 广域数值特征 (RobustScaler): {engineer.wide_numeric_features}")
        print(f"  - 常规数值特征 (StandardScaler): {engineer.normal_numeric_features}")
        print(f"  - 离散特征 (KBinsDiscretizer): {engineer.discrete_features}")
        print(f"  - 交互特征: PolynomialFeatures(degree=2, interaction_only=True)")
        print(f"  - 特征选择: L1正则化")
        
    except Exception as e:
        print(f"特征工程失败: {str(e)}")
        return 1
    
    # 步骤3: 模型训练
    print("\n[3/5] 模型训练...")
    try:
        trainer = ModelTrainer(random_state=42)
        
        # 先划分数据集（防止数据泄露）
        X_train, X_test, y_train, y_test = trainer.split_data(
            X, y, test_size=0.25, stratify=True
        )
        print(f"训练集: {len(X_train)} 条, 测试集: {len(X_test)} 条")
        
        # 构建完整训练管道
        full_pipeline = trainer.build_full_training_pipeline(feature_pipeline)
        
        # 执行交叉验证
        print("执行5折交叉验证...")
        cv_results = trainer.cross_validate(X_train, y_train, cv=5)
        print(f"交叉验证结果 - 平均F1: {cv_results['mean_score']:.4f} (±{cv_results['std_score']:.4f})")
        
        # 在完整训练集上训练
        print("在训练集上训练最终模型...")
        trainer.train()
        print("模型训练完成")
        
    except Exception as e:
        print(f"模型训练失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 步骤4: 模型评估
    print("\n[4/5] 模型评估...")
    try:
        evaluator = ModelEvaluator()
        
        # 在测试集上进行预测
        y_pred = trainer.predict(X_test)
        y_pred_proba = trainer.predict_proba(X_test)
        
        evaluator.set_predictions(y_test, y_pred, y_pred_proba)
        
        # 生成并打印评估报告
        evaluator.print_evaluation_summary()
        
        # 保存评估结果
        evaluator.save_results_to_json(EVALUATION_OUTPUT_PATH)
        print(f"评估结果已保存到: {EVALUATION_OUTPUT_PATH}")
        
    except Exception as e:
        print(f"模型评估失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 步骤5: 保存模型
    print("\n[5/5] 保存模型...")
    try:
        trainer.save_model(MODEL_OUTPUT_PATH)
        engineer.save_pipeline(FEATURE_PIPELINE_PATH)
        print(f"模型管道已保存到: {MODEL_OUTPUT_PATH}")
        print(f"特征管道已保存到: {FEATURE_PIPELINE_PATH}")
        
    except Exception as e:
        print(f"保存模型失败: {str(e)}")
        return 1
    
    print("\n" + "=" * 60)
    print("流程完成!")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
