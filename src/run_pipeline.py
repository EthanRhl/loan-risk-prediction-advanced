import os
import sys
import pandas as pd
import numpy as np

# 将项目根目录加入 sys.path，以便导入 config
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model_training import LoanRiskModel, compare_models
from config import DATA_CONFIG, OUTPUT_CONFIG


def main():
    """主函数"""
    print("=" * 60)
    print("贷款风险预测系统（Home Credit Default Risk）")
    print("=" * 60)

    # 创建输出目录
    os.makedirs(OUTPUT_CONFIG['model_dir'], exist_ok=True)
    os.makedirs(OUTPUT_CONFIG['result_dir'], exist_ok=True)
    os.makedirs(OUTPUT_CONFIG['figure_dir'], exist_ok=True)

    # 数据加载
    print("\n数据加载...")
    preprocessor = DataPreprocessor(max_missing_ratio=0.5)

    if not os.path.exists(DATA_CONFIG['train_file']):
        print(f"错误: 训练数据未找到: {DATA_CONFIG['train_file']}")
        print("请先下载数据集到 data/ 目录")
        return

    train_df = preprocessor.load_data(DATA_CONFIG['train_file'])

    # 数据预处理
    print("\n数据预处理...")
    train_processed = preprocessor.preprocess(train_df, is_train=True)

    # 检查缺失值
    missing = preprocessor.check_missing(train_processed)
    print(f"\n剩余缺失值情况（仅展示前 10 列）:")
    print(missing.head(10))

    # 特征工程
    print("\n特征工程...")
    fe = FeatureEngineer()
    train_featured = fe.create_features(train_processed)

    # 准备训练数据
    print("\n准备训练数据...")
    feature_cols = preprocessor.get_feature_names(train_featured, target_col='TARGET')

    X = train_featured[feature_cols].values
    y = train_featured['TARGET'].values

    print(f"特征数量: {len(feature_cols)}")
    print(f"样本数量: {len(y)}")
    print(f"正样本比例: {y.mean():.2%}")

    # 划分训练 / 验证集
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n训练集: {len(y_train)} 样本, 正样本: {y_train.sum()}")
    print(f"验证集: {len(y_val)} 样本, 正样本: {y_val.sum()}")

    # 模型对比
    print("\n模型对比...")
    results = compare_models(X_train, X_val, y_train, y_val, use_smote=True)

    # 训练最佳模型
    print("\n训练最佳模型...")
    best_model_type = results.loc[results['AUC'].idxmax(), '模型']

    model = LoanRiskModel(model_type=best_model_type)
    model.train(X_train, y_train, use_smote=True)

    # 寻找最优阈值以提升召回率
    print("\n寻找最优阈值...")
    optimal_threshold = model.find_optimal_threshold(X_val, y_val, target_recall=0.30)

    # 使用最优阈值评估
    final_metrics = model.evaluate(X_val, y_val, threshold=optimal_threshold)

    # 特征重要性分析
    print("\n特征重要性分析...")
    importance = model.get_feature_importance(feature_cols)
    if importance is not None:
        print("\nTOP 10 重要特征:")
        print(importance.head(10).to_string(index=False))

        # 保存特征重要性
        importance_path = os.path.join(OUTPUT_CONFIG['result_dir'], 'feature_importance.csv')
        importance.to_csv(importance_path, index=False)

    # 保存模型
    print("\n保存模型...")
    model_path = os.path.join(OUTPUT_CONFIG['model_dir'], f'loan_risk_model_{best_model_type}.pkl')
    model.save_model(model_path)

    # 生成评估报告
    print("\n生成评估报告...")
    report = f"""贷款风险预测模型评估报告（Home Credit Default Risk）
{'=' * 50}
模型类型: {best_model_type}
分类阈值: {optimal_threshold:.4f}

评估指标:
- 准确率: {final_metrics['accuracy']:.4f}
- 精确率: {final_metrics['precision']:.4f}
- 召回率: {final_metrics['recall']:.4f}
- F1分数: {final_metrics['f1']:.4f}
- AUC: {final_metrics['auc']:.4f}

混淆矩阵:
TN={model.metrics['confusion_matrix'][0, 0]}, FP={model.metrics['confusion_matrix'][0, 1]}
FN={model.metrics['confusion_matrix'][1, 0]}, TP={model.metrics['confusion_matrix'][1, 1]}

优化措施:
1.使用 SMOTE 过采样处理类别不平衡
2.使用 class_weight='balanced' 代价敏感学习
3.调整分类阈值以平衡召回率和精确率

TOP 5 重要特征:
"""
    if importance is not None:
        for _, row in importance.head(5).iterrows():
            report += f"- {row['feature']}: {row['importance']:.4f}\n"

    report_path = os.path.join(OUTPUT_CONFIG['result_dir'], 'model_evaluation.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(report)
    print(f"\n评估报告已保存至: {report_path}")

    print("\n" + "=" * 60)
    print("模型训练完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
