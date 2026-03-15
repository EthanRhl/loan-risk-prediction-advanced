import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# 导入自定义模块
from src.data_preprocessing import LoanDataPreprocessor
from src.model_training import LoanRiskModel, train_and_compare_models
from src.visualization import plot_model_results


def main():
    """主函数"""
    print("=" * 70)
    print("贷款风险预测系统")
    print("作者：任惠霖")
    print(f"运行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # 检查数据文件
    data_file = "data/application_train.csv"  # Home Credit数据集

    if not os.path.exists(data_file):
        print(f"⚠️  警告：数据文件 {data_file} 不存在")
        print("请从Kaggle下载 Home Credit Default Risk 数据集")
        print("下载地址：https://www.kaggle.com/c/home-credit-default-risk/data")
        print("\n使用模拟数据进行演示...")

        # 创建模拟数据
        np.random.seed(42)
        n_samples = 10000
        df = pd.DataFrame({
            'SK_ID_CURR': range(1, n_samples + 1),
            'CNT_CHILDREN': np.random.randint(0, 5, n_samples),
            'AMT_INCOME_TOTAL': np.random.uniform(20000, 500000, n_samples),
            'AMT_CREDIT': np.random.uniform(50000, 1000000, n_samples),
            'AMT_ANNUITY': np.random.uniform(5000, 50000, n_samples),
            'DAYS_BIRTH': -np.random.randint(5000, 25000, n_samples),
            'DAYS_EMPLOYED': -np.random.randint(0, 15000, n_samples),
            'FLAG_OWN_CAR': np.random.choice(['Y', 'N'], n_samples),
            'FLAG_OWN_REALTY': np.random.choice(['Y', 'N'], n_samples),
            'NAME_INCOME_TYPE': np.random.choice(['Working', 'Pensioner', 'Commercial'], n_samples),
            'CODE_GENDER': np.random.choice(['M', 'F'], n_samples),
            'TARGET': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])  # 30%违约率
        })

        # 保存模拟数据
        os.makedirs("data", exist_ok=True)
        df.to_csv(data_file, index=False)
        print(f"✓ 模拟数据已保存至：{data_file}")

    # 数据预处理
    print("\n" + "=" * 70)
    print("数据预处理")
    print("=" * 70)

    preprocessor = LoanDataPreprocessor()
    df = preprocessor.load_data(data_file)

    # 转换目标变量（如果是二分类）
    if 'TARGET' in df.columns:
        target_col = 'TARGET'
    elif 'loan_status' in df.columns:
        target_col = 'loan_status'
        df[target_col] = df[target_col].apply(lambda x: 1 if x > 0 else 0)
    else:
        print("未找到目标变量列")
        return

    # 预处理
    X_train, X_test, y_train, y_test, df_processed = preprocessor.fit_transform(df, target_col)

    # 模型训练与对比
    print("\n" + "=" * 70)
    print("模型训练与对比")
    print("=" * 70)

    # 多模型对比
    comparison_results = train_and_compare_models(X_train, X_test, y_train, y_test)

    # 训练最佳模型
    print("\n" + "=" * 70)
    print("训练最佳模型")
    print("=" * 70)

    best_model = LoanRiskModel(model_type='random_forest')
    best_model.train(X_train, y_train)
    test_metrics = best_model.evaluate(X_test, y_test)

    # 保存结果
    print("\n" + "=" * 70)
    print("保存结果")
    print("=" * 70)

    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # 保存模型
    model_path = "models/loan_risk_model.pkl"
    best_model.save_model(model_path)

    # 保存评估结果
    results_path = "results/model_evaluation.txt"
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write("贷款风险预测模型评估报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"模型类型：随机森林\n")
        f.write(f"训练时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("测试集指标:\n")
        for k, v in test_metrics.items():
            f.write(f"  {k}: {v:.4f}\n")
        f.write(
            f"\n交叉验证AUC: {best_model.metrics['cv_auc_mean']:.4f} (+/- {best_model.metrics['cv_auc_std']:.4f})\n")

    print(f"✓ 模型已保存至：{model_path}")
    print(f"✓ 评估报告已保存至：{results_path}")

    # 特征重要性
    print("\n" + "=" * 70)
    print("特征重要性分析")
    print("=" * 70)

    feature_importance = best_model.get_feature_importance(X_train.columns)
    if feature_importance is not None:
        print("Top 10 重要特征:")
        print(feature_importance.head(10).to_string(index=False))

        # 保存特征重要性
        feature_importance.to_csv("results/feature_importance.csv", index=False)
        print("特征重要性已保存至：results/feature_importance.csv")

    # 7. 总结
    print("\n" + "=" * 70)
    print("项目完成总结")
    print("=" * 70)
    print(f"✓ 数据预处理：完成")
    print(f"✓ 模型训练：完成 (3个模型对比)")
    print(f"✓ 模型评估：完成 (AUC: {test_metrics['auc']:.4f})")
    print(f"✓ 结果保存：完成")
    print("=" * 70)

    return test_metrics


if __name__ == "__main__":
    try:
        metrics = main()
        print("\n程序运行成功!")
    except Exception as e:
        print(f"\n程序运行失败：{str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)