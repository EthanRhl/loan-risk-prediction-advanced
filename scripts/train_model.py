
"""
贷款风险预测模型训练脚本
使用Scikit-learn构建分类模型
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
import joblib
import json
import os
from datetime import datetime


def load_data():
    """加载数据（使用Home Credit简化版）"""
    # 这里使用模拟数据，实际可替换为真实数据
    np.random.seed(42)
    n_samples = 10000

    data = {
        'EXT_SOURCE_1': np.random.uniform(0, 1, n_samples),
        'EXT_SOURCE_2': np.random.uniform(0, 1, n_samples),
        'EXT_SOURCE_3': np.random.uniform(0, 1, n_samples),
        'DAYS_BIRTH': np.random.uniform(-20000, -5000, n_samples),
        'DAYS_EMPLOYED': np.random.uniform(-10000, 0, n_samples),
        'AMT_INCOME_TOTAL': np.random.uniform(20000, 500000, n_samples),
        'AMT_CREDIT': np.random.uniform(50000, 1000000, n_samples),
        'DAYS_REGISTRATION': np.random.uniform(-5000, 0, n_samples),
        'FLAG_OWN_CAR': np.random.choice([0, 1], n_samples),
        'FLAG_OWN_REALTY': np.random.choice([0, 1], n_samples),
        'CNT_CHILDREN': np.random.choice([0, 1, 2, 3, 4], n_samples),
        'TARGET': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])  # 30%违约率
    }

    return pd.DataFrame(data)


def preprocess_data(df):
    """数据预处理"""
    df = df.copy()

    # 处理负值（DAYS_*字段）
    for col in ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION']:
        df[col] = -df[col] / 365  # 转为年

    # 处理异常值
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].clip(0, 50)

    # 收入对数变换
    df['AMT_INCOME_TOTAL'] = np.log1p(df['AMT_INCOME_TOTAL'])
    df['AMT_CREDIT'] = np.log1p(df['AMT_CREDIT'])

    return df


def train_and_evaluate():
    """训练模型并评估"""
    print("=" * 60)
    print("开始模型训练")
    print("=" * 60)

    # 加载数据
    df = load_data()
    print(f"数据加载完成：{len(df)} 条记录")

    # 预处理
    df = preprocess_data(df)

    # 特征和标签
    feature_cols = [col for col in df.columns if col != 'TARGET']
    X = df[feature_cols]
    y = df['TARGET']

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"训练集：{len(X_train)} 条，测试集：{len(X_test)} 条")

    # 训练逻辑回归模型
    print("\n训练逻辑回归模型...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)

    # 训练随机森林模型
    print("训练随机森林模型...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    # 评估
    models = {
        'Logistic Regression': lr_model,
        'Random Forest': rf_model
    }

    results = {}

    for name, model in models.items():
        print(f"\n{name} 评估结果:")
        print("-" * 40)

        # 预测
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # 计算指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        print(f"准确率 (Accuracy):  {accuracy:.4f}")
        print(f"精确率 (Precision): {precision:.4f}")
        print(f"召回率 (Recall):    {recall:.4f}")
        print(f"F1分数:             {f1:.4f}")
        print(f"AUC-ROC:            {auc:.4f}")

        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n混淆矩阵:\n{cm}")

        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'confusion_matrix': cm.tolist()
        }

    # 保存最佳模型
    best_model_name = max(results, key=lambda x: results[x]['auc'])
    best_model = models[best_model_name]

    model_path = "models/risk_prediction_model.pkl"
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, model_path)
    print(f"\n✓ 最佳模型已保存：{model_path} ({best_model_name})")

    # 保存评估报告
    report = {
        'training_time': datetime.now().isoformat(),
        'dataset_size': len(df),
        'train_size': len(X_train),
        'test_size': len(X_test),
        'features': feature_cols,
        'models_evaluated': len(models),
        'best_model': best_model_name,
        'results': results,
        'feature_importance': (
            list(zip(feature_cols, rf_model.feature_importances_.tolist()))
            if hasattr(rf_model, 'feature_importances_') else []
        )
    }

    report_path = "models/evaluation_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"✓ 评估报告已保存：{report_path}")
    print("=" * 60)

    return report


if __name__ == "__main__":
    train_and_evaluate()