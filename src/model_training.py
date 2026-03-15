import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score
import joblib
import warnings

warnings.filterwarnings('ignore')


class LoanRiskModel:
    """贷款风险预测模型"""

    def __init__(self, model_type='logistic'):
        """
        初始化模型
        参数: model_type: 模型类型 ('logistic', 'random_forest', 'gradient_boosting')
        """
        self.model_type = model_type
        self.model = None
        self.metrics = {}

        if model_type == 'logistic':
            self.model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42,
                                                class_weight='balanced')
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
        else:
            raise ValueError(f"不支持的模型类型：{model_type}")

    def train(self, X_train, y_train):
        """训练模型"""
        print(f"正在训练 {self.model_type} 模型...")
        self.model.fit(X_train, y_train)
        print("模型训练完成")

        # 训练集预测
        y_train_pred = self.model.predict(X_train)
        y_train_proba = self.model.predict_proba(X_train)[:, 1]

        self.metrics['train'] = {
            'accuracy': accuracy_score(y_train, y_train_pred),
            'precision': precision_score(y_train, y_train_pred, zero_division=0),
            'recall': recall_score(y_train, y_train_pred, zero_division=0),
            'f1': f1_score(y_train, y_train_pred, zero_division=0),
            'auc': roc_auc_score(y_train, y_train_proba) if len(np.unique(y_train)) > 1 else 0
        }

        print(f"训练集准确率：{self.metrics['train']['accuracy']:.4f}")
        print(f"训练集AUC：{self.metrics['train']['auc']:.4f}")

        return self.model

    def evaluate(self, X_test, y_test):
        """评估模型"""
        print(f"正在评估模型...")

        # 预测
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        # 计算指标
        self.metrics['test'] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0
        }

        # 交叉验证
        cv_scores = cross_val_score(self.model, X_test, y_test, cv=5, scoring='roc_auc')
        self.metrics['cv_auc_mean'] = cv_scores.mean()
        self.metrics['cv_auc_std'] = cv_scores.std()

        print("=" * 50)
        print("模型评估结果")
        print("=" * 50)
        print(f"测试集准确率：{self.metrics['test']['accuracy']:.4f}")
        print(f"测试集精确率：{self.metrics['test']['precision']:.4f}")
        print(f"测试集召回率：{self.metrics['test']['recall']:.4f}")
        print(f"测试集F1分数：{self.metrics['test']['f1']:.4f}")
        print(f"测试集AUC：{self.metrics['test']['auc']:.4f}")
        print(f"5折交叉验证AUC：{self.metrics['cv_auc_mean']:.4f} (+/- {self.metrics['cv_auc_std']:.4f})")
        print("=" * 50)

        return self.metrics['test']

    def predict(self, X):
        """预测"""
        return self.model.predict(X)

    def predict_proba(self, X):
        """预测概率"""
        return self.model.predict_proba(X)

    def get_feature_importance(self, feature_names):
        """获取特征重要性"""
        if hasattr(self.model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance
        elif hasattr(self.model, 'coef_'):
            importance = pd.DataFrame({
                'feature': feature_names,
                'importance': np.abs(self.model.coef_[0])
            }).sort_values('importance', ascending=False)
            return importance
        else:
            return None

    def save_model(self, path):
        """保存模型"""
        joblib.dump(self.model, path)
        print(f"模型已保存至：{path}")

    def load_model(self, path):
        """加载模型"""
        self.model = joblib.load(path)
        print(f"模型已加载：{path}")


def train_and_compare_models(X_train, X_test, y_train, y_test):
    """训练并比较多个模型"""
    print("=" * 60)
    print("多模型对比实验")
    print("=" * 60)

    models = {
        '逻辑回归': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        '随机森林': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced'),
        '梯度提升': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    }

    results = []

    for name, model in models.items():
        print(f"\n训练 {name}...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            '模型': name,
            '准确率': accuracy_score(y_test, y_pred),
            '精确率': precision_score(y_test, y_pred, zero_division=0),
            '召回率': recall_score(y_test, y_pred, zero_division=0),
            'F1分数': f1_score(y_test, y_pred, zero_division=0),
            'AUC': roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0
        }

        results.append(metrics)
        print(f"  准确率：{metrics['准确率']:.4f}, AUC：{metrics['AUC']:.4f}")

    results_df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("模型对比结果")
    print("=" * 60)
    print(results_df.to_string(index=False))

    # 找出最佳模型
    best_model = results_df.loc[results_df['AUC'].idxmax()]
    print(f"\n最佳模型：{best_model['模型']} (AUC: {best_model['AUC']:.4f})")

    return results_df


if __name__ == "__main__":
    # 测试代码
    print("模型训练模块测试")
    model = LoanRiskModel(model_type='logistic')
    print("模型初始化成功")