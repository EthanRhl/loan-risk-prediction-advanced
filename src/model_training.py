import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')


class LoanRiskModel:
    """贷款风险预测模型"""
    
    def __init__(self, model_type='lightgbm', random_state=42):
        """
        初始化模型
        
        参数:
            model_type: 模型类型 ('logistic', 'random_forest', 'xgboost', 'lightgbm')
            random_state: 随机种子
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.metrics = {}
        
        self._init_model()
    
    def _init_model(self):
        """初始化模型"""
        if self.model_type == 'logistic':
            self.model = LogisticRegression(
                max_iter=1000, 
                random_state=self.random_state,
                class_weight='balanced'  # 代价敏感学习
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1
            )
        elif self.model_type == 'xgboost':
            try:
                from xgboost import XGBClassifier
                # 计算类别权重
                self.model = XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=self.random_state,
                    scale_pos_weight=3,  # 正负样本比例
                    use_label_encoder=False,
                    eval_metric='auc',
                    n_jobs=-1
                )
            except ImportError:
                print("XGBoost未安装，使用梯度提升替代")
                self.model = GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=5,
                    random_state=self.random_state
                )
        elif self.model_type == 'lightgbm':
            try:
                from lightgbm import LGBMClassifier
                self.model = LGBMClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=self.random_state,
                    class_weight='balanced',
                    verbose=-1,
                    n_jobs=-1
                )
            except ImportError:
                print("LightGBM未安装，使用梯度提升替代")
                self.model = GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=5,
                    random_state=self.random_state
                )
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
    
    def train(self, X_train, y_train, use_smote=False):
        """
        训练模型
        
        参数:
            X_train: 训练特征
            y_train: 训练标签
            use_smote: 是否使用SMOTE过采样
        """
        print(f"\n训练 {self.model_type} 模型...")
        
        # SMOTE过采样
        if use_smote:
            try:
                from imblearn.over_sampling import SMOTE
                smote = SMOTE(random_state=self.random_state, sampling_strategy=0.5)
                X_train, y_train = smote.fit_resample(X_train, y_train)
                print(f"SMOTE过采样后: {len(y_train)} 样本")
                print(f"正样本比例: {y_train.mean():.2%}")
            except ImportError:
                print("imbalanced-learn未安装，跳过SMOTE")
        
        # 训练模型
        self.model.fit(X_train, y_train)
        
        # 训练集评估
        y_train_pred = self.model.predict(X_train)
        y_train_proba = self.model.predict_proba(X_train)[:, 1]
        
        self.metrics['train'] = {
            'accuracy': accuracy_score(y_train, y_train_pred),
            'precision': precision_score(y_train, y_train_pred, zero_division=0),
            'recall': recall_score(y_train, y_train_pred, zero_division=0),
            'f1': f1_score(y_train, y_train_pred, zero_division=0),
            'auc': roc_auc_score(y_train, y_train_proba)
        }
        
        print(f"训练集 AUC: {self.metrics['train']['auc']:.4f}")
        print(f"训练集 召回率: {self.metrics['train']['recall']:.4f}")
        
        return self.model
    
    def evaluate(self, X_test, y_test, threshold=0.5):
        """
        评估模型
        
        参数:
            X_test: 测试特征
            y_test: 测试标签
            threshold: 分类阈值
        """
        print(f"\n评估模型 (阈值={threshold})...")
        
        # 预测概率
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        # 根据阈值预测
        y_pred = (y_proba >= threshold).astype(int)
        
        # 计算指标
        self.metrics['test'] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_proba)
        }
        
        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        self.metrics['confusion_matrix'] = cm
        
        # 打印结果
        print("=" * 50)
        print("模型评估结果")
        print("=" * 50)
        print(f"准确率: {self.metrics['test']['accuracy']:.4f}")
        print(f"精确率: {self.metrics['test']['precision']:.4f}")
        print(f"召回率: {self.metrics['test']['recall']:.4f}")
        print(f"F1分数: {self.metrics['test']['f1']:.4f}")
        print(f"AUC: {self.metrics['test']['auc']:.4f}")
        print("-" * 50)
        print("混淆矩阵:")
        print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
        print(f"  FN={cm[1,0]}, TP={cm[1,1]}")
        print("=" * 50)
        
        return self.metrics['test']
    
    def find_optimal_threshold(self, X_test, y_test, target_recall=0.3):
        """
        寻找最优阈值以达到目标召回率
        
        参数:
            X_test: 测试特征
            y_test: 测试标签
            target_recall: 目标召回率
        """
        from sklearn.metrics import precision_recall_curve
        
        y_proba = self.model.predict_proba(X_test)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
        
        # 找到最接近目标召回率的阈值
        idx = np.argmin(np.abs(recalls - target_recall))
        optimal_threshold = thresholds[idx] if idx < len(thresholds) else 0.5
        
        print(f"\n最优阈值: {optimal_threshold:.4f}")
        print(f"对应召回率: {recalls[idx]:.4f}")
        print(f"对应精确率: {precisions[idx]:.4f}")
        
        return optimal_threshold
    
    def get_feature_importance(self, feature_names):
        """获取特征重要性"""
        if hasattr(self.model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        elif hasattr(self.model, 'coef_'):
            importance = pd.DataFrame({
                'feature': feature_names,
                'importance': np.abs(self.model.coef_[0])
            }).sort_values('importance', ascending=False)
        else:
            importance = None
        
        return importance
    
    def save_model(self, filepath):
        """保存模型"""
        import joblib
        joblib.dump(self.model, filepath)
        print(f"模型已保存至: {filepath}")
    
    def load_model(self, filepath):
        """加载模型"""
        import joblib
        self.model = joblib.load(filepath)
        print(f"模型已加载: {filepath}")


def compare_models(X_train, X_test, y_train, y_test, use_smote=True):
    """
    对比多个模型
    
    参数:
        X_train, X_test, y_train, y_test: 训练测试数据
        use_smote: 是否使用SMOTE
    """
    print("=" * 60)
    print("多模型对比实验")
    print("=" * 60)
    
    models = ['logistic', 'random_forest', 'xgboost', 'lightgbm']
    results = []
    
    for model_type in models:
        try:
            model = LoanRiskModel(model_type=model_type)
            model.train(X_train, y_train, use_smote=use_smote)
            metrics = model.evaluate(X_test, y_test)
            
            results.append({
                '模型': model_type,
                '准确率': metrics['accuracy'],
                '精确率': metrics['precision'],
                '召回率': metrics['recall'],
                'F1分数': metrics['f1'],
                'AUC': metrics['auc']
            })
        except Exception as e:
            print(f"{model_type} 模型训练失败: {e}")
    
    results_df = pd.DataFrame(results)
    
    print("\n" + "=" * 60)
    print("模型对比结果")
    print("=" * 60)
    print(results_df.to_string(index=False))
    
    # 找出最佳模型
    best_idx = results_df['AUC'].idxmax()
    print(f"\n最佳模型: {results_df.loc[best_idx, '模型']} (AUC: {results_df.loc[best_idx, 'AUC']:.4f})")
    
    return results_df


if __name__ == '__main__':
    # 测试代码
    from sklearn.datasets import make_classification
    
    # 创建不平衡测试数据
    X, y = make_classification(
        n_samples=1000, 
        n_features=20, 
        n_informative=10,
        n_redundant=5,
        weights=[0.92, 0.08],  # 8%正样本
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"训练集: {len(y_train)} 样本, 正样本比例: {y_train.mean():.2%}")
    print(f"测试集: {len(y_test)} 样本, 正样本比例: {y_test.mean():.2%}")
    
    # 测试单个模型
    model = LoanRiskModel(model_type='lightgbm')
    model.train(X_train, y_train, use_smote=True)
    model.evaluate(X_test, y_test)
    
    # 寻找最优阈值
    optimal_threshold = model.find_optimal_threshold(X_test, y_test, target_recall=0.3)
    model.evaluate(X_test, y_test, threshold=optimal_threshold)
