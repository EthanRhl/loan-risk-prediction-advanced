#项目配置文件
# 数据配置
DATA_CONFIG = {
    'train_file': 'data/application_train.csv',
    'test_file': 'data/application_test.csv',
    'target_col': 'TARGET',
    'id_col': 'SK_ID_CURR'
}

# 模型配置
MODEL_CONFIG = {
    'random_state': 42,
    'test_size': 0.2,
    'cv_folds': 5
}

# 类别不平衡处理配置
IMBALANCE_CONFIG = {
    'use_smote': True,
    'smote_sampling_strategy': 0.5,  # 少数类采样比例
    'use_class_weight': True
}

# 特征工程配置
FEATURE_CONFIG = {
    'max_missing_ratio': 0.5,       # 缺失值超过此比例的特征将被删除
    'correlation_threshold': 0.9,   # 高相关性特征阈值
    'n_top_features': 50            # 保留的 top 特征数量（如需按重要性筛选）
}

# 输出配置
OUTPUT_CONFIG = {
    'model_dir': 'models/',
    'result_dir': 'results/',
    'figure_dir': 'results/figures/'
}
