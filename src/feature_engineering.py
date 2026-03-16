import pandas as pd
import numpy as np


class FeatureEngineer:
    """特征工程类"""

    def __init__(self):
        self.feature_importance = None

    def create_features(self, df: pd.DataFrame):
        """创建新特征（仅使用 Home Credit 数据集中真实存在的字段）"""
        df = df.copy()

        # 比率特征
        if all(col in df.columns for col in ['AMT_CREDIT', 'AMT_INCOME_TOTAL']):
            df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + 1)

        if all(col in df.columns for col in ['AMT_ANNUITY', 'AMT_INCOME_TOTAL']):
            df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1)

        if all(col in df.columns for col in ['AMT_CREDIT', 'AMT_ANNUITY']):
            df['CREDIT_ANNUITY_RATIO'] = df['AMT_CREDIT'] / (df['AMT_ANNUITY'] + 1)

        if all(col in df.columns for col in ['AMT_GOODS_PRICE', 'AMT_CREDIT']):
            df['GOODS_CREDIT_RATIO'] = df['AMT_GOODS_PRICE'] / (df['AMT_CREDIT'] + 1)

        # 时间特征（Home Credit 中 DAYS_BIRTH / DAYS_EMPLOYED 等为负数天数）
        if 'DAYS_BIRTH' in df.columns:
            df['AGE_YEARS'] = -df['DAYS_BIRTH'] / 365

        if 'DAYS_EMPLOYED' in df.columns:
            df['EMPLOYMENT_YEARS'] = -df['DAYS_EMPLOYED'] / 365

        if all(col in df.columns for col in ['DAYS_EMPLOYED', 'DAYS_BIRTH']):
            df['EMPLOYMENT_AGE_RATIO'] = df['EMPLOYMENT_YEARS'] / (df['AGE_YEARS'] + 1)

        if 'DAYS_REGISTRATION' in df.columns:
            df['REGISTRATION_YEARS'] = -df['DAYS_REGISTRATION'] / 365

        if 'DAYS_ID_PUBLISH' in df.columns:
            df['ID_PUBLISH_YEARS'] = -df['DAYS_ID_PUBLISH'] / 365

        # 统计特征
        if all(col in df.columns for col in ['AMT_INCOME_TOTAL', 'CNT_FAM_MEMBERS']):
            df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'] + 1)

        if all(col in df.columns for col in ['AMT_INCOME_TOTAL', 'CNT_CHILDREN']):
            # CNT_CHILDREN 可能为 0，+1 避免除零
            df['INCOME_PER_CHILD'] = df['AMT_INCOME_TOTAL'] / (df['CNT_CHILDREN'] + 1)

        # 外部评分组合（EXT_SOURCE_1/2/3 是 Home Credit 中重要的外部评分）
        ext_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
        existing_ext_cols = [col for col in ext_cols if col in df.columns]

        if len(existing_ext_cols) > 0:
            df['EXT_SOURCE_MEAN'] = df[existing_ext_cols].mean(axis=1)
            df['EXT_SOURCE_MAX'] = df[existing_ext_cols].max(axis=1)
            df['EXT_SOURCE_MIN'] = df[existing_ext_cols].min(axis=1)
            df['EXT_SOURCE_STD'] = df[existing_ext_cols].std(axis=1)

        # 文档提交特征（FLAG_DOCUMENT_* 是 Home Credit 中的一类标志列）
        flag_docs = [col for col in df.columns if col.startswith('FLAG_DOCUMENT_')]
        if flag_docs:
            df['DOC_COUNT'] = df[flag_docs].sum(axis=1)

        #  联系方式完整度
        contact_cols = [
            'FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE',
            'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL'
        ]
        existing_contact = [col for col in contact_cols if col in df.columns]
        if existing_contact:
            df['CONTACT_COUNT'] = df[existing_contact].sum(axis=1)

        # 地址匹配特征（LIVE_CITY_NOT_WORK_CITY 在 Home Credit 中存在）
        if all(col in df.columns for col in ['REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY']):
            df['ADDRESS_MISMATCH'] = (
                df['REG_CITY_NOT_LIVE_CITY'] +
                df['REG_CITY_NOT_WORK_CITY'] +
                df.get('LIVE_CITY_NOT_WORK_CITY', 0)
            )

        print(f"特征工程完成，新增特征后数据形状: {df.shape}")
        return df

    def select_features_by_importance(self, df: pd.DataFrame,
                                      feature_importance: pd.DataFrame,
                                      top_n: int = 50,
                                      target_col: str = 'TARGET'):
        """根据特征重要性选择特征"""
        top_features = feature_importance.head(top_n)['feature'].tolist()

        # 确保目标列和 ID 列保留
        keep_cols = top_features + ['SK_ID_CURR']
        if target_col in df.columns:
            keep_cols.append(target_col)

        selected_cols = [col for col in keep_cols if col in df.columns]

        print(f"选择 TOP {top_n} 重要特征")
        return df[selected_cols]

    def remove_high_correlation(self, df: pd.DataFrame,
                               threshold: float = 0.9,
                               target_col: str = 'TARGET'):
        """移除高相关性特征"""
        # 排除 ID 和目标列
        exclude_cols = ['SK_ID_CURR', target_col]
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        # 计算相关矩阵
        corr_matrix = df[feature_cols].corr().abs()

        # 找出高相关特征对
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

        print(f"移除 {len(to_drop)} 个高相关性特征")
        return df.drop(columns=to_drop)

    def get_feature_summary(self, df: pd.DataFrame):
        """获取特征摘要"""
        summary = pd.DataFrame({
            'dtype': df.dtypes,
            'missing_count': df.isnull().sum(),
            'missing_ratio': df.isnull().sum() / len(df) * 100,
            'unique_count': df.nunique(),
            'min': df.min(numeric_only=True),
            'max': df.max(numeric_only=True),
            'mean': df.mean(numeric_only=True),
            'std': df.std(numeric_only=True)
        })

        return summary


if __name__ == '__main__':
    # 简单测试：使用 Home Credit 真实数据前 n 行
    import os
    import sys

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)

    from config import DATA_CONFIG

    fe = FeatureEngineer()

    if not os.path.exists(DATA_CONFIG['train_file']):
        print(f"训练数据未找到: {DATA_CONFIG['train_file']}")
        print("请先从 Kaggle 下载数据到 data/ 目录")
    else:
        # 只读前 1000 行做测试
        train_df = pd.read_csv(DATA_CONFIG['train_file'], nrows=1000)
        result = fe.create_features(train_df)

        print("新增特征:")
        new_cols = [col for col in result.columns if col not in train_df.columns]
        print(result[new_cols].head())
