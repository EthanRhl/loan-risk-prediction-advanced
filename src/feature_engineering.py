import pandas as pd
import numpy as np


class FeatureEngineer:
    """特征工程类"""
    
    def __init__(self):
        self.feature_importance = None
    
    def create_features(self, df: pd.DataFrame):
        """创建新特征"""
        df = df.copy()
        
        # 比率特征
        df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + 1)
        df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1)
        df['CREDIT_ANNUITY_RATIO'] = df['AMT_CREDIT'] / (df['AMT_ANNUITY'] + 1)
        df['GOODS_CREDIT_RATIO'] = df['AMT_GOODS_PRICE'] / (df['AMT_CREDIT'] + 1)
        
        # 时间特征
        df['AGE_YEARS'] = -df['DAYS_BIRTH'] / 365
        df['EMPLOYMENT_YEARS'] = -df['DAYS_EMPLOYED'] / 365
        df['EMPLOYMENT_AGE_RATIO'] = df['EMPLOYMENT_YEARS'] / (df['AGE_YEARS'] + 1)
        df['REGISTRATION_YEARS'] = -df['DAYS_REGISTRATION'] / 365
        df['ID_PUBLISH_YEARS'] = -df['DAYS_ID_PUBLISH'] / 365
        
        # 统计特征
        df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'] + 1)
        df['INCOME_PER_CHILD'] = df['AMT_INCOME_TOTAL'] / (df['CNT_CHILDREN'] + 1)
        
        # 外部评分组合
        ext_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
        existing_ext_cols = [col for col in ext_cols if col in df.columns]
        
        if len(existing_ext_cols) > 0:
            df['EXT_SOURCE_MEAN'] = df[existing_ext_cols].mean(axis=1)
            df['EXT_SOURCE_MAX'] = df[existing_ext_cols].max(axis=1)
            df['EXT_SOURCE_MIN'] = df[existing_ext_cols].min(axis=1)
            df['EXT_SOURCE_STD'] = df[existing_ext_cols].std(axis=1)
        
        # 文档提交特征
        flag_docs = [col for col in df.columns if col.startswith('FLAG_DOCUMENT_')]
        if flag_docs:
            df['DOC_COUNT'] = df[flag_docs].sum(axis=1)
        
        # 联系方式完整度
        contact_cols = ['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 
                       'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL']
        existing_contact = [col for col in contact_cols if col in df.columns]
        if existing_contact:
            df['CONTACT_COUNT'] = df[existing_contact].sum(axis=1)
        
        # 地址匹配特征
        if 'REG_CITY_NOT_LIVE_CITY' in df.columns and 'REG_CITY_NOT_WORK_CITY' in df.columns:
            df['ADDRESS_MISMATCH'] = (df['REG_CITY_NOT_LIVE_CITY'] + 
                                      df['REG_CITY_NOT_WORK_CITY'] + 
                                      df.get('LIVE_CITY_NOT_WORK_CITY', 0))
        
        print(f"特征工程完成，新增特征后数据形状: {df.shape}")
        return df
    
    def select_features_by_importance(self, df: pd.DataFrame, 
                                       feature_importance: pd.DataFrame,
                                       top_n: int = 50,
                                       target_col: str = 'TARGET'):
        """根据特征重要性选择特征"""
        top_features = feature_importance.head(top_n)['feature'].tolist()
        
        # 确保目标列和ID列保留
        keep_cols = top_features + ['SK_ID_CURR']
        if target_col in df.columns:
            keep_cols.append(target_col)
        
        selected_cols = [col for col in keep_cols if col in df.columns]
        
        print(f"选择TOP {top_n}重要特征")
        return df[selected_cols]
    
    def remove_high_correlation(self, df: pd.DataFrame, 
                                threshold: float = 0.9,
                                target_col: str = 'TARGET'):
        """移除高相关性特征"""
        # 排除ID和目标列
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
    # 测试代码
    fe = FeatureEngineer()
    
    test_df = pd.DataFrame({
        'SK_ID_CURR': [1, 2, 3],
        'TARGET': [0, 0, 1],
        'AMT_INCOME_TOTAL': [100000, 200000, 150000],
        'AMT_CREDIT': [500000, 400000, 300000],
        'AMT_ANNUITY': [30000, 25000, 20000],
        'AMT_GOODS_PRICE': [450000, 380000, 280000],
        'DAYS_BIRTH': [-12000, -15000, -10000],
        'DAYS_EMPLOYED': [-2000, -3000, -1500],
        'DAYS_REGISTRATION': [-5000, -6000, -4000],
        'DAYS_ID_PUBLISH': [-1000, -2000, -800],
        'CNT_FAM_MEMBERS': [2, 3, 1],
        'CNT_CHILDREN': [0, 1, 0],
        'EXT_SOURCE_1': [0.5, 0.6, 0.4],
        'EXT_SOURCE_2': [0.7, 0.8, 0.5],
        'EXT_SOURCE_3': [0.6, 0.7, 0.3]
    })
    
    result = fe.create_features(test_df)
    print("新增特征:")
    new_cols = [col for col in result.columns if col not in test_df.columns]
    print(result[new_cols].head())
