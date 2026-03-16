import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DataPreprocessor:
    """数据预处理类"""

    def __init__(self, max_missing_ratio=0.5):
        """
        初始化
        参数:
            max_missing_ratio: 缺失值超过此比例的特征将被删除
        """
        self.max_missing_ratio = max_missing_ratio
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.dropped_cols = []
        self.numeric_cols = []
        self.categorical_cols = []

    def load_data(self, filepath: str):
        """加载数据"""
        print(f"加载数据: {filepath}")
        df = pd.read_csv(filepath)
        print(f"数据形状: {df.shape}")
        return df

    def check_missing(self, df: pd.DataFrame):
        """检查缺失值"""
        missing = df.isnull().sum()
        missing_ratio = missing / len(df) * 100
        missing_df = pd.DataFrame({
            'missing_count': missing,
            'missing_ratio': missing_ratio
        }).sort_values('missing_ratio', ascending=False)

        return missing_df[missing_df['missing_count'] > 0]

    def handle_missing(self, df: pd.DataFrame, is_train=True):
        """处理缺失值"""
        df = df.copy()

        if is_train:
            # 删除缺失值过多的列
            missing_ratio = df.isnull().sum() / len(df)
            self.dropped_cols = missing_ratio[missing_ratio > self.max_missing_ratio].index.tolist()
            print(f"删除缺失值过多的列: {len(self.dropped_cols)} 个")
            df = df.drop(columns=self.dropped_cols, errors='ignore')

            # 记录数值列和类别列
            self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            self.categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        else:
            df = df.drop(columns=self.dropped_cols, errors='ignore')

        # 填充数值列缺失值
        for col in self.numeric_cols:
            if col in df.columns and df[col].isnull().any():
                if is_train:
                    median_val = df[col].median()
                    self.label_encoders[f'{col}_median'] = median_val
                df[col] = df[col].fillna(self.label_encoders.get(f'{col}_median', 0))

        # 填充类别列缺失值
        for col in self.categorical_cols:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna('Unknown')

        return df

    def encode_categorical(self, df: pd.DataFrame, is_train=True):
        """编码类别变量"""
        df = df.copy()

        for col in self.categorical_cols:
            if col not in df.columns:
                continue

            if is_train:
                le = LabelEncoder()
                df[col] = df[col].astype(str)
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
            else:
                le = self.label_encoders.get(col)
                if le:
                    df[col] = df[col].astype(str)
                    # 处理未见过的类别
                    df[col] = df[col].apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )

        return df

    def handle_outliers(self, df: pd.DataFrame):
        """处理异常值（DAYS_EMPLOYED = 365243 为典型异常值）"""
        df = df.copy()

        if 'DAYS_EMPLOYED' in df.columns:
            df['DAYS_EMPLOYED_ANOM'] = (df['DAYS_EMPLOYED'] == 365243).astype(int)
            df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
            df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].fillna(df['DAYS_EMPLOYED'].median())

        return df

    def scale_features(self, df: pd.DataFrame, is_train=True):
        """特征标准化"""
        # 排除 ID 和目标列
        exclude_cols = ['SK_ID_CURR', 'TARGET']
        scale_cols = [col for col in self.numeric_cols if col not in exclude_cols]
        scale_cols = [col for col in scale_cols if col in df.columns]

        if is_train:
            df[scale_cols] = self.scaler.fit_transform(df[scale_cols])
        else:
            df[scale_cols] = self.scaler.transform(df[scale_cols])

        return df

    def preprocess(self, df: pd.DataFrame, is_train=True):
        """完整预处理流程"""
        print(f"预处理 {'训练' if is_train else '测试'} 数据...")

        # 处理异常值
        df = self.handle_outliers(df)

        # 处理缺失值
        df = self.handle_missing(df, is_train)

        # 编码类别变量
        df = self.encode_categorical(df, is_train)

        print(f"预处理完成，数据形状: {df.shape}")
        return df

    def get_feature_names(self, df: pd.DataFrame, target_col='TARGET'):
        """获取特征列名"""
        feature_cols = [col for col in df.columns if col not in ['SK_ID_CURR', target_col]]
        return feature_cols


if __name__ == '__main__':
    # 简单测试：使用 Home Credit 真实数据前 n 行
    import os
    import sys

    # 将项目根目录加入 sys.path，以便导入 config
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)

    from config import DATA_CONFIG

    preprocessor = DataPreprocessor(max_missing_ratio=0.5)

    if not os.path.exists(DATA_CONFIG['train_file']):
        print(f"训练数据未找到: {DATA_CONFIG['train_file']}")
        print("请先下载数据到 data/ 目录")
    else:
        # 为了演示只读前 1000 行
        train_df = pd.read_csv(DATA_CONFIG['train_file'], nrows=1000)
        print("原始数据:")
        print(train_df.head())

        train_processed = preprocessor.preprocess(train_df, is_train=True)
        print("预处理后数据:")
        print(train_processed.head())
