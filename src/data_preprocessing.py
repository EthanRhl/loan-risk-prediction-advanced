import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')


class LoanDataPreprocessor:
    """贷款数据预处理器"""

    def __init__(self):
        self.label_encoders = {}
        self.feature_columns = []

    def load_data(self, file_path):
        """加载数据"""
        print(f"正在加载数据：{file_path}")
        df = pd.read_csv(file_path)
        print(f"数据形状：{df.shape}")
        return df

    def handle_missing_values(self, df):
        """处理缺失值"""
        print("处理缺失值...")

        # 数值型特征用中位数填充
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        # 类别型特征用众数填充
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown')

        missing_count = df.isnull().sum().sum()
        print(f"剩余缺失值数量：{missing_count}")

        return df

    def encode_categorical_features(self, df, categorical_cols):
        """编码类别型特征"""
        print("编码类别型特征...")

        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                print(f"  - {col}: {len(le.classes_)}个类别")

        return df

    def create_features(self, df):
        """创建衍生特征"""
        print("创建衍生特征...")

        # 负债收入比
        if 'loan_amnt' in df.columns and 'annual_inc' in df.columns:
            df['debt_to_income'] = df['loan_amnt'] / (df['annual_inc'] + 1)

        # 贷款期限（月）
        if 'term' in df.columns:
            df['term_months'] = df['term'].apply(lambda x: int(x.split()[0]) if isinstance(x, str) else x)

        # 年龄分段
        if 'age' in df.columns:
            df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 55, 100],
                                     labels=['<25', '25-35', '35-45', '45-55', '55+'])

        print(f"新增特征数量：{df.shape[1]}")

        return df

    def prepare_features_and_target(self, df, target_col='loan_status'):
        """准备特征和目标变量"""
        print("准备特征和目标变量...")

        # 排除的列
        exclude_cols = [target_col, 'id', 'member_id', 'emp_title', 'title',
                        'zip_code', 'addr_state', 'earliest_cr_line']

        # 获取特征列
        self.feature_columns = [col for col in df.columns
                                if col not in exclude_cols
                                and df[col].dtype in [np.int64, np.float64, np.int32, np.float32]]

        print(f"特征数量：{len(self.feature_columns)}")

        X = df[self.feature_columns]
        y = df[target_col] if target_col in df.columns else None

        return X, y

    def split_data(self, X, y, test_size=0.2, random_state=42):
        """划分训练集和测试集"""
        print(f"划分数据集 (测试集比例：{test_size})...")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if y is not None else None
        )

        print(f"  训练集：{X_train.shape[0]} 样本")
        print(f"  测试集：{X_test.shape[0]} 样本")

        return X_train, X_test, y_train, y_test

    def fit_transform(self, df, target_col='loan_status'):
        """完整预处理流程"""
        print("=" * 50)
        print("开始数据预处理")
        print("=" * 50)

        # 处理缺失值
        df = self.handle_missing_values(df)

        # 创建特征
        df = self.create_features(df)

        # 准备特征和目标
        X, y = self.prepare_features_and_target(df, target_col)

        # 划分数据集
        X_train, X_test, y_train, y_test = self.split_data(X, y)

        print("=" * 50)
        print("数据预处理完成")
        print("=" * 50)

        return X_train, X_test, y_train, y_test, df


if __name__ == "__main__":
    # 测试代码
    print("数据预处理模块测试")
    preprocessor = LoanDataPreprocessor()
    print("预处理器初始化成功")