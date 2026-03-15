# 贷款风险预测系统

## 项目简介

本项目基于Home Credit Default Risk数据集，使用机器学习算法构建贷款违约风险预测模型。针对信贷风控场景中常见的类别不平衡问题，采用SMOTE过采样、代价敏感学习等技术优化模型召回率，提升对高风险客户的识别能力。

## 数据来源

- **数据集名称**：Home Credit Default Risk
- **数据来源**：Kaggle (https://www.kaggle.com/competitions/home-credit-default-risk)
- **数据内容**：贷款申请者的详细信息及历史贷款记录
- **目标变量**：TARGET（1=违约，0=正常还款）

## 技术栈

| 类别 | 工具/技术 |
|------|-----------|
| 编程语言 | Python 3.12 |
| 机器学习 | Scikit-learn, XGBoost, LightGBM |
| 数据处理 | Pandas, NumPy |
| 类别不平衡处理 | SMOTE, ADASYN, class_weight |
| 可视化 | Matplotlib, Seaborn |
| 开发环境 | PyCharm 2024.3.5, Anaconda |

## 项目结构

```
loan-risk-prediction-advanced/
├── README.md                    # 项目文档
├── requirements.txt             # Python依赖
├── config.py                    # 配置文件
├── data/                        # 数据目录
│   ├── application_train.csv    # 训练数据（Kaggle下载）
│   └── application_test.csv     # 测试数据
├── src/
│   ├── data_preprocessing.py    # 数据预处理
│   ├── feature_engineering.py   # 特征工程
│   ├── model_training.py        # 模型训练
│   ├── model_evaluation.py      # 模型评估
│   └── run_pipeline.py          # 主运行脚本
├── models/                      # 保存的模型
├── results/                     # 结果输出
│   ├── model_evaluation.txt     # 评估报告
│   └── feature_importance.csv   # 特征重要性
└── notebooks/
    └── loan_prediction.ipynb    # 分析笔记本
```

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
conda create -n loan_risk python=3.12
conda activate loan_risk

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

从Kaggle下载数据集：
```bash
# 方式1：手动下载
# 访问 https://www.kaggle.com/competitions/home-credit-default-risk/data
# 下载 application_train.csv 和 application_test.csv 到 data/ 目录

# 方式2：使用Kaggle CLI
kaggle competitions download -c home-credit-default-risk
unzip home-credit-default-risk.zip -d data/
```

### 3. 运行模型

```bash
# 运行完整流程
python src/run_pipeline.py

# 或使用Jupyter Notebook
jupyter notebook notebooks/loan_prediction.ipynb
```

## 核心优化点

### 1. 类别不平衡处理

原始数据中违约客户仅占8%，导致模型偏向预测为正常还款。采用以下方法处理：

```python
# 方法1：SMOTE过采样
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# 方法2：代价敏感学习
model = LGBMClassifier(class_weight='balanced')

# 方法3：调整分类阈值
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
```

### 2. 特征工程

| 特征类型 | 示例特征 | 说明 |
|----------|----------|------|
| 比率特征 | 收入/负债比、收入/信用额度 | 反映还款能力 |
| 时间特征 | 就业年限、年龄 | 稳定性指标 |
| 统计特征 | 历史贷款次数、历史违约次数 | 信用历史 |
| 交叉特征 | 收入×年龄、信用额度/收入 | 组合特征 |

### 3. 模型选择与调优

| 模型 | 优势 | 适用场景 |
|------|------|----------|
| 逻辑回归 | 可解释性强 | 基线模型 |
| 随机森林 | 抗过拟合 | 特征重要性分析 |
| XGBoost | 性能优秀 | 生产环境 |
| LightGBM | 训练速度快 | 大规模数据 |

## 模型评估指标

在信贷风控场景中，需要平衡以下指标：

| 指标 | 说明 | 目标 |
|------|------|------|
| AUC | 整体区分能力 | > 0.75 |
| 召回率 | 违约客户识别率 | > 0.30 |
| 精确率 | 预测违约的准确性 | > 0.20 |
| F1分数 | 精确率与召回率平衡 | > 0.25 |

**注意**：在风控场景中，召回率比精确率更重要，因为漏掉高风险客户的代价更大。

## 模型性能对比

| 模型 | AUC  | 召回率  | 精确率  | F1分数 |
|------|------|------|------|------|
| 逻辑回归 | 0.63 | 0.56 | 0.12 | 0.20 |
| 随机森林 | 0.71 | 0.14 | 0.23 | 0.18 |
| XGBoost | 0.76 | 0.21 | 0.20 | 0.25 |
| LightGBM | 0.76 | 0.09 | 0.40 | 0.15 |

## 特征重要性分析

TOP 10 重要特征：
1. EMERGENCYSTATE_MODE
2. FLAG_PHONE
3. REG_CITY_NOT_WORK_CITY
4. FLAG_OWN_CAR
5. FLAG_DOCUMENT_3
6. REGION_RATING_CLIENT
7. ADDRESS_MISMATCH
8. NAME_EDUCATION_TYPE
9. FLAG_OWN_REALTY
10. CODE_GENDER


## 项目亮点

1. **类别不平衡处理**：SMOTE + 代价敏感学习 + 阈值调整
2. **多模型对比**：逻辑回归、随机森林、XGBoost、LightGBM
3. **完整评估体系**：AUC、召回率、精确率、F1、混淆矩阵
4. **特征工程**：比率特征、交叉特征、统计特征
5. **可解释性**：特征重要性分析、SHAP值分析

## 学习要点

- 类别不平衡问题的处理方法
- 信贷风控场景的模型评估指标选择
- 特征工程在风控模型中的应用
- XGBoost/LightGBM参数调优
- 模型可解释性分析方法

## 注意事项

- 本项目仅用于学习交流
- 数据来源于Kaggle公开数据集
- 模型结果仅供参考，不构成实际信贷决策依据

