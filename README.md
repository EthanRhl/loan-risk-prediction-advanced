# 贷款风险预测系统

## 项目简介

本项目基于Home Credit Default Risk数据集，利用机器学习算法构建贷款违约风险预测模型，旨在提前识别高违约风险客户，降低金融机构的坏账率。

## 数据集
- 数据集名称：Home Credit Default Risk
- 数据来源：Kaggle
- 数据内容：包含贷款申请者的详细信息及历史贷款记录

## 技术栈
- 编程语言：Python 3.12
- 机器学习库：Scikit-learn
- 数据处理：Pandas, NumPy
- 可视化：Matplotlib, Seaborn

## 核心成果
- 最佳模型：梯度提升，AUC最高达0.7503
- 测试集性能：准确率0.9195，召回率0.0189

## 快速开始

### 1. 克隆项目

- git clone https://github.com/EthanRhl/loan-risk-prediction-advanced.git
- cd loan-risk-prediction-advanced

### 2. 安装依赖
- pip install -r requirements.txt

### 3. 准备数据
- 从Kaggle下载 Home Credit Default Risk 数据集，将 application_train.csv 放入 data/ 目录。

### 4. 运行项目
- python run_pipeline.py

### 5. 查看结果
- 模型文件：models/loan_risk_model.pkl
- 评估报告：results/model_evaluation.txt
- 特征重要性：results/feature_importance.csv

## 项目结构
loan-risk-prediction-advanced/
├── README.md                 # 项目文档
├── requirements.txt          # 依赖列表
├── run_pipeline.py           # 主运行脚本
├── data/                     # 数据目录
│   └── application_train.csv
├── src/                      # 源代码
│   ├── data_preprocessing.py # 数据预处理
│   ├── model_training.py     # 模型训练
│   └── visualization.py      # 可视化
├── models/                   # 保存的模型
├── results/                  # 结果输出
└── loan_prediction.ipynb     # 分析Notebook
