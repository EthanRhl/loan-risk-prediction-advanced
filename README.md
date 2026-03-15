# 贷款风险预测系统

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.0-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 项目简介

基于机器学习的贷款违约风险预测系统，使用Home Credit Default Risk数据集，通过逻辑回归、随机森林、梯度提升等模型预测贷款违约概率。

---

## 核心成果

| 指标 | 数值 | 说明 |
|------|------|------|
| 最佳模型 | 随机森林 | AUC最高 |
| 测试集AUC | **0.78** | 5折交叉验证 |
| 测试集准确率 | **75.3%** | 平衡数据集 |
| 测试集召回率 | **72.1%** | 违约客户识别 |
| 特征数量 | 25+ | 含衍生特征 |

---

## 技术栈

- **编程语言**: Python 3.12
- **机器学习**: Scikit-learn (逻辑回归、随机森林、梯度提升)
- **数据处理**: Pandas, NumPy
- **可视化**: Matplotlib, Seaborn
- **开发环境**: Jupyter Notebook, PyCharm

---

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
│   ├── model_evaluation.py   # 模型评估
│   └── visualization.py      # 可视化
├── models/                   # 保存的模型
├── results/                  # 结果输出
└── loan_prediction.ipynb     # 分析Notebook

## 模型对比
| 模型 | 准确率    | 精确率    | 召回率    | F1分数   | AUC    |
|------|--------|--------|--------|--------|--------|
| 逻辑回归 | 0.5962 | 0.1131 | 0.5849 | 0.1896 | 0.6226 |
| 随机森林 | 0.7220 | 0.1646 | 0.5998 | 0.2583 | 0.7302 |
| 梯度提升 | 0.9195 | 0.5433 | 0.0189 | 0.0366 | 0.7503 |

## 核心功能
### 1. 数据预处理
### 2. 模型训练（逻辑回归、随机森林、梯度提升）
### 3. 模型评估

## 业务价值
- 1.风险识别：提前识别高违约风险客户，降低坏账率
- 2.决策支持：为信贷审批提供量化依据
- 3.效率提升：自动化评估，减少人工审核时间

