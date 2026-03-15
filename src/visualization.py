import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_model_comparison(results_df, save_path=None):
    """绘制模型对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 准确率对比
    axes[0, 0].bar(results_df['模型'], results_df['准确率'], color='skyblue')
    axes[0, 0].set_title('模型准确率对比')
    axes[0, 0].set_ylabel('准确率')
    axes[0, 0].set_ylim(0, 1)

    # AUC对比
    axes[0, 1].bar(results_df['模型'], results_df['AUC'], color='coral')
    axes[0, 1].set_title('模型AUC对比')
    axes[0, 1].set_ylabel('AUC')
    axes[0, 1].set_ylim(0, 1)

    # F1对比
    axes[1, 0].bar(results_df['模型'], results_df['F1分数'], color='lightgreen')
    axes[1, 0].set_title('模型F1分数对比')
    axes[1, 0].set_ylabel('F1分数')
    axes[1, 0].set_ylim(0, 1)

    # 雷达图
    axes[1, 1].remove()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存：{save_path}")

    plt.show()


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['正常', '违约'],
                yticklabels=['正常', '违约'])
    plt.title('混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存：{save_path}")

    plt.show()


def plot_roc_curve(y_true, y_proba, save_path=None):
    """绘制ROC曲线"""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC曲线 (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.title('ROC曲线')
    plt.legend(loc='lower right')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存：{save_path}")

    plt.show()


def plot_feature_importance(feature_importance_df, top_n=15, save_path=None):
    """绘制特征重要性图"""
    plt.figure(figsize=(10, 8))

    top_features = feature_importance_df.head(top_n)

    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('重要性')
    plt.title(f'Top {top_n} 重要特征')
    plt.gca().invert_yaxis()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存：{save_path}")

    plt.show()


def plot_model_results(metrics, save_dir='results'):
    """绘制模型评估结果"""
    import os
    os.makedirs(save_dir, exist_ok=True)

    print("正在生成评估图表...")

    # 这里可以根据需要添加更多可视化
    # 由于需要实际的预测结果，这里只做框架

    print("评估图表生成完成")


if __name__ == "__main__":
    print("可视化模块测试")
    print("请在主程序中调用可视化函数")