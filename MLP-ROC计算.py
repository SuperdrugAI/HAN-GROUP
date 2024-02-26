# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 18:59:17 2023

@author: hp
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import os

# 读取数据
data = pd.read_csv(r'C:\Users\hp\Desktop\ZSM 数据 代码 图\临床尿液 F1.csv', encoding='gb2312')
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练MLP分类器
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, alpha=0.0000001,
                    solver='sgd', verbose=0, tol=1e-4, random_state=1,
                    learning_rate_init=.1)
mlp.fit(X_train_scaled, y_train)

# 计算测试集的预测概率
y_score = mlp.predict_proba(X_test_scaled)[:, 1]

# 计算ROC曲线
fpr, tpr, thresholds = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

# 绘制ROC曲线
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# 保存ROC数据到桌面
roc_data = pd.DataFrame({'FPR': fpr, 'TPR': tpr, 'Thresholds': thresholds})
desktop_path = os.path.join(os.environ['USERPROFILE'], 'Desktop')
roc_data.to_csv(os.path.join(desktop_path, 'roc_data F1.csv'), index=False)
