import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import auc

# 假设这里已经正确加载了数据
H1 = pd.read_csv('C:\\Users\\hp\\Desktop\\ZSM 数据 代码 图\\roc_data H1.csv')
E1 = pd.read_csv('C:\\Users\\hp\\Desktop\\ZSM 数据 代码 图\\roc_data E1.csv')
F1 = pd.read_csv('C:\\Users\\hp\\Desktop\\ZSM 数据 代码 图\\roc_data F1.csv')

# 计算各数据集的AUC值
auc_1 = auc(H1['FPR'], H1['TPR'])* 100
auc_2 = auc(E1['FPR'], E1['TPR'])* 100
auc_3 = auc(F1['FPR'], F1['TPR'])* 100

# 绘制ROC曲线
plt.figure(figsize=(10,8),dpi=600)
plt.plot(H1['FPR'], H1['TPR'], label=f'Healthy Vs E.coli/E.faecalis (AUC = {auc_1:.1f}%)', lw=4,alpha=0.9,color = 'lightgreen')
plt.plot(E1['FPR'], E1['TPR'], label=f'E.coli Vs Healthy/E.faecalis (AUC = {auc_2:.1f}%)', lw=2,alpha=1,color = 'red')
plt.plot(F1['FPR'], F1['TPR'], label=f'E.faecalis Vs Healthy/E.coli (AUC = {auc_3:.1f}%)', lw=7,alpha=0.8,color = 'lightblue')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)
plt.title('MLP Classification ROC Curve ', fontsize=24)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.legend(loc='lower right', frameon=False, fontsize=16)

plt.show()
