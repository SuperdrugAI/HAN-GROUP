import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.colors as colors
# 加载数据
data = pd.read_csv(r'C:\Users\24582\Desktop\临床血清.xlsl', encoding='gb2312')
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=6)

# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 应用PCA
pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# 训练MLP分类器
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, alpha=1e-4,
                    solver='sgd', verbose=0, tol=1e-4, random_state=1,
                    learning_rate_init=.1)
mlp.fit(X_train_pca, y_train)

# 计算训练集和测试集的准确率
train_accuracy = mlp.score(X_train_pca, y_train) * 100
test_accuracy = mlp.score(X_test_pca, y_test) * 100

# 绘制3D图，增加图形尺寸
fig = plt.figure(dpi=600, figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# 设置PCA坐标范围
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

z_min = X_train_pca[:, 2].min()
z_max = X_train_pca[:, 2].max()

z_values = np.linspace(z_min, z_max + 0.5 , 5)

colors = ['#9945e0','#7effb8', '#fc315c','yellow','green'] 

for z_val in z_values:
    grid = np.c_[xx.ravel(), yy.ravel(), np.full(xx.ravel().shape, z_val)]
    Z = mlp.predict(grid)
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, zdir='z', offset=z_val, alpha=0.22,cmap=plt.cm.rainbow)
labels = ['E.faecalis', 'Healthy', 'E.coli','1','2']
for i, label in zip(range(5), labels):
    ax.scatter(X_test_pca[y_test == i, 0], X_test_pca[y_test == i, 1], X_test_pca[y_test == i, 2], 
               c=colors[i], label=label, alpha=0.9,s=50)

ax.text2D(0.1, 0.95, '3D Classification Diagram of MLP', transform=ax.transAxes, fontsize=24)
ax.view_init(elev=30, azim=45)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.grid(True) 
ax.set_xlabel('Factor 1', fontsize=18)
ax.set_ylabel('Factor 2', fontsize=18)
ax.set_zlabel('Factor 3', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=16)
ax.legend(fontsize=12)
ax.legend(loc='lower left', bbox_to_anchor=(0.78, 0.78),frameon=False,fontsize=16)
plt.show()

y_pred = mlp.predict(X_test_pca)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 8), dpi=600)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
disp.plot(cmap='Blues', values_format='d')
plt.tick_params(axis='both', which='major', labelsize=12)
plt.show()
