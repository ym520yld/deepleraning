# sklearn自带一些常用数据集，加州的房价预测
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import numpy as np
from sklearn.preprocessing import StandardScaler

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
# print(data.shape)
# print(target.shape)

# 把数据按照7:3 分割成训练集和测试集
# 用训练集求W
# 用测试集看下w是否准确
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# 数据进行归一化
sta = StandardScaler()
# 求出归一化参数
sta.fit(X_train)

X_train = sta.transform(X_train)
X_test = sta.transform(X_test)

# 构建一个线性回归模型,此时的w和w0是随机生成的
model = linear_model.LinearRegression()
# 使用梯度下降法求解w和w0
model.fit(X_train, y_train)
# 打印w0
print(model.intercept_)
# 打印w
print(model.coef_)
print('测试集残差平方和:{:.2f}'.format(np.mean((model.predict(X_test) - y_test) ** 2)))
print('训练集残差平方和:{:.2f}'.format(np.mean((model.predict(X_train) - y_train) ** 2)))
