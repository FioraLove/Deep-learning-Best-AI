"""
姓名	工资(元)	房屋面积(平方)	可贷款金额(元)
张三	6000	58	30000
李四	9000	77	55010
王五	11000	89	73542
赵六	15000	54	63201
孙七  12000 60    ？？
"""

"""
Sklearn的API如下：
class sklearn.linear_model.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=None)
fit_intercept : 布尔值，是否使用偏置项，默认是 True。
normalize : 布尔值，是否启用归一化，默认是 False。当 fit_intercept 被置为 False 的时候，这个参数会被忽略。当该参数为 True 时，数据会被归一化处理。
copy_X : 布尔值，默认是 True，如果为 True，x 参数会被拷贝不会影响原来的值，否则会被复写。
n_jobs：数值或者布尔，如果设置了，则多核并行处理。

属性如下：
coef_：x 的权重系数大小
intercept_：偏置项大小
"""
from sklearn.linear_model import LinearRegression

# 生成x矩阵
x_data = [
    [6000, 58],
    [9000, 77],
    [11000, 89],
    [15000, 54]
]
# 输出y矩阵
y_data = [
    30000, 55010, 73542, 63201
]
#  线性回归
model = LinearRegression()
# 线性回归建模
model.fit(x_data, y_data)
print('方程为：y={w1}x1+{w2}x2+{b}'.format(w1=round(model.coef_[0], 2),
                                       w2=round(model.coef_[1], 2),
                                       b=model.intercept_))
x_test = [[12000, 60]]
print('贷款金额为：', model.predict(x_test)[0])
