import sklearn
from planar_utils import load_planar_dataset, plot_decision_boundary
import matplotlib.pyplot as plt
import numpy as np

X, Y = load_planar_dataset()
clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T, Y.T)

plot_decision_boundary(lambda x: clf.predict(x), X, Y)  # 绘制决策边界
plt.title("Logistic Regression")  # 图标题
LR_predictions = clf.predict(X.T)  # 预测结果
print("逻辑回归的准确性： %d " % float((np.dot(Y, LR_predictions) +
                               np.dot(1 - Y, 1 - LR_predictions)) / float(Y.size) * 100) +
      "% " + "(正确标记的数据点所占的百分比)")
plt.show()