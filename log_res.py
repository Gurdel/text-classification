import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model

print('Ivan Martsilenko', 'var6 - .237')

a = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5]
b = [1.76, 0.12, 2.72, -0.5, 3.33, - 3.6, 4.33, -8.1, 5.83, -16]
class_1 = [0.357, -0.4, -1.3, -2.16, -3.58]
X = np.array([[a[i], b[i]] for i in range(len(a))])
plt.plot(*zip(*X), marker='o', color='r', ls='') 
plt.show()
Y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

# Fit the data to a logistic regression model.
clf = sklearn.linear_model.LogisticRegression()
clf.fit(X, Y)

# Retrieve the model parameters.
b = clf.intercept_[0]
w1, w2 = clf.coef_.T
# Calculate the intercept and gradient of the decision boundary.
c = -b/w2
m = -w1/w2

# Plot the data and the classification with the decision boundary.
xmin, xmax = 0, 5
ymin, ymax = -18, 7
xd = np.array([xmin, xmax])
yd = m*xd + c
plt.plot(xd, yd, 'k', lw=1, ls='--')
plt.plot(*zip(*X), marker='o', color='r', ls='') 
plt.fill_between(xd, yd, ymin, color='blue', alpha=0.2)
plt.fill_between(xd, yd, ymax, color='orange', alpha=0.2)

plt.scatter(*X[Y==0].T, s=8, alpha=0.5)
plt.scatter(*X[Y==1].T, s=8, alpha=0.5)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.ylabel("x_2")
plt.xlabel("x_1")

plt.show()