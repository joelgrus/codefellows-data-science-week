# scikit-learn

# What is scikit-learn?

It's the most popular machine learning library for Python.

# What is machine learning?

Good question, let's crib from Wikipedia:

> Machine learning explores the construction and study of algorithms that can
> learn from and make predictions on data. Such algorithms operate by
> building a model from example inputs in order to make data-driven
> predictions or decisions, rather than
> following strictly static program instructions.

# Can I learn machine learning in a day?

Most certainly not.

# The fundamental idea

There may be some sort of structure in our data
that's not apparent to the naked eye.  By making some assumptions
about the data, we can "learn" the structure and use it to make predictions
about new data.

# Supervised v Unsupervised

In *supervised* learning, our data has some set of labels ("correct answers")
that we're trying to learn how to predict.  In *unsupervised* learning
there is no single "correct" answer, instead we might be trying to (say)
identify clusters.

# The fundamental problem

Typically we have a sample of data from some larger population.
We would like to learn the population structure, but not any
idiosyncracies of our sample.

# The solution

Typically we split the data into a training set and a test set.
We train a model on the training set, and we check it on the test set.
If it performs well on the test set, then we assume it's captured
the structure in the underlying population.  If it performs poorly on the test set,
then we assume we captured idioscyncracies in the training data.

# Test train split

```
import numpy as np
from sklearn.cross_validation import train_test_split
a = np.arange(10).reshape((5, 2))
b = range(5)
a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=0.33, random_state=42)

In [73]: a
Out[73]:
array([[0, 1],
       [2, 3],
       [4, 5],
       [6, 7],
       [8, 9]])

In [74]: b
Out[74]: [0, 1, 2, 3, 4]

In [75]: a_train
Out[75]:
array([[4, 5],
       [0, 1],
       [6, 7]])

In [76]: b_train
Out[76]: [2, 0, 3]
```

# The scikit-learn workflow

```
# create the model
model = SomeModel(parameter1=1, parameter2=2)

# train the model using inputs and known outputs
model.train(x_train, y_train)

# create predictions from new data
predictions = model.predict(x_test)

# see how we did
plt.scatter(predictions, y_test)
```

# Measuring Performance

How do we know how good our model is?  Looking at the scatterplot is good,
but for various types of models we can look at different measures:

# Mean Square Error

Each test prediction has an *error* compared to its actual value.
If we just average the errors, positive and negative errors might cancel
each other out.  So sometimes we look at the *mean square error*.

```
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, predictions)
```

It's not clear what this number means on its own, but you can use it to compare
different models.

# R-squared

The R^2 value measures how much of the variation in the y-value is captured
by the model.  It is defined as

1 - SS_residual / SS_total

where SS_residual is the *total* squared error of our predictions,
and SS_total is the total squared variation of the actual values (relative to their mean).

If this number is 1, SS_residual must be 0, which means we predicted perfectly.
If this number is 0, SS_residual equals SS_total, which is just how you'd do
if you predicted the mean of the test data.  If this number is negative, it means
that you're doing worse than just predicting the mean of the test data.

```
from sklearn.metrics import r2_score
r2 = r2_score(y_test, predictions)
```

# Confusion Matrix

For classification problems, the confusion matrix shows how many inputs
in each class were classified into each class. Imagine the simple case where
we are classifying things as either `True` or `False`.  The confusion matrix
consists of the counts

```
[[TP, FN],
 [FP, TN]]
```

where TP is the true positives (predict True, actual True), FN is the false
negatives (predict False, actual True), FP is the false positives (predict True,
actual False), and TN is the true negatives (predict False, actual False).

```
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predicted)
```

# Precision and Recall

Still in the case of classification, we often speak about precision and recall.
Precision is the fraction of positive predictions that are *correct*; recall is
the fraction of positive examples that we predicted positive. That is,

Precision = TP / (TP + FP)

Recall = TP / (TP + FN)

```
from sklearn.metrics import precision_score, recall_score
precision = precision_score(y_test, predicted)
recall = recall_score(y_test, predicted)
```

# A Tour Of Some Common Machine Learning Models

# k Means Clustering

Given a dataset and a number *k* of clusters, group the points into *k*
clusters that are as "tight" as possible.

```
from sklearn.cluster import KMeans
data = np.random.rand(100,2)
model = KMeans(n_clusters=5)
model.fit(data)
```

```
In [95]: model.cluster_centers_
Out[95]:
array([[ 0.1887559 ,  0.25786778],
       [ 0.21884906,  0.79470713],
       [ 0.75571261,  0.11922741],
       [ 0.62498061,  0.50819928],
       [ 0.73698435,  0.8420976 ]])


In [96]: model.predict(data)
Out[96]:
array([1, 1, 0, 2, 2, 0, 0, 3, 3, 2, 4, 0, 1, 3, 4, 1, 3, 0, 1, 2, 2, 3, 0,
      2, 1, 0, 4, 0, 1, 2, 0, 3, 3, 4, 0, 2, 4, 3, 2, 1, 4, 3, 4, 1, 1, 4,
      1, 0, 1, 0, 2, 3, 4, 3, 3, 4, 3, 0, 3, 4, 2, 4, 1, 2, 3, 4, 2, 1, 0,
      3, 3, 2, 3, 3, 2, 2, 0, 4, 4, 0, 4, 4, 3, 1, 1, 3, 3, 1, 0, 0, 4, 2,
      2, 1, 4, 1, 2, 2, 1, 1], dtype=int32)
```


```
for x, y in model.cluster_centers_:
  plt.plot(x, y, color='black', marker='*')
output = zip(data, model.predict(data))
for cluster, color in zip([0,1,2,3,4], ['r','g','b','y','m']):
  points = [p for p, c in output if c == cluster]
  x, y = zip(*points)
  plt.scatter(x, y, color=color)
plt.show()
```

[add picture]

# k nearest neighbors

Given a labeled dataset, predict a value for a new point by taking the majority
label from the *k* nearest neighbors:

```
>>> X = [[0], [1], [2], [3]]
>>> y = [0, 0, 1, 1]
>>> from sklearn.neighbors import KNeighborsClassifier
>>> neigh = KNeighborsClassifier(n_neighbors=3)
>>> neigh.fit(X, y)
KNeighborsClassifier(...)
>>> print(neigh.predict([[1.1]]))
[0]
>>> print(neigh.predict_proba([[0.9]]))
[[ 0.66666667  0.33333333]]
```

Good: doesn't assume any structure in the data.
Bad: doesn't assume any structures in the data.

# Linear Regression

Given some input variables `X` and an output variable `y`, assume a linear relationship

y = x_1 * beta_1 + ... + x_n * beta_n

```
from sklearn.linear_model import LinearRegression
x = np.random.random([100, 3])

# y = 5 * x_1 + 10 * x_2 - 2 * x_3 + 5 + noise
y = np.dot(x, [5,10,-2]) + 10 + 5 * np.random.random(100)

model = LinearRegression()
model.fit(x, y)

In [127]: model.coef_
Out[127]: array([ 5.18845306,  9.67420044, -1.14505447])

predictions = model.predict(x)

plt.scatter(predictions, y)
plt.xlabel("predicted value")
plt.ylabel("actual value")
plt.show()
```

# Regularized Regression

Sometimes you have a lot of variables, and things get lost in the noise:

```
x = 0.1 * np.random.random([100,100])
beta = np.zeros(100)
beta[0] = 5
beta[1] = 10
beta[2] = -2
y = np.dot(x, beta)

model = LinearRegression()
model.fit(x, y)

In [173]: np.round(model.coef_,1)
Out[173]:
array([ 5.6,  8.6, -0.3, -1.4, -1.8, -1.6, -1. ,  3.9, -1.9,  1.3,  1.6,
        2.3, -1.8, -1.3, -1.9,  0.8,  3.3, -0.1, -0.4,  0.2,  0.4, -0.2,
       -0.8,  1. ,  1.1, -1.2,  2.4,  2.6, -5.5,  1.2, -0.2, -0. ,  2.8,
        1.7, -2.3, -1.2, -1.1,  0.6, -2.2, -1.9, -0.9, -2.2, -1.1,  2.2,
       -0.9, -4. , -1.2, -0.5,  2.2,  2.1,  1.1,  0.3, -6.1,  3.5, -0.1,
        0.4, -0.9, -0.7, -2.1, -0.9,  1.4, -3. ,  2. , -2.4,  1.2,  0.2,
       -0.6, -0.7,  1.9,  2.1,  1.8,  0.2,  0.5,  2.1,  1.8,  2. ,  2. ,
       -1.4, -3.4, -1.8,  0.8,  0.3,  1.5,  1.7, -4.4, -2.1, -0.1,  2.5,
       -4.7, -1.4, -2.4,  2.2, -2.4, -1.1,  0.6,  1.5,  0.6, -1.3,  0.2,
        0.5])
```

*regularization* applies a penalty to larger coefficients.  "Ridge" penalizes
according to the square of the coefficients:

```
from sklearn.linear_model import Ridge
model = Ridge(alpha=1)
model.fit(x,y)
np.round(model.coef_, 1)
Out[176]:
array([ 0.4,  0.7, -0. , -0.1,  0. , -0. , -0. ,  0.1, -0. ,  0.2,  0. ,
        0. , -0. , -0.1,  0. ,  0.1, -0.1,  0. ,  0. ,  0. ,  0.1, -0.1,
       -0.1,  0.1,  0. , -0.1, -0.1,  0. ,  0. , -0.1,  0. ,  0. ,  0.1,
        0.2,  0.2, -0.1,  0.1, -0. ,  0.1, -0.1,  0. ,  0. , -0.1, -0. ,
        0.1, -0. , -0.1,  0. , -0. ,  0.1,  0. , -0.2, -0.2,  0.1, -0.1,
       -0.1, -0.1, -0.1,  0. ,  0. ,  0. , -0.1, -0.1,  0. ,  0. , -0.1,
        0. ,  0. , -0. , -0.1, -0.1, -0.1,  0. ,  0. ,  0.1,  0.1, -0. ,
        0.1,  0.1,  0.1,  0. , -0. , -0. ,  0. ,  0. , -0. ,  0.1,  0. ,
        0.1,  0. ,  0. , -0.1, -0.1, -0.1, -0.1,  0.1,  0. , -0. ,  0. ,
        0.1])
```

Whereas "lasso" penalizes according to the absolute value of the coefficients.
It forces more of them to be zero:

```
from sklearn.linear_model import Lasso
model = Lasso(alpha=0.001)
model.fit(x,y)
np.round(model.coef_, 1)
Out[180]:
array([ 3.8,  8.7, -0.4, -0. ,  0. , -0. ,  0. ,  0. , -0. ,  0. , -0. ,
       -0. ,  0. , -0. ,  0. ,  0. , -0. , -0. ,  0. , -0. ,  0. , -0. ,
       -0. ,  0. ,  0. , -0. , -0. ,  0. , -0. ,  0. , -0. , -0. ,  0. ,
        0. ,  0. , -0. ,  0. , -0. ,  0. , -0. ,  0. ,  0. , -0. ,  0. ,
        0. , -0. ,  0. ,  0. , -0. ,  0. , -0. , -0. , -0. ,  0. , -0. ,
       -0. , -0. , -0. ,  0. ,  0. ,  0. , -0. , -0. ,  0. , -0. , -0. ,
        0. , -0. ,  0. , -0. , -0. , -0. , -0. , -0. ,  0. ,  0. , -0. ,
        0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. , -0. , -0. ,  0. ,  0. ,
        0. , -0. ,  0. , -0. , -0. , -0. , -0. ,  0. ,  0. , -0. ,  0. ,
        0. ])
```

choosing the right `alpha` can be tricky, you usually have to try different values.

# Logistic Regression

Instead of predicting the value of an output, we want to predict a 0 or 1
to do some sort of classfication.  We still compute a linear function of the inputs
but we transform it to lie between 0 and 1.
Then we can interpret it as a probability:

```
from sklearn.linear_model import LogisticRegression
x = np.random.random([100, 3])
# predict 1 if bigger than 20, 0 otherwise
y = np.dot(x, [5,10,-2]) + 10 + 5 * np.random.random(100) > 20

model = LogisticRegression()
model.fit(x, y)

model.coef_
# not very meaningful
Out[138]: array([[ 0.78373639,  3.56743894, -1.08936704]])

# an array of (false_probability, true_probability)
predictions = model.predict_proba(x)

plt.scatter(predictions[:,1], y)
plt.xlabel("predicted probability")
plt.ylabel("actual value")
plt.show()
```

# Decision Tree

A decision tree splits the dataset on its various attributes and uses that
to create a prediction.

[TODO: pic of decision tree]

It can be a good choice when you have a lot of qualitative attributes.

```
play_tennis = [
 {'id':'d1','outlook':'sunny','temp':'hot','humidity':'high','wind':'weak','play':'-'},
 {'id':'d2','outlook':'sunny','temp':'hot','humidity':'high','wind':'strong','play':'-'},
 {'id':'d3','outlook':'overcast','temp':'hot','humidity':'high','wind':'weak','play':'+'},
 {'id':'d4','outlook':'rain','temp':'mild','humidity':'high','wind':'weak','play':'+'},
 {'id':'d5','outlook':'rain','temp':'cool','humidity':'normal','wind':'weak','play':'+'},
 {'id':'d6','outlook':'rain','temp':'cool','humidity':'normal','wind':'strong','play':'-'},
 {'id':'d7','outlook':'overcast','temp':'cool','humidity':'normal','wind':'strong','play':'+'},
 {'id':'d8','outlook':'sunny','temp':'mild','humidity':'high','wind':'weak','play':'-'},
 {'id':'d9','outlook':'sunny','temp':'cool','humidity':'normal','wind':'weak','play':'+'},
 {'id':'d10','outlook':'rain','temp':'mild','humidity':'normal','wind':'weak','play':'+'},
 {'id':'d11','outlook':'sunny','temp':'mild','humidity':'normal','wind':'strong','play':'+'},
 {'id':'d12','outlook':'overcast','temp':'mild','humidity':'high','wind':'strong','play':'+'},
 {'id':'d13','outlook':'overcast','temp':'hot','humidity':'normal','wind':'weak','play':'+'},
 {'id':'d14','outlook':'rain','temp':'mild','humidity':'high','wind':'strong','play':'-'}
]

df = DataFrame.from_dict(play_tennis)
humidity   id   outlook play  temp    wind
0      high   d1     sunny    -   hot    weak
1      high   d2     sunny    -   hot  strong
2      high   d3  overcast    +   hot    weak
3      high   d4      rain    +  mild    weak
4    normal   d5      rain    +  cool    weak
5    normal   d6      rain    -  cool  strong
6    normal   d7  overcast    +  cool  strong
7      high   d8     sunny    -  mild    weak
8    normal   d9     sunny    +  cool    weak
9    normal  d10      rain    +  mild    weak
10   normal  d11     sunny    +  mild  strong
11     high  d12  overcast    +  mild  strong
12   normal  d13  overcast    +   hot    weak
13     high  d14      rain    -  mild  strong
```

scikit-learn doesn't like categorical variables, only numeric ones.
We can fix this with `get_dummies`:

```
categories = Series(['a', 'b', 'c', 'a'])
0    a
1    b
2    c
3    a
dtype: object

pd.get_dummies(categories)

   a  b  c
0  1  0  0
1  0  1  0
2  0  0  1
3  1  0  0
```
And then apply to only the columns we want:

```
x = pd.get_dummies(df[['humidity', 'outlook', 'temp', 'wind']])
y = df['play'].values
```

and now create the decision tree.  At this point the scikit-learn workflow
should be pretty old hat:

```
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(x, y)
```

And now we can compare the predictions with the actual:

```
In [38]: y == model.predict(x)
Out[38]:
array([ True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True], dtype=bool)
```

[TODO: add pic of tree]

Decision trees have an unfortunate tendency to overfit.  We frequently instead
use `sklearn.ensemble.RandomForestClassifier`, which builds different
decision trees off different subsets of the data and combines them.
