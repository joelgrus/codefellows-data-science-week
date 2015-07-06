# NumPy

# What is NumPy?

NumPy is the basis for most Python data analysis.

# Convention

```
import numpy as np
```

# `ndarray`

Its fundamental tool is the `ndarray`, a multidimensional array.
At the most basic level you can think of them as `list`s.

```
In [2]: data_list = [1, 2, 3, 4, 5]
In [3]: data_array = np.array(data_list)
In [4]: data_array
Out[4]: array([1, 2, 3, 4, 5])
```

# multidimensional arrays

You can also create multidimensional arrays the same way:

```
In [10]: array_2d = np.array([[1, 2, 3], [4, 5, 6]])
In [11]: array_2d
Out[11]: array([[1, 2, 3],
                [4, 5, 6]])
```

# properties

Arrays have various properties:

```
In [12]: data_array.shape
Out[12]: (5,)

In [13]: array_2d.shape
Out[13]: (2, 3)
```

# shape

You can change the shape of an array as well:

```
In [31]: array_2d.reshape([3, 2])
Out[31]: array([[1, 2],
                [3, 4],
                [5, 6]])

In [32]: array_2d.reshape([6,])
Out[32]: array([1, 2, 3, 4, 5, 6])

In [35]: array_2d.reshape([6, 1])
Out[35]: array([[1],
                [2],
                [3],
                [4],
                [5],
                [6]])
```

# vectorization

Arrays have a useful feature called vectorization that allows you to
do operations on them (quickly and) without using for loops or list comprehensions.

If the arrays are the same size, this is easy:

```
In [18]: array1 = np.array([1,2,3])
In [19]: array2 = np.array([4,5,6])
In [20]: array1 + array2
Out[20]: array([5, 7, 9])
In [21]: array1 * array2
Out[21]: array([ 4, 10, 18])
```

# broadcasting

Even if the arrays aren't the same size, NumPy does what's called broadcasting to make the smaller array the same size as the larger. This means that, for instance, to add a single value to an array you can just use:

```
In [14]: array_2d + 1
Out[14]: array([[2, 3, 4],
                [5, 6, 7]])
```

Similarly, you can add a row to every row of a multidimensional array, or a column to every column of a multidimensional array.

```
In [15]: array_2d + np.array([-1, 0, 1])
Out[15]: array([[0, 2, 4],
                [3, 5, 7]])

In [17]: array_2d + np.array([[-1], [1]])
Out[17]: array([[0, 1, 2],
                [5, 6, 7]])
```
# Getting data out of an array

You can also extract elements from an array as you would do from a list:

```
In [21]: data_array[0]
Out[21]: 1

In [22]: array_2d[1][1]
Out[22]: 5
```

And get slices:

```
In [23]: data_array[2:3]
Out[23]: array([3])

In [24]: array_2d[:][:1]
Out[24]: array([[1, 2, 3]])
```

# Applying functions to arrays

You can also apply many functions to arrays:

```
In [25]: np.exp(array_2d)
Out[25]: array([[   2.71828183,    7.3890561 ,   20.08553692],
                [  54.59815003,  148.4131591 ,  403.42879349]])

In [26]: np.sqrt(data_array)
Out[26]: array([ 1.        ,  1.41421356,  1.73205081,  2.        ,  2.23606798])
```

# Boolean operations

What's often more interesting is boolean operations:

```
In [28]: data_array
Out[28]: array([1, 2, 3, 4, 5])
In [29]: data_array != 3
Out[29]: array([ True,  True, False,  True,  True], dtype=bool)
```

Which gives an array of Booleans that you can use to index into the original array
and get back only the part that meets the criteria:

```
In [30]: data_array[data_array != 3]
Out[30]: array([1, 2, 4, 5])
```

# Using NumPy

Generally speaking you won't use numpy directly that much,
but most data libraries use it, so you need to know about it.
