# Visualization

# matplotlib

matplotlib is the most commonly used Python plotting library.  
In particular, we use its `pyplot` module which (as convention)
we import as `plt`

```
import matplotlib.pyplot as plt
```

# Making Charts

To use `matplotlib` we accumulate state in the `plt` object
and then display it using `show`:

```
plt.plot([1,3],  # x values
         [2,4])  # y values
plt.title("This is just a sample graph")
plt.xlabel("This is just an x-axis")
plt.ylabel("This is just a y-axis")
plt.show()
```
[TODO: add picture]

as you can see, `plt.plot` plots line[s] connecting the points you give it.

# Aside: An Unzipping Trick

You can use `zip` to zip lists together

```
In [181]: zip(range(5), ['a','b','c','d','e'])
Out[181]: [(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd'), (4, 'e')]
```

You can also use it to unzip lists of pairs:

```
In [182]: zip(*[(0,'a'), (1, 'b'), (2, 'c')])
Out[182]: [(0, 1, 2), ('a', 'b', 'c')]
```

The `*` does argument unpacking, which means the above call is equivalent to

```
In [183]: zip((0,'a'), (1, 'b'), (2, 'c'))
Out[183]: [(0, 1, 2), ('a', 'b', 'c')]
```

This is useful when we have a list of points and need to split them into
a list of x-coordinates and a list of y-coordinates, which is how `matplotlib`
wants them.

# Bar Charts

```
plt.bar(range(10), np.random.rand(10))
plt.title("A random bar chart")
```

[TODO: show pic]

# Scatter Plots

You can supply a lot of (cryptic) options to matplotlib.  For instance,
you can specify `color`, `marker`, and `label` for each series you plot.

```
s = range(10)
plt.scatter(xs, 5 * np.random.rand(10) + xs, color='r', marker='*', label='series1')
plt.scatter(xs, 5 * np.random.rand(10) + xs, color='g', marker='o', label='series2')
plt.title("A scatterplot with two series")
plt.legend(loc=9)
```

[TODO: show pic]

If you specify labels, you can get a legend as shown.
