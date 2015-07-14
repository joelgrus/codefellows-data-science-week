# IPython

# What is IPython

A better Python shell with lots of magical features.

# Lots of features

We'll only talk about a few of our favorites here.

# %paste magic

It's hard to paste code into the Python shell:

```
for x in range(10):
  print x

  print x + 1
```

doesn't work:

```
>>> for x in range(10):
...   print x
...
0
1
2
3
4
5
6
7
8
9
>>>   print x + 1
  File "<stdin>", line 1
    print x + 1
    ^
IndentationError: unexpected indent
```

Whereas in IPython just type `%paste` (or `paste`) and it just works.

# tab completion

Let's say you want to know what methods an object has

```
x = { 1 : 2, 3 : 4 }
```

just type

```
x.
```

and hit tab:

```
In [5]: x.
x.clear       x.has_key     x.itervalues  x.setdefault  x.viewkeys
x.copy        x.items       x.keys        x.update      x.viewvalues
x.fromkeys    x.iteritems   x.pop         x.values
x.get         x.iterkeys    x.popitem     x.viewitems
```

# interactive help

OK, so you want to know what `x.values` does.  Type

```
In [5]: x.values?
Type:        builtin_function_or_method
String form: <built-in method values of dict object at 0x7fd383aa9280>
Docstring:   D.values() -> list of D's values
```

# IPython notebooks

You can also use IPython to produce interactive notebooks
that let you mix code and text:

```
$ ipython notebook
[I 10:55:41.950 NotebookApp] Using existing profile dir: u'/home/joel/.ipython/profile_default'
[I 10:55:41.955 NotebookApp] Using MathJax from CDN: https://cdn.mathjax.org/mathjax/latest/MathJax.js
[I 10:55:41.995 NotebookApp] Serving notebooks from local directory: /home/joel/src/codefellows-data-science-week/data
[I 10:55:41.995 NotebookApp] 0 active kernels
[I 10:55:41.995 NotebookApp] The IPython Notebook is running at: http://localhost:8888/
```

If you go to that URL, you'll see a list of notebooks, and options to create new ones.

[demo]

# See Also

IPython can do lots more, one could probably teach a week course just on
IPython.  [Read the documentation](http://ipython.org/documentation.html).
