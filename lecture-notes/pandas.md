# pandas

# What is pandas?

pandas is a (the) data frame library for Python.  it is your new best friend.

# Conventions

When we use pandas, we will always do

```
from pandas import Series, DataFrame
import pandas as pd
```

# Series

The first fundamental pandas construct is a `Series`.  Think of a Series as a
Python dict all of whose values have the same type (and all of whose keys have the same type too)

```
In [4]: series = Series({'a' : 1, 'c' : 3, 'e' : 7, 'b' : 4})

In [5]: series
Out[5]:
a    1
b    4
c    3
e    7
dtype: int64
```

(Notice how the keys got sorted.)

# Indexes

The keys are referred to as the index:

```
In [6]: series.index
Out[6]: Index([u'a', u'b', u'c', u'e'], dtype='object')
```

And you can reference elements by their index:

```
In [9]: series['a']
Out[9]: 1

In [11]: series[['b','e', 'f']]
Out[11]:
b     4
e     7
f   NaN
dtype: float64
```
Notice that you get a NaN (Not a Number) for index elements that don't exist.

# Boolean Indexes

You can also index in with booleans like we did in numpy arrays:

```
In [12]: series > 3
Out[12]:
a    False
b     True
c    False
e     True
dtype: bool

In [13]: series[series > 3]
Out[13]:
b    4
e    7
dtype: int64
```

# Arithmetic

And you can do the same kinds of arithmetic that we did with numpy arrays as well

```
In [15]: series + 1
Out[15]:
a    2
b    5
c    4
e    8
dtype: int64

In [16]: np.exp(series)
Out[16]:
a       2.718282
b      54.598150
c      20.085537
e    1096.633158
dtype: float64
```

# Working with multiple series

When you work with multiple series, the index gets respected:

```
In [17]: series2 = Series([10, 20, 30, 40, 50], index=['a','b','c','d','e'])

In [18]: series + series2
Out[18]:
a    11
b    24
c    33
d   NaN
e    57
dtype: float64
```

Notice that the a value got added to the a value, and so on. The original series had no d value, so we got a NaN for that term.

# Dealing with missing values

```
In [19]: series3 = series + series2

In [20]: series3.isnull()
Out[20]:
a    False
b    False
c    False
d     True
e    False
dtype: bool
```

We can use that to get rid of the NaN values:

```
In [21]: series3[~series3.isnull()]
Out[21]:
a    11
b    24
c    33
e    57
dtype: float64
```

or more simply we can just

```
In [22]: series3.dropna()
Out[22]:
a    11
b    24
c    33
e    57
dtype: float64
```

# Dealing with missing values differently

You also might want to do something other than drop the `NaN` values.

```
In [23]: series2.add(series, fill_value=0)
Out[23]:
a    11
b    24
c    33
d    40
e    57
dtype: float64
```

You could even do (bad idea!)

```
In [24]: series2.add(series, fill_value=10)
Out[24]:
a    11
b    24
c    33
d    50
e    57
dtype: float64
```

# DataFrames

Now that you understand Series, you can think of a DataFrame (conceptually)
as a dict whose keys are column names and whose values are series.

```
In [25]: df = DataFrame({ 'x' : series, 'y' : series2 })

In [26]: df
Out[26]:
    x   y
a   1  10
b   4  20
c   3  30
d NaN  40
e   7  50
```

# Referencing columns

You can reference each series by its column name

```
In [27]: df['y']
Out[27]:
a    10
b    20
c    30
d    40
e    50
Name: y, dtype: int64
```

And get specific rows with another indexing:

```
In [30]: df['y'][['b','d']]
Out[30]:
b    20
d    40
Name: y, dtype: int64
```

# Referencing rows

You can reference rows using .loc

```
In [28]: df.loc[['b','d']]
Out[28]:
    x   y
b   4  20
d NaN  40
```

# Adding columns

Adding columns is as easy as assigning them a name:

```
In [31]: df['z'] = np.sqrt(df['x'])

In [32]: df
Out[32]:
    x   y         z
a   1  10  1.000000
b   4  20  2.000000
c   3  30  1.732051
d NaN  40       NaN
e   7  50  2.645751
```

# Loading Data from a file

We use different functions depending on the file format

```
df = pd.read_csv('path/to/csv/file')
df = pd.read_table('path/to/tab/delimited/file')
```

# Manipulating Data

Let's start with some fake sales data:


```
df = DataFrame({
  'salesperson' : Series(['Adam', 'Bill', 'Carla', 'Denise',
                          'Adam', 'Bill', 'Carla', 'Denise']),
  'item' : Series(['X', 'X', 'X', 'X', 'Y', 'Y', 'Y', 'Z']),
  'quantity' : Series([10, 15, 30, 7, 12, 19, 15, 18]),
  'revenue' : Series([50, 90, 140, 150, 13, 20, 16, 1000])})
```

which looks as you might expect:

```
In [35]: df
Out[35]:
item  quantity  revenue salesperson
0    X        10       50        Adam
1    X        15       90        Bill
2    X        30      140       Carla
3    X         7      150      Denise
4    Y        12       13        Adam
5    Y        19       20        Bill
6    Y        15       16       Carla
7    Z        18     1000      Denise
```

# Describing things

If we had a lot of rows, we might want to see just the first few:

```
In [49]: df.head()
Out[49]:
  item  quantity  revenue salesperson
0    X        10       50        Adam
1    X        15       90        Bill
2    X        30      140       Carla
3    X         7      150      Denise
4    Y        12       13        Adam
```

We can get a summary of the numeric fields:


```
In [47]: df.describe()
Out[47]:
       quantity      revenue
count    8.0000     8.000000
mean    15.7500   184.875000
std      7.0051   333.829485
min      7.0000    13.000000
25%     11.5000    19.000000
50%     15.0000    70.000000
75%     18.2500   142.500000
max     30.0000  1000.000000
```



# Number of distinct items:

```
In [42]: df.item  # alternative to bracket notation
Out[42]:
0    X
1    X
2    X
3    X
4    Y
5    Y
6    Y
7    Z
Name: item, dtype: object

In [44]: df.item.unique()
Out[44]: array(['X', 'Y', 'Z'], dtype=object)

In [45]: df.item.nunique()
Out[45]: 3
```

# Correlations

Correlation measures how two variables move in tandem about their means.

The `corr` method finds pairwise correlations for all the variables in
the data frame:

```
In [50]: df.corr()
Out[50]:
          quantity   revenue
quantity  1.000000  0.157716
revenue   0.157716  1.000000
```

# Group by

Let's say we want to find the total revenue by salesperson:

```
In [52]: df.groupby('salesperson')['revenue'].sum()
Out[52]:
salesperson
Adam             63
Bill            110
Carla           156
Denise         1150
Name: revenue, dtype: int64
```

or the total quantity by item:

```
In [53]: df.groupby('item')['quantity'].sum()
Out[53]:
item
X       62
Y       46
Z       18
Name: quantity, dtype: int64
```

# Merging

We can also merge multiple data frames.  This is similar to a SQL join:

```
regions = DataFrame({
  'person' : Series(['Adam', 'Bill', 'Carla', 'Denise', 'Edgar']),
  'region' : Series(['West', 'East', 'West', 'East', 'East'])
  })
```

Imagine now we want to find sales by region, we need to merge the `regions`
data frame with the sales data frame.  We can do that with pd.merge:

```
In [57]: sales = pd.merge(df, regions, left_on='salesperson', right_on='person')

In [58]: sales
Out[58]:
  item  quantity  revenue salesperson  person region
0    X        10       50        Adam    Adam   West
1    Y        12       13        Adam    Adam   West
2    X        15       90        Bill    Bill   East
3    Y        19       20        Bill    Bill   East
4    X        30      140       Carla   Carla   West
5    Y        15       16       Carla   Carla   West
6    X         7      150      Denise  Denise   East
7    Z        18     1000      Denise  Denise   East
```

Now we have person in there twice, so let's get rid of it

```
In [59]: del sales['person']

In [60]: sales
Out[60]:
  item  quantity  revenue salesperson region
0    X        10       50        Adam   West
1    Y        12       13        Adam   West
2    X        15       90        Bill   East
3    Y        19       20        Bill   East
4    X        30      140       Carla   West
5    Y        15       16       Carla   West
6    X         7      150      Denise   East
7    Z        18     1000      Denise   East
```

# Aside: Outer join

By default, `merge` does an inner join, which only keeps rows that match
in both tables.  This means the 'Edgar' row got thrown away.  If we don't want
that behavior we can do an outer join.

```
In [61]: pd.merge(df, regions, left_on='salesperson', right_on='person', how='outer')
Out[61]:
  item  quantity  revenue salesperson  person region
0    X        10       50        Adam    Adam   West
1    Y        12       13        Adam    Adam   West
2    X        15       90        Bill    Bill   East
3    Y        19       20        Bill    Bill   East
4    X        30      140       Carla   Carla   West
5    Y        15       16       Carla   Carla   West
6    X         7      150      Denise  Denise   East
7    Z        18     1000      Denise  Denise   East
8  NaN       NaN      NaN         NaN   Edgar   East
```

Notice that we get a bunch of NaN values for the row with no matches.

# Working with the merged data

Revenue by region:

```
In [62]: sales.groupby('region')['revenue'].sum()
Out[62]:
region
East      1260
West       219
Name: revenue, dtype: int64
```

What if we want revenue by region by item?

One way is to group by multiple columns

```
In [63]: sales.groupby(['region', 'item'])['revenue'].sum()
Out[63]:
region  item
East    X        240
        Y         20
        Z       1000
West    X        190
        Y         29
Name: revenue, dtype: int64
```

# pivot table

Another possiblity is to use `pivot_table`, which will put one field in the index
and one field in the columns:

```
In [68]: pt = sales.pivot_table(index='region', columns='item', values='revenue', aggfunc=np.sum, fill_value=0)

In [69]: pt
Out[69]:
item      X   Y     Z
region
East    240  20  1000
West    190  29     0

In [70]: pt['X']
Out[70]:
region
East      240
West      190
Name: X, dtype: int64

In [71]: pt.loc['East']
Out[71]:
item
X        240
Y         20
Z       1000
Name: East, dtype: int64
```

# And more

There's so much more you can do with pandas, check out the extensive documentation.
