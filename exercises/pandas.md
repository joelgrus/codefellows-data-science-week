# pandas

In these exercises we'll be working with a dataset from the Washington DC
bikeshare program that I stole from Erin Shellman.  It consists of three files:

* stations.tsv : information about each bike share station
* usage_2012.tsv : one row for each bike share rental in 2012
* daily_weather.tsv : one row of Washington DC weather for each day in 2012

1. Load the data (using `pd.read_table`) and start exploring it.

2. Compute the average temperature by season ('season_desc').
(The temperatures are numbers between 0 and 1, but don't worry about that.
Let's say that's the Shellman temperature scale.)

I get

```
season_desc
Fall           0.711445
Spring         0.321700
Summer         0.554557
Winter         0.419368
```

Which clearly looks wrong.  Figure out what's wrong with the original data
and fix it.

3. Various of the columns represent dates or datetimes, but out of the box
`pd.read_table` won't treat them correctly.  This makes it hard to (for example)
compute the number of rentals by month.  Fix the dates and compute the number
of rentals by month.

4.  
