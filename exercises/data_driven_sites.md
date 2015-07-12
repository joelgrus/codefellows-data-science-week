# data driven sites

Our goal here is to build a website that allows us to choose a station
and visualize the time series of daily rental counts for that station.

In particular, the website should have three components:

1. a text input
2. a list of stations (filtered by the value of the text input)
3. a chart

That is, if you type 'x' into the box, you should see the stations

* Fairfax Dr & Wilson Blvd
* Fairfax Village
* N Randolph St & Fairfax Dr

and if you click on "Fairfax Dr & Wilson Blvd" the graph should show the
daily rental counts for that stop.

This is a somewhat open-ended problem.  Here are some thoughts:

1. You'll need a Pyramid server with two different routes.  One should take
a text snippet and return the list of stations that match that snippet
(`re.search` might be your friend).  The other should take a station name
and return the time series of daily rental counts for that station.

2. To do that, your Pyramid server will probably need to use some `pandas`.
So that the website will be snappy, you probably want to precompute
the time series for each station, and also the list of stations.

3. Your JavaScript will need two types of event listeners.  
One will need to watch the text input and (make an ajax call to)
refresh the list of stations whenever the input changes ('keyup').
The other will watch the station names and (make an ajax call to)
refresh the graph whenever someone clicks on a station name.

4. The station names have lots of &s and /s in them.
These are tricky to pass in as URL parameters.  Either you'll need to
use some kind of surrogate keys, or else you'll need to use

`encodeURIComponent` (on the javascript side), and
`urllib.unquote` (on the python side)

5. If you're really committed, you can use CSS to make the site look really nice.
