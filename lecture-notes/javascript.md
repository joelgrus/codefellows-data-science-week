# JavaScript

# What is JavaScript

It's the primary programming language of the web.
It has nothing to do with Java.

# What is Javascript

It's a dynamic language that (in some ways) is like Python.  It has more of a
"C-style" syntax.  (curly braces, etc..)

# Trying JavaScript

In Chrome if you go to "Developer Tools" there is a console where you can type in
JavaScript commands:

```
> var x = 1
< undefined
> x + 2
< 3
```

# In HTML

More broadly, you can stick JavaScript in your HTML files:

```
<!doctype html>
<body>
  <script>
  var x = 1;
  // one of those stupid popups you hate
  alert(x + 1);
  </script>
</body>
```

# Basics

Use `var` to declare a variable, otherwise it's a global.

Fundamental data types are arrays (like Python lists)

```
> var a = []
< undefined
> a.push(1)
< 1
> a.push(2)
< 2
> a
< [1, 2]
> a.length
< 2
> a[0] = 10
< 10
> a
< [10, 2]
```

and objects (like Python dicts).

```
> var obj = {}
< undefined
> obj['x'] = 3
< 3
> obj.x
< 3
> obj.y
< undefined
```

Can do classes and other object-oriented stuff, but we're going
to keep it really simple.

# Functions

It's pretty straightforward to define functions:

```
function addOne(x) {
  return x + 1;
}
```

# Callbacks

JavaScript tends to do a lot of asynchronous operations, which have the flavor
"request x, go off and do other stuff, but call this function when x is ready"

```
requestValueAsynchronously('x', function(result) {
  console.log(result);
  });
// the program doesn't wait for the result to be ready but keeps going.
// whenever the result is ready, the callback function will get called on it
```

This is in contrast to a "blocking style":

```
var x = requestValueSynchronously('x');
// code blocks until x is ready
console.log(x);
```

# Manipulating the DOM

When we use JavaScript with webpages, we typically want to manipulate them.
Frequently people will use tools like `jQuery`, but we'll just use plain JavaScript.

```
// get the element with id "my-element";
var elt = document.getElementById("my-element");
// clobber whatever it contains
elt.innerHTML = "<b>BOLD</b>";
```

# Events

If we want our page to be interactive, we have to use *events*.  For example,
if we want a div to alert when someone clicks on it, we can

```
<div id="alert-div>
...
var div = document.getElementById("alert-div");
div.addEventListener('click', function() {
  alert("someone clicked on me!");
});
```

Similarly, if we want to do something whenever someone types in an input box:

```
<input id="my-input">
...
var myInput = document.getElementById("my-input");
myInput.addEventListener('keyup', function() {
  console.log(myInput.value);
});
```

There are lots of other events you can handle, check the documentation.

# Making AJAX Calls

Is less easy if you don't use jQuery.  Here's some code I cribbed from StackOverflow:

```
// from http://stackoverflow.com/questions/8567114/how-to-make-an-ajax-call-without-jquery/18324384#18324384
function callAjax(url, callback){
    var xmlhttp;
    // compatible with IE7+, Firefox, Chrome, Opera, Safari
    xmlhttp = new XMLHttpRequest();
    xmlhttp.onreadystatechange = function(){
        if (xmlhttp.readyState == 4 && xmlhttp.status == 200){
            callback(xmlhttp.responseText);
        }
    }
    xmlhttp.open("GET", url, true);
    xmlhttp.send();
}
```

you pass it a URL and a callback, and it will retrieve the data at the URL
and pass it to the callback function.  

# JSON

When we transfer data (say, from a web service to a web page), it needs to be
serialized to text.  This is commonly done with JSON, the JavaScript Object Notation.

In particular, say we are building a Flask server to return an array of points:

```
@app.route('/points/')
def points():
    points = [{'x' : 1, 'y' : 2}, {'x' : 3, 'y' : 4}]
    return ???
```

we can't return Python lists or dicts, so we need to serialize it to JSON.

```
import json
serialized = json.dumps(points)
```

Then in the JavaScript callback, we need to deserialize it

```
function callback(serialized) {
  var points = JSON.parse(serialized);
  for (var i = 0; i < points.length; i++) {
    var point = points[i];
    console.log(point.x + " " + point.y);
  }
}
```

# Putting it all together:

* Create a Flask application that serves data at URLs
* Create an HTML page with some controls and some JS code
* Have the JS code make ajax calls to the Flash application
* Use callbacks to modify the page when the data returns
