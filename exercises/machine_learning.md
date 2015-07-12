# Machine Learning

The goal here is to predict what makes for a successful bike station.
We'll use linear regression.

1. To start with, we'll need to compute the number of rentals per station
per day. Use pandas to do that.

2. a. Our `stations` data has a huge number of quantitative attributes: `fast_food`,
`parking`, `restaurant`, etc...  Some of them are encoded as 0 or 1 (for absence
or presence), others represent counts.  To start with, run a simple linear
regression where the input (x) variables are all the various station attributes
and the output (y) variable is the average number of rentals per day.

b. Plot the predicted values (`model.predict(x)`) against the actual values and
see how they compare.

c. In this case, there are 129 input variables and only 185 rows which means we're
very likely to overfit.  Look at the model coefficients and see if anything
jumps out as odd.

d. Go back and split the data into a training set and a test set.  Train the model
on the training set and evaluate it on the test set.  How does it do?

3. a. Since we have so many variables, this is a good candidate for regularization.
In particular, since we'd like to eliminate a lot of them, `lasso` seems like a
good candidate.  Build a lasso model on your training data for various values of
alpha.  Which variables survive?

b. How does this model perform on the test set?

4. No matter how high I make `alpha`, the coefficient on `crossing` ("number of
nearby crosswalks") never goes away.  Try a simple linear regression on just
that variable.
