# Project 3

## Basic Linear Regression
In this section, we'll perform a basic linear regression on our Charleston housing data from Zillow.  

### Setting Up Environment
First, let's go ahead and import our libraries and relevant methods we'll need.
We'll need pandas and numpy, as well as linear regression and k-fold cross validation sklearn.

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
```
Next, we'll use `pandas` to read in our dataset.  Our data is a csv detailing the asking prices of houses in Charleston.

```python
ask = pd.read_csv("charleston_ask.csv")
```

Now that we have our data, we want to specify our features and targets. Our features in this case will be the number of beds, number of baths, and square feet of each home.  Our target will be the asking price.

```python
X = ask[['beds', 'baths', 'sqft']]
Y = ask['prices']
```

Now, lets define our model.

```python
lin_reg = LinearRegression()
```

Easy as pie.  Now we're ready to train once we implement our k-folds cross validation technique.

### K-Folds Cross Validation

Why should we use cross validation?  K-Folds allows us to use all our data, with a similar base idea to the normal train test split.  We never want to train with just the data, because we become extremely vulnerable to overfitting.  Furthermore, this methodology allows us to generate metrics and statistics about our model.

Let's define our kfolds here:
```python
kf = KFold(n_splits = 5, shuffle=True)
```

The parameter for number of splits is variable.  It's definitely a parameter we can optimize if we choose to do something like gridsearch later down the road, but typically, the higher this value, the less biased our model is.  As a consequence, we do have higher variance, and with small datasets, it's possible to repeat combinations of your data.  We can go ahead and just use 5 for now.

### Training

Training can be done in different ways, but here's some code following what we did in class:

The basic idea is to take the mean of every model we train on each split.

```python
train_scores = []
test_scores = []

for idxTrain, idxTest in kf.split(X):
  Xtrain = X.iloc[idxTrain, :]
  Xtest = X.iloc[idxTest, :]
  ytrain = Y.iloc[idxTrain]
  ytest = Y.iloc[idxTest]

  lin_reg.fit(Xtrain, ytrain)

  train_scores.append(lin_reg.score(Xtrain, ytrain))
  test_scores.append(lin_reg.score(Xtest, ytest))

print('Training: ' + format(np.mean(train_scores), '.3f'))
print('Testing: ' + format(np.mean(test_scores), '.3f'))
```

This outputs:

```
Training: 0.019
Testing: -0.007
```

### Intepreting Results

The above training and testing values we've calculated represent R<sup>2</sup> values.  An R<sup>2</sup> value can also be called the coefficient of determination.  A value of 1 means that the regression explains all the variability around the mean, indicating that our model is perfect, amazing, fantastic.  A value of 0 means the opposite - that none of the variability is explained, indicating our model is quite terrible, inaccurate, not a good fit.  

In our case, we have an extremely low value for the training scores, which is already a bad sign.  We even managed to achieve a negative value for our testing score. It's pretty normal to see a higher training score as opposed to a testing score, but a large difference is indicative of overfitting.  Currently, our model is terrible.  Let's see what we optimizations we can do in order to improve it.

## Standardizing Features

Looking at the data, the scale of the data widely varies.  The numbers we have for house area, number of bathrooms, and number of bedrooms are completely different measurements.  In this case, we want to go ahead and standardize these values and optimize our model.

Let's go ahead and import `StandardScaler` from `sklearn`, which can easily do this transformation for us.

```python
from sklearn.preprocessing import StandardScaler as SS
```

Now let's go ahead and traing our linear regression again, but making sure to transform and standardize our features.



```python
train_scores = []
test_scores = []

for idxTrain, idxTest in kf.split(X):
  Xtrain = X.iloc[idxTrain, :]
  Xtest = X.iloc[idxTest, :]
  ytrain = Y.iloc[idxTrain]
  ytest = Y.iloc[idxTest]
  
  #Standardize our features
  Xtrain = ss.fit_transform(Xtrain)
  Xtest = ss.transform(Xtest)
  
  lin_reg.fit(Xtrain, ytrain)

  train_scores.append(lin_reg.score(Xtrain, ytrain))
  test_scores.append(lin_reg.score(Xtest, ytest))

print('Training: ' + format(np.mean(train_scores), '.3f'))
print('Testing: ' + format(np.mean(test_scores), '.3f'))
```

We get the output:

```
Training: 0.020
Testing: -0.002
```

Note that we are still using 5 splits.  

### Interpreting Results

Standardizing features is a very powerful tool that can help us optimize our models.  In our case, our model did slightly improve. Percentage-wise, we had a large increase in accuracy in the test data, and a minimal increase in the training accuracy.  However, overall, these values are still nowhere near where we want them to be.  Testing is still negative, and training is still basically 0.  Our linear regression is performing quite poorly.

## Ridge Regression

Perhaps a linear regression isn't the best model we can fit to our data.  Another model we will try is a Ridge Regression.  Let's go ahead and import it from `sklearn`.

```python
from sklearn.linear_model import Ridge
```

### Training Model
After choosing an alpha value for our model (we can brute force search the most optimal one in a range to find it), we can go ahead and run our model.

In our case, we will stick to 5 folds, and keep standardizing the features.

```python
rid_reg = Ridge(alpha=75.758)
train_scores = []
test_scores = []
for idxTrain, idxTest in kf.split(X):
  Xtrain = X.iloc[idxTrain, :]
  Xtest = X.iloc[idxTest, :]
  ytrain = Y.iloc[idxTrain]
  ytest = Y.iloc[idxTest]
  Xtrain = ss.fit_transform(Xtrain)
  Xtest = ss.transform(Xtest)
  rid_reg.fit(Xtrain, ytrain)

  train_scores.append(rid_reg.score(Xtrain, ytrain))
  test_scores.append(rid_reg.score(Xtest, ytest))

print('Training: ' + format(np.mean(train_scores), '.3f'))
print('Testing: ' + format(np.mean(test_scores), '.3f'))
```

We get the following:

```
Training: 0.019
Testing: 0.017
```

### Interpreting Results

Our values again are not great.  They're quite a bit better than the values we had previously, with our test score rising significantly and being relatively close to the training score.


## Actual Price

Let's go ahead and run the models from previous exactly as they are, but we'll use the asking price data instead.

Read in the data:

```python
act = pd.read_csv("charleston_act.csv")
```

### Unstandardized Linear Regression
Running our first linear regression model from above, we get the following result:
```
Training: 0.006
Testing: -0.024
```
Again, these values are not great, and are pretty unchanged from before.

### Standardized Linear Regression
Running our second linear regression model which uses standardized features from above, we get the following result:
```
Training: 0.005
Testing: -0.009
```
These values are still not great, and our model still isn't well fit to the data.


### Ridge Regression
Running our ridge regression model same as above yields the following results:
```
Training: 0.004
Testing: -0.001
```

Same as the other models, we haven't really improved much from when we used the asking price.  

## Adding Zip Codes

Let's go ahead and use the actual price, with the zip code data added.

```python
act = pd.read_csv("charleston_act.csv")

X = act.drop(columns='prices')
Y = act['prices']
```

### Unstandardized Linear Regression
Running our first linear regression model from above yields the following results:
```
Training: 0.344
Testing: 0.252
```
These results are much much better.  We have a better fit overall, not a fantastic one, but definitely a step up from not being fit at all.  Our model's predictive power is still mediocre even though we have zip codes incorporated.  Since the training score is quite a bit higher than the testing score, we may be seeing some overfitting happening here.

### Standardized Linear Regression
Running our linear regression model with standardized features from above yields the following results:

```
Training: 0.343
Testing: 0.265
```

Standardizing doesn't really seem to make much of a difference, but the accuracies are up significantly from before.  Again, the zip codes are helping greatly here.

### Ridge Regression
Running our ridge regression model same as above, with 5 folds and standardized features, we get the following:
```
Training: 0.340
Testing: 0.291
```

This result is again much better, and shows that the zip codes are strong in helping us determine actual house pricing.  We can speculate that the reason why zip codes are so important in this calculation may be that similarly priced houses tend to be grouped together in the same areas.  Because zip codes are an indicator of geographic location, similar zip codes may imply similar prices.

## Conclusion

Choosing the above ridge regression trained using k-folds for cross validation, our highest R<sup>2</sup> value was 0.291, which isn't fantastic.  This model is only slightly overfit, since our training result is 0.340, and higher than our testing R<sup>2</sup>.  However, it is quite normal to get a higher value for training.  It seems that location correlates strongly with the result of our data, and the number of beds/baths/sqft isn't as strong of an indicator.  In order to increase predictive power, it may be necessary to look for additional features.  One such feature may be to look at the type of building for sale, since we can also use condo vs house vs townhouse vs new construction as features.  Intuitively, these may lead to strong price variations.  

Another improvement we could make outside of just Zillow housing data would be joining this data with other geocoded data on local crime, poverty, or school statistics.  Population of areas may also be a good indicator of demand, meaning prices may also be higher for housing.  Even further, taking a look at the overall income level of the area the house is located in might help determine the prices in that area.
