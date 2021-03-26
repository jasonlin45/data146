# Midterm

## Import Libraries and Dataset
First, importing the relevant libraries and functions we will need
```python
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler as SS
import numpy as np
import pandas as pd
```
Next, store the dataset we will be using
```python
data = fetch_california_housing(as_frame=True)
X = data.data
X_names = data.feature_names
y = data.target
df = data.frame
```
The `as_frame=True` parameter brings in the data as a `pandas` dataframe

## Set up KFolds Training Method
Creating a function for kfolds with various parameters allows us to easily train different models without having to type the same code over and over.
```python
def DoKFold(model, X, y, k, standardize=False, random_state=146):
    from sklearn.model_selection import KFold
    if standardize:
        from sklearn.preprocessing import StandardScaler as SS
        ss = SS()

    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

    train_scores = []
    test_scores = []

    for idxTrain, idxTest in kf.split(X):
        Xtrain = X.iloc[idxTrain, :]
        Xtest = X.iloc[idxTest, :]
        ytrain = y[idxTrain]
        ytest = y[idxTest]

        if standardize:
            Xtrain = ss.fit_transform(Xtrain)
            Xtest = ss.transform(Xtest)

        model.fit(Xtrain, ytrain)

        train_scores.append(r2_score(ytrain, model.predict(Xtrain)))
        test_scores.append(r2_score(ytest, model.predict(Xtest)))

    return train_scores, test_scores
```

## Question 15
Which of the below features is most strongly correlated with the target?
1. MedInc (median income)
2. AveRooms (average number of rooms)
3. AveBedrms (average number of bedrooms)
4. HouseAg (average house age)

Running `df.corr()` lets us see the Pearson's Correlation Coefficient for each variable in our dataset.

|             |      MedInc |   HouseAge |    AveRooms |   AveBedrms |   Population |    AveOccup |    Latitude |   Longitude |   MedHouseVal |
|:------------|------------:|-----------:|------------:|------------:|-------------:|------------:|------------:|------------:|--------------:|
| MedInc      |  1          | -0.119034  |  0.326895   |  -0.0620401 |   0.00483435 |  0.0187662  | -0.0798091  | -0.0151759  |     0.688075  |
| HouseAge    | -0.119034   |  1         | -0.153277   |  -0.0777473 |  -0.296244   |  0.0131914  |  0.0111727  | -0.108197   |     0.105623  |
| AveRooms    |  0.326895   | -0.153277  |  1          |   0.847621  |  -0.0722128  | -0.00485229 |  0.106389   | -0.0275401  |     0.151948  |
| AveBedrms   | -0.0620401  | -0.0777473 |  0.847621   |   1         |  -0.0661974  | -0.0061812  |  0.0697211  |  0.0133444  |    -0.0467005 |
| Population  |  0.00483435 | -0.296244  | -0.0722128  |  -0.0661974 |   1          |  0.0698627  | -0.108785   |  0.0997732  |    -0.0246497 |
| AveOccup    |  0.0187662  |  0.0131914 | -0.00485229 |  -0.0061812 |   0.0698627  |  1          |  0.00236618 |  0.00247582 |    -0.0237374 |
| Latitude    | -0.0798091  |  0.0111727 |  0.106389   |   0.0697211 |  -0.108785   |  0.00236618 |  1          | -0.924664   |    -0.14416   |
| Longitude   | -0.0151759  | -0.108197  | -0.0275401  |   0.0133444 |   0.0997732  |  0.00247582 | -0.924664   |  1          |    -0.0459666 |
| MedHouseVal |  0.688075   |  0.105623  |  0.151948   |  -0.0467005 |  -0.0246497  | -0.0237374  | -0.14416    | -0.0459666  |     1         |

After looking at this table, it's easy to see that the most correlated feature is Median Income (MedInc), with an R<sup>2</sup> of 0.688.

## Question 16
​
If the features are standardized, the correlations from the previous question do not change.
​
```python
Xs = SS().fit_transform(X)
sdf = pd.DataFrame(Xs, index=X.index, columns=X.columns)
sdf['MedHouseVal'] = y
sdf.corr()
```
The correlations are the same as previous.

## Question 17

If we were to perform a linear regression using only the feature identified in question 15,
what would be the coefficient of determination?
Enter your answer to two decimal places, for example: 0.12

Lets go ahead and actually do the regression.

```python
x = data.data[['MedInc']]
lin_reg = LinearRegression()
np.round(lin_reg.fit(x, y).score(x, y), 2)
```
Out:
```python
0.47
```
We have an R<sup>2</sup> value of 0.47

## Question 18 
​
Let's take a look at how a few different regression methods perform on this data.

Start with a linear regression.

Standardize the data

Perform a K-fold validation using:
k=20 \ 
shuffle=True \
random_state=146 \
What is the mean R2 value on the test folds?  Enter your answer to 5 decimal places, for example: 0.12345

```python
train_scores, test_scores = DoKFold(lin_reg,X,y,20, standardize=True)
print('Training: ' + format(np.mean(train_scores), '.5f'))
print('Testing: ' + format(np.mean(test_scores), '.5f'))
```
Out:
```
Training: 0.60630
Testing: 0.60198
```

## Question 19

Next, try Ridge regression.

To save you some time, I've determined that you should look at 101 equally spaced values between 20 and 30 for alpha.

Use the same settings for K-fold validation as in the previous question.

For the optimal value of alpha in this range, what is the mean R2 value on the test folds?
Enter your answer to 5 decimal places, for example: 0.12345

```python
a_range = np.linspace(20, 30, 101)
# a_range = np.linspace(5, 15, 100)
# a_range = np.linspace(7, 8, 100)

k = 20

avg_tr_score=[]
avg_te_score=[]

for a in a_range:
    rid_reg = Ridge(alpha=a)
    train_scores,test_scores = DoKFold(rid_reg,X,y,k,standardize=True)
    avg_tr_score.append(np.mean(train_scores))
    avg_te_score.append(np.mean(test_scores))

idx = np.argmax(avg_te_score)
print('Optimal alpha value: ' + format(a_range[idx], '.3f'))
print('Training score for this value: ' + format(avg_tr_score[idx],'.3f'))
print('Testing score for this value: ' + format(avg_te_score[idx], '.5f'))
```
Out:
```
Optimal alpha value: 25.800
Training score for this value: 0.606
Testing score for this value: 0.60201
```
## Question 20

Next, try Lasso regression.  Look at 101 equally spaced values between 0.001 and 0.003.

Use the same settings for K-fold validation as in the previous 2 questions.

For the optimal value of alpha in this range, what is the mean R2 value on the test folds?
Enter you answer to 5 decimal places, for example: 0.12345

```python
a_range = np.linspace(0.002, 0.003, 101)

k = 20

avg_tr_score=[]
avg_te_score=[]

for a in a_range:
    print(a)
    las_reg = Lasso(alpha=a)
    train_scores,test_scores = DoKFold(las_reg,X,y,k,standardize=True)
    avg_tr_score.append(np.mean(train_scores))
    avg_te_score.append(np.mean(test_scores))

idx = np.argmax(avg_te_score)
print('Optimal alpha value: ' + format(a_range[idx], '.5f'))
print('Training score for this value: ' + format(avg_tr_score[idx],'.3f'))
print('Testing score for this value: ' + format(avg_te_score[idx], '.5f'))
```

Optimal R2 value on test folds: 0.60213

## Question 21
Let's look at some of what these models are estimating.

Refit a linear, Ridge, and Lasso regression to the entire (standardized) dataset.

No need to do any train/test splits or K-fold validation here. Use the optimal alpha values you found previously.

Which of these models estimates the smallest coefficient for the variable that is least correlated
(in terms of absolute value of the correlation coefficient) with the target?
​
```python
lin_reg.fit(X, y)
rid_reg = Ridge(25.8)
rid_reg.fit(X, y)
las_reg = Lasso(0.00186)
las_reg.fit(X, y)
```
Least correlated: AveOccup (Average Occupation)

Ridge regression esimates -0.03925, the lowest

## Question 22 
​
Which of the above models estimates the smallest coefficient for the variable that is most correlated
(in terms of the absolute value of the correlation coefficient) with the target?
​
Most correlated: MedInc (Median Income)

Lasso Regression estimates 0.82, the lowest

## Question 23
If we had looked at MSE instead of R2 when doing our Ridge regression (question 19),
would we have determined the same optimal value for alpha, or something different?

Our new training method:

```python
def DoKFold(model, X, y, k, standardize=False, random_state=146):
# def DoKFold(model, X, y, k, standardize=False):
    from sklearn.model_selection import KFold
    if standardize:
        from sklearn.preprocessing import StandardScaler as SS
        ss = SS()

    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    # kf = KFold(n_splits=k, shuffle=True)

    train_scores = []
    test_scores = []

    for idxTrain, idxTest in kf.split(X):
        Xtrain = X.iloc[idxTrain, :]
        Xtest = X.iloc[idxTest, :]
        ytrain = y[idxTrain]
        ytest = y[idxTest]

        if standardize:
            Xtrain = ss.fit_transform(Xtrain)
            Xtest = ss.transform(Xtest)

        model.fit(Xtrain, ytrain)

        train_scores.append(mean_squared_error(ytrain, model.predict(Xtrain)))
        test_scores.append(mean_squared_error(ytest, model.predict(Xtest)))
        
        
    return train_scores, test_scores
```

This returns a different alpha result.

## Question 24
If we had looked at MSE instead of R2 when doing our Lasso regression (question 20),
what would we have determined the optimal value for alpha to be?
Enter your answer to 5 decimal places, for example: 0.12345

Running the above lasso regression with the new training method gives us an
optimal alpha value of 0.00300
