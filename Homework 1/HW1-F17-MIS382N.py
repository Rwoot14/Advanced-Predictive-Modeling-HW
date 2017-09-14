
# coding: utf-8

# 
# # <p style="text-align: center;">MIS 382N: ADVANCED PREDICTIVE MODELING - MSBA</p>
# # <p style="text-align: center;">Assignment 1</p>
# ## <p style="text-align: center;">Total points: 75</p>
# ## <p style="text-align: center;">Due: Tuesday, September 13 submitted via Canvas by 11:59 p</p>
# 
# Your homework should be written in a **Jupyter notebook**. You may work in groups of two if you wish. Only one student per team needs to submit the assignment on Canvas.  But be sure to include name and UTEID for both students.  Homework groups will be created and managed through Canvas, so please do not arbitrarily change your homework group.  
# 
# Also, please make sure your code runs and the graphics (and anything else) are displayed in your notebook before submitting. (%matplotlib inline)

# # Question 1: Applications of machine learning (10 pts)
# 
# Read the [article](http://www.datasciencecentral.com/profiles/blogs/20-data-science-systems-used-by-amazon-to-operate-its-business) "21 data science systems used by Amazon to operate its business" and pick any two of the data science systems used by Amazon according to this blog.
# 
# (5 pts each) For each of these two system you have chosen:
# 
# What kind of machine learning problem is involved (e.g. classification, regression, clustering, outlier detection,...)? Speculate on what kind of data may be needed and how the results can be useful to the company.
# 
# 
# ## Answer
# Supply chain optimization (I). Sites selection for warehouses to minimize distribution costs (proximity to vendors, balanced against proximity to consumers). How many warehouses are needed, and what capacity each of them should have. 
# 
# The ideal data would be demographic\orders data on both the vendors and nearby population. Amazon would need to know the order volume of nearby populations along with projected growth which would be a forecasting problem. Also Amazon would need to compare that to vendor/transportation data to see what the optimal placement of warehouses would be, and how much capacity they're likely to have. The results from this analysis would increase efficiency in delivery time, and an increase in margins by decrease the costs associated with transportation. 
# 
# 
# Fake reviews detection. They still have tons of progress to make in this area: at least categorizing users would be a first step, so that buyers know what kind of user produced a specific review; then relevancy algorithms must be used to assess how relevant a review is for a specific product, knowing that most likes and stars assigned by users are biased - partly because most normal people don't have time or interest to write a review. Indeed, fake reviews is a lucrative business taking advantages of inefficiencies in platforms such as Amazon. The best solution is to remove user-generated reviews and replace them, for each product, by number of sales over the last 30 days.
# 
# The data needed for this problem would be reviews on products, possibly, if available, data that turks have already went through on a sample data set that has fake reviews in it. This data would be used as a training set for a classification model that would learn to predict fake reviews. This model would look at the attributes of the real reviews and see if there are differences between the fake reviews and real reviews. Depending on how accurate the model is at detecting fake reviews, it would provide enormous value in parsing down the reviews which need inspection to see if they're fake or not. 
# 
# 
# 
# 
# 
# 

# # Question 2: Maximum likelihood estimate (10 pts)
# 
# Suppose a manager at an internet sales company wants to estimate how fast his salesperson is generating successful leads. Instead of recording the time for each lead, the time taken to generate the next 5 leads are recorded, i.e., there is one recording (denoting the elapsed time) for every 5 consecutive leads. For a specific salesperson, the time intervals recorded are {1,3,1.5,4,2,7,1.2,2,4,3.1} hours. 
# 
# A statistician suggests that if these time intervals are assumed to arise by i.i.d. sampling from the following distribution:
# $$ p(t) = \frac{1}{C \times \theta^{5}}t^{4}exp^{-\frac{t}{\theta}},$$
# (where C is a normalizing constant). Therefore, if $\theta$ can be estimated, then he can provide detailed information
# about the lead generation process, including average rates, variances etc.
# 
# Find the Maximum Likelihood estimate for $\theta$ based on the recorded observations.
# 
# 
# ## Answer

# # Question 3: Multiple Linear Regression in Python (25 pts)
# 
# Use the following code to import the boston housing dataset and linear models in python.
# 
# 

# In[349]:

boston = datasets.load_boston()

X = boston.data
y = boston.target
features=boston.feature_names


# The dataset information can be found [here](http://scikit-learn.org/stable/datasets/index.html#boston-house-prices-dataset).
# 
# a. (3 pts) Print the shape (number of rows and columns) of the feature matrix, and print the first 5 rows.
# 
# b.  (6 pts) Using ordinary least squares, fit a multiple linear regression (MLR) on all the feature variables using the entire dataset (506 rows). Report the regression coefficient of each input feature and evaluate the model using mean squared error (MSE).  Example of ordinary least squares in Python is shown in Section 1.1.1 of http://scikit-learn.org/stable/modules/linear_model.html.
# 
# c.  (6 pts) Split the data into a training set and a test set.  Use the first 400 rows for training set and remaining rows for test set.  Fit an MLR using the training set.  Evaluate the trained model using the training set and the test set, respectively.  Compare the two MSE values thus obtained.
# 
# d.  (6 pts) Do you think your MLR model is reasonable for this problem? You may look at the distribution of residuals to provide an informed answer.
# 
# e. (5 pts) Use the following code to add new features to the dataset.  You should have 26 variables now.  Note that this code adds one squared term for each variable; in practice one may introduce only a few terms based on domain knowledge or experimentation.  Repeat (c) and report the MSE values of the training set and the test set, respectively.

# In[159]:

X = np.concatenate((X, np.square(X)), axis=1)


# ## Answer
# 
# 

# In[160]:

from sklearn import linear_model
from pandas import Series, DataFrame
import pandas as pd
get_ipython().magic(u'pylab inline')
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
import numpy as np
# Special packages
import statsmodels.api as sm
from patsy import dmatrices


# # A

# In[256]:

#x = np.empty(506)
#x.fill(1)
#X = np.insert(X,0,x, axis = 1)
DataFrame(X)


# # B

# In[254]:

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X,y)

# Make predictions using the testing set
y_pred = regr.predict(X)

# The coefficients
print('Coefficients: \n', DataFrame((zip(features, regr.coef_))).set_index(0))
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y,y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y,y_pred))


# In[255]:

regr.intercept_


# # C

# In[121]:




# In[239]:

X_train=X[:400]
y_train=y[:400]
X_test=X[400:]
y_test=y[400:]


# In[240]:

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# In[241]:

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train,y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test,y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test,y_pred))


# # D

# In[242]:

resid=y_pred-y_test


# In[243]:

plt.scatter(y_pred, resid)
plt.xlabel('y_pred', fontsize=18)
plt.ylabel('Residual', fontsize=16)


# # E

# In[145]:

X = np.concatenate((X, np.square(X)), axis=1)


# In[146]:

X_train=X[1:400]
y_train=y[1:400]
X_test=X[401:506]
y_test=y[401:506]


# In[147]:

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train,y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test,y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test,y_pred))


# In[148]:

resid=y_pred-y_test
plt.scatter(y_pred, resid)
plt.xlabel('y_pred', fontsize=18)
plt.ylabel('Residual', fontsize=16)


# # Question 4: Ridge and Lasso Regression (25 points)
# 

# Using the same boston data from before, in this question you will explore the application of Lasso and Ridge regression using sklearn package in Python. The following code will split the data into training and test set using [train_test_split](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) with **random state 20** and **test_size = 0.33**.  Note: lambda is called alpha in sklearn.

# In[433]:

boston = datasets.load_boston()

X = boston.data
y = boston.target
features=boston.feature_names


# In[434]:

from sklearn.model_selection import train_test_split
X = np.concatenate((X, np.square(X)), axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.33, random_state=20)


# 1) Use sklearn.linear_model.Lasso and sklearn.linear_model.Ridge classes to do a [5-fold cross validation](http://scikit-learn.org/stable/auto_examples/exercises/plot_cv_diabetes.html#example-exercises-plot-cv-diabetes-py) using sklearn's [KFold](http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.KFold.html). For the sweep of the regularization parameter, we will look at a grid of values ranging from $\lambda = 10^{10}$ to $\lambda = 10^{-2}$. In Python, you can consider this range of values as follows:
# 
#       import numpy as np
# 
#       alphas =  10**np.linspace(10,-2,100)*0.5
# 
#   Report the best chosen $\lambda$ based on cross validation. The cross validation should happen on your training data using  average MSE as the scoring metric. (8pts)
# 
# 2) Run ridge and lasso for all of the alphas specified above (on training data), and plot the coefficients learned for each of them - there should be one plot each for lasso and ridge, so a total of two plots; the plots for different features for a method should be on the same plot (e.g. Fig 6.6 of JW). What do you qualitatively observe when value of the regularization parameter is changed? (7pts)
# 
# 3) Run least squares regression, ridge, and lasso on the training data. For ridge and lasso, use only the best regularization parameter. Report the prediction error (MSE) on the test data for each. (5pts)
# 
# 4) Run lasso again with cross validation using [sklearn.linear_model.LassoCV](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html). Set the cross validation parameters as follows:
# 
#     LassoCV(alphas=None, cv=10, max_iter=10000)
# 
# Report the best $\lambda$ based on cross validation. Run lasso on the training data using the best $\lambda$ and report the coefficeints for 26 variables. What do you observe from these coefficients? (5pts)
# 
# ## Answer
# 
# 

# In[435]:

DataFrame(X)


# In[ ]:




# In[436]:

from sklearn import datasets
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


# # 1

# In[437]:

# Lasso setup for cross validating the shrinkage parameter with K_fold cv.
lasso = Lasso(random_state=0,normalize=True)
alphas = 10**np.linspace(10,-2,100)*0.5

tuned_parameters = [{'alpha': alphas}]
n_folds = 5

# Uses function gridsearch to iterate the lasso model over the k_folds
clf = GridSearchCV(lasso, tuned_parameters, cv=n_folds, refit=False,scoring='neg_mean_squared_error')#refit needs to be true?? need to standardize x's??
clf.fit(X_train, y_train)
#stores the scores of the cross validation
scores = clf.cv_results_['mean_test_score']
scores_std = clf.cv_results_['std_test_score']

########
# CODE NOT USED 
########
#plt.figure().set_size_inches(8, 6)
#plt.semilogx(alphas, scores)

# plot error lines showing +/- std. errors of the scores
#std_error = scores_std / np.sqrt(n_folds)

#plt.semilogx(alphas, scores + std_error, 'b--')
#plt.semilogx(alphas, scores - std_error, 'b--')

# alpha=0.2 controls the translucency of the fill color
#plt.fill_between(alphas, scores + std_error, scores - std_error, alpha=0.2)

#plt.ylabel('CV score +/- std error')
#plt.xlabel('alpha')
#plt.axhline(np.max(scores), linestyle='--', color='.5')
#plt.xlim([alphas[0], alphas[-1]])


# In[438]:

lasso_param=clf.best_params_
print(clf.best_params_)


# ### The optimal lambda for the cross validated lasso regression is the alpha show above. The lambda is relatively small meaning that the model needs to be more complex to get the optimal MSE. We chose to normalize the data because the data were all of different magnitudes/units and needed to be normalized so the appropriate lambda could be chosen for all the parameters. 

# In[439]:

from sklearn.linear_model import Ridge


# In[440]:

# Ridge Regression
ridge = Ridge(random_state=0,normalize=True)
alphas = 10**np.linspace(10,-2,100)*0.5

tuned_parameters = [{'alpha': alphas}]
n_folds = 5

clf = GridSearchCV(ridge, tuned_parameters, cv=n_folds, refit=False,scoring='neg_mean_squared_error')
clf.fit(X_train, y_train)
scores = clf.cv_results_['mean_test_score']
scores_std = clf.cv_results_['std_test_score']
#plt.figure().set_size_inches(8, 6)
#plt.semilogx(alphas, scores)

# plot error lines showing +/- std. errors of the scores
#std_error = scores_std / np.sqrt(n_folds)

#plt.semilogx(alphas, scores + std_error, 'b--')
#plt.semilogx(alphas, scores - std_error, 'b--')

# alpha=0.2 controls the translucency of the fill color
#plt.fill_between(alphas, scores + std_error, scores - std_error, alpha=0.2)

#plt.ylabel('CV score +/- std error')
#plt.xlabel('alpha')
#plt.axhline(np.max(scores), linestyle='--', color='.5')
#plt.xlim([alphas[0], alphas[-1]])


# In[441]:

ridge_param=clf.best_params_
print(clf.best_params_)


# In[ ]:




# ### The optimal lambda for the cross validated Ridge regression is the alpha show above. The lambda is the same as the lasso regression.  We chose to normalize the data because the data were all of different magnitudes/units and needed to be normalized so the appropriate lambda could be chosen.

# # 2

# In[442]:

lasso = Lasso(random_state=0, normalize=True)
alphas = 10**np.linspace(10,-2,100)*0.5
tuned_parameters = [{'alpha': alphas}]
n_folds = 5
coefs = []
for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(X_train, y_train)
    coefs.append(lasso.coef_)


ax = plt.gca()
ax.plot(alphas*2, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')


# ### For the lasso regression The Coef.'s go to 0 relatively quickly for a small lambda. This means that the model will be similar to the OLS model. 

# In[443]:

ridge = Ridge(random_state=0, normalize= True)
alphas = 10**np.linspace(10,-2,100)*0.5
tuned_parameters = [{'alpha': alphas}]
n_folds = 5
coefs = []
for a in alphas:
    ridge.set_params(alpha=a)
    ridge.fit(X_train, y_train)
    coefs.append(ridge.coef_)


ax = plt.gca()
ax.plot(alphas*2, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')


# ### The Ridge regression model has a larger lambda than the lasso regression with the coef. converging to 0. This means that the model is getting further away from the normal OLS model and is able to shrink the coef. more. 

# ### For both of the plots as lambda increases, the coef. of both models either reach 0 (for lasso) or converge to 0 (for ridge). This is because lambda is shrinks the ceof.'s to prevent overfitting. 

# # 3

# In[448]:

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train,y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)


# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test,y_pred))


# ### The Ridge regression does better than the lasso regression but worse than the ridge regression. which indicates there is a need for a shrinkage parameter and that not all the variables contain meaningful information

# In[449]:

lasso=linear_model.Lasso(alpha=lasso_param['alpha'])


# In[450]:

lasso_fit=lasso.fit(X_train, y_train)
y_pred=lasso_fit.predict(X_test)
mean_squared_error(y_test, y_pred)


# ### The Lasso regression has the worst MSE compared to the ridge and ols regression. 

# In[451]:

Ridge=linear_model.Ridge(alpha=ridge_param['alpha'])


# In[452]:

ridge_fit=Ridge.fit(X_train, y_train)
y_pred=ridge_fit.predict(X_test)
mean_squared_error(y_test, y_pred)


# ### The Ridge regression does has the best MSE. This is likley because while the ridge regression shrinks the paramaters of the model, it still uses information from the other variables

# # 4

# In[453]:

lasso=LassoCV(fit_intercept = True,alphas=None, cv=10, max_iter=10000, normalize= True)
lasso_fit=lasso.fit(X_train, y_train)
y_pred=lasso_fit.predict(X_test)
mean_squared_error(y_test, y_pred)


# ### Once the new variables were added into the model the Lasso regression had a much better MSE. This is likley due to the added variables bringing in new information for the model to learn from. 

# In[454]:

lasso_fit.alpha_


# ### The Lambda reported from the model is shown above and is very small, the lasso model doesnt need to shrink the coef. that much

# In[455]:

lasso_fit.coef_


# In[456]:

DataFrame((zip(features, lasso_fit.coef_))).set_index(0)


# ### From the lasso regression estimated coef, INDUS, ZN, and NOX contain no information for the response variable and were shrunk to 0 by the lasso model. 

# # Question 5  (5 pts)
# 
# A regression model that includes "interaction terms" (i.e. quadratic terms of the form $x_ix_j$) as predictors in addition to the linear terms is clearly more general than a corresponding model that employs the same independent variables but only uses the linear terms. Outline two situations where the simpler (less general) model would be preferred to the more powerful model that includes interactive terms.
# 
# ## Answer
# 
# 1: If the data were fit perfectly by a linear approxamation adding interaction terms to the model would only increase the variance of the model, with no significant decrease in bias. An example might be years of education as an undergraduate and total amount paid in tuition. 
# 
# 2: If the data were extermely noisy, adding more interaction terms might be just 'chasing noise'. An example might be predicting stock market prices. If you fit the data really well with a complicated model you will likley just be chasing the noise of the stock market and wont make good predictions out of sample. A much simplier model would be suffice to adequetly predict stock market prices, like yesterdays stock price. 

# In[ ]:



