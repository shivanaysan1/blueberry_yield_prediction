#!/usr/bin/env python
# coding: utf-8

# # Load Python Packages

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

from sklearn import metrics

from sklearn import set_config
set_config(display="diagram")

# Packages to prevent any warnings from being displayed (not neccessary but useful)
import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Importing the dataset into a pandas dataframe
df = pd.read_csv('blueberry.csv')
df


# # Primary Exploratory Data Analysis

# In[3]:


# Printing the first 5 rows of the dataset
df.head()


# In[4]:


# Printing the last 5 rows of the dataset
df.tail()


# In[5]:


# Printing the number of columns in the dataset
len(df.columns)


# In[6]:


# Printing the number of rows in the dataset
len(df)


# In[7]:


# Printing the shape of the data frame (ensuring it is 2-dimensional)
df.shape


# In[8]:


# Printing the size of the data frame (number of rows x number of columns)
df.size


# In[9]:


# Printing the labels of the columns
df.columns


# In[10]:


# Printing basic in formation about each column (label, number of non-null values, data type)
df.info()


# In[11]:


# Printing the basic statistical information of all columns (mean, standard deviation, minimum value, maximum value and
# percentiles [25%, 50%, 75%])
df.describe()


# In[12]:


# Printing the data types of each column
# All columns are of float64 data type except of the 'Row#' column (not important column)
df.dtypes


# # Data Preprocessing

# In[13]:


# Finding if there are any null values in each column
# No null values are to be found
df.isna().sum()


# In[14]:


# Finding if there are any duplicated values in each column
# No duplicated vales are to be found
df.duplicated().sum()


# In[15]:


# Dropping unneccesary columns
df.drop(['Row#'], axis = 'columns', inplace = True)
df


# In[16]:


# Checking the skewness of the target variable
sns.kdeplot(df['yield'], color='r')
print('Skewness value of Yield:', df['yield'].skew())
# The skewness is negative and is within the acceptable range


# In[17]:


# Checking for outliers in all columns
def show_outlier(features):
    fig,ax = plt.subplots(int(np.ceil(len(df.columns)/4)),4,figsize = (30,15))
    ax = np.ravel(ax)
    for i,col in enumerate(df.columns):
        sns.boxplot(ax = ax[i], x = df[col], color= "red")
    fig.suptitle("Box plots of all data ",fontsize = 20)
    plt.tight_layout(pad=3)
    plt.show()

show_outlier(df)
# Most outlier values are not extreme
# No outliers are removed to preserve context


# # Feature Engineering

# In[18]:


# Creating new features (variables) by combining existing features 
df['totalBeeDensity'] = df['honeybee'] + df['bumbles'] + df['andrena'] + df['osmia']
df['averageTRange'] = (df['AverageOfUpperTRange'] + df['AverageOfLowerTRange'])/2


# In[19]:


# Dropping unrelevant features that have been either used to create new features or are unneccesary
df.drop(['MaxOfUpperTRange', 'MinOfUpperTRange', 'AverageOfUpperTRange', 'MaxOfLowerTRange', 'MinOfLowerTRange', 'AverageOfLowerTRange', 'RainingDays', 'fruitset', 'fruitmass', 'seeds'], axis = 'columns', inplace = True)
df


# # Secondary Exploratory Data Analysis

# In[20]:


# Creating a heatmap of correlation values between variables
plt.figure(figsize = (30,20))
pearson = df.corr()
sns.heatmap(pearson, annot=True, cmap = plt.cm.Reds)
plt.show()
# The darker the box, the more positively correlated the variables (1 = most positively correlated)
# The lighter the box, the more negatively correlated the variables (-1 = most negatively correlated)


# In[21]:


# Creating scatterplots between variables
fig, ax = plt.subplots(3, 3, figsize = (40, 20))
for var, subplot in zip(df, ax.flatten()):
    sns.scatterplot(x = var, y = 'yield',  data = df, ax = subplot, hue = 'yield' )


# In[22]:


# Prining the correlation values between variables
df.corr()


# In[23]:


# Finding the features that are most correlated (more than 0.5) with the target variable
core_target = abs(pearson["yield"])
relevant_features = core_target[core_target>0.5]
relevant_features
# The features are 'clonesize', 'AverageRainingDays' and 'yield' (target variable)


# # Feature Selection

# In[24]:


# Selecting the features (all variables except 'yield')
features = df.drop('yield', axis = 1)
features


# In[25]:


# Checking the shape of the feature matrix (must be 2-dimensional)
features.shape


# In[26]:


# Selecting the target variable to be 'yield'
target = df['yield']
target


# In[27]:


# Checking the shape of the target vector (must be 1-dimensional)
target.shape


# # Data Modelling & Testing

# In[28]:


X = features
y = target

# Applying standard scaling to the feature matrix
scaler = StandardScaler()
X = scaler.fit_transform(X)


# In[29]:


# Splitting the data set into a training set and test set
# The split is 80% : 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state = 42)

# Checking the shape of the training and test sets
X_train.shape,X_test.shape,y_train.shape,y_test.shape


# ## Linear Regression

# In[30]:


from sklearn.linear_model import LinearRegression

lgr = LinearRegression()
lgr.fit(X_train, y_train)


# In[31]:


print("Accuracy score:", lgr.score(X_test, y_test))


# ## KNN Regression

# In[32]:


from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors = 12, p = 1)
knn.fit(X_train, y_train)


# In[33]:


print("Accuracy score:", knn.score(X_test, y_test))


# In[34]:


knn.get_params()


# In[35]:


leaf_size = [i for i in range(0,101)]
n_neighbors = [i for i in range(0,101)]
p = [i for i in range(0,101)]

hyperparemeter_grid = {'leaf_size':leaf_size, 'n_neighbors':n_neighbors, 'p':p}


# In[36]:


random_cv = RandomizedSearchCV(estimator = knn, param_distributions = hyperparemeter_grid, cv = 5, n_iter = 50, 
                               scoring = 'neg_mean_absolute_error', n_jobs = 4, verbose = 5, return_train_score = True, 
                               random_state = 42)


# In[37]:


random_cv.fit(X_train, y_train)


# In[38]:


random_cv.best_estimator_


# In[39]:


knn_tuned = KNeighborsRegressor(leaf_size = 4, n_neighbors = 2, p = 84)


# In[40]:


knn_tuned.fit(X_train, y_train)


# In[41]:


print("Accuracy score:", knn_tuned.score(X_test, y_test))


# ## Decision Tree Regression

# In[42]:


from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor(max_depth = 8, random_state = 0)
dtr.fit(X_train, y_train)


# In[43]:


print("Accuracy score:", dtr.score(X_test, y_test))


# In[44]:


dtr.get_params()


# In[45]:


max_depth = [i for i in range(0,101)]
random_state = [i for i in range(0,101)]
min_samples_leaf = [5, 10, 20, 50, 100]

hyperparemeter_grid = {'max_depth':max_depth, 'random_state':random_state, 'min_samples_leaf':min_samples_leaf}


# In[46]:


random_cv = RandomizedSearchCV(estimator = dtr, param_distributions = hyperparemeter_grid, cv = 5, n_iter = 50, 
                               scoring = 'neg_mean_absolute_error', n_jobs = 4, verbose = 5, return_train_score = True, 
                               random_state = 42)


# In[47]:


random_cv.fit(X_train, y_train)


# In[48]:


random_cv.best_estimator_


# In[49]:


dtr_tuned = DecisionTreeRegressor(max_depth = 21, min_samples_leaf = 5, random_state = 22)


# In[50]:


dtr_tuned.fit(X_train, y_train)


# In[51]:


print("Accuracy score:", dtr_tuned.score(X_test, y_test))


# ## Random Forest Regression

# In[52]:


from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)


# In[53]:


print("Accuracy score:", rfr.score(X_test, y_test))


# In[54]:


rfr.get_params()


# In[55]:


bootstrap = [True, False]
max_depth = [i for i in range(0,101)]
max_features = ['auto', 'sqrt', 'log2']
min_samples_leaf = [i for i in range(0,11)]
min_samples_split = [i for i in range(0,11)]
n_estimators = [i for i in range(0,1000)]

hyperparemeter_grid = {'bootstrap':bootstrap, 'max_depth':max_depth, 'max_features':max_features,
                       'min_samples_leaf':min_samples_leaf, 'min_samples_split':min_samples_split,
                       'n_estimators':n_estimators}


# In[56]:


random_cv = RandomizedSearchCV(estimator = rfr, param_distributions = hyperparemeter_grid, cv = 5, n_iter = 50, 
                               scoring = 'neg_mean_absolute_error', n_jobs = 4, verbose = 5, return_train_score = True, 
                               random_state = 42)


# In[57]:


random_cv.fit(X_train, y_train)


# In[58]:


random_cv.best_estimator_


# In[59]:


rfr_tuned = RandomForestRegressor(max_depth = 58, min_samples_leaf = 2, min_samples_split = 5, n_estimators = 788)


# In[60]:


rfr_tuned.fit(X_train, y_train)


# In[61]:


print("Accuracy score:", rfr_tuned.score(X_test, y_test))


# ## XGBoost Regression

# In[62]:


import xgboost

xgb = xgboost.XGBRegressor()
xgb.fit(X_train, y_train)


# In[63]:


print("Accuracy score:", xgb.score(X_test, y_test))


# In[64]:


xgb.get_params()


# In[65]:


base_score = [i/10.0 for i in range(0,11)]
booster = ['gbtree', 'gblinear', 'dart']
learning_rate = [i/10.0 for i in range(0,11)]
max_depth = [i for i in range(0,11)]
min_child_weight = [i for i in range(0,11)]
n_estimators = [i for i in range(0,500)]

hyperparemeter_grid = {'n_estimators':n_estimators, 'max_depth':max_depth, 'learning_rate':learning_rate,
                       'min_child_weight':min_child_weight, 'booster':booster, 'base_score':base_score }


# In[66]:


random_cv = RandomizedSearchCV(estimator = xgb, param_distributions = hyperparemeter_grid, cv = 5, n_iter = 50, 
                               scoring = 'neg_mean_absolute_error', n_jobs = 4, verbose = 5, return_train_score = True, 
                               random_state = 42)


# In[67]:


random_cv.fit(X_train, y_train)


# In[68]:


random_cv.best_estimator_


# In[69]:


xgb_tuned = xgboost.XGBRegressor(base_score=0.0, booster='gbtree', callbacks=None,
                                 colsample_bylevel=None, colsample_bynode=None,
                                 colsample_bytree=None, early_stopping_rounds=None,
                                 enable_categorical=False, eval_metric=None, feature_types=None,
                                 gamma=None, gpu_id=None, grow_policy=None, importance_type=None,
                                 interaction_constraints=None, learning_rate=0.1, max_bin=None,
                                 max_cat_threshold=None, max_cat_to_onehot=None,
                                 max_delta_step=None, max_depth=7, max_leaves=None,
                                 min_child_weight=8, monotone_constraints=None,
                                 n_estimators=355, n_jobs=None, num_parallel_tree=None,
                                 predictor=None, random_state=None)


# In[70]:


xgb_tuned.fit(X_train, y_train)


# In[71]:


print("Accuracy score:", xgb_tuned.score(X_test, y_test))


# ## AdaBoost Regression

# In[72]:


from sklearn.ensemble import AdaBoostRegressor

abt = AdaBoostRegressor()
abt.fit(X_train,y_train)


# In[73]:


print("Accuracy score:", abt.score(X_test, y_test))


# In[74]:


abt.get_params()


# In[75]:


n_estimators = [i for i in range(0,100)]
learning_rate = [i/10.0 for i in range(0,11)]
random_state = [i for i in range(0,11)]
loss = ['linear', 'square', 'exponential']

hyperparemeter_grid = {'n_estimators':n_estimators, 'learning_rate':learning_rate, 'random_state':random_state,
                       'loss':loss}


# In[76]:


random_cv = RandomizedSearchCV(estimator = abt, param_distributions = hyperparemeter_grid, cv = 5, n_iter = 50, 
                               scoring = 'neg_mean_absolute_error', n_jobs = 4, verbose = 5, return_train_score = True, 
                               random_state = 42)


# In[77]:


random_cv.fit(X_train, y_train)


# In[78]:


random_cv.best_params_


# In[79]:


abt_tuned = AdaBoostRegressor(random_state = 6, n_estimators = 96, loss = "square", learning_rate = 0.6)


# In[80]:


abt_tuned.fit(X_train,y_train)


# In[81]:


print("Accuracy score:", abt_tuned.score(X_test, y_test))


# ## CatBoost Regression

# In[82]:


from catboost import CatBoostRegressor

cbt = CatBoostRegressor()
cbt.fit(X_train, y_train)


# In[83]:


print("Accuracy score:", cbt.score(X_test, y_test))


# In[84]:


cbt.get_all_params()


# In[85]:


depth = [i for i in range(0,17)]
iterations = [i for i in range(0,1000)]
learning_rate = [i/10.0 for i in range(0,11)]
l2_leaf_reg = [i for i in range(0,11)]

hyperparemeter_grid = {'depth':depth, 'iterations':iterations, 'learning_rate':learning_rate,'l2_leaf_reg':l2_leaf_reg}


# In[86]:


random_cv = RandomizedSearchCV(estimator = cbt, param_distributions = hyperparemeter_grid, cv = 5, n_iter = 50, 
                               scoring = 'neg_mean_absolute_error', n_jobs = 4, verbose = 5, return_train_score = True, 
                               random_state = 42)


# In[87]:


random_cv.fit(X_train, y_train)


# In[88]:


random_cv.best_params_


# In[89]:


cbt_tuned = CatBoostRegressor(learning_rate = 0.1, l2_leaf_reg = 8, iterations = 546, depth = 5)


# In[90]:


cbt_tuned.fit(X_train, y_train)


# In[91]:


print("Accuracy score:", cbt_tuned.score(X_test, y_test))


# # Stacking Regression Pipeline

# In[92]:


from prettytable import PrettyTable

t = PrettyTable(['Model', 'Accuracy Score'])
t.add_row(['Linear Regression', '%.3f' % lgr.score(X_test, y_test)])
t.add_row(['K-Nearest Neighbour (KNN) Regression', '%.3f' % knn_tuned.score(X_test, y_test)])
t.add_row(['Decision Tree Regression', '%.3f' % dtr_tuned.score(X_test, y_test)])
t.add_row(['Random Forest Regression', '%.3f' % rfr_tuned.score(X_test, y_test)])
t.add_row(['XGBoost Regression', '%.3f' % xgb_tuned.score(X_test, y_test)])
t.add_row(['AdaBoost Regression', '%.3f' % abt_tuned.score(X_test, y_test)])
t.add_row(['CatBoost Regression', '%.3f' % cbt_tuned.score(X_test, y_test)])
print(t)


# In[93]:


from sklearn.ensemble import StackingRegressor
from sklego.linear_model import LADRegression
from sklearn.pipeline import Pipeline, make_pipeline

estimators = [("pipe_rfr", rfr_tuned), ("pipe_xgb", xgb_tuned), ("pipe_cbt", cbt_tuned)]
stacking_regressor = StackingRegressor(estimators = estimators, final_estimator = LADRegression(alpha = 0.001))


# In[94]:


final_pipe = Pipeline(steps = [('stacking_regressor', stacking_regressor)])
final_pipe


# In[95]:


stacked_regressor =  final_pipe.fit(X_train, y_train)


# # Model Evaluation

# In[96]:


print("Accuracy Score:", stacked_regressor.score(X_test, y_test))


# In[97]:


print("Adjusted R-Squared Score:", 1 - (1-stacked_regressor.score(X, y))*(len(y)-1)/(len(y)-X.shape[1]-1))


# In[98]:


y_pred = stacked_regressor.predict(X_test)


# In[99]:


print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[100]:


Mod

