#!/usr/bin/env python
# coding: utf-8

# # Credit Card Applications
# A notebook that looks to automate the process of credit card applications (time is money). Going to focus mainly on the classification algorithms so not too much analysis.

# ## 1. Load & Import Data

# In[2]:


import pandas as pd
import numpy as np

cc_apps = pd.read_csv('datasets/cc_approvals.data', header=None)

print(cc_apps.head())


# ## 2. Inspect Applications
# 
# The probable features in a typical credit card application are <code>Gender</code>, <code>Age</code>, <code>Debt</code>, <code>Married</code>, <code>BankCustomer</code>, <code>EducationLevel</code>, <code>Ethnicity</code>, <code>YearsEmployed</code>, <code>PriorDefault</code>, <code>Employed</code>, <code>CreditScore</code>, <code>DriversLicense</code>, <code>Citizen</code>, <code>ZipCode</code>, <code>Income</code> and finally the <code>ApprovalStatus</code>. This gives us a pretty good starting point, and we can map these features with respect to the columns in the output.   </p>
# <p>As we can see from our first glance at the data, the dataset has a mixture of numerical and non-numerical features. This can be fixed with some preprocessing, but before we do that, let's learn about the dataset a bit more to see if there are other dataset issues that need to be fixed.</p>

# In[3]:


cc_apps_description = cc_apps.describe()
print(cc_apps_description)

print("\n")

cc_apps_info = cc_apps.info()
print(cc_apps_info)

print("\n")
print(cc_apps.tail())


# ## 3. Filling Missing Values
# 
# Missing values in the dataset are marked with a '?'

# In[4]:


cc_apps = cc_apps.replace('?', np.nan)


# In[5]:


cc_apps.fillna(cc_apps.mean(), inplace=True)

cc_apps.isnull().sum()


# The missing values have been sorted for numeric columns, now for categorical columns.

# In[6]:


for col in cc_apps.columns:
    # Check if the column is of object type
    if cc_apps[col].dtype == 'object':
        # Impute with the most frequent value
        cc_apps = cc_apps.fillna(cc_apps[col].value_counts().index[0])

# Count the number of NaNs in the dataset and print the counts to verify
cc_apps.isna().sum()


# ## 4. Preprocessing Data

# In[7]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in cc_apps.columns:
    # Compare if the dtype is object
    if cc_apps[col].dtype=='object':
    # Use LabelEncoder to do the numeric transformation
        cc_apps[col]=le.fit_transform(cc_apps[col])


# In[8]:


from sklearn.model_selection import train_test_split
# Drop the features 11 and 13 and convert the DataFrame to a NumPy array
cc_apps = cc_apps.drop([11, 13], axis=1)
cc_apps = cc_apps.to_numpy()

X,y = cc_apps[:,0:13] , cc_apps[:,13]

X_train, X_test, y_train, y_test = train_test_split(X,
                                y,
                                test_size=0.33,
                                random_state=42)


# In[9]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX_train = scaler.fit_transform(X_train)
rescaledX_test = scaler.transform(X_test)


# ## 5. Logistic Regression
# 
# Essentially a classification task to decide whether applciation will be approved or not

# In[14]:


from sklearn.linear_model import LogisticRegression
# Instantiate a LogisticRegression classifier with default parameter values
logreg = LogisticRegression()

logreg.fit(rescaledX_train, y_train)


# In[15]:


y_pred = logreg.predict(rescaledX_test)


# In[19]:


from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix


# In[18]:


print("Accuracy of logistic regression classifier: ", logreg.score(rescaledX_test, y_test))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[20]:


plot_confusion_matrix(logreg, rescaledX_test, y_test)


# Do some grid search analysis

# In[25]:


from sklearn.model_selection import GridSearchCV

tol = [0.01,0.001,0.0001]
max_iter = [250, 300, 400]

param_grid = dict(tol=tol, max_iter=max_iter)


# In[26]:


grid_model = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5)

rescaledX = scaler.fit_transform(X)

grid_model_result = grid_model.fit(X, y)

best_score, best_params = grid_model_result.best_score_, grid_model_result.best_params_
print("Best: %f using %s" % (best_score, best_params))


# ## 6. Random Forest

# In[27]:


from sklearn.ensemble import RandomForestClassifier


# In[28]:


rf_clf = RandomForestClassifier()


# In[30]:


rf_clf.fit(rescaledX_train, y_train)


# In[31]:


rf_preds = rf_clf.predict(X_test)


# In[32]:


print("Accuracy of random forest classifier: ", rf_clf.score(rescaledX_test, y_test))
print(confusion_matrix(y_test, rf_preds))
print(classification_report(y_test, rf_preds))


# Slightly higher accuracy but f1-score & recall are slightly lower

# ## 7. XGBoost

# In[33]:


import xgboost as xgb


# In[34]:


churn_dmatrix = xgb.DMatrix(data=rescaledX_test, label=y_test)


# In[36]:


params = {"objective":"binary:logistic", 'max_depth':4}


# In[37]:


cv_results = xgb.cv(dtrain = churn_dmatrix, params=params, nfold=4, num_boost_round=10, metrics='error', as_pandas=True)


# In[38]:


print("Accuracy: %f" %((1-cv_results["test-error-mean"]).iloc[-1]))


# Add some tuning to the xgboost model

# In[39]:


gbm_param_grid = {
    'learning_rate': [0.1, 0.01, 0.001],
    'n_estimators': [10, 20, 50, 100],
    'subsample': [0.3, 0.5, 0.7, 0.9],
    'max_depth': [4,8,12]
}


# In[41]:


gbm = xgb.XGBClassifier(params={"objective":"binary:logistic"})


# In[43]:


grid_gbm = GridSearchCV(estimator=gbm,param_grid=gbm_param_grid,
                             
                        cv=4, verbose=1)


# In[44]:


grid_gbm.fit(rescaledX_train, y_train)


# In[46]:


print("Best parameters found: ",grid_gbm.best_params_)
print("Highest accuracy found: ", grid_gbm.best_score_)


# In[ ]:




