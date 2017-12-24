##############################################################################
#
#  Introduction
#  -------------
#  Logistic Regression is a Machine Learning classification algorithm that is
#  used to predict the probability of a categorical dependent variable.
#  In logistic regression, the dependent variable is a binary variable that
#  contains data coded as 1 (yes, success, etc.) or 0 (no, failure, etc.). 
#  In other words, the logistic regression model predicts P(Y=1) as
#  a function of X.

#  Logistic Regression Assumptions
#  --------------------------------
#   * Binary logistic regression requires the dependent variable to be binary.
#   * For a binary regression, the factor level 1 of the dependent variable
#     should represent the desired outcome.
#   * Only the meaningful variables should be included.
#   * The independent variables should be independent of each other. 
#     That is, the model should have little or no multicollinearity.
#   * The independent variables are linearly related to the log odds.
#   * Logistic regression requires quite large sample sizes.
#
#  Goal
#  -----
#  The classification goal is to predict whether the client will
#  subscribe (1/0) to a term deposit (variable y).
#
#  Reference: https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
#
##############################################################################

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt 
plt.rc("font", size=14)

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn import metrics


import statsmodels.api as sm

import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


"""
Input variables
----------------

    - age (numeric)
    - job : type of job (categorical: 'admin', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown')
    - marital : marital status (categorical: 'divorced', 'married', 'single', 'unknown')
    - education (categorical: 'basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 'professional.course', 'university.degree', 'unknown')
    - default: has credit in default? (categorical: 'no', 'yes', 'unknown')
    - housing: has housing loan? (categorical: 'no', 'yes', 'unknown')
    - loan: has personal loan? (categorical: 'no', 'yes', 'unknown')
    - contact: contact communication type (categorical: 'cellular', 'telephone')
    - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
    - day_of_week: last contact day of the week (categorical: 'mon', 'tue', 'wed', 'thu', 'fri')
    - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). The duration is not known before a call is performed, also, after the end of the call, y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model
    - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
    - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
    - previous: number of contacts performed before this campaign and for this client (numeric)
    - poutcome: outcome of the previous marketing campaign (categorical: 'failure', 'nonexistent', 'success')
    - emp.var.rate: employment variation rate - (numeric)
    - cons.price.idx: consumer price index - (numeric)
    - cons.cons.idx: consumer confidence index - (numeric)
    - euribor3m: euribor 3 month rate (numeric)
    - nr.employed: number of employees - (numeric)

Predict variable (desired target):
------------------------------------

    - y: has the client subscribed a term deposit? (binary: 1, means Yes, 0 means No)
"""

def main():

    data = pd.read_csv('banking.csv', header=0)
    data = data.dropna()
    print (data.shape)
    print (list(data.columns) )

    # The education column of the dataset has many categories and we need to reduce the categories for a better modelling
    data["education"].unique()

    # Let us group 'basic.4y', 'basic.9y' and 'basic.6y' together and call them 'Basic'.
    data['education']=np.where(data['education'] =='basic.9y', 'Basic', data['education'])
    data['education']=np.where(data['education'] =='basic.6y', 'Basic', data['education'])
    data['education']=np.where(data['education'] =='basic.4y', 'Basic', data['education'])

    data["education"].unique()

    print data.head()

    # Data exploration
    print "value counts:", data['y'].value_counts()

    sns.countplot(x='y', data=data, palette='hls')
    #plt.show()
    plt.savefig('count_plot')

    data.groupby('y').mean()

    # We can calculate categorical means for other categorical variables such as education and marital status to get a more detailed sense of our data.
    data.groupby('job').mean()
    data.groupby('marital').mean()
    data.groupby('education').mean()

    print data.head()

    # Visualizations
    #%matplotlib inline
    pd.crosstab(data.job,data.y).plot(kind='bar')
    plt.title('Purchase Frequency for Job Title')
    plt.xlabel('Job')
    plt.ylabel('Frequency of Purchase')
    plt.savefig('purchase_fre_job')
    #plt.show()

    # The frequency of purchase of the deposit depends a great deal on the job title.
    # Thus, the job title can be a good predictor of the outcome variable.

    # Marital Status vs Purchase
    table=pd.crosstab(data.marital,data.y)
    table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
    plt.title('Stacked Bar Chart of Marital Status vs Purchase')
    plt.xlabel('Marital Status')
    plt.ylabel('Proportion of Customers')
    plt.savefig('mariral_vs_pur_stack')
    #plt.show()

    # Education vs Purchase
    table=pd.crosstab(data.education,data.y)
    table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
    plt.title('Stacked Bar Chart of Education vs Purchase')
    plt.xlabel('Education')
    plt.ylabel('Proportion of Customers')
    plt.savefig('edu_vs_pur_stack')

    # Day of Week
    # Day of week may not be a good predictor of the outcome.
    pd.crosstab(data.day_of_week,data.y).plot(kind='bar')
    plt.title('Purchase Frequency for Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Frequency of Purchase')
    plt.savefig('pur_dayofweek_bar')

    # Month
    pd.crosstab(data.month,data.y).plot(kind='bar')
    plt.title('Purchase Frequency for Month')
    plt.xlabel('Month')
    plt.ylabel('Frequency of Purchase')
    plt.savefig('pur_fre_month_bar')

    # Age
    data.age.hist()
    pd.crosstab(data.age,data.y).plot(kind='bar')
    plt.title('Histogram of Age')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.savefig('hist_age')
    #plt.show()

    # Most of the customers of the bank in this dataset are in the age range of 30-40
    pd.crosstab(data.poutcome,data.y).plot(kind='bar')
    plt.title('Purchase Frequency for Poutcome')
    plt.xlabel('Poutcome')
    plt.ylabel('Frequency of Purchase')
    plt.savefig('pur_fre_pout_bar')

    #Create dummy variables
    #That is variables with only two values, zero and one.

    cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
    for var in cat_vars:
        cat_list='var'+'_'+var
        cat_list = pd.get_dummies(data[var], prefix=var)
        data1=data.join(cat_list)
        data=data1
    cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
    data_vars=data.columns.values.tolist()
    to_keep=[i for i in data_vars if i not in cat_vars]

    print data.head()

    data_final=data[to_keep]
    print "Final Columns: ", data_final.columns.values

    data_final_vars=data_final.columns.values.tolist()
    y=['y']
    X=[i for i in data_final_vars if i not in y]

    # Feature Selection
    """
    Recursive Feature Elimination (RFE) is based on the idea to repeatedly construct a model and choose either the best or worst performing feature, setting the feature aside and then repeating the process with the rest of the features. This process is applied until all features in the dataset are exhausted. The goal of RFE is to select features by recursively considering smaller and smaller sets of features.
    """
    logreg = LogisticRegression()
    print "Performing Recursive Feature Elimination..."
    rfe = RFE(logreg, 18)
    rfe = rfe.fit(data_final[X], data_final[y] )
    print(rfe.support_)
    print(rfe.ranking_)


    cols=["previous", "euribor3m", "job_blue-collar", "job_retired", "job_services", "job_student", "default_no", 
      "month_aug", "month_dec", "month_jul", "month_nov", "month_oct", "month_sep", "day_of_week_fri", "day_of_week_wed", 
      "poutcome_failure", "poutcome_nonexistent", "poutcome_success"] 
    X=data_final[cols]
    y=data_final['y']

    # Implementing the model
    #logit_model=sm.Logit(y,X)
    #result=logit_model.fit()
    #print(result.summary())

    # Logistic Regression Model Fitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)

    # Predicting the test set results and calculating the accuracy
    y_pred = logreg.predict(X_test)
    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

if __name__ == "__main__":
    main()
