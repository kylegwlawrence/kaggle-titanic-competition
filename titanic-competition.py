
# %% Imports
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
#from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys

#get script directory
here = os.path.dirname(os.path.abspath(__file__))

#%% a to z wireless hierarchy only
train_df = pd.read_csv('train.csv')

# %% inspect passengerid column
train_df.info()
train_df.describe()
# next - split into categorical and numerical and see how each correlates with survival
#%%
categoricals = ['Sex','Pclass','Embarked']
numericals = ['Age','SibSp','Parch','Fare','Survived']
# to check correlation of categoricals, plot frequency of survival (or percent of total that survived?) on bar graph
#%% print a bar plot of survival rates for each category 
def cat_plots(cat):
    survived = train_df[train_df['Survived'] == 1][['Survived', cat]
        ].groupby([cat]).agg({'Survived': 'count'})
    not_survived = train_df[train_df['Survived'] == 0][['Survived', cat]
        ].groupby([cat]).agg({'Survived': 'count'})
    survive_rate = survived/(not_survived+survived)
    
    ax = survive_rate[['Survived']].plot(
        kind='bar', title="{} Survival rate".format(cat), figsize=(5, 5), legend=False, fontsize=12)
    ax.set_xlabel("{}".format(cat), fontsize=12)
    ax.set_ylabel("% passengers in each {} that survived".format(cat), fontsize=12)
    plt.show()

for cat in categoricals:
    cat_plots(cat)

#%% print a corerlation matrix to show relationship between survival and predictor variables and multicollinearity
corrMatrix = train_df[numericals].corr()
plt.figure(figsize=(5, 5))
sn.heatmap(corrMatrix, annot=True, linewidths=1)
plt.show()

# %%bin the ages and plot survival rate
def histogram_age(cut_num_bins):
    train_df['bin_age_{}'.format(cut_num_bins)] = pd.cut(train_df['Age'], bins=cut_num_bins)
    
    survived = train_df[train_df['Survived'] == 1][['Survived', 'bin_age_{}'.format(cut_num_bins)]
    ].groupby(['bin_age_{}'.format(cut_num_bins)]).agg({'Survived': 'count'})
    
    not_survived = train_df[train_df['Survived'] == 0][['Survived', 'bin_age_{}'.format(cut_num_bins)]
        ].groupby(['bin_age_{}'.format(cut_num_bins)]).agg({'Survived': 'count'})
    
    survive_rate = survived/(not_survived+survived)

    ax = survive_rate[['Survived']].plot(
        kind='bar', title="Age Histogram Survival rate", figsize=(8, 5), legend=False, fontsize=12)
    ax.set_xlabel("Age", fontsize=12)
    ax.set_ylabel(
        "% survived", fontsize=12)
    plt.show()
    train_df.drop(['bin_age_{}'.format(cut_num_bins)],axis=1,inplace=True)

# look at histograms with different bin sizes
for n in range(8,9):
    histogram_age(n)

#%%
exploration_results = """
    More likely to survive if female than male. This makes sense given the women and childern first strategy upon evacuation.
    \n
    First class passengers are more likely to survive than second class passengers and second class passengers are more likely to survive than third class passengers. This makes sense given the class priority upon evacuation.
    \n 
    I looked up the definitions of the Embarked codes. C = Cherbourg, Q = Queenstown, S = Southampton. Survival is more likely for passengers from Cherbourg than anywhere else. Survival of passengers from Southampton is least likely. There might be a correlation between embarked location and class.
    \n
    Investigating correlation amongst numericals columns, Fare is the most strongly correlated numerical category with Survived. This means the more a passenger paid, the more likely they were to survive. Parch is also positively correlted with survival although weakly. Age and SibSp are both weakly negatively correlated with Survived.
    \n
    Multicollinearity: there is medium positive correlation between Parch and SibSp. Might make more sense to make an indicator out of this such as is alone, has a child, has a sibling, is with a party (>2 ppl). Another possible feature is to split binary cols for num people in party - 1, 2, 3 or more
    \n
    Histograms of different age bins across survival rate shows that younger passengers are more likely to survive than older passengers in general. The highest survival rate when splitting age across 8 bins is 0-10 years old (~60%) and the lowest survival rate is the last bin, 70 to 80 years old (~20%)
"""
print(exploration_results)

# Data preprocessing

#%%addres missing values
# -drop cabin - majority are missing
train_df.drop(['Cabin'],axis=1,inplace=True)

# %%-missing in Age - impute average age. use .item() to convert from shape(1,) to integer
mean_age = train_df[~train_df['Age'].isna()][['Age']].mean().item()
train_df['Age'].fillna(value=mean_age,inplace=True)

# %% Impute C for missing embarked values snice majority came from C. there are only two missing values
train_df['Embarked'].fillna(value='C', inplace=True)

# %%missing values for fare - use average fare by class
fare_mean_by_class = train_df[train_df['Fare'] >0][['Fare', 'Pclass']
    ].groupby(['Pclass']).agg({'Fare': 'mean'})
dict_fare_mean_by_class = fare_mean_by_class.to_dict()['Fare']
#%% add mean by class to zero fares
def impute_mean_class_fare(Pclass, Fare):
    if Fare==0:
        if Pclass==1:
            return dict_fare_mean_by_class[1]
        elif Pclass==2:
            return dict_fare_mean_by_class[2]
        elif Pclass == 3:
            return dict_fare_mean_by_class[3]
    elif Fare!=0:
        return Fare

train_df['Fare'] = train_df.apply(
    lambda x: impute_mean_class_fare(
        x['Pclass']
        , x['Fare']), axis=1)
# %% data transformation - Fare
train_df['logFare'] = np.log(train_df['Fare'])

# %%compare Fare and logFare to explain reason for transformation
hist_fare = train_df[['Fare']].hist(bins=25)
hist_log_fare = train_df[['logFare']].hist(bins=25)

# feature engineering  -create some features from existing features
# 1. is_alone (Parch and Sibsp = 0)
# 2. is_party (sum of Parch and Sibsp>1)
# 3. class X logFare = will put more weight on higher prices in classes 2 and 3

#%% is alone
def is_alone(Parch, SibSp):
    if Parch+SibSp==0:
        return 1
    else:
        return 0
train_df['IsAlone'] = train_df.apply(
    lambda x: is_alone(
        x['Parch'], x['SibSp']), axis=1)
cat_plots('IsAlone')

#%% is with party
def is_with_party(Parch, SibSp):
    if Parch+SibSp > 1:
        return 1
    else:
        return 0
train_df['IsWithParty'] = train_df.apply(
    lambda x: is_with_party(
        x['Parch'], x['SibSp']), axis=1)
cat_plots('IsWithParty')

train_df['PclassxlogFare'] = train_df['Pclass']*train_df['logFare']
hist_Pclass_times_log_fare = train_df[['PclassxlogFare']].hist(bins=25)
# %% one hot encode the categorical variables
one_hot_cats = ['Sex','Embarked','Pclass']
train_df = pd.get_dummies(train_df, columns=one_hot_cats, prefix=one_hot_cats)
# drop some redundant columns. These will be the "base" vars = 1. Also with isalone and iswithparty, base assumption is they are in party of 2
train_df.drop(labels=['Sex_female','Embarked_S','Pclass_1'],axis=1,inplace=True)

# %% model training - split data
X = train_df[['Age','IsAlone','IsWithParty','PclassxlogFare','Sex_male','Embarked_C','Embarked_Q','Pclass_2','Pclass_3']]
y = train_df['Survived']
X_train, X_test, Y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=101)

#%%1. logistic regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
print('Logistic regression score = {}'.format(acc_log))
confusion_matrix(y_test, Y_pred)



#----------------code below is not mine --------------------#
#%% 2. support vector machine
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
print('Support vector machine score = {}'.format(acc_svc))
confusion_matrix(y_test, Y_pred)

# %% 3. k nearesty neaighbour
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
print('KNN score = {}'.format(acc_knn))
confusion_matrix(y_test, Y_pred)

# %% 4. gaussian naive bayes
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
print('Gaussian Naive Bayes score = {}'.format(acc_gaussian))
confusion_matrix(y_test, Y_pred)

# %% 5. perceptron
perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
print('Perceptron score = {}'.format(acc_perceptron))
confusion_matrix(y_test, Y_pred)

# %% 6. linear svc
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
print('Linear SVC score = {}'.format(acc_linear_svc))
confusion_matrix(y_test, Y_pred)

# %% 7. stochastic gradient
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
print('Stochastic gradient score = {}'.format(acc_sgd))
confusion_matrix(y_test, Y_pred)

# %% 8. Decision tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
print('Decision tree score = {}'.format(acc_decision_tree))
confusion_matrix(y_test, Y_pred)

# %% 9. random forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print('Random forest score = {}'.format(acc_random_forest))
confusion_matrix(y_test, Y_pred)

# %% put models scores in a dataframe
models = pd.DataFrame({'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
    'Random Forest', 'Naive Bayes', 'Perceptron', 'Stochastic Gradient Decent',
    'Linear SVC', 'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, acc_random_forest, acc_gaussian, acc_perceptron,
    acc_sgd, acc_linear_svc, acc_decision_tree]})

models.sort_values(by='Score', ascending=False, ignore_index=True)

#%% Create a list which contains classifiers
classifiers = []
classifiers.append(LogisticRegression())
classifiers.append(SVC())
classifiers.append(KNeighborsClassifier(n_neighbors=5))
classifiers.append(GaussianNB())
classifiers.append(Perceptron())
classifiers.append(LinearSVC())
classifiers.append(SGDClassifier())
classifiers.append(DecisionTreeClassifier())
classifiers.append(RandomForestClassifier())

len(classifiers)
# %%Create a list which contains cross validation results for each classifier
cv_results = []
for classifier in classifiers:
    cv_results.append(cross_val_score(classifier, X_train,
        Y_train, scoring='accuracy', cv=10))
# %% k-fold cross validation
cv_mean = []
cv_std = []
for cv_result in cv_results:
    cv_mean.append(cv_result.mean())
    cv_std.append(cv_result.std())
    
cv_res = pd.DataFrame({'Cross Validation Mean': cv_mean, 'Cross Validation Std': cv_std, 'Algorithm': [
    'Logistic Regression', 'Support Vector Machines', 'KNN', 'Gausian Naive Bayes', 'Perceptron', 'Linear SVC', 'Stochastic Gradient Descent', 'Decision Tree', 'Random Forest']})
cv_res.sort_values(by='Cross Validation Mean',
    ascending=False, ignore_index=True)

# %%
