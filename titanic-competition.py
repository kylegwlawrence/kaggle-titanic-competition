
# %% Imports
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

# %% notes
# -the Kaggle data preview does a good job of describing each column

# -survived is the output (y)
# - passengerid is an index and doesn't matter for testing, only submission. same with name and ticket
# -cabin is missing the most values. Don't test specific cabin yet, just test p class
# -age is missing some. not sure how to impute this. try imputing average age first, then average or mean age by class/sex or combo
# -should age be bucketed as a categorical variable? Try different methods
# -SibSp - Number of siblings/spouses aboard
# -Parch - number of parents/children aboard
# -Fare might correlate with Class
# -Embarked has values S, C, Q or nan

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
    Multicollinearity: there is medium positive correlation between Parch and SibSp. Might make more sense to make an indicator out of this such as is alone, has a child, has a sibling, is with a party (>2 ppl)
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

