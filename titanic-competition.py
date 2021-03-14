
# %% Imports
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys
start_time = time.time()

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

#%%
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
for n in range(3,9):
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
    Histograms of different age bins across survival rate shows that younger passengers are more likely to survive than older passengers in general. The highest survival rate when splitting age across 8 bins is 0-10 years old (~60%) and the lowest survival rate is the last bin, 70 to 80 years old (~20%)
"""




# %%
