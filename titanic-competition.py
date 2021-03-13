
# %% Imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta
from sqlalchemy import create_engine
import sqlalchemy
import os
import math
import time
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
numericals = ['Age','SibSp','Parch','Fare']
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




# %%
