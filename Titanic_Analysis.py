import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Some of the questions that we try to answer are
# 1.) Who were the passengers on the Titanic? (Ages,Gender,Class,..etc)
# 2.) What deck were the passengers on and how does that relate to their class?
# 3.) Where did the passengers come from?
# 4.) Who was alone and who was with family?
#
# Then we'll dig deeper, with a broader question:
#
# 5.) What factors helped someone survive the sinking?


# To answer the first question
# 1.) Who were the passengers on the Titanic? (Ages,Gender,Class,..etc)
titanic_df = pd.read_csv('train.csv')
print(titanic_df.columns)
sns.factorplot('Sex',data=titanic_df,kind="count",hue='Pclass')
plt.title("Understanding the sex of the people who travelled in various classes")
sns.factorplot('Pclass',data=titanic_df,kind="count",hue='Sex')
plt.title("Change in the axis to visualise better")
print()
def male_female_child(passenger):
    age,sex = passenger
    if age<16:
        return 'child'
    else:
        return sex


titanic_df['person'] = titanic_df[['Age','Sex']].apply(male_female_child,axis=1)
print(titanic_df[0:10])
sns.factorplot('Pclass',data=titanic_df,kind='count',hue='person')
print()
#to check for age of the people on overall we create histogram
figure = plt.figure()
figure.add_subplot(1,1,1)
titanic_df['Age'].hist(bins=70,rwidth=0.75)
plt.title("Age histogram to understand the overall passengers")

print("The mean age of the passengers is ",titanic_df['Age'].mean())
print("The frequency of the people travelling with different sex are \n",titanic_df['person'].value_counts())

#To analyse further lets use kde plot
fig = sns.FacetGrid(titanic_df,hue='Sex',aspect=4)
fig.map(sns.kdeplot,'Age',shade=True)
oldest = titanic_df['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()

#To include the children
fig = sns.FacetGrid(titanic_df,hue='person',aspect=4)
fig.map(sns.kdeplot,'Age',shade=True)
oldest = titanic_df['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()

#To study the effect of the same on the classes
fig = sns.FacetGrid(titanic_df,hue='Pclass',aspect=4)
fig.map(sns.kdeplot,'Age',shade=True)
oldest = titanic_df['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()

# To answer the second question
# 2.) What deck were the passengers on and how does that relate to their class?
#To study the cabin deck remove the null values

deck = titanic_df['Cabin'].dropna()
print(deck.head())
levels = []
for level in deck:
    levels.append(level[0])
cabin_df = DataFrame(levels)
cabin_df.columns = ['Cabin']
sns.factorplot('Cabin',data=cabin_df,kind='count',palette='winter_d')
#cleaning the data to get rid of the T value that makes no sense
cabin_df = cabin_df[cabin_df != 'T']
sns.factorplot('Cabin',data=cabin_df,kind='count',palette='summer')

# To answer the third question
# 3.) Where did the passengers come from?
#To understand from where the passengers came from lets plot the embarked column
sns.factorplot('Embarked',data=titanic_df,kind='count',hue='Pclass',order=['C','Q','S'])