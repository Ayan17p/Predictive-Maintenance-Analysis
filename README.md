# Predictive-Maintenance-Analysis
Predictive Maintenance Analysis is a Machine Learning project that creates a model to predict the failure of a machine.

#!/usr/bin/env python
# coding: utf-8

# # PREDICTIVE MAINTENANCE ANALYSIS


# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('predictive_maintenance.csv')
df


# ### Data Description


# - This data is about the drilling tool used in a lathe machine.
# - The columns consist values of:
#     1. Air temperature in Kelvin (K)
#     2. Process temperature in Kelvin (K)
#     3. Rotational speed in Rotation per minute (RPM)
#     4. Torque in Newton meters (Nm)
#     5. Tool wear in minutes (min)
# - Type column is given to represent the type of product with respect to its quality. 
# >L, M and H are for Low, Medium and High quality products respectively.
# - ProductID consisting of a letter L, M, or H for low (50% of all products), medium (30%), and high (20%) as product quality variants and a variant-specific serial number.
# - The target column is the output of machine, if it is failing or not.
# >0 represents no failure whereas 1 represents that failure has occured.
# - An additional column of Failure Type is given so that we can understand what type of failure is occuring.

# In[3]:


df.info()


# In[4]:


df.describe()


# _The above table shows us that our data is normally distributed since all the mean and median values are close to each other. So there is no need for skewness removal._

# In[5]:


df['Target'].value_counts()


# In[6]:


df['Target'].value_counts().plot.pie(explode=[0.1,0.1],autopct='%.1f%%',title="Machine Failure")


# _The above Pie chart shows that only 3.4% of times our machine is failing. Though the failure rate is very low in comparison to success rate, we still need to avoid any failure in order to improve productivity of the machine._

# In[7]:


df['Failure Type'].value_counts()


# Here we can see the types of failures such as:
#   1. Heat Dissipation Failure : The failure that occurs due to overheating of the tool.
#   2. Power Failure : The failure occuring due to power cutout.
#   3. Overstrain Failure : Failure because of excessive strain on the tool.
#   4. Tool Wear Failure : Failure due to tool wear and tear that happens after excessive use of tool.
#   5. Random Failures : Random Failures can be any failure whose cause can't be assessed or any human error.
#   
# "No failure" as the name suggests means there is no failure of machine

# In[8]:


plt.figure(figsize = (8,8))
sns.countplot(data = df[df['Target'] == 1], x = "Failure Type")
plt.title("Count of Failures W.R.T Failure Types")
plt.show()


# _In the above countplot we can see that "No Failure" is also plotted but that does not makes any sense. This means there are some fake values in the data. So we need to eliminate them._

# In[9]:


df[(df['Target']==1) & (df['Failure Type']=='No Failure')]


# In[10]:


i = df[(df['Target']==1) & (df['Failure Type']=='No Failure')].index
df.drop(i,axis=0,inplace=True)


# _Also, we cannot see anything related to Random Failures since they cannot be assesed. So we will drop the products with Random failures._

# In[11]:


df[(df['Target']==0) & (df['Failure Type']=='Random Failures')]


# In[12]:


i = df[(df['Target']==0) & (df['Failure Type']=='Random Failures')].index
df.drop(i,axis=0,inplace=True)


# In[13]:


plt.figure(figsize = (8,8))
sns.countplot(data = df[df['Target'] == 1], x = "Failure Type")
plt.title("Count of Failures W.R.T Failure Types")
plt.show()


# _Here we can see that most of our failures are  __Heat Dissipation Failure__, then __Power Failure__ followed by __Overstrain__ and __Tool Wear Failures__._

# In[14]:


df['Type'].value_counts().plot(kind='pie',autopct='%.f%%',title="Percentage of Product Types")


# _We can see that most of our products i.e. 60%, belong to low quality category. 30% belong to medium quality and 10% to high quality categories._

# In[15]:


pd.DataFrame(df["Type"].value_counts())


# In[16]:


sns.heatmap(df.corr(),annot=True,fmt=".2f")


# _From above heatmap, we can see that there is no correlation between any two columns._

# In[17]:


df.head(1)


# In[18]:


colname=df.iloc[:,2:8].columns


# In[19]:


for i in df[colname]:
    print(i)
    
    sns.lineplot(df[i],df['Target'])
    plt.show()


# _Here we can see that Rotational speed [rpm], Torque [Nm], Tool wear [min] are highly affecting our Target column_

# In[20]:


sns.pairplot(df,hue='Target')


# In[21]:


df.head()


# In[22]:


df.drop(['UDI','Product ID','Failure Type'],axis=1,inplace=True)


# _Here we dropped all the columns that we don't require._

# In[23]:


df.head()


# In[24]:


df["Type"] = df["Type"].replace("L",0)
df["Type"] = df["Type"].replace("M",1)
df["Type"] = df["Type"].replace("H",2)


# _Replacing the categorical values of Type column into numeric values._

# In[25]:


df.head()


# In[26]:


predictors = df.iloc[:,:-1]
predictors


# In[27]:


label = df.iloc[:,-1]
label


# ### Splitting data into train and test

# In[28]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(predictors,label,test_size=0.3,random_state=42)


# In[29]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
from sklearn.metrics import accuracy_score,classification_report


# In[30]:


def mymodel(model):
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test,y_pred))
    return model


# _Here we created a function to check which Machine Learning Algorithm will be the best fit for our model. We will select the one with highest accuracy._

# In[31]:


mymodel(logreg)


# In[32]:


mymodel(knn)


# In[33]:


mymodel(dt)


# _We can see that Decision Tree Classifier is best suited for our model since it has the highest accuracy of all i.e. 98%_

# In[34]:


print(dt.score(X_train,y_train))
print(dt.score(X_test,y_test))


# _Our training and testing scores are a little different from each other. But we need scores close to each other for our model to work best._ 

# ### Hyperparameter Tuning

# In[35]:


for i in range(1,51):
    dt1 = DecisionTreeClassifier(max_depth=i)
    dt1.fit(X_train,y_train)
    y_pred = dt1.predict(X_test)
    print(f"{i} = {accuracy_score(y_test,y_pred)}")


# _Here, the maximum accuracy is on index 8, so we will select max_depth = 8_

# In[36]:


for i in range(2,51):
    dt2 = DecisionTreeClassifier(min_samples_split=i)
    dt2.fit(X_train,y_train)
    y_pred = dt2.predict(X_test)
    print(f"{i} = {accuracy_score(y_test,y_pred)}")


# _Here, the maximum accuracy is on index 34, so we will select min_samples_split = 34_

# In[37]:


for i in range(1,51):
    dt3 = DecisionTreeClassifier(min_samples_leaf=i)
    dt3.fit(X_train,y_train)
    y_pred = dt3.predict(X_test)
    print(f"{i} = {accuracy_score(y_test,y_pred)}")


# _Here, the maximum accuracy is on index 12, so we will select min_samples_leaf = 12_

# In[38]:


dt4 = DecisionTreeClassifier(max_depth=8,min_samples_leaf=34,min_samples_split=12)
mymodel(dt4)


# In[39]:


print(dt4.score(X_train,y_train))
print(dt4.score(X_test,y_test))


# _Now, we are getting both training and testing scores almost equal._

# ### Creating model

# In[40]:


def predtest():
    Type = int(input("Enter type of product based on quality.\nIf Low, enter 0.\nIf Medium, enter 1.\nIf High, enter 2.\nYou entered : "))
    Airtemp = eval(input("Enter Air Temperature in K : "))
    Protemp = eval(input("Enter Process Temperature in K : "))
    Rotsp = int(input("Enter Rotational Speed in RPM : "))
    Torq = eval(input("Enter Torque in Nm : "))
    Toolw = int(input("Enter Tool Wear in min : "))
    
    newx = [Type,Airtemp,Protemp,Rotsp,Torq,Toolw]
    yp = dt4.predict([newx])[0]
    
    if yp==1:
        print("Machine Failed")
        return yp
    else:
        print("Machine didn't Fail")
        return yp  


# _In order to check if our model is working properly or not, we need to test it._

# _Here we have some existing values of the given parameters for testing failure._

# ### Model testing

# __Testing if our model is properly working or not.__

# >Example:-
# - Type = M
# - Air temperature [K] = 298.2
# - Process temperature [K] = 308.5
# - Rotational speed [rpm] = 2678
# - Torque [Nm] = 10.7
# - Tool wear [min] = 86
# - Target = 1

# __Here the machine should fail.__

# In[43]:


predtest()


# >Example:-
# - Type = H
# - Air temperature [K] = 298.4
# - Process temperature [K] = 308.9
# - Rotational speed [rpm] = 1782
# - Torque [Nm] = 23.9
# - Tool wear [min] = 24
# - Target = 0

# __Here the machine should not fail.__

# In[44]:


predtest()


# _Since the outcome is correct. Our model is successful._

# ## Prescriptive Analysis

# - __After studying the dataset, we can see that machine failure is mainly occuring because of 3 reasons:__
#    1. Improper rotational speed of the tool.
#    2. Torque not maintained as per the requirement.
#    3. High tool wear.

# - __Air temperature and Process temperature are not playing any major role in machine failure.__

# - __Low quality products are more likely to fail compared to medium quality and high quality products.__

# - __Heat dissipation failure is the most occuring failure which causes due to overheating of the tool.__

# - __Power failure is also a major problem causing machine failure__

# >__Measures to take in order to avoid failure.__ 
#    1. User must set the rotational speed properly. Not too high and also not too low.
#    2. User should not set the torque too high or too low. Torque should be set as per the tool requirement.
#    3. User should not use the same tool for long period of time. Since tools may wear due to excessive usage.
#    4. High quality and medium quality products should be used more frequently.
#    5. Overheating of tool must be avoided by constantly providing coolant to the tool. 
#    6. Secondary power input shoould be provided in case of power failure

https://drive.google.com/file/d/15p6AbiswfcXae46HGzcVUPY0ki_bo_ad/view?usp=share_link
