#!/usr/bin/env python
# coding: utf-8

# ### About the Data
1. Day_of_week: The day of the week when the accident occurred.
2.Age_band_of_driver: The age group or band of the driver involved in the accident.
3.Sex_of_driver: The gender of the driver involved in the accident.
4.Educational_level: The educational level of the driver involved in the accident.
5.Vehicle_driver_relation: Relationship of the driver with the vehicle (e.g., owner, renter).
6.Driving_experience: Experience level of the driver in terms of years.
7.Type_of_vehicle: Type of vehicle involved in the accident (e.g., car, truck, motorcycle).
8.Owner_of_vehicle: Ownership status of the vehicle (e.g., self-owned, company-owned).
9.Service_year_of_vehicle: Number of years the vehicle has been in service.
10.Vehicle_movement: Movement or action of the vehicle before or during the accident.
11.Casualty_class: Classification of the casualty (e.g., driver, passenger, pedestrian).
12.Sex_of_casualty: Gender of the casualty involved in the accident.
13.Age_band_of_casualty: Age group or band of the casualty involved in the accident.
14.Casualty_severity: Severity of the casualty (e.g., minor injury, serious injury, fatality).
15.Work_of_casuality: Occupation or work status of the casualty.
16.Fitness_of_casuality: Fitness status of the casualty.
17.Pedestrian_movement: Movement of any pedestrians involved in the accident.
18.Cause_of_accident: The cause or reason for the accident.
19.Accident_severity: Severity of the accident itself.
# # Importing the libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,f1_score,precision_score,recall_score


# In[2]:


#Reading the dataset
df = pd.read_csv('RTA Dataset.csv')


# In[3]:


df.head(5)


# In[4]:


#Checking the information
df.info()

Their are "29" object columns and "2" numerical(Integer) Columns
# In[5]:


#checking for null values
df.isnull().sum()


# In[6]:


print(31-15) #Their are 16 columns with null values present


# In[7]:


#describing the data for numerical data
df.describe()


# In[8]:


# Describing the data including categorical columns
df.describe(include='object')


# In[9]:


sns.countplot(x='Day_of_week', data=df)
plt.title('Accidents by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Number of Accidents')
plt.show()


# In[10]:


plt.figure(figsize=(8, 6))
df['Sex_of_driver'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Distribution of Driver Sex')
plt.ylabel('')
plt.show()


# In[11]:


plt.figure(figsize=(10, 8))
heatmap_data = df.groupby(['Day_of_week', 'Age_band_of_driver']).size().unstack(fill_value=0)
sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt='d')
plt.title('Accidents by Day of Week and Age Band of Driver')
plt.xlabel('Age Band of Driver')
plt.ylabel('Day of Week')
plt.show()


# In[12]:


#Checking for unique/ different type of values present in "Educational_level" column
df['Educational_level'].unique()


# In[13]:


#Calculating the mode of "Educational_level"
ELM=df['Educational_level'].mode()
ELM


# In[14]:


#Calculating the mode of "Vehicle_driver_relation"
VDRM = df['Vehicle_driver_relation'].mode()
VDRM


# In[15]:


#Calculating the mode of "Driving_experience"
DE = df['Driving_experience'].mode()
DE


# In[16]:


print("The percentage of data that is null:")
df.isnull().sum()/len(df)*100


# In[17]:


# created an function to replace the null values with mode value
def null_value_treatment(col):
    for i in df:
        if df[i].dtypes=='object':
            df[i].fillna(df[i].mode()[0],inplace=True)
        else:
            df[i].fillna(df[i].median(),inplace=True)


# In[18]:


for i in df:
    null_value_treatment(i)


# In[19]:


df.isnull().sum()/len(df)*100


# In[20]:


df.sample(10)


# After filling the null values with mode, their are some columns that has "na" and "unknown" present.

# In[21]:


df['Age_band_of_casualty'].unique()


# In[22]:


df['Service_year_of_vehicle'].unique()


# In[23]:


df['Service_year_of_vehicle'].mode()[0]


# In[24]:


df['Age_band_of_casualty'].mode()[0]


# ##### As we can se that the mode value of that column is null

# In[25]:


#Checking the correlation
df.corr()


# ###### As we can see that "Number_of_vehicles_involved" has correlation with "Number_of_casualties" of 0.2134. A correlation of 0.213 is not particularly strong, but it’s not necessarily “bad.

# In[26]:


df.skew()


# ##### A skewness value of 1.3234, 2.344 indicates that the data is positively skewed (right-skewed) and -3.833 indicates that the data is left skewed

# In[27]:


# Converting the categorical data into numerical data
LE = LabelEncoder()


# In[28]:


def Categorical_numerical(col):
    for i in df:
        df[col] = LE.fit_transform(df[col])


# In[29]:


for i in df:
    Categorical_numerical(i)


# In[30]:


df.head(4)


# In[31]:


#treating the outlier with zscore
from scipy.stats import zscore
def outlier_treatment(col):
    zscore1=(abs(zscore(df[col])))
    outlier=zscore1>+3
    median=df[col].median()
    df.loc[outlier,col]=median


# In[32]:


for i in df:
    outlier_treatment(i)


# ### Training the Model

# In[33]:


X = df.drop('Accident_severity',axis=1)


# In[34]:


y = df.Accident_severity


# ## Balancing the data using smote

# In[35]:


from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)
df_resampled = pd.concat([pd.DataFrame(X_resampled), pd.DataFrame(y_resampled)], axis=1)


# In[36]:


# Splitting the data into training and testing
from sklearn.model_selection import train_test_split


# In[37]:


x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.8,random_state=0)


# ## Logistic Regression 

# In[38]:


from sklearn.linear_model import LogisticRegression
LR=LogisticRegression()


# In[39]:


LR.fit(x_train,y_train)


# In[40]:


pred = LR.predict(x_test)


# ### Evaluating the Model

# In[41]:


accuracy_score(y_test,pred)*100


# In[43]:


predict = LR.predict(x_test)


# In[44]:


confusion_matrix(y_test,predict)


# ## DecisionTreeClassifier()

# In[45]:


#Decision Tree classifier
DTC = DecisionTreeClassifier()


# In[46]:


DTC.fit(x_train,y_train)


# In[47]:


y_pred = DTC.predict(x_test)


# In[48]:


y_pred


# ### Evaluating the model

# In[49]:


accuracy_score(y_test,y_pred)*100


# In[50]:


f1_score(y_test,y_pred,average='weighted')*100


# In[51]:


confusion_matrix(y_test,y_pred)


# ## K NEIGHBORS CLASSIFIER

# In[52]:


from sklearn.neighbors import KNeighborsClassifier


# In[53]:


KNN = KNeighborsClassifier(n_neighbors=25)


# In[54]:


KNN.fit(x_train,y_train)


# In[55]:


knn_pred=KNN.predict(x_test)


# In[56]:


y_test


# ### Evaluating the Model

# In[57]:


accuracy_score(y_test,knn_pred)*100


# In[58]:


f1_score(y_test,knn_pred,average='weighted')*100


# In[59]:


confusion_matrix(y_test,knn_pred)


# In[60]:


print("The true predictions are:")
8460+0


# In[61]:


print("The False predictions are:")
1393+0


# In[62]:


KNN.score(x_train,y_train)


# In[63]:


KNN.score(x_test,y_test)


# # Random Forest Classifier

# In[64]:


from sklearn.ensemble import RandomForestClassifier


# In[65]:


RFC = RandomForestClassifier()
RFC.fit(x_train,y_train)


# In[66]:


model_pred = RFC.predict(x_test)


# In[67]:


y_test


# ### Evaluating the model

# In[68]:


accuracy_score(y_test,model_pred)*100


# In[69]:


confusion_matrix(y_test,model_pred)


# In[70]:


print("The True predictions are:")
(4+845)


# In[71]:


print("The False predictions are:")
2+1389


# In[72]:


precision = precision_score(y_test, model_pred)
recall = recall_score(y_test, model_pred)

# Print the results
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")


# In[73]:


f1_score(y_test,model_pred)*100


# ## Conclusion
1. Logistic Regression Model:
Algorithm Used: Logistic Regression
Performance Metrics:
Accuracy: 85.8621%
F1-score: 76.32%

2. Random Forest Model:
Algorithm Used: Random Forest
Performance Metrics:
Accuracy: 85.88%
F1-score: 0.57%

3. K-nearest Neighbour:
Algorithm Used: KNN Classifier
Performance Metrics:
Accuracy: 85.86%
F1-score: 79.33%
    
4. Decision Tree
Algorithm Used: Decsison tree Classifier
Performance Metrics:
Accuracy: 75.69%
F1-score: 76.32%
# The analysis concludes with the following results:
# 
# Logistic Regression achieved an accuracy of approximately 85.86%.
# Decision Tree Classifier had an accuracy of around 75.69%.
# K-Nearest Neighbors (KNN) achieved an accuracy of 85.86%.
# Random Forest Classifier had an accuracy of approximately 85.88%.
# The models were evaluated based on accuracy, F1-score, and confusion matrices. 
# 
# Further fine-tuning and feature engineering may improve model performance. 

# In[ ]:




