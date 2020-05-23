#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score 

from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score 

from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.compose import make_column_transformer
import pandas as pd
import numpy as np


data = pd.read_csv('apy.csv')
df = pd.DataFrame(data)
data1 = pd.read_excel('profit.xlsx')
df1 = pd.DataFrame(data1)


req = ['State_Name','Crop_Year','Crop','Profit per Kg']
df1=df1[req]
df1 = df1[ df1['Crop_Year']== 2010]
df1 = df1.reset_index()
df1.drop(columns = ['index','Crop_Year'], inplace=True)
df1['State_Name'] = df1['State_Name'].str.lower() 
df1['Crop'] = df1['Crop'].str.lower()


df = df.drop('Production', axis=1)
df = df[df['Crop_Year']==2010]
df=df.reset_index()
df.drop(['index','Crop_Year'], inplace=True, axis=1)
names=['State_Name','District_Name','Season','Area','Crop']
df=df[names]
df['Season'] = df['Season'].str.rstrip()
df['State_Name'] = df['State_Name'].str.rstrip()

x = list(df.columns)
x.remove('Area')
for i in x:
    df[i] = df[i].str.lower()

    

# X = df.iloc[:,0:3]
# y = df.iloc[:,3]

#encoding data                    
he5 = LabelEncoder()
he4 = LabelEncoder()                                            
df1["State_Name"] = he5.fit_transform(df1["State_Name"])      
df1["Crop"] = he4.fit_transform(df1["Crop"])

        
he1 = LabelEncoder()
he2 = LabelEncoder()                      
he3 = LabelEncoder()                                            
df["State_Name"] = he1.fit_transform(df["State_Name"])          
df["District_Name"] = he2.fit_transform(df["District_Name"])         
df["Season"] = he3.fit_transform(df["Season"]) 

X = df.iloc[:,0:4].to_numpy()


# he_state = OneHotEncoder(sparse=False)
# he_district = OneHotEncoder(sparse=False)                 
# he_season = OneHotEncoder(sparse=False)                                          
# x1 = he_state.fit_transform([df.iloc[:, 0]])          
# x2 = he_district.fit_transform([df.iloc[:, 1]])         
# x3 = he_season.fit_transform([df.iloc[:, 2]]) 
        
# onehotencoder_dict={}

# onehotencoder_dict[0] = he_state
# onehotencoder_dict[1] = he_district
# onehotencoder_dict[2] = he_season

# X = make_column_transformer(
#         (OneHotEncoder(sparse=False),
#         ['State_Name', 
#         'District_Name', 
#         'Season']),
#          remainder = 'passthrough').fit_transform(df.iloc[:, 0 : 4])
# onehotencoder_dict[0] = onehot_encoder

label_encoder = LabelEncoder() 
y = label_encoder.fit_transform(df['Crop']) 


# In[25]:


df1['Crop'].unique()


# In[26]:


df['Crop'].unique()


# In[42]:


df['State_Name'].unique()


# In[36]:


df1.head()


# In[27]:


he4.classes_


# In[ ]:





# In[ ]:





# In[2]:


k_scores = []
for k in range(1, 37):
    k_scores.append(
    cross_val_score(
      KNeighborsClassifier(n_neighbors = k),
      X,
      y,
      cv = 2,
      scoring = 'accuracy'
    ).mean()
  )


# In[3]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


# In[4]:


estimators = []
model1 = KNeighborsClassifier(n_neighbors = k_scores.index(max(k_scores)) + 1)
estimators.append(('kNN', model1))

model2 = GaussianNB() 
estimators.append(('Naive bayes', model2))

# model3 = RandomForestClassifier(n_estimators=11, random_state=1)
# estimators.append(('Random forest', model3))

model4 = DecisionTreeClassifier()
estimators.append(('Decision tree', model4))

# create the ensemble model
# ensemble = VotingClassifier(estimators, voting='hard')
# results = model_selection.cross_val_score(ensemble, X, Y, cv=kfold)
# print(results.mean())

# Voting Classifier with hard voting 
vot_hard = VotingClassifier(estimators, voting ='hard') 
vot_hard.fit(X_train, y_train)
y_pred = vot_hard.predict(X_test)
print(y_pred)
temp=label_encoder.inverse_transform(y_pred)
print(temp)
# using accuracy_score metric to predict accuracy 
score = accuracy_score(y_test, y_pred)#, normalize=False) 
print(score) 


# In[5]:


import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split


X1 = df1.iloc[:,0:2]
y1 = df1.iloc[:,2]

# X_new = make_column_transformer((OneHotEncoder(sparse=False),['State_Name','Crop']), remainder = 'passthrough').fit_transform(X)

#split the model
X_train1,X_test1,y_train1,y_test1 = train_test_split(X1,y1,test_size = 0.2,random_state = 0)

regressor = LinearRegression()
regressor.fit(X_train1,y_train1)

print(regressor.intercept_)
print(regressor.coef_)

y_pred1 = regressor.predict(X_test1)

from sklearn import metrics
import math
print(math.sqrt(metrics.mean_squared_error(y_test1,y_pred1)))


# In[ ]:


from tkinter import *
from tkinter import messagebox
from tkinter.ttk import *

window = Tk()
window.title("Crop Recommendation System")
window.geometry('1640x1640')
window.configure(background='light blue')


lbl = Label(window, text="Enter details for Prediction: ",background="light blue",foreground="red",font=("bold", 30))
lbl.place(x=420,y=53)
#lbl.config(background="red")

lbl = Label(window, text="Enter your state:",foreground="dark green",background="light blue",font=("bold", 15))
lbl.place(x=250,y=150)
name = Entry(window,width=50)
name.place(x=800,y=150)
name.focus_set() 


lbl = Label(window, text="Enter your district:",foreground="dark green",background="light blue",font=("bold", 15))
lbl.place(x=250,y=200)
dname = Entry(window,width=50)
dname.place(x=800,y=200)


lbl = Label(window, text="Enter Season:",foreground="dark green",background="light blue",font=("bold", 15))
lbl.place(x=250,y=250)
sea = Entry(window,width=50)
sea.place(x=800,y=250)


lbl = Label(window, text="Enter area (in hectares):",foreground="dark green",background="light blue",font=("bold", 15))
lbl.place(x=250,y=300)
area = Spinbox(window, from_=1, width=10)
area.place(x=800,y=300)

def checkit():
    if (name.get() == "" or dname.get()=="" or sea.get()=="" or area.get()==""):
        messagebox.showinfo('Invalid', 'Please enter all the fields')
    else:        
        feat=[]
        f1 = name.get()
        f2 = dname.get()
        f3 = sea.get()
        f4 = float(area.get())
        feat.append(he1.transform([f1]))
        feat.append(he2.transform([f2]))
        feat.append(he3.transform([f3]))
        feat.append(f4)
        
        
#         X = make_column_transformer(
#         (OneHotEncoder(sparse = False),
#         ['State_Name', 
#         'District_Name', 
#         'Season']),
#         remainder = 'passthrough').fit_transform(df.iloc[:, 0 : 4])
        
        pred=vot_hard.predict([feat])
        pred=label_encoder.inverse_transform(pred)
        
        c="You should grow " +str(pred[0])
        messagebox.showinfo('Prediction',c)
        
#         feat.append(np.asscalar(he5.transform(['maharashtra'])))
#         feat.append(np.asscalar(he4.transform(['urdbean'])))
    
#         y_pred = regressor.predict([feat])
        
        
        feat2 = []
        feat2.append(np.asscalar(he5.transform([f1])))
        feat2.append(np.asscalar(he4.transform([pred])))
        pred2 = regressor.predict([feat2])
        
        fpro = round(pred2[0],2)
        c="Estimated profit(per kg) will be   ₹  "+str(fpro)
        messagebox.showinfo('Prediction',c)
        #lbl = Label(window, text=x)
        #lbl.grid(column=5, row=25)
        feat=[]
        feat2=[]
                
#submit = Button(window, text="Submit",width=20, command=checkit).place(x=550,y=650) 
submit = Button(window, text="Submit",width=20,command=checkit)
submit.place(x=550,y=650) 
#menu = Menu(window)
#menu.add_command(label='File')
#window.config(menu=menu)

window.mainloop()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[78]:


df.loc[:,'Crop']


# In[92]:


x = df.groupby('Crop').size()
z = list(df.groupby('Crop').size())
z
zind = list(x.index)
print(zind)
print(z)


# In[ ]:


sl.plt.bar()


# In[99]:


import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize']=(20,30)
objects = zind
y_pos = np.arange(len(zind))
performance = z

plt.barh(y_pos, performance, align='center', alpha=1)
plt.yticks(y_pos, objects)
plt.xlabel('count')
plt.title('Crops in dataset')

plt.show()


# In[85]:


np.arange(6)


# In[ ]:





# In[64]:


import matplotlib.pyplot as plt

plt.scatter(X1.iloc[:,0], y1)
plt.plot(X_test1.iloc[:,0], y_pred1, color='red')
plt.show()


# In[66]:


import seaborn as sns

tips = sns.load_dataset("tips")


# In[67]:


tips


# In[69]:


x = df1.plot.pie(y='Crop', figsize=(5, 5))


# In[70]:


x.show()


# In[ ]:




