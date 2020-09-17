#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

telecommunication = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
#telecommunication = telecommunication.head()

# convert str to num: yes = 1, no = 0
column_convert = ['Partner', 'Dependents','PhoneService', 'PaperlessBilling', 'Churn']
for i in column_convert:
    telecommunication[i].replace('Yes', 1, True)
    telecommunication[i].replace('No', 0, True)

# check is null, if null, use column mean value to fill the space 
telecommunication = telecommunication.convert_objects(convert_numeric = True)
check_null = telecommunication.isnull().any()
for i in check_null:
    if i == True:
        null_index = check_null[check_null.values == True].index
        column_mean = telecommunication[null_index].mean()
        telecommunication = telecommunication.fillna(column_mean)

# make the ratio of churn:not_churn = 1:1
churn_y = telecommunication[telecommunication.Churn == 1]
churn_n = telecommunication[telecommunication.Churn == 0]

num_sample = min(len(churn_y), len(churn_n))
if num_sample == len(churn_y):
    churn_n = churn_n.sample(n = num_sample)
else:
    churn_y = churn_y.sample(n = num_sample)

tele_yn = pd.concat([churn_y, churn_n])

# heatmap to show the correlation
corr1 = tele_yn.corr()
mask = np.zeros_like(corr1, dtype = np.bool)
mask[np.triu_indices_from(mask)] = True

plt.subplots(figsize = (18, 11))
corr_map = sns.heatmap(corr1, mask = mask, annot = True, vmax = 1, 
            center = 0.2, cmap = 'BuPu_r', linewidths = 1, fmt = '.4f')
plt.title('Heatmap1: Correlation Between Churn and Others')
plt.savefig('corr_map.png', dpi=200)
sns.set_style("darkgrid")

# covert dummy variable to 1 and 0
column_name = list(tele_yn)
tele_customer_id = tele_yn[column_name[0]]
tele_dummy = tele_yn[column_name[1:]]
tele_dummy = pd.get_dummies(tele_dummy)
tele_new = pd.concat([tele_customer_id, tele_dummy], axis = 1)

# horizontal bar graph to show correlation
corr2 = tele_new.corr()
corr_churn = corr2['Churn'].sort_values(ascending = True)

plt.figure(figsize=(15, 15))
plt.axvline(0, color='b', alpha = 0.4)
corr_churn_n = corr_churn.iloc[:len(corr_churn)-1]
corr_churn_n = corr_churn_n.round(4)
corr_bar = corr_churn_n.plot.barh(alpha = 0.8, color =  'pink')   
for x, y in enumerate(corr_churn_n):
    plt.text(y+0.02, x, '%s'%y, ha = 'center',va = 'bottom')
plt.title('Correlation Between All Variables')   
plt.savefig('corr_bar.png', dpi=200, bbox_inches='tight')

# heatmap 2
plt.figure(figsize=(90, 55))
corr3 = tele_new[tele_new.Churn == 1].corr()
corr_map = sns.heatmap(corr3, annot = True, vmax = 1, 
            center = 0.2, cmap = 'OrRd', linewidths = 1, fmt = '.4f')
plt.title('Heatmap2: Correlation Between All Variables')
plt.savefig('corr_map2.png', dpi=200)

# remove variables with corr = 1
corr2_copy = corr2.copy()
column_name3 = list(corr2_copy)
dict_corr1 = {}
for i, r in corr2_copy.iterrows():
    temp = []
    for a in column_name3:
        if r[a] == 1 and a != i:
            temp.append(a)
            dict_corr1[i] = temp
dict_corr1 = pd.DataFrame(dict_corr1)
column_name4 = list(dict_corr1)
column_name4 = column_name4[1:]

# facetgrid, tenure，monthly_charge, gender, churn
temp_churn = tele_yn.copy()
column_convert.append('SeniorCitizen')
for i in column_convert:
    temp_churn[i].replace(1, 'Yes', True)
    temp_churn[i].replace(0, 'No', True)
pal = dict(Yes="seagreen", No="gray")
g = sns.FacetGrid(temp_churn, col="gender", hue="Churn", 
                  palette=pal, hue_order=["Yes", "No"], 
                  hue_kws = dict(marker = ['^', 'v']))
g = (g.map(plt.scatter, "MonthlyCharges", "tenure", alpha = 0.6).add_legend()) 


# logistic model
logistic_x = tele_new.drop(['customerID','Churn'], axis = 1)
column_name4 = list(logistic_x)
#logistic_x = tele_new.drop(['customerID','Churn', 'gender_Female', 'gender_Male'], axis = 1)
#logistic_x = logistic_x.drop(column_name4, axis = 1)
logistic_x = logistic_x.as_matrix()
logistic_y = tele_new['Churn'].as_matrix()


model=LogisticRegression(fit_intercept=True,solver='liblinear')
model=model.fit(logistic_x,logistic_y)

print(model.coef_)
print(model.intercept_)
print(model.score(logistic_x,logistic_y))
coef_var = {}
for i in range(len(model.coef_[0, :])):
    coef_var[column_name4[i]] = model.coef_[0, i]
coef_var = pd.DataFrame.from_dict(coef_var, orient='index')

#coef_var.to_csv('coef_var.csv')

# find the churn rate of different group
churn = telecommunication['Churn']
churn_rate = churn[churn == 0].count()/churn.count()
print(churn_rate)

# graph for each variable 
def draw_qualitative(column):
    pic_name_bar = 'churn_' + column + '_bar' + '.png'
    pic_name_pie = 'churn_' + column + '_pie' + '.png'
    plt.clf()
    
    plt.title('churn_' + column + '_pie')
    if column == 'PaymentMethod':
        temp_churn[column].value_counts().plot.pie(startangle = 20, 
                  autopct='%.2f', fontsize=12, figsize = (5, 5))
    else:
        temp_churn[column].value_counts().plot.pie(autopct='%.2f', fontsize=12, figsize = (5, 5))
    plt.legend(bbox_to_anchor=(1, 0.7))
    plt.savefig(pic_name_pie, dpi=200, bbox_inches='tight')
    
    temp_churn.groupby(column)['Churn'].value_counts().unstack().plot.bar()
    plt.title('churn_' + column + '_bar')
    plt.savefig(pic_name_bar, dpi=200, bbox_inches='tight')
    plt.show()

def draw_quantitative(column):
    pic_name_bar = 'churn_' + column + '.png'
    
    graph = temp_churn.groupby('Churn')[column].plot.kde()
    graph = sns.kdeplot(temp_churn[column], shade = True)
    plt.title('churn_' + column)
    plt.savefig(pic_name_bar, dpi=200, bbox_inches='tight')
    plt.show()

column_name2 = column_name.copy()
column_name2.remove('customerID')
column_name2.remove('tenure')
column_name2.remove('MonthlyCharges')
column_name2.remove('TotalCharges')
column_name2.remove('Churn')

quantitative  = ['tenure', 'MonthlyCharges']

for i in column_name2:
    draw_qualitative(i)
    
for i in quantitative:
    draw_quantitative(i)

# graph for service 
column_service = temp_churn.copy()
column_service_all = column_service.iloc[:, 9:15]
column_service_dsl = column_service[column_service.InternetService == 'DSL'].iloc[:, 9:15]
column_service_fiber_optic = column_service[column_service.InternetService == 'Fiber optic'].iloc[:, 9:15]
   
def draw_service(dataframe):
    service = {}
    pic_name = dataframe + '.png'
    dataframe = eval(dataframe)
    for i in dataframe:
        service[i] = dataframe[i].value_counts()
    service = pd.DataFrame(service)
    service = service.stack().unstack(0)
    if str(dataframe) == str(column_service_all):
        service = service[['Yes', 'No', 'No internet service']]
    else:
        service = service[['Yes', 'No']]
    service.plot.bar(stacked = True)
    plt.title(pic_name)
    plt.legend(bbox_to_anchor=(1, 0.25))
    plt.savefig(pic_name, dpi=200, bbox_inches='tight')
    plt.show()

draw_service('column_service_all')
draw_service('column_service_dsl')
draw_service('column_service_fiber_optic')
