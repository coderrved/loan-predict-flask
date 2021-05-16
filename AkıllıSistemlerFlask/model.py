import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
import warnings
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None


ds = pd.read_csv("C:\\Users\\oem\\Desktop\\Akilli_Sistemler\\AkıllıSistemlerFlask\\train_kredi_tahmini.csv")


def missingValues(ds):
    missingValue=ds.isnull().sum()
    missingValuePercent=100*ds.isnull().sum()/len(ds)
    missingValueTab=pd.concat([missingValue,missingValuePercent],axis=1)
    missingValueTable=missingValueTab.rename(
    columns ={0:'Eksik Degerler',1:'% Degeri'})
    return missingValueTable

missingValues(ds)

ds_crop = ds.dropna(subset=["Gender","Married"])   

ds_crop[['Self_Employed']]=ds_crop[['Self_Employed']].replace('No',0)
ds_crop[['Self_Employed']]=ds_crop[['Self_Employed']].replace('Yes',1)

ds_crop[['Dependents']]=ds_crop[['Dependents']].replace('0',0)
ds_crop[['Dependents']]=ds_crop[['Dependents']].replace('1',1)
ds_crop[['Dependents']]=ds_crop[['Dependents']].replace('2',2)
ds_crop[['Dependents']]=ds_crop[['Dependents']].replace('3+',3)

ds_crop[['Loan_Status']]=ds_crop[['Loan_Status']].replace('N',0)
ds_crop[['Loan_Status']]=ds_crop[['Loan_Status']].replace('Y',1)

ds_fill=ds_crop.fillna(ds_crop.median())     

missingValues(ds_fill)

ds_fill['Dependents'] = ds_fill['Dependents'].astype('int64')
ds_fill['Self_Employed'] = ds_fill['Self_Employed'].astype('int64')
ds_fill['CoapplicantIncome'] = ds_fill['CoapplicantIncome'].astype('int64')
ds_fill['LoanAmount'] = ds_fill['LoanAmount'].astype('int64')
ds_fill['Loan_Amount_Term'] = ds_fill['Loan_Amount_Term'].astype('int64')
ds_fill['Credit_History'] = ds_fill['Credit_History'].astype('int64')

features = ['Dependents','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History']
ds_x = ds_fill.loc[:,features]
ds_y = ds_fill.loc[:,['Loan_Status']]

smote = SMOTE()

X_sm, y_sm = smote.fit_resample(ds_x, ds_y)

X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size = 0.4, random_state = 120)

from sklearn.tree import DecisionTreeClassifier

decision_tree_model = DecisionTreeClassifier().fit(X_train,y_train)

y_pred1 = decision_tree_model.predict(X_test)

print(accuracy_score(y_test,y_pred1))
ilkDurum = accuracy_score(y_test,y_pred1)
print(ilkDurum)

cart_model = DecisionTreeClassifier()

decision_tree_params = {"max_depth":[3,4,5,6,7,8,9,10],
               "min_samples_split":[1,2,3,4,5,10,20,30],
                       "max_leaf_nodes":[1,2,3,4,5,10,20,30]}

decision_tree_params

decision_tree_cv_model = GridSearchCV(cart_model,decision_tree_params,cv=10).fit(X_train,y_train)

parameters_decision_tree=decision_tree_cv_model.best_params_

print(parameters_decision_tree)

decision_tree_tuned = DecisionTreeClassifier(max_depth=parameters_decision_tree["max_depth"],min_samples_split=parameters_decision_tree["min_samples_split"],max_leaf_nodes=parameters_decision_tree["max_leaf_nodes"]).fit(X_train,y_train)  # Düzenlemeler yapılacak.

y_pred = decision_tree_tuned.predict(X_test)

print(accuracy_score(y_test,y_pred))
ikinciDurum = accuracy_score(y_test,y_pred)
print(ikinciDurum)

if ilkDurum > ikinciDurum:
    print("ilk durum büyük")
    print(ilkDurum)
    pickle.dump(decision_tree_model, open('model.pkl','wb'))
else:
    print("ikinci durum büyük")
    print(ikinciDurum)
    pickle.dump(decision_tree_tuned, open('model.pkl','wb'))
