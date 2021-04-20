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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
import warnings
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None


ds = pd.read_csv("C:\\Users\\oem\\Desktop\\JupyterNotebook\\train_kredi_tahmini.csv")       # CSV dosyasını okudum.


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

X_sm = StandardScaler().fit_transform(X_sm)

X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size = 0.4, random_state = 120)

from sklearn.tree import DecisionTreeClassifier

rf_model = RandomForestClassifier().fit(X_train,y_train)
y_pred = rf_model.predict(X_test)
ilkDurum  = accuracy_score(y_test,y_pred)
print(ilkDurum)

rf_model1 = RandomForestClassifier(max_features=5,
                                min_samples_split=2,
                                n_estimators=1000)
rf_tuned=rf_model1.fit(X_train,y_train)
y_pred1 = rf_tuned.predict(X_test)
ikinciDurum  = accuracy_score(y_test,y_pred1)
print(ikinciDurum)




if ilkDurum > ikinciDurum:
    print("ilk durum büyük")
    print(ilkDurum)
    pickle.dump(rf_model, open('model_rf.pkl','wb'))
else:
    print("ikinci durum büyük")
    print(ikinciDurum)
    pickle.dump(rf_tuned, open('model_rf.pkl','wb'))
