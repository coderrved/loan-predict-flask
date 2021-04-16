from flask import Flask,render_template,request
import numpy as np
import pandas as pd
import seaborn as sns
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
import warnings
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None

app = Flask(__name__)


def missingValues(ds):
        missingValue=ds.isnull().sum()
        missingValuePercent=100*ds.isnull().sum()/len(ds)
        missingValueTab=pd.concat([missingValue,missingValuePercent],axis=1)
        missingValueTable=missingValueTab.rename(
        columns ={0:'Eksik Degerler',1:'% Degeri'})
        return missingValueTable


# gender ve married verileri bos olan satirlar silindi ve Maskeleme yapiyorum. 
# Ilgili sutun adlarındaki ilgili degerleri degistiriyorum boylece verileri numeric yapmis oluyorum

def mask(ds):  
    ds_crop = ds.dropna(subset=["Gender","Married"])
    ds_crop[['Self_Employed']]=ds_crop[['Self_Employed']].replace('No',0)
    ds_crop[['Self_Employed']]=ds_crop[['Self_Employed']].replace('Yes',1)
    ds_crop[['Dependents']]=ds_crop[['Dependents']].replace('0',0)
    ds_crop[['Dependents']]=ds_crop[['Dependents']].replace('1',1)
    ds_crop[['Dependents']]=ds_crop[['Dependents']].replace('2',2)
    ds_crop[['Dependents']]=ds_crop[['Dependents']].replace('3+',3)
    ds_crop[['Loan_Status']]=ds_crop[['Loan_Status']].replace('N',0)
    ds_crop[['Loan_Status']]=ds_crop[['Loan_Status']].replace('Y',1)
    return ds_crop

# Nan hucreler median degere gore dolduruldu.

def fillNaFunc(ds_crop):
    ds_fill=ds_crop.fillna(ds_crop.median())
    return ds_fill

def fillTypeConvert(ds_fill):
    ds_fill['Dependents'] = ds_fill['Dependents'].astype('int64')
    ds_fill['Self_Employed'] = ds_fill['Self_Employed'].astype('int64')
    ds_fill['CoapplicantIncome'] = ds_fill['CoapplicantIncome'].astype('int64')
    ds_fill['LoanAmount'] = ds_fill['LoanAmount'].astype('int64')
    ds_fill['Loan_Amount_Term'] = ds_fill['Loan_Amount_Term'].astype('int64')
    ds_fill['Credit_History'] = ds_fill['Credit_History'].astype('int64')
    return ds_fill

def smoteProcessXSM(ds_fill):
    features = ['Dependents','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History']
    ds_x = ds_fill.loc[:,features]
    ds_y = ds_fill.loc[:,['Loan_Status']]

    smote = SMOTE()
    X_sm, y_sm = smote.fit_resample(ds_x, ds_y)
    return X_sm

def smoteProcessYSM(ds_fill):
    features = ['Dependents','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History']
    ds_x = ds_fill.loc[:,features]
    ds_y = ds_fill.loc[:,['Loan_Status']]

    smote = SMOTE()
    X_sm, y_sm = smote.fit_resample(ds_x, ds_y)
    return y_sm

def denemeFunc(X_sm):
    
    X_sm = StandardScaler().fit_transform(X_sm)
    return X_sm

def decisionTreeClassifierModel(X_test,X_train,y_test,y_train,number1,number2,number3,number4,number5,number6,number7):
    #data = [[0], [0], [2583], [2358],[120],[360],[1]]
    data = [[number1], [number2], [number3], [number4],[number5],[number6],[number7]]
    scaler = StandardScaler()
    scaler.fit(data)
    scaler.mean_
    newArray = scaler.transform(data)
    newArray2 = newArray.reshape(1,7)
    #print(newArray2)

    decision_tree_model = DecisionTreeClassifier().fit(X_train,y_train)
    y_pred1 = decision_tree_model.predict(X_test)
    #print(accuracy_score(y_test,y_pred1))
    decision_tree_model = DecisionTreeClassifier()
    decision_tree_params = {"max_depth":[1,2,3,4,5,10,20,30],
               "min_samples_split":[1,2,3,4,5,10,20,30],
                       "max_leaf_nodes":[1,2,3,4,5,10,20,30]}
    #decision_tree_cv_model = GridSearchCV(decision_tree_model,decision_tree_params,cv=10).fit(X_train,y_train)
    decision_tree_tuned = DecisionTreeClassifier(max_depth=5,min_samples_split=2,max_leaf_nodes=4).fit(X_train,y_train)  # Düzenlemeler yapılacak.
    y_pred1 = decision_tree_tuned.predict(X_test)
    #linear_predict_data=np.array([[0.71669001,0.71669001, 1.67329007, 1.46510366, 0.60565725,-0.38359174, 0.71576473]])
    linear_predict_data=newArray2

    y_pred1 = decision_tree_tuned.predict(linear_predict_data)
    print(y_pred1)
    return y_pred1


@app.route('/',methods=["GET","POST"])
def hello_world():
    if request.method == "POST":
        number1 = request.form.get("number1")
        number2 = request.form.get("number2")
        number3 = request.form.get("number3")
        number4 = request.form.get("number4")
        number5 = request.form.get("number5")
        number6 = request.form.get("number6")
        number7 = request.form.get("number7")   
        print("form bilgileri")
        print(number1)
        print(number2)
        print(number3)
        print(number4)
        print(number5)
        print(number6)
        print(number7)

        ds = pd.read_csv("C:\\Users\\oem\\Desktop\\JupyterNotebook\\train_kredi_tahmini.csv") #csv dosyasi okuma
    
        ds_crop = mask(ds)

        ds_fill=fillNaFunc(ds_crop)
    
        ds_fill = fillTypeConvert(ds_fill)
        print("************")

        print("************")
        #smoteProcess(ds_fill)

        print("************")
        X_sm = smoteProcessXSM(ds_fill)
        y_sm = smoteProcessYSM(ds_fill)
        X_sm = denemeFunc(X_sm)
        print("************")

        X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size = 0.4, random_state = 120)
        gelenDeger = decisionTreeClassifierModel(X_test,X_train,y_test,y_train,number1,number2,number3,number4,number5,number6,number7)

        if gelenDeger[0] == 0:
            donenDeger = "UYGUN DEĞİLDİR"
        else:
            donenDeger = "UYGUNDUR"

        return render_template("index.html",donenDeger=donenDeger)
    if request.method == "GET":
        

        return render_template("index.html")

    """
    ds = pd.read_csv("C:\\Users\\oem\\Desktop\\JupyterNotebook\\train_kredi_tahmini.csv") #csv dosyasi okuma
    
    ds_crop = mask(ds)

    ds_fill=fillNaFunc(ds_crop)
    
    ds_fill = fillTypeConvert(ds_fill)
    print("************")

    print("************")
    #smoteProcess(ds_fill)

    print("************")
    X_sm = smoteProcessXSM(ds_fill)
    y_sm = smoteProcessYSM(ds_fill)
    X_sm = denemeFunc(X_sm)
    print("************")
    X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size = 0.4, random_state = 120)
    gelenDeger = decisionTreeClassifierModel(X_test,X_train,y_test,y_train)

    if gelenDeger[0] == 0:
        donenDeger = "Verilmez"
    else:
        donenDeger = "Verilir"

    return render_template("index.html")
    """

@app.route("/toplam", methods=["GET","POST"])
def toplam():
    if request.method == "POST":
        number1 = request.form.get("number1")
        number2 = request.form.get("number2")
        number3 = request.form.get("number3")
        number4 = request.form.get("number4")
        number5 = request.form.get("number5")
        number6 = request.form.get("number6")
        number7 = request.form.get("number7")       
        x = np.array([number1,number2,number3,number4,number5,number6,number7])
        ds = pd.read_csv("C:\\Users\\oem\\Desktop\\JupyterNotebook\\train_kredi_tahmini.csv") #csv dosyasi okuma
    
        ds_crop = mask(ds)

        ds_fill=fillNaFunc(ds_crop)
    
        ds_fill = fillTypeConvert(ds_fill)
        print("************")

        print("************")
        #smoteProcess(ds_fill)

        print("************")
        X_sm = smoteProcessXSM(ds_fill)
        y_sm = smoteProcessYSM(ds_fill)
        X_sm = denemeFunc(X_sm)
        print("************")
        X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size = 0.4, random_state = 120)
        gelenDeger = decisionTreeClassifierModel(X_test,X_train,y_test,y_train,number1,number2,number3,number4,number5,number6,number7)

        if gelenDeger[0] == 0:
            donenDeger = "Verilmez"
        else:
            donenDeger = "Verilir"

        return render_template("number.html",donenDeger=donenDeger)
    if request.method == "GET":
        return render_template("number.html")

if __name__ == '__main__':
    app.run(debug=True)