from flask import Flask,render_template,request
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
model_rf = pickle.load(open('model_rf.pkl', 'rb'))


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
        
        data = [number1, number2, number3, number4,number5,number6,number7]
        np_array=np.asarray(data)
        data = np_array.reshape(1,7)

        gelenDeger = model.predict(data)
        print(gelenDeger)
        gelenDeger2 = model_rf.predict(data)
        print(gelenDeger2)
        
        if gelenDeger[0] == gelenDeger2[0]:
            if gelenDeger[0] == 0:
                donenDeger = "UYGUN DEĞİLDİR"
                return render_template("index.html",donenDeger=donenDeger)
            else:
                donenDeger = "UYGUNDUR"
                return render_template("index.html",donenDeger=donenDeger)
        else:
            if gelenDeger[0] == 0:
                donenDeger = "DT'YE UYGUN DEĞİLDİR"
                donenDeger2 = "RF'E GORE UYGUNDUR"
                return render_template("index.html",donenDeger=donenDeger, donenDeger2=donenDeger2)
            else:
                donenDeger2 = "RF'E GORE UYGUN DEĞİLDİR"
                donenDeger = "DT'YE UYGUNDUR"
                return render_template("index.html",donenDeger=donenDeger, donenDeger2=donenDeger2)
        
        
    if request.method == "GET":
        return render_template("index.html")


if __name__ == '__main__':
    app.run(debug=True)