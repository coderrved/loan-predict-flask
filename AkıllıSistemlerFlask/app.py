from flask import Flask,render_template,request
from sklearn.preprocessing import StandardScaler
import pickle


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
        
        data = [[number1], [number2], [number3], [number4],[number5],[number6],[number7]]
        scaler = StandardScaler()
        scaler.fit(data)
        scaler.mean_
        newArray = scaler.transform(data)
        newArray2 = newArray.reshape(1,7)

        gelenDeger = model.predict(newArray2)
        gelenDeger2 = model_rf.predict(newArray2)
        
        print("Decision Tree: ", gelenDeger[0])
        print("Random Forest: ", gelenDeger2[0])
        #gelenDeger[0] = 1

        if gelenDeger[0] == gelenDeger2[0]:
            if gelenDeger[0] == 0:
                donenDeger = "UYGUN DEĞİLDİR"
                return render_template("index.html",donenDeger=donenDeger)
            else:
                donenDeger = "UYGUNDUR"
                return render_template("index.html",donenDeger=donenDeger)
        else:
            if gelenDeger[0] == 0:
                donenDeger = "UYGUN DEĞİLDİR"
                donenDeger2 = "RF'E GORE UYGUNDUR"
                return render_template("index.html",donenDeger=donenDeger, donenDeger2=donenDeger2)
            else:
                donenDeger2 = "RF'E GORE UYGUN DEĞİLDİR"
                donenDeger = "UYGUNDUR"
                return render_template("index.html",donenDeger=donenDeger, donenDeger2=donenDeger2)
        
    if request.method == "GET":
        return render_template("index.html")


if __name__ == '__main__':
    app.run(debug=True)