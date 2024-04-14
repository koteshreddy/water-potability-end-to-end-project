import pickle
import os
from flask import Flask,request,app,jsonify,url_for,render_template,json
import numpy as np
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('random_water_potable.pkl','rb'))
scaled = pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home(): 
    return render_template('home.html')


@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scaled.transform(np.array(list(data.values())).reshape(1,-1))
    #output=model.predict(new_data)
    pred = model.predict(pd.Series(data).to_frame().T)
    print(pred[0])
    numpy_array = np.array([pred[0]], dtype=np.int64)

# Convert NumPy int64 to Python int
    numpy_array_as_list = numpy_array.tolist()

# Now you can serialize the list to JSON
    json_string = json.dumps(numpy_array_as_list)
    return json_string


if __name__=="__main__":
    app.run(debug=True)
   



