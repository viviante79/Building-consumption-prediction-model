#!/usr/bin/env python
# coding: utf-8

# In[1]:


# basics :
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# preprocessing :
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler

# Model
import xgboost as xgb
from xgboost import XGBRegressor

# Export :
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import joblib
import traceback


# In[2]:


app = Flask(__name__)


# In[3]:


@app.route("/upload")
def upload_file():
    return render_template("upload.html", template_folder="templates")


# In[9]:


@app.route("/uploader", methods = ["POST"])
def uploader_file() :
    f = request.files["upload_file"]
    f.save(secure_filename(f.filename))
    if model : 
        try :
            # Lecture du fichier
            df = pd.read_csv(f.filename)
            df = df.drop(columns=["Unnamed: 0", "ID"])
            
            # Imputations des variables numériques :
            imputer = KNNImputer(n_neighbors=5)
            numericals = df.select_dtypes(np.number)
            numcol = df.select_dtypes(np.number).columns
            numericals = pd.DataFrame(imputer.fit_transform(numericals), columns=numcol)
            
            # Encodage des variables catégorielles :
            transformer = OneHotEncoder(sparse=False)
            categoricals = pd.DataFrame(transformer.fit_transform(df.select_dtypes("object")))
            feature_names = transformer.get_feature_names_out(input_features=None)
            categoricals = categoricals.astype("int")
            categoricals.columns = feature_names
            
            # Passage au logarithme :
            numericals = np.log(numericals, where=numericals>0)
            
            # Standardisation des données :
            scaler = RobustScaler()
            numericals[["usage_1", "usage_2", "usage_3", "age","nbre_batiments","nbre_etages","surface","energystar"]] = scaler.fit_transform(numericals[["usage_1", "usage_2", "usage_3", "age","nbre_batiments","nbre_etages","surface","energystar"]])
            
            # Jointure des deux types de variables :
            data = numericals.join(categoricals)
            
            # Modèle :
            query = data.drop(columns=["conso","co2","energystar"])
            print(query)
            prediction = list(model.predict(query))
            print(list(model.predict(query)))
            
            dictionnary = dict(zip(data.index, prediction))
            
            return jsonify(str(dictionnary))
        except :
            return jsonify({"trace" : traceback.format_exc()})
        else :
            print ("Train the model first")
            return "No model here to use"

if __name__ == "__main__" :
    model = joblib.load("XGB_model")
    print ("Model Loaded")
    app.run(host="localhost", port=5000, debug=True)

