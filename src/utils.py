import os
import sys 
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import r2_score
from src.exception import CustomException
from src.logger import logging

def save_model(file_path,obj):
    try:
        with open(file_path,"wb") as f:
            pickle.dump(obj,f)
    except Exception as e:
        raise CustomException(e,sys)
    
def load_model(file_path):
    try:
        with open(file_path,"rb") as f:
            model = pickle.load(f)
        return model        
    except Exception as e:
        raise CustomException(e,sys)

def evaluation_report(xtrain,ytrain,xtest,ytest,model):
    report_list = []
    try:
        for i,j in model.items():
            model=j
            model.fit(xtrain,ytrain)
            y_pred = model.predict(xtest)
            score = r2_score(y_pred,ytest)
            report_list.append([i,score])
        report = pd.DataFrame(report_list,columns = ["model","score"]).sort_values(by=["score"],ascending=False)
        logging.info("Report generated")
        return report
    except Exception as e:
        raise CustomException(e,sys)
    