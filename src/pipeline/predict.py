from dataclasses import dataclass
from src.utils import load_model
import sys 
import os
from src.exception import CustomException
from src.logger import logging
import numpy as np
import pandas as pd

@dataclass
class PredictConfig:
    preprocessor_path = "artifact/preprocessor.pkl"
    model_path = "artifact/model.pkl"


class Prediction:
    def __init__(self):
        self.path = PredictConfig()

    def upload_models(self):
        try: 
            logging.info("loading path initiated")
            transform_path = self.path.preprocessor_path
            model_pred_path = self.path.model_path 
            logging.info("loading models")           
            preprocessor = load_model(transform_path)
            model = load_model(model_pred_path)
            return(preprocessor,model)
        
        except Exception as e:
            raise CustomException(e,sys)
        

if __name__=="__main__":
    predictor = Prediction()

    # Load preprocessor and model
    preprocessor, model = predictor.upload_models()
    feed = preprocessor.transform(pd.DataFrame([["female","group B","bachelor's degree","standard","none",72,74]],
                                               columns=["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course",
                                                        "reading_score","writing_score"]))
    print(feed)
    print(model.predict(feed))