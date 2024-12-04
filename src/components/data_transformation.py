import os
import sys
from src.exception import CustomException
from src.logger import logging

from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.utils import save_model


from src.components.data_ingestion import DataIngestion,DataIngestionConfig

@dataclass
class DataTransformationConfig:
    preproccessor_obj_path = os.path.join("artifact","preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.preprocessor_path = DataTransformationConfig()

    def data_preprocessor(self):
        try:
            numerical_col = ["writing_score", "reading_score"]
            catogorical_col = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            num_pipe = Pipeline(
                steps=[
                    ("missing",SimpleImputer(strategy="median")),
                    ("transforming",StandardScaler())
                ]
            )

            cat_pipe = Pipeline(
                steps = [
                    ("missing",SimpleImputer(strategy="most_frequent")),
                    ("encoding",OneHotEncoder())
                ]
            )
            logging.info("numerical and catogorical - pipelines created")

            transformer = ColumnTransformer([
                ("numerical",num_pipe,numerical_col),
                ("catogorical",cat_pipe,catogorical_col)
            ])
            logging.info("Column transformation and preprocessor is created")
            return transformer
        except Exception as e:
            raise CustomException(e,sys)
        

    def Transformation(self,train_path,test_path):
        try:
            df_train = pd.read_csv(train_path)
            df_test = pd.read_csv(test_path)
            logging.info("train&test data obtained")

            preprocessor = self.data_preprocessor()
            logging.info("preprocessor data obtained")

            target = "math_score"
            x_train = df_train.drop(target,axis=1)
            y_train = df_train[target]
            x_test = df_test.drop(target,axis=1)
            y_test = df_test[target]
            logging.info("train_test_features_target_split is done")

            x_train_arr = preprocessor.fit_transform(x_train) 
            x_test_arr = preprocessor.transform(x_test)

            train_arr = np.c_[x_train_arr,np.array(y_train)]
            test_arr = np.c_[x_test_arr,np.array(y_test)]
            logging.info("converted train_test data back to array")

            save_model(
                file_path = self.preprocessor_path.preproccessor_obj_path,
                obj = preprocessor
            )

            return(train_arr,test_arr,self.preprocessor_path.preproccessor_obj_path)

        except Exception as e:
            raise CustomException(e,sys)
    


