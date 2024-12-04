import sys
import os
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_model,evaluation_report

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


@dataclass
class ModelConfig:
    model_path = os.path.join("artifact","model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_saving_path = ModelConfig()


    def initiate_model_training(self,train_arr,test_arr):
        try:
            logging.info("Splitting Training and Testing datasets")
            x_train,y_train,x_test,y_test = (train_arr[:,:-1],train_arr[:,-1],test_arr[:,:-1],test_arr[:,-1])

            model_dict = {
                "LinearRegression":LinearRegression(),
                "RandomForest":RandomForestRegressor(),
                "KnnRegression":KNeighborsRegressor(),
                "DecisionTree":DecisionTreeRegressor(),
                "XGBoost":XGBRegressor()
            }

            report = evaluation_report(x_train,y_train,x_test,y_test,model_dict)
            if report.loc[0]["score"]>0.7:
                final_model = model_dict[report.loc[0]["model"]]
            
            logging.info(f"Best model found {final_model}")

            save_model(file_path=self.model_saving_path.model_path,obj=final_model)
            
            return report
        
        except Exception as e:
            raise CustomException(e,sys)

