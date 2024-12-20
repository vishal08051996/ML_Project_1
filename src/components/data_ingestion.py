import sys
import os
from src.exception import CustomException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifact","data.csv")
    test_data_path: str = os.path.join("artifact","test.csv")
    train_data_path: str = os.path.join("artifact","train.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered into reading raw data mode")

        try:
            df = pd.read_csv("stud.csv")
            logging.info("Reading raw data as dataframe")
            logging.info("creating directory for raw data")
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            logging.info("Directory for raw data - done and initiated saving file")
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("Saved raw data - Train_test_split initiated")
            train_data,test_data = train_test_split(df,test_size=0.2,random_state=44)
            train_data.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_data.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Saved train&test data")

            return(self.ingestion_config.train_data_path,self.ingestion_config.test_data_path)

        except Exception as e:
            raise CustomException(e,sys)
        
