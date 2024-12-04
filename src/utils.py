import os
import sys 
import numpy as np
import pandas as pd
import pickle
from src.exception import CustomException
from src.logger import logging

def save_model(file_path,obj):
    try:
        with open(file_path,"wb") as f:
            pickle.dump(obj,f)
    except Exception as e:
        raise CustomException(e,sys)