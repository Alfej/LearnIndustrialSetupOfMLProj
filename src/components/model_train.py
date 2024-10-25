import os
import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from perpetual import PerpetualBooster
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainingConfig:
    trained_model_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.modelTrainingconfig = ModelTrainingConfig()

    def initiateModelTraining(self, train_array, test_array):
        try:
            logging.info("Splitting train and test input data")
            x_train,y_train,x_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            # print(pd.DataFrame(x_train).info())
            models = {
                        # "logistic regression":LogisticRegression(),
                        "Xgboost": XGBClassifier(),
                        "Decision Tree": DecisionTreeClassifier(),
                        "KNN classification":KNeighborsClassifier(),
                        "Randomforest classification":RandomForestClassifier(),
                        "catboost classification":CatBoostClassifier(),
                        "ExtraTreeClassifier": ExtraTreeClassifier(),
                        "AdaBoostClassifier": AdaBoostClassifier(),
                        "GradientBoostingClassifier": GradientBoostingClassifier(),
                        "Light GBM": LGBMClassifier(),
                        "PerpetualBooster": PerpetualBooster()
                    }
            logging.info("experimenting with models")
            model_report: dict = evaluate_model(x_train = x_train,y_train=y_train,x_test = x_test,y_test=y_test,models = models)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            
            logging.info(f"Found best model {best_model_name}")

            save_object(file_path=self.modelTrainingconfig.trained_model_path, obj= best_model)

            predicted_values = best_model.predict(x_test)
            roc_auc_value = roc_auc_score(y_test,predicted_values)

            return roc_auc_value

        except Exception as e:
            raise CustomException(e,sys)
            
