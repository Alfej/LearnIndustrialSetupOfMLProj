import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler,OrdinalEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import os

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransform:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        try:
            label_encode_cols = [
                'loan_grade'
            ]
            OneHot_encode_cols = [
                'loan_intent'
                ,'cb_person_default_on_file'
                ,'person_home_ownership'
            ]

            oneHot_pipeline = Pipeline(
                steps=[
                  ("one_hot_encoder",OneHotEncoder())
                ]
            )
            LabelEncode_pipeline = Pipeline(
                steps=[
                  ("label_encoder",OrdinalEncoder())
                ]
            )
            
            preprocessor = ColumnTransformer(
                [
                    ("Label_encode_pipeline",LabelEncode_pipeline,label_encode_cols),
                    ("oneHot_encode_pipeline",oneHot_pipeline,OneHot_encode_cols)
                ]
            )

            logging.info("Categorical columns encoding completed")

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test completed")

            logging.info("obtaining preprocessing oject")

            preprocessing_obj = self.get_data_transformer_object()

            target_col = 'loan_status'

            input_feature_train_df = train_df.drop(columns=[target_col,'id'],axis=1)
            target_feature_train_df = train_df[target_col]

            input_feature_test_df = test_df.drop(columns=[target_col,'id'],axis=1)
            target_feature_test_df = test_df[target_col]


            logging.info("Applying preprocessing on train data")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr,np.array(input_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr,np.array(input_feature_test_df)
            ]

            logging.info("completed transformation")

            logging.info("Saving object")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj

            )

            logging.info("Object is saved")

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
             
        except Exception as e:
            raise CustomException(e,sys)
        
        