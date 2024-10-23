import os 
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transform import DataTransform,DataTransformationConfig

# Creating this classes that will create a train,test and raw data path for us
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts','train.csv')
    test_data_path: str = os.path.join('artifacts','test.csv')
    raw_data_path: str = os.path.join('artifacts','raw.csv')

class DataIngestion:
    # This init will fetch the path that are created above
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    # This is the method created to fetch the data from the source to part it in train test split
    def initiate_data_ingestion(self):
        # Always use try and except to catch any exception is encountered 
        logging.info("Entered data ingestion method or component")
        try:
            # Here we are reading data from the known place instead we can use any database connetion as well
            df = pd.read_csv('Notebook/Data/train.csv')
            logging.info("Loaded the dataset to dataframe")

            # Creating the derectory to store this data, If it is aready exsited it will use it
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            # Writing raw dataset
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train Test split initiated")
            # performing train test split
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)

            # writing train test dataset
            train_set.to_csv(self.ingestion_config.train_data_path,index = False,header = True)
            test_set.to_csv(self.ingestion_config.test_data_path,index = False,header = True)

            logging.info("Train test split completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)
            

if __name__=="__main__":
    obj = DataIngestion()
    train_path,test_path = obj.initiate_data_ingestion()

    data_trans = DataTransform()
    data_trans.initiate_data_transformation(train_path,test_path)
