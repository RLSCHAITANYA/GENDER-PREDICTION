import os
import sys
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from src.utils import evaluate_models,save_object
from src.exception import CustomException
from src.logger import logging

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            xtrain, ytrain, xtest, ytest = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models={
                "Random Forest":RandomForestClassifier(),
                "Decision Tree":DecisionTreeClassifier(),
                "Logistic Regression":LogisticRegression(),
                "Naive Bayes":GaussianNB(),
                "KNN-Classifier":KNeighborsClassifier(),
            }
            params={
                "Random Forest": {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 10]},
                "Decision Tree": {'max_depth': [3, 5, 10]},
                "Logistic Regression": {'C': [0.1, 1, 10]},
                "Naive Bayes": {},
                "KNN-Classifier": {'n_neighbors': [3, 5, 7]}
            }

            model_report:dict = evaluate_models(xtrain,ytrain,xtest,ytest,models,params)

            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')
            # To get best model score from dictionary 
            model_score = list(model_report.values())[0]
            best_model_name = list(model_report.keys())[0]
            best_model = models[best_model_name]

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model

            )
            

            return best_model_name,model_score
        except Exception as e:
            raise CustomException(e,sys)