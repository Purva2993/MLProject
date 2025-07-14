import sys
import os

import pandas as pd
import numpy as np
import dill #library which will help us create pickle files
from src.exception import CustomException
from src.logger import logging

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys) from e
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for model_name, model in models.items():
            logging.info(f"Training and tuning model: {model_name}")

            # Get params safely
            params_to_use = param.get(model_name, {})

            if params_to_use:
                # Perform Grid Search if we have params
                gs = GridSearchCV(model, params_to_use, cv=3)
                gs.fit(X_train, y_train)
                best_model = gs.best_estimator_
            else:
                # Otherwise, just fit the model directly
                model.fit(X_train, y_train)
                best_model = model

            # Predict and score
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

            # Replace the model in the dictionary with the trained one
            models[model_name] = best_model

        return report

    except Exception as e:
        raise CustomException(e, sys)
