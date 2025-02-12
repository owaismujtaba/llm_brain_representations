import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import pdb

import config as config
from src.utils.graphics import styled_print

class ModelTrainer:
    def __init__(self, model_name, subject_id, val_size=0.15):
        styled_print("📊", "Initializing ModelTrainer Class", "yellow", panel=True)
        self.name = model_name
        self.subjet_id = subject_id
        self.val_size = val_size
        self.dir = config.TRAINED_DIR
        self.sub_dir = Path(self.dir, subject_id)
        self.model_dir = Path(self.sub_dir,  'Mapping', model_name)
        self.model_path = Path(self.model_dir, f'{model_name}.h5')
        os.makedirs(self.model_dir, exist_ok=True)

        
        print("✅ ModelTrainer Initialization Complete ✅")

    def train_model(self, model, X, y):
        self.model = model
        print("🔧 Starting Model Training 🔧")
        print(f"🟢 Initial Data Shapes: X={X.shape}, y={y.shape}")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        print(f"📊 Training Data Shapes: X_train={X_train.shape}, y_train={y_train.shape}")
        print(f"📊 Test Data Shapes: X_test={X_test.shape}, y_test={y_test.shape}")

        history = self.model.train(X_train, y_train)
        print("✅ Model training completed")
        try:
            history_path = Path(self.model_dir, 'history.csv')
            history.to_csv(history_path)
            print(f"💾 Training history saved at: {history_path}")
           
            model.save(self.model_path)
            print(f"💾 Model saved at: {self.model_path}")
        except:
            print('History saving not allowd')
            self.model_type='Reg'
        self.evaluate_model(X_test, y_test)

    def evaluate_model(self, X, y):
        print("🔍 Evaluating Model 🔍")
        print(f"🟢 Input Data Shapes: X={X.shape}, y={y.shape}")
        if self.model_type =='Reg':
            predictions = self.model.model.predict(X)
        else:
            predictions = self.model.predict(X)
        print(f"📊 Predictions Shape: {predictions.shape}")
        predicted_flat = predictions.flatten()
        y_flatten = y.flatten()
        mse = mean_squared_error(y_flatten, predicted_flat)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_flatten, predicted_flat)

        print(f"📊 RMSE {rmse}, MSE {mse}, 'R2 {r2}")

        np.save(str(Path(self.model_dir, 'metrics.npy')), np.array([mse, rmse, r2]))
        self.metrices = [mse, rmse, r2]
        print(f"💾 Metrics values saved at: {str(Path(self.model_dir, 'metrics.npy'))}")