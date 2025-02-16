import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
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
        print("🔧 Starting Model Training with 5-Fold Cross Validation 🔧")
        print(f"🟢 Initial Data Shapes: X={X.shape}, y={y.shape}")

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"\n🔄 Fold {fold + 1}/5")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            #print(f"📊 Training Data Shapes: X_train={X_train.shape}, y_train={y_train.shape}")
            #print(f"📊 Validation Data Shapes: X_val={X_val.shape}, y_val={y_val.shape}")

            history = self.model.train(X_train, y_train)
            

            
            try:
                history_path = Path(self.model_dir, 'history.csv')
                history.to_csv(history_path)
                print(f"💾 Training history saved at: {history_path}")

                model.save(self.model_path)
                print(f"💾 Model saved at: {self.model_path}")
            except:
                print('⚠️ History saving not allowed')
                self.model_type = 'Reg'
            score = self.evaluate_model(X_val, y_val, fold)
            print(f"✅ Fold {fold + 1} Score: {score}")
            fold_scores.append(score)
        
       
    def evaluate_model(self, X, y, fold):
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
        pcc = np.corrcoef(y_flatten, predicted_flat)[0, 1]

        print(f"📊 RMSE {rmse}, MSE {mse}, 'R2 {r2}, PCC {pcc}")


        #mse_per_sample = [mean_squared_error(y[i], predictions[i]) for i in range(y.shape[0])]
        ##rmse_per_sample = [np.sqrt(mse) for mse in mse_per_sample]
        #2_per_sample = [r2_score(y[i], predictions[i]) for i in range(y.shape[0])]

        #print(r2_per_sample)

        np.save(str(Path(self.model_dir, f'Fold_{fold}_metrics.npy')), np.array([mse, rmse, r2, pcc]))
        self.metrices = [mse, rmse, r2]
        print(f"💾 Metrics values saved at: {str(Path(self.model_dir, f'Fold_{fold}_metrics.npy'))}")