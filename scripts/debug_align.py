import joblib
import pandas as pd
from pathlib import Path
import numpy as np

def main():
    # load model and training split
    model = joblib.load('models_artifacts/TCGA-LUAD/primary_model.joblib')
    X_train = pd.read_parquet('models_artifacts/TCGA-LUAD/splits/X_train.parquet')
    print('TRAINING COLS:', len(X_train.columns))
    print('SAMPLE COLS:', list(X_train.columns)[:20])
    
    # extract final estimator coefficients
    if hasattr(model, 'steps'):
        est = model.steps[-1][1]
        coef = getattr(est, 'coef_', None)
        if coef is not None:
            coef = np.asarray(coef).ravel()
            print('COEF SHAPE:', coef.shape)
            print('COEF NONZEROS:', int((coef != 0).sum()))
            print('COEF MIN/MAX:', float(coef.min()), float(coef.max()))
            # top 5 by abs
            idx = np.argsort(-np.abs(coef))[:5]
            print('TOP 5 COEF IDS:', idx.tolist())
            print('TOP 5 VALUES:', coef[idx].tolist())
            print('TOP 5 NAMES:', [X_train.columns[i] for i in idx])

if __name__ == '__main__':
    main()
