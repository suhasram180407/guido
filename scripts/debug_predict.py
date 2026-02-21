import joblib
import pandas as pd
import numpy as np

def main():
    # step-by-step reproduce the API flow
    model = joblib.load('models_artifacts/TCGA-LUAD/primary_model.joblib')
    X_train = pd.read_parquet('models_artifacts/TCGA-LUAD/splits/X_train.parquet')
    
    # load CSV as in the API
    df = pd.read_csv('input.csv')
    print('INPUT CSV SHAPE:', df.shape)
    print('INPUT CSV COLS SAMPLE:', list(df.columns)[:10])
    
    # mimic the API reindex call
    sample = df.iloc[0:1]
    train_cols = list(X_train.columns)
    X = sample.reindex(columns=train_cols, fill_value=0)
    print('REINDEXED X SHAPE:', X.shape)
    print('REINDEXED X COLS SAMPLE:', list(X.columns)[:10])
    
    # run predict
    pred = model.predict(X)
    print('PREDICTION:', pred)
    
    # extract coef
    if hasattr(model, 'steps'):
        est = model.steps[-1][1]
        coef = getattr(est, 'coef_', None)
        if coef is not None:
            coef = np.asarray(coef).ravel()
            print('COEF SHAPE:', coef.shape, '| TRAIN_COLS:', len(train_cols))
            print('COEF NONZEROS:', int((coef != 0).sum()))
            print('COEF MIN/MAX:', float(coef.min()), float(coef.max()))
            # check X shape alignment
            print('X.shape[1]:', X.shape[1], 'vs coef shape:', coef.shape)

if __name__ == '__main__':
    main()
