import joblib
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def main():
    # load splits
    X_train = pd.read_parquet('models_artifacts/TCGA-LUAD/splits/X_train.parquet')
    y_train = pd.read_parquet('models_artifacts/TCGA-LUAD/splits/y_train.parquet').squeeze()
    
    # drop metadata columns (non-ENSG)
    drop_cols = [c for c in X_train.columns if not c.startswith('ENSG')]
    X_clean = X_train.drop(columns=drop_cols)
    print('Dropped', len(drop_cols), 'metadata cols; now', X_clean.shape[1], 'ENSG columns')
    
    # select top 50 variance genes
    import numpy as np
    variances = X_clean.var(axis=0)
    top_genes = variances.nlargest(50).index.tolist()
    X_small = X_clean.loc[:, top_genes]
    print('Using top 50 variance genes:', X_small.shape)
    
    # rebuild model with smaller alpha to prevent total sparsity
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('enet', ElasticNet(alpha=0.001, l1_ratio=0.3, max_iter=5000, random_state=42))
    ])
    pipe.fit(X_small, y_train)
    
    # save
    joblib.dump(pipe, 'models_artifacts/TCGA-LUAD/primary_model.joblib')
    
    coef = pipe.steps[-1][1].coef_
    nonzero = int((coef!=0).sum())
    print('Saved model artifact with', nonzero, 'nonzero coefficients')
    
    # also cache the clean split (so API uses clean X)
    X_small.to_parquet('models_artifacts/TCGA-LUAD/splits/X_train.parquet')

if __name__ == '__main__':
    main()
