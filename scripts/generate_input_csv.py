import pandas as pd
import numpy as np

def main():
    # read actual training split column names
    X_train = pd.read_parquet('models_artifacts/TCGA-LUAD/splits/X_train.parquet')
    cols = list(X_train.columns)
    n_genes = len(cols)
    n_samples = 20
    
    # generate random expression-like values (uniform 0..5)
    np.random.seed(42)
    data = np.random.uniform(0, 5, size=(n_samples, n_genes))
    df = pd.DataFrame(data, columns=cols)
    
    df.to_csv('input_corrected.csv', index=False)
    print(f'Generated input_corrected.csv with {n_samples} samples and {n_genes} features')

if __name__ == '__main__':
    main()
