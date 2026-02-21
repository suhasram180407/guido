import pandas as pd
import numpy as np

def main():
    # read actual training split column names
    X_train = pd.read_parquet('models_artifacts/TCGA-LUAD/splits/X_train.parquet')
    cols = list(X_train.columns)
    n_genes = len(cols)
    n_samples = 20
    
    # generate higher expression values to trigger positive prediction
    # Use values in range 7-15 (much higher than the previous 0-5 range)
    np.random.seed(123)  # different seed for different data
    data = np.random.uniform(7, 15, size=(n_samples, n_genes))
    df = pd.DataFrame(data, columns=cols)
    
    df.to_csv('input_positive.csv', index=False)
    print(f'Generated input_positive.csv with {n_samples} samples and {n_genes} features')
    print(f'Value range: {data.min():.2f} - {data.max():.2f}')

if __name__ == '__main__':
    main()
