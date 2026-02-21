import pandas as pd
from pathlib import Path

P = Path('models_artifacts/TCGA-LUAD/splits/X_train.parquet')
def main():
    df = pd.read_parquet(P)
    print('N_COLUMNS', len(df.columns))
    print('COLUMNS SAMPLE:', list(df.columns)[:50])

if __name__ == '__main__':
    main()
