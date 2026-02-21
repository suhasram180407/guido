import joblib
from pathlib import Path

MODEL_PATH = Path('models_artifacts/TCGA-LUAD/primary_model.joblib')

def main():
    m = joblib.load(MODEL_PATH)
    print('MODEL TYPE:', type(m))
    print('HAS predict_proba:', hasattr(m, 'predict_proba'))
    print('HAS decision_function:', hasattr(m, 'decision_function'))
    print('HAS coef_:', hasattr(m, 'coef_'))
    print('HAS feature_importances_:', hasattr(m, 'feature_importances_'))
    # if sklearn Pipeline
    try:
        steps = getattr(m, 'steps', None)
        if steps:
            print('Pipeline steps:')
            for name, step in steps:
                print(' -', name, type(step))
                print('   attrs:', [a for a in dir(step) if not a.startswith('_')][:20])
    except Exception as e:
        print('Error inspecting steps:', e)

    # print coefficient stats from final estimator if present
    try:
        if getattr(m, 'steps', None):
            final = m.steps[-1][1]
            coef = getattr(final, 'coef_', None)
            if coef is not None:
                import numpy as _np
                coef = _np.asarray(coef).ravel()
                print('FINAL ESTIMATOR:', type(final))
                print('COEF SHAPE:', coef.shape)
                print('COEF MIN/MAX/MEAN:', float(coef.min()), float(coef.max()), float(coef.mean()))
                print('COEF NONZERO COUNT:', int((coef != 0).sum()))
            else:
                print('Final estimator has no coef_')
    except Exception as e:
        print('Error reading final estimator coef:', e)
    except Exception as e:
        print('Error inspecting steps:', e)

if __name__ == '__main__':
    main()
