#!/usr/bin/env python3
import joblib,sys
from pathlib import Path
p=Path('models_artifacts/TCGA-LUAD/primary_model_tuned.joblib')
if not p.exists():
    print('Tuned model missing',p)
    sys.exit(2)
mdl=joblib.load(p)
print('Loaded object type:',type(mdl))
try:
    from sklearn.pipeline import Pipeline
    print('Is Pipeline:', isinstance(mdl, Pipeline))
    if isinstance(mdl, Pipeline):
        last=mdl.steps[-1][1]
        print('Final estimator type:', type(last))
        print('Has predict_proba:', hasattr(last,'predict_proba'))
    else:
        print('Has predict_proba:', hasattr(mdl,'predict_proba'))
except Exception as e:
    print('Inspect error:', e)
