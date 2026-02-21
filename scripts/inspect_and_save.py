#!/usr/bin/env python3
import joblib,sys,json
from pathlib import Path
p=Path('models_artifacts/TCGA-LUAD/primary_model_tuned.joblib')
out=Path('results/TCGA-LUAD/inspect_tuned_model.json')
out.parent.mkdir(parents=True,exist_ok=True)
if not p.exists():
    out.write_text(json.dumps({'error':'missing','path':str(p)}))
    sys.exit(2)
mdl=joblib.load(p)
res={'loaded_type':str(type(mdl))}
try:
    from sklearn.pipeline import Pipeline
    res['is_pipeline']=isinstance(mdl, Pipeline)
    if res['is_pipeline']:
        last=mdl.steps[-1][1]
        res['final_estimator']=str(type(last))
        res['has_predict_proba']=hasattr(last,'predict_proba')
    else:
        res['has_predict_proba']=hasattr(mdl,'predict_proba')
except Exception as e:
    res['inspect_error']=str(e)
out.write_text(json.dumps(res,indent=2))
print('Wrote',out)
