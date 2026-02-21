import requests, json, pprint

url = 'http://127.0.0.1:8001/api/ui/predict'
# Example sample with multiple genes
sample = {"TP53": 2.3, "EGFR": 0.5, "KRAS": 1.1, "BRCA1": 0.9, "BRCA2": 1.0, "GENE1": 0.7}
print('SENT SAMPLE:', json.dumps(sample))
try:
    r = requests.post(url, data={'project_id':'TCGA-LUAD', 'disease_name':'lung_adenocarcinoma', 'json_payload':json.dumps(sample)}, timeout=120)
    print('STATUS', r.status_code)
    try:
        pprint.pprint(r.json())
    except Exception:
        print(r.text)
except Exception as e:
    print('REQUEST ERROR:', str(e))
