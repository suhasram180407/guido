import requests
import sys

URL = 'http://127.0.0.1:8002/api/ui/predict'

def main():
    files = {'file': ('input_positive.csv', open('input_positive.csv', 'rb'))}
    data = {'project_id': 'TCGA-LUAD', 'disease_name': 'lung_adenocarcinoma'}
    try:
        r = requests.post(URL, files=files, data=data, timeout=180)
        print('STATUS', r.status_code)
        if r.status_code == 200:
            result = r.json()
            pred = result.get('prediction', {})
            print(f"\nPrediction: {pred.get('label')}")
            print(f"Probability: {pred.get('prob')}")
            print(f"Model: {pred.get('model')}")
            
            # Show top 5 features
            features = result.get('features', [])[:5]
            if features:
                print("\nTop 5 features:")
                for f in features:
                    print(f"  {f['name']}: {f['value']:.4f}")
        else:
            print(r.text)
    except Exception as e:
        print('ERR', str(e))

if __name__ == '__main__':
    main()
