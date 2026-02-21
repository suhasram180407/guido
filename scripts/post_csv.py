import requests
import sys

URL = 'http://127.0.0.1:8001/api/ui/predict'

def main():
    files = {'file': ('input_corrected.csv', open('input_corrected.csv', 'rb'))}
    data = {'project_id': 'TCGA-LUAD', 'disease_name': 'lung_adenocarcinoma'}
    try:
        r = requests.post(URL, files=files, data=data, timeout=180)
        print('STATUS', r.status_code)
        try:
            print(r.json())
        except Exception:
            print(r.text)
    except Exception as e:
        print('ERR', str(e))

if __name__ == '__main__':
    main()
