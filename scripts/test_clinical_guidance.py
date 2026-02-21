"""Test script to check clinical guidance in prediction response"""
import requests
import json

CSV_PATH = "input_positive.csv"
API_URL = "http://127.0.0.1:8002/api/ui/predict"

try:
    with open(CSV_PATH, "rb") as f:
        files = {"file": (CSV_PATH, f, "text/csv")}
        data = {
            "project_id": "TCGA-LUAD",
            "disease_name": "lung_adenocarcinoma"
        }
        
        response = requests.post(API_URL, files=files, data=data, timeout=120)
        
    print(f"STATUS {response.status_code}\n")
    
    if response.status_code == 200:
        result = response.json()
        
        print(f"Prediction: {result.get('prediction')}")
        prob = result.get('probability')
        if prob is not None:
            print(f"Probability: {prob:.4f}\n")
        else:
            print(f"Probability: {prob}\n")
        
        # Check if report exists
        if "report" in result:
            report = result["report"]
            
            # Print clinical guidance if present
            if "clinical_guidance" in report:
                guidance = report["clinical_guidance"]
                print("="*60)
                print("CLINICAL GUIDANCE")
                print("="*60)
                print(f"\nWhen to Visit Doctor:\n{guidance.get('when_to_visit_doctor')}\n")
                print(f"Follow-up Timeline:\n{guidance.get('follow_up_timeline')}\n")
                
                print("Symptoms to Watch:")
                for symptom in guidance.get('symptoms_to_watch', []):
                    print(f"  • {symptom}")
                
                print("\nLifestyle Recommendations:")
                for rec in guidance.get('lifestyle_recommendations', []):
                    print(f"  • {rec}")
                
                print("\nEmergency Signs (Seek Immediate Care):")
                for sign in guidance.get('emergency_signs', []):
                    print(f"  ⚠️  {sign}")
                print("="*60)
            else:
                print("⚠️  No clinical guidance found in report")
                print("\nReport keys:", list(report.keys()))
                if "error" in report:
                    print(f"Error message: {report['error']}")
        else:
            print("⚠️  No report found in response")
            print("\nResponse keys:", list(result.keys()))
    else:
        print(f"ERROR: {response.text}")
        
except Exception as e:
    print(f"ERR {e}")
