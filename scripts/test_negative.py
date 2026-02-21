"""Test negative prediction with fallback verification"""
import requests
import json

CSV_PATH = "input_corrected.csv"
API_URL = "http://127.0.0.1:8002/api/ui/predict"

with open(CSV_PATH, "rb") as f:
    resp = requests.post(
        API_URL,
        files={"file": (CSV_PATH, f, "text/csv")},
        data={"project_id": "TCGA-LUAD", "disease_name": "lung_adenocarcinoma"},
        timeout=120,
    )

print(f"STATUS {resp.status_code}")
r = resp.json()
pred = r.get("prediction", {})
print(f"Prediction label: {pred.get('label')}")
print(f"Prediction prob:  {pred.get('prob')}")

rpt = r.get("report", {})
verdict = rpt.get("overall_system_verdict", "N/A")
print(f"\nVerdict: {verdict[:150]}")

cg = rpt.get("clinical_guidance", {})
print(f"\nWhen to visit doctor: {cg.get('when_to_visit_doctor', 'N/A')[:150]}")
print(f"Symptoms to watch: {len(cg.get('symptoms_to_watch', []))} items")
print(f"Lifestyle recs:    {len(cg.get('lifestyle_recommendations', []))} items")
print(f"Emergency signs:   {len(cg.get('emergency_signs', []))} items")

# Check for bad patterns
s = json.dumps(rpt).upper()
bad = ["INSUFFICIENT VALIDATED", "FAILED:", "NOT APPLICABLE", "UNABLE TO EVALUATE", "[OBJECT OBJECT]"]
found = [p for p in bad if p in s]
if found:
    print(f"\n!! BAD PATTERNS FOUND: {found}")
else:
    print("\n✓ No bad patterns — report is clean!")
