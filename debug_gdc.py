import requests
import json

GDC_ENDPOINT = 'https://api.gdc.cancer.gov'
project_id = 'TCGA-LUAD'

# Test 1: Find all expression files
print("=== Query 1: All expression files ===")
filters = {
    "op": "in",
    "content": {"field": "cases.project.project_id", "value": [project_id]}
}
params = {"filters": json.dumps(filters), "format": "JSON", "size": "10"}
resp = requests.get(f"{GDC_ENDPOINT}/files", params=params, timeout=30)
data = resp.json()
print(f"Total files: {data['data']['pagination']['total']}")

# Test 2: Get just expression files
print("\n=== Query 2: Expression quantification files ===")
filters = {
    "op": "and",
    "content": [
        {"op": "in", "content": {"field": "cases.project.project_id", "value": [project_id]}},
        {"op": "=", "content": {"field": "data_type", "value": "Gene Expression Quantification"}},
    ]
}
params = {"filters": json.dumps(filters), "format": "JSON", "size": "1", "sort": "file_name"}
resp = requests.get(f"{GDC_ENDPOINT}/files", params=params, timeout=30)
data = resp.json()
total = data['data']['pagination']['total']
print(f"Expression files: {total}")

if data['data']['hits']:
    first = data['data']['hits'][0]
    print(f"First file: {first['file_name']}")
    print(f"Workflow: {first.get('analysis', {}).get('workflow_type', 'N/A')}")

# Test 3: Fetch multiple to see workflow types
print("\n=== Query 3: Check workflow types ===")
params = {"filters": json.dumps(filters), "format": "JSON", "size": "50"}
resp = requests.get(f"{GDC_ENDPOINT}/files", params=params, timeout=30)
data = resp.json()
workflows = {}
for hit in data['data']['hits']:
    wf = hit.get('analysis', {}).get('workflow_type', 'Unknown')
    workflows[wf] = workflows.get(wf, 0) + 1

print(f"Workflow type distribution:")
for wf, count in sorted(workflows.items(), key=lambda x: -x[1]):
    print(f"  {wf}: {count}")
