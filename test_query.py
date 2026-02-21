from src.data.acquisition import _gdc_files_query

hits = _gdc_files_query('TCGA-LUAD')
print(f'✅ Found {len(hits)} files')
if hits:
    print(f'First file: {hits[0]["file_name"]}')
    print(f'Samples of IDs:')
    for h in hits[:3]:
        print(f'  - {h["file_name"][:60]}...')
