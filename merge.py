import json

with open('GNN.ipynb', 'r', encoding='utf-8') as f:
    nb1 = json.load(f)

with open('KG20C.ipynb', 'r', encoding='utf-8') as f:
    nb2 = json.load(f)

nb1['cells'] += nb2['cells']

with open('GUIGNARD_LASNE_BOUMARD.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb1, f, ensure_ascii=False, indent=1)

print(f'Done - {len(nb1["cells"])} cellules au total')
