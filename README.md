# Graph Machine Learning — GUIGNARD · LASNE · BOUMARD

**Auteurs :** Quentin GUIGNARD · Corentin LASNE · Zacharie BOUMARD  
**Cours :** Graph Machine Learning — CentraleSupélec  
**Notebook :** `GUIGNARD_LASNE_BOUMARD.ipynb`

---

## Structure du projet

```
Graph_ML/
├── GUIGNARD_LASNE_BOUMARD.ipynb   # Notebook principal (Axe 1 + Axe 2)
├── README.md
├── requirements.txt
└── data/
    ├── Planetoid/
    │   └── PubMed/                # Téléchargé automatiquement par PyG
    └── KG20C/
        ├── train.txt
        ├── valid.txt
        ├── test.txt
        ├── all_entity_info.txt
        └── all_relation_info.txt
```

---

## Axe 1 — PubMed (GNN)

**Dataset :** [PubMed Planetoid](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html)

### Caractéristiques du dataset

| Caractéristique | Valeur |
|---|---|
| Nœuds | 19 717 articles scientifiques |
| Arêtes | 44 338 relations de citation |
| Features | 500 (TF-IDF, dictionnaire médical) |
| Classes | 3 (Diabète type 1, type 2, expérimental) |
| Homophilie observée | 0.802 |

### Ce que nous avons exploré

#### 1. Exploration du graphe
- Statistiques de base : densité, degré moyen 4.50, distribution exponentielle
- **Homophilie** : h_obs = 0.802, h_norm = 0.693, z-score = 176.62
- **Centralités** : Degree, PageRank, Betweenness (k=800 paires)

#### 2. Diffusion / Influence
- Independent Cascade (IC) — comparaison de 4 heuristiques de graines : PageRank, Degree, Betweenness, Aléatoire (K=5, proba=0.05, moyenne sur 20 runs)

#### 3. Détection de communautés
- **Louvain** : 38 communautés, modularité Q = **0.7709**
- **k-core** : k_max = 10, 137 nœuds, 1 104 arêtes

#### 4. Shallow Embeddings — baseline
- **Node2Vec** (p=1, q=1) — topologie seule, sans features TF-IDF

#### 5. GNN — Node Classification
- GCN et GraphSAGE avec K couches variables ; over-smoothing visible à K ≥ 4

#### 6. GNN — Link Prediction

| Modèle    | Test AUC | Test AP |
|-----------|----------|---------|
| GCN       | 0.955    | 0.956   |
| GAT       | 0.942    | 0.932   |
| GraphSAGE | 0.860    | 0.862   |

- Split : `RandomLinkSplit(is_undirected=True, num_val=0.1, num_test=0.1)`
- Decoder : produit scalaire normalisé ; Loss : BCE

### Choix non retenus

- **Node Classification** : tâche canonique de PubMed, mais le projet cible la Link Prediction
- **Algorithme glouton MI** : trop coûteux sur 20k nœuds ; PageRank utilisé comme heuristique
- **DeepWalk / Walktrap** : redondants — Node2Vec(p=1,q=1) ≡ DeepWalk ; Louvain > Walktrap sur grands graphes creux
- **Hard Negatives** : `structured_negative_sampling` incompatible avec le pipeline `RandomLinkSplit` actuel
- **Modèles translationnels sur PubMed** : graphe homogène — KGE réservés à l'Axe 2

---

## Axe 2 — KG20C (Knowledge Graph Embedding)

**Dataset :** KG20C — extrait du graphe de connaissances Freebase centré sur ~20 types de personnalités célèbres

### Caractéristiques du dataset

| Caractéristique | Valeur |
|---|---|
| Entités | ~2 000 |
| Types de relations | 21 |
| Triplets train / valid / test | ~22 000 / ~2 800 / ~2 800 |
| Tâche | Complétion de KG + classification downstream |

### Ce que nous avons exploré

#### Partie 1 — Limites d'un GNN classique
- GNN = message-passing homogène ; incapable de modéliser la **sémantique relationnelle** (21 types de relations) sans architecture spécifique

#### Partie 2 — Limites de TransE
- TransE : **h + r ≈ t** — efficace pour relations 1-1, échoue sur les relations **1-N / N-1 / N-N**

#### Partie 3 — Espaces complexes (DistMult, ComplEx, RotatE)
- **DistMult** : score bilinéaire diagonal — symétrique, ne modélise pas les relations asymétriques
- **ComplEx** : extension hermitienne — gère l'asymétrie mais pas l'inversion exacte
- **RotatE** : **h ∘ r = t** dans ℂ — modélise symétrie, antisymétrie, inversion, composition

#### Partie 4 — Entraînement et évaluation (RotatE, `embedding_dim=100`, `num_epochs=50`)

| Métrique   | Valeur  |
|------------|---------|
| MRR        | 0.113   |
| Hits@1     | 4.9 %   |
| Hits@3     | 8.2 %   |
| Hits@10    | 20.0 %  |
| AMRI       | 0.790   |

**Asymétrie tête / queue :**

| Métrique | Head  | Tail  |
|----------|-------|-------|
| Hits@10  | 11.3 % | 28.7 % |

La prédiction de queue est plus facile : l'entité-objet est souvent unique et prévisible ; l'entité-sujet dépend du contexte global. Le rang moyen est nettement supérieur au rang médian (distribution à queue lourde, relations 1-N).

#### Partie 5 — Utilisation downstream
- Embeddings RotatE → **Random Forest** de classification
- Accuracy ~81 %, Macro F1 ~40 % (déséquilibre de classes marqué)
- t-SNE : clusters visibles correspondant aux types d'entités

---

## Installation

```bash
pip install torch torch-geometric pandas networkx python-louvain scikit-learn matplotlib pykeen
# Optionnel (requis pour Node2Vec) :
pip install torch-cluster
```

Ou via le fichier de dépendances :

```bash
pip install -r requirements.txt
```

---

## Références

### Axe 1 — GNN / PubMed
- Yang et al. (2016). *Revisiting Semi-Supervised Learning with Graph Embeddings*. ICML.
- Kipf & Welling (2017). *Semi-Supervised Classification with Graph Convolutional Networks*. ICLR.
- Hamilton et al. (2017). *Inductive Representation Learning on Large Graphs (GraphSAGE)*. NeurIPS.
- Veličković et al. (2018). *Graph Attention Networks*. ICLR.
- Grover & Leskovec (2016). *node2vec: Scalable Feature Learning for Networks*. KDD.
- Blondel et al. (2008). *Fast unfolding of communities in large networks*. J. Stat. Mech.
- Kempe, Kleinberg & Tardos (2003). *Maximizing the spread of influence through a social network*. KDD.

### Axe 2 — KGE / KG20C
- Sun et al. (2019). *RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space*. ICLR.
- Yang et al. (2015). *Embedding Entities and Relations for Learning and Inference in KBs (DistMult)*. ICLR.
- Trouillon et al. (2016). *Complex Embeddings for Simple Link Prediction (ComplEx)*. ICML.
- Bordes et al. (2013). *Translating Embeddings for Modeling Multi-relational Data (TransE)*. NeurIPS.
- Ali et al. (2021). *PyKEEN: A Python Library for Training and Evaluating KGE Models*. JMLR.
