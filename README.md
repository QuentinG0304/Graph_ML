# Graph Machine Learning — Axe 1 : PubMed (GNN)

**Auteurs :** Quentin GUIGNARD · Corentin LASNE · Zacharie BOUMARD  
**Cours :** Graph Machine Learning — CentraleSupélec  
**Dataset :** [PubMed Planetoid](https://pytorch-geometric.readthedocs.io/en/2.5.1/generated/torch_geometric.datasets.Planetoid.html)

---

## Description du dataset

PubMed est un graphe de **citations scientifiques biomédicales** :

| Caractéristique | Valeur |
|---|---|
| Nœuds | 19 717 articles scientifiques |
| Arêtes | 44 338 relations de citation |
| Features | 500 (TF-IDF, dictionnaire médical) |
| Classes | 3 (Diabète type 1, type 2, expérimental) |
| Homophilie observée | 0.802 |

---

## Ce que nous avons exploré

### 1. Exploration du graphe (*GraphProperties*)
- Statistiques de base (densité, degré moyen, distribution des degrés)
- **Homophilie** : h_obs = 0.802, h_norm = 0.693, z-score = 176.62 — confirmation forte que les articles proches citent des articles de même thématique
- **Centralités** : Degree, PageRank, Betweenness Centrality (approximé sur k=800 paires)

### 2. Diffusion / Influence (*lInfluenceMaximisation*)
- Implémentation du modèle **Independent Cascade (IC)**
- Graines : les 5 articles à plus fort PageRank
- Résultat : **117 nœuds atteints** en **13 étapes de propagation**
- Justification du choix PageRank comme heuristique de sélection (voir section "Non exploré")

### 3. Détection de communautés (*communityDetection*)
- **Louvain** (python-louvain) : **38 communautés**, modularité Q = **0.7709**
- **k-core** : k_max = 10, sous-graphe dense de **137 nœuds** et **1 104 arêtes** (densité × 80 vs graphe complet)

### 4. Shallow Embeddings — baseline (*graphShallowEmbeddings*)
- **Node2Vec** (PyTorch Geometric) avec paramètres p=1, q=1
- Sert de **baseline structurale** : utilise uniquement la topologie du graphe, sans les features TF-IDF
- Comparaison implicite avec les GNNs qui exploitent les deux

> ⚠️ `torch-cluster` requis pour Node2Vec. En cas d'absence, une alternative sklearn (PCA + KMeans sur les features) est proposée automatiquement.

### 5. GNN — Node Classification (*GNN*)
- **GCN_Dynamic** et **GraphSAGE_Dynamic** avec K couches variables (k=1 à 6)
- Analyse de l'**over-smoothing** : la précision plafonne puis chute à partir de K=4-5
- Comparaison loss curves et accuracy GCN vs GraphSAGE

### 6. GNN — Link Prediction (*GNN*)
Pipeline complet avec les trois architectures :

| Modèle | Principe | Val AUC | Test AUC |
|---|---|---|---|
| GCN Encoder | Agrégation spectrale | — | — |
| GraphSAGE Encoder | Agrégation spatiale + sampling | — | — |
| GAT Encoder | Attention multi-têtes (heads=4) | — | — |

> Les valeurs exactes sont affichées dans le notebook après exécution complète.

**Détails d'implémentation :**
- Split : `RandomLinkSplit(is_undirected=True, num_val=0.1, num_test=0.1, add_negative_train_samples=True)` — 50% liens positifs, 50% négatifs générés aléatoirement
- Loss : `F.binary_cross_entropy_with_logits` sur `edge_label`
- Decoder : produit scalaire normalisé entre embeddings de nœuds
- Métriques : **AUC-ROC** et **Average Precision (AP)** (sklearn)

---

## Choix méthodologiques : ce que nous n'avons volontairement pas fait

### Node Classification sur PubMed
> **Pourquoi intéressant :** PubMed est historiquement conçu pour classer les articles dans 3 catégories (Diabète type 1, 2, expérimental) — c'est la tâche canonique du dataset.  
> **Pourquoi non fait :** Le projet se concentre sur la **Link Prediction**. Nous détournons volontairement le dataset de sa tâche principale pour évaluer la capacité des GNNs à prédire des citations (arêtes), plus représentatif des cas d'usage en biomédical (ex. prédiction d'interactions médicamenteuses ou de Drug-Target).

### Algorithme glouton pour la Maximisation d'Influence
> **Pourquoi intéressant :** Le cours présente un algorithme glouton fondé sur la **sous-modularité** pour trouver les k meilleures graines avec garantie d'approximation (1 - 1/e).  
> **Pourquoi non fait :** Sur PubMed (≈20k nœuds, ≈44k arêtes), la complexité est O(k · n · R) avec R simulations Monte-Carlo — prohibitif en pratique. Le **PageRank est utilisé comme heuristique** : les nœuds à fort PageRank sont des hubs de diffusion naturels, ce qui est une approximation standard et justifiée dans la littérature.

### DeepWalk et Walktrap
> **Pourquoi intéressant :** DeepWalk et Walktrap sont des méthodes classiques de marches aléatoires présentées en cours.  
> **Pourquoi non fait :** **Node2Vec généralise DeepWalk** (p=1, q=1 ≡ DeepWalk), il est donc redondant d'implémenter les deux. Pour la détection de communautés, **Louvain est plus rapide et plus précis** que Walktrap sur les grands graphes creux comme PubMed.

### Modèles Translationnels (TransE, RotatE, ComplEx)
> **Pourquoi intéressant :** Ces modèles sont présentés dans le cours pour la complétion de graphes de connaissances (Knowledge Graph Completion).  
> **Pourquoi non fait :** PubMed est un **graphe homogène** (un seul type de nœud : "Article", un seul type de relation : "Cite"). Les modèles translationnels sont conçus pour les **graphes hétérogènes / multi-relationnels**. Ils seront utilisés dans l'**Axe 2** sur des Knowledge Graphs à types multiples de relations.

---

## Extension : Application aux Graphes Biomédicaux (Cancer / PPI)

La même méthodologie est directement transposable aux **réseaux d'interactions protéine–protéine (PPI)** dans la recherche sur le cancer :

| Étape | PubMed (Axe 1) | Cancer / PPI (extension) |
|---|---|---|
| Nœuds | Articles scientifiques | Protéines / gènes |
| Arêtes | Citations | Interactions biologiques |
| Features | TF-IDF (500 mots-clés) | Expression génique, séquence AA |
| Tâche 1 | Link Prediction | Drug-Target Prediction |
| Tâche 2 | Détection de communautés | Identification de complexes protéiques |
| Communautés | Thématiques de recherche | Voies métaboliques / signalisation |

---

## Structure du projet

```
Graph_ML/
├── GNN.ipynb          # Notebook principal (Axe 1 — PubMed)
├── README.md          # Ce fichier
├── requirements.txt   # Dépendances Python
└── data/
    └── Planetoid/
        └── PubMed/    # Téléchargé automatiquement par PyG
```

---

## Installation

```bash
pip install torch torch-geometric pandas networkx python-louvain scikit-learn matplotlib
# Optionnel (requis pour Node2Vec) :
pip install torch-cluster
```

Ou via le fichier de dépendances :

```bash
pip install -r requirements.txt
```

---

## Références

- Yang, Z., Cohen, W. W., & Salakhutdinov, R. (2016). *Revisiting Semi-Supervised Learning with Graph Embeddings*. ICML.
- Kipf, T. N., & Welling, M. (2017). *Semi-Supervised Classification with Graph Convolutional Networks*. ICLR.
- Hamilton, W., Ying, R., & Leskovec, J. (2017). *Inductive Representation Learning on Large Graphs*. NeurIPS.
- Veličković, P. et al. (2018). *Graph Attention Networks*. ICLR.
- Grover, A., & Leskovec, J. (2016). *node2vec: Scalable Feature Learning for Networks*. KDD.
- Blondel, V. D. et al. (2008). *Fast unfolding of communities in large networks*. Journal of Statistical Mechanics.
- Kempe, D., Kleinberg, J., & Tardos, É. (2003). *Maximizing the spread of influence through a social network*. KDD.
