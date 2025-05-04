# 🧠 Perceptron Multicouche — Implémentation Manuelle en Python

Ce projet propose une implémentation à la main d’un perceptron multicouche (**MLP**, Multi-Layer Perceptron) réalisée en Python, sans utiliser de bibliothèques de deep learning comme TensorFlow ou PyTorch. L'objectif est de comprendre de manière concrète et mathématique le fonctionnement interne d’un réseau de neurones, depuis la propagation avant jusqu'à la rétropropagation du gradient.

## 📁 Structure du projet

Le répertoire principal contient les fichiers suivants :

- `network_trainning_perception.py` : Script principal d'entraînement du réseau  
- `network_inference_percpetron.py` : Script d'inférence (prédiction) sur de nouvelles données  
- `data_0.npz` : Données d'entraînement et de test (contenant `x_train`, `y_train`, etc.)  
- `W_up0.npy`, `W_up1.npy`, etc. : Fichiers de poids du réseau sauvegardés après l'entraînement

## 🧮 Présentation du perceptron

Le **perceptron** est une unité de calcul neuronale qui applique une transformation linéaire suivie d'une fonction d'activation. Formellement, il fonctionne ainsi :

- `z = w·x + b` : combinaison linéaire des entrées `x` avec des poids `w` et biais `b` (le biais n’est pas utilisé ici)  
- `a = σ(z)` : application d'une fonction d'activation, ici la **sigmoïde** : `σ(z) = 1 / (1 + exp(-z))`

Ce modèle simple permet de classer des données **linéairement séparables**. Pour résoudre des problèmes plus complexes, on utilise plusieurs couches de perceptrons.

## 🧱 Réseaux de neurones multicouches

Le **perceptron multicouche** est un empilement de plusieurs couches de neurones :

- Une **couche d'entrée** (ex. les pixels d'une image ou un vecteur de caractéristiques)  
- Une ou plusieurs **couches cachées** (apprentissage de représentations intermédiaires)  
- Une **couche de sortie** (représentation finale, souvent sous forme de vecteur de classes)

Chaque couche applique la transformation suivante :

- `a(i+1) = σ(W(i) · a(i))`

où :
- `W(i)` est la matrice des poids de la couche `i`  
- `a(i)` est le vecteur d’activations de la couche `i`  
- `σ` est la fonction d’activation (sigmoïde ici)

## 🔁 Apprentissage par rétropropagation

L’apprentissage repose sur une boucle en 4 étapes :

1. **Propagation avant (forward)** : Les données sont passées à travers toutes les couches pour produire une sortie.  
2. **Calcul de l’erreur** : On compare la sortie réelle à la sortie attendue (label) via une fonction de perte.  
3. **Rétropropagation du gradient** : L’erreur est propagée couche par couche dans le sens inverse afin de calculer les gradients.  
4. **Mise à jour des poids** : Chaque poids est ajusté selon l’algorithme de descente de gradient.

### Détails mathématiques

- Fonction d’activation : `σ(z) = 1 / (1 + exp(-z))`  
- Dérivée de la sigmoïde : `σ'(z) = σ(z) * (1 - σ(z))`  
- Calcul du gradient d’erreur sur la couche de sortie : `δ_L = y - a_L`  
- Propagation de l’erreur dans les couches : `δ_l = (W_{l+1} · δ_{l+1}) ⊙ σ'(a_l)`  
- Mise à jour des poids : `W = W + α · (a_prev^T · δ)`

où :
- `⊙` désigne le produit élément par élément  
- `α` est le taux d’apprentissage

## ⚙️ Explication du code

Le code fonctionne sans dépendances lourdes (juste `numpy`, `matplotlib`, et `tqdm`).

### Initialisation du réseau

Les poids sont initialisés aléatoirement pour chaque couche dans la fonction `structure_network0`. Chaque couche reçoit une matrice de poids de taille adaptée au nombre de neurones.

### Entraînement

La fonction `trainning` gère une passe complète d’entraînement sur une donnée :
- Calcul des activations pour chaque couche (forward)  
- Calcul de l’erreur finale  
- Propagation des gradients en remontant  
- Mise à jour des poids par la règle de la descente de gradient

Les activations et gradients sont stockés dans des listes pour chaque couche.

L’apprentissage global est effectué sur toutes les données d'entraînement, via des boucles sur les images (`steps_data`) et des itérations internes (`steps_in`).

### Prédiction

Le script `network_inference_percpetron.py` applique la fonction `prediction` qui reproduit simplement la propagation avant sur une donnée de test. On compare ensuite la sortie du réseau avec le label réel.

### Évaluation

Le taux de réussite est évalué sur les données de test via la proportion de prédictions correctes.

Extrait de sortie attendu :

> Le taux de succès des prédictions est de 96.5%

## ✅ À retenir

- Ce projet met en œuvre **tout le mécanisme d’un réseau de neurones** à la main : propagation, activation, dérivées, gradients, mises à jour.  
- Le format choisi (sans bibliothèques haut niveau) est idéal pour l’apprentissage et l’expérimentation.  
- Le modèle fonctionne sur deux classes (classification binaire), mais peut être généralisé.

## 🚀 Pistes d’amélioration

- Utiliser plus de classes (ex. 10 pour MNIST)  
- Ajouter d'autres fonctions d’activation (ReLU, tanh)  
- Ajouter une régularisation (L2)  
- Utiliser des mini-batchs au lieu de traiter une image à la fois  
- Ajouter une interface visuelle pour explorer les prédictions

## 👤 Auteur

Julien de Saint Angel
* [Julien de SAINT ANGEL](mailto:juliencine17@gmail.com)

