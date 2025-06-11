# 🔥 Prédiction du Risque de Feu en Corse à partir de Données Météo

## 🧠 Objectif

Ce projet vise à **prédire le risque d'incendie (feu)** dans le temps pour chaque zone géographique de la **Corse**, en s’appuyant sur un **modèle de survie** basé sur des données **météorologiques** et des **données d’historique d’incendies**.

---

## 🗂️ Données utilisées

### 🔸 Données d’incendies (BDIFF)
- Source : [BDIFF - Base de Données des Incendies de Forêt en France](https://bdiff.agriculture.gouv.fr/)
- Période : 2006 à 2024
- Variables :
  - Date et lieu du feu
  - Surface brûlée
  - Localisation (commune, latitude, longitude)

### 🔸 Données météorologiques
- Source : [Météo-France](https://donneespubliques.meteofrance.fr/)
- Données quotidiennes par station météo en Corse
- Variables :
  - Température, humidité, vent, précipitations, etc.
  - Données synchronisées avec les dates et localisations des feux

---

## ⚙️ Modélisation

### 📌 Problématique
> Estimer la **probabilité qu’un feu se déclenche dans une zone donnée à un horizon t (7j, 30j, 60j...)**, en fonction des conditions météo récentes.

### 🔍 Modèle principal
- **XGBoost Regressor** avec l’objectif `survival:cox` (modèle de survie)
- Pipeline de traitement avec :
  - Imputation des données manquantes (`SimpleImputer`)
  - Standardisation (`StandardScaler`)
- Apprentissage supervisé à partir de :
  - `event` = feu ou non
  - `duration` = temps d’attente jusqu’au feu

### 🔬 Modèle de base de risque
- Utilisation d’un **modèle de Cox fictif (lifelines)** pour estimer la **fonction de risque cumulatif de base** \( H₀(t) \)
- Calcul final de la **fonction de survie** \( S(t|x) \) via :
  
  \[
  S(t|x) = \exp\left(-H₀(t) \cdot \exp(f(x))\right)
  \]

  où \( f(x) \) est la prédiction du modèle (log hazard ratio).

---

## 🗺️ Visualisation

### 📍 Carte interactive
- Affichage du risque de feu par zone sur une carte (Plotly ScatterMapbox)
- Possibilité de sélectionner l’horizon temporel (7j, 30j, etc.)

### 📈 Courbes de probabilité
- Courbe d’évolution du risque dans le temps pour une **ville sélectionnée**.

---

## 📊 Évaluation

- **C-index (test)** : ~0.80
- Permet de mesurer la capacité du modèle à bien classer les zones par risque relatif.

---

## 🛠️ À venir

- Ajout de données topographiques (pente, altitude, type de végétation)
- Raffinement du modèle (feature engineering avancé, tuning)
- Déploiement d’une application web interactive

---

## 👤 Auteurs

- Projet réalisé dans le cadre d’une reconversion professionnelle en data science
- Développé avec Python, Scikit-learn, XGBoost, Lifelines, Plotly

---




