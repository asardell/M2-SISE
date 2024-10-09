# Chapitre 5 : Les bases du machine learning

## Objectifs

Voici les objectifs de ce module :
- [x] Différencier classification et régression
- [x] Evaluer un modèle
- [x] Optimiser les paramètres d'un modèle
- [x] Interpréter un modèle

- [Chapitre 5 : Les bases du machine learning](#chapitre-5--les-bases-du-machine-learning)
  - [Objectifs](#objectifs)
  - [Classification](#classification)
    - [Variable cible](#variable-cible)
    - [Variables explicatives](#variables-explicatives)
    - [Echantillonnage](#echantillonnage)
    - [Arbre de décision](#arbre-de-décision)
    - [Evaluation de modèle](#evaluation-de-modèle)
    - [Validation croisée](#validation-croisée)
    - [Optimisation des paramètres](#optimisation-des-paramètres)
    - [Courbe ROC pour comparer des modèles](#courbe-roc-pour-comparer-des-modèles)
    - [D'autres méthodes](#dautres-méthodes)
      - [KNN](#knn)
      - [Régression logistique](#régression-logistique)
      - [Random Forest](#random-forest)
      - [Aller plus loin avec SMOTE pour ré équilibrer les classes](#aller-plus-loin-avec-smote-pour-ré-équilibrer-les-classes)
      - [Aller plus loin avec SHAP pour interpréter ses modèles](#aller-plus-loin-avec-shap-pour-interpréter-ses-modèles)
  - [Régression](#régression)
    - [Variable cible](#variable-cible-1)
    - [Variables explicatives](#variables-explicatives-1)
    - [Echantillonnage](#echantillonnage-1)
    - [Régression linéaire simple/multiple](#régression-linéaire-simplemultiple)
    - [Evaluation de modèle](#evaluation-de-modèle-1)
    - [D'autres méthodes](#dautres-méthodes-1)
      - [Ridge](#ridge)
      - [Lasso](#lasso)
      - [Elasticnet](#elasticnet)
      - [Arbre de régression](#arbre-de-régression)
  - [Liens utiles](#liens-utiles)


## Classification

### Variable cible

1. Analyse de la répartition de la variable cible (`Etiquette_DPE`)

```python
df['Etiquette_DPE'].value_counts(normalize=True)
```

2. Recode la variable cible en une nouvelle colonne binaire `passoire_energetique`

```python
df['passoire_energetique'] = df['Etiquette_DPE'].isin(['F', 'G'])
```
3. Analyse de la répartition de la nouvelle variable binaire `passoire_energetique`

```python
df['passoire_energetique'].value_counts(normalize=True)
```

### Variables explicatives

1. Selection des variables explicatives

```python
# Vérification des données manquantes
ls_variables_explicatives = ['Année_construction','Surface_habitable_logement','Coût_total_5_usages','Coût_ECS','Coût_chauffage','Coût_éclairage','Coût_auxiliaires','Coût_refroidissement','Type_énergie_n°1']
```

2. Inspection des Données Manquantes

```python
# Vérification des données manquantes
df[ls_variables_explicatives].isnull().sum()
#Remplacer les valeurs NA par la moyenne des colonnes
for col in ls_variables_explicatives:
    try:
        df[col] = df[col].fillna(df[col].mean())
    except:
        print(f"Erreur sur la colonne {col}")
```

3.  Analyse des Distributions des Variables Explicatives

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Distribution de la surface
plt.figure(figsize=(8, 6))
sns.histplot(df['Surface_habitable_logement'], bins=30, kde=True)
plt.title('Distribution de la Surface habitable')
plt.xlabel('Surface (m²)')
plt.ylabel('Fréquence')
plt.show()
```

4. Analyse des Corrélations entre les Variables Explicatives

```python
# Calcul de la matrice de corrélation
corr_matrix = df[ls_variables_explicatives[:-1]].corr()

# Affichage de la matrice de corrélation sous forme de heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Matrice de Corrélation')
plt.show()
```

5. Analyse des liens entre les Variables Explicatives et la Variable Cible

```python
# Boxplot de chaque variable explicative par rapport à la passoire énergétique
for col in ls_variables_explicatives[:-1]:
    plt.figure(figsize=(8, 6))
    
    # Tracer le boxplot sans les outliers
    sns.boxplot(x='passoire_energetique', y=col, data=df, showfliers=False)
    
    # Automatiser le titre avec le nom de la colonne
    plt.title(f'Boxplot de {col} en Fonction de la Passoire Énergétique')
    
    plt.xlabel('Passoire Énergétique (True = F/G, False = A à E)')
    plt.ylabel(f'{col}')  # Automatiser l'étiquette de l'axe y avec le nom de la colonne
    plt.show()

```
6. Encodage des variables catégorielles

```python
# Concaténer les deux listes : ls_variables_explicatives et ['passoire_energetique']
df = df[ls_variables_explicatives + ['passoire_energetique']]
df = pd.get_dummies(df, columns=['Type_énergie_n°1'], drop_first=True)
```

### Echantillonnage

1. Créer un objet `X` avec les variables explicatives

```python
# Utiliser set.difference() pour exclure la colonne cible de ls_variables_explicatives
X =df[df.columns.difference(['passoire_energetique'])]
```

2. Crée un objet `Y` avec la variable à expliquer

```python
Y = df['passoire_energetique']
```

3.  Scinder l'échantillon en train / test

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                    test_size = 0.30,
                                                    stratify = Y,
                                                    random_state = 42)
```

4. Répartition des classes dans `y_train` et `y_test`

```python
y_train.value_counts(normalize=True)
y_test.value_counts(normalize=True)
```

5. Afficher un extrait de  `X_train` et `X_test`
   
```python
print(X_train.shape)
print(X_test.shape)
X_test.head()
```

### Arbre de décision

1. Lancer l'apprentissage du modèle sur l'échantillon d'entrainement

```python
from sklearn.tree import DecisionTreeClassifier
model_arbre = DecisionTreeClassifier(max_depth=3 , min_samples_leaf=50, min_samples_split=100)
model_arbre = model_arbre.fit(X_train,y_train)
```

2. Afficher l'arbre de décision

```python
from sklearn.tree import plot_tree
plt.figure(figsize=(16,4))
plot_tree(model_arbre,feature_names = list(X.columns),filled=True, fontsize=10)
plt.show()
```

3. Prédire sur l'échantillon test

```python
y_pred = model_arbre.predict(X_test)
y_pred
```

4. Afficher les probabilités d'appartenance aux classes

```python
y_pred_proba = model_arbre.predict_proba(X_test)
y_pred_proba[0:10]
```

### Evaluation de modèle

1. Calculer la matrice de confusion

```python
from sklearn.metrics import confusion_matrix
mc = pd.DataFrame(confusion_matrix(y_test,y_pred),
                  columns=['pred_0','pred_1'],
                  index=['obs_0','obs_1'])

mc

#ou 

pd.crosstab(y_test,y_pred, colnames=['pred'], rownames=['obs'], margins=True)
```

2. Calculer le taux de bonne prédiction

```python
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))
```

3. Calculer le rappel et la précision

```python
from sklearn.metrics import recall_score, precision_score
print('recall :' + str(recall_score(y_test,y_pred,average='binary', pos_label=0)))
print('precision : ' + str(precision_score(y_test,y_pred, average='binary', pos_label=0)))
```

4. Calculer le `f1-score`

```python
from sklearn.metrics import f1_score
print('f1_score : ' + str(f1_score(y_test,y_pred, average='binary', pos_label=0)))
```

5. Calculer ces métriques avec l'approche macro non pondéré

```python
print('recall : ' + str(recall_score(y_test,y_pred,average='macro')))
print('precision : ' + str(precision_score(y_test,y_pred, average='macro')))
print('f1_score : ' + str(f1_score(y_test,y_pred, average='macro')))
```

6. Calculer ces métriques avec l'approche macro pondéré

```python
print('recall : ' + str(recall_score(y_test,y_pred,average='weighted')))
print('precision : ' + str(precision_score(y_test,y_pred, average='weighted')))
print('f1_score : ' + str(f1_score(y_test,y_pred, average='weighted')))
```

### Validation croisée

```python
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import classification_report

# Création d'un modèle d'arbre de décision
model_arbre_cv = DecisionTreeClassifier(random_state=42)

# Mise en œuvre de la validation croisée
# Ici, nous utilisons une validation croisée à 5 plis
cv_scores = cross_val_score(model_arbre_cv, X_train, y_train, cv=5, scoring='f1_macro')

# Affichage des scores de validation croisée
print(f"Scores de validation croisée : {cv_scores}")
print(f"Moyenne des scores de validation croisée : {cv_scores.mean()}")

# Entraînement du modèle sur l'ensemble d'entraînement
model_arbre_cv.fit(X_train, y_train)

# Prédictions sur l'ensemble de test
y_pred_arbre_cv = model_arbre_cv.predict(X_test)

# Évaluation du modèle
print(classification_report(y_test, y_pred_arbre_cv))
print(f"Accuracy sur l'ensemble de test : {accuracy_score(y_test, y_pred_arbre_cv)}")
```

### Optimisation des paramètres

1. Configurer les fenêtres de recherches

```python
import numpy as np
#cette fois-ci on utilise numpy pour générer des séquences à la place des listes
parameters = {'max_depth' : np.arange(start = 1, stop = 10, step = 1) ,
              'min_samples_leaf' : np.arange(start = 5, stop = 250, step = 50),
              'min_samples_split' : np.arange(start = 10, stop = 500, step = 50)}
# Calculer le nombre de valeurs pour chaque paramètre
total_combinaisons = (
    len(parameters['max_depth']) *
    len(parameters['min_samples_leaf']) *
    len(parameters['min_samples_split'])
)

print(f"Nombre total de modèles à tester: {total_combinaisons}")
```

2. Apprentissage avec Grid Search

```python
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
model_arbre_grid = DecisionTreeClassifier()
score = make_scorer(f1_score, pos_label=0)
model_arbre_grid = GridSearchCV(model_arbre_grid, parameters, scoring = score, verbose = 2, cv = 5)
model_arbre_grid.fit(X_train, y_train)
```

3. Afficher le meilleur modèle

```python
print("Voici les paramètres du meilleure modèle : " + str(model_arbre_grid.best_estimator_))
print("Voici le "  + str(model_arbre_grid.scorer_) + " du meilleure modèle : " + str(model_arbre_grid.best_score_))
```

4. Prédire avec la meilleure combinaison de paramètres

```python
# Prédictions sur l'ensemble de test
y_pred_arbre_grid = model_arbre_grid.predict(X_test)
```

5. Evaluer le modèle avec la meilleure combinaison de paramètres

```python
# Évaluation du modèle
print(classification_report(y_test, y_pred_arbre_grid))
print(f"Accuracy sur l'ensemble de test : {accuracy_score(y_test, y_pred_arbre_grid)}")
print('recall :' + str(recall_score(y_test,y_pred,average='binary', pos_label=0)))
print('precision : ' + str(precision_score(y_test,y_pred, average='binary', pos_label=0)))
print('f1_score : ' + str(f1_score(y_test,y_pred, average='binary', pos_label=0)))
```

### Courbe ROC pour comparer des modèles

1. Obtener les probabilités de prédiction  des modèles

```python
# Prédire les probabilités pour la classe positive (1)
y_proba_cv = model_arbre_cv.predict_proba(X_test)[:, 1]
y_proba_grid = model_arbre_grid.predict_proba(X_test)[:, 1]
```

2. Calculer les Valeurs de la Courbe ROC

```python
from sklearn.metrics import roc_curve, roc_auc_score
# Calculer les courbes ROC
fpr1, tpr1, _ = roc_curve(y_test, y_proba_cv)
fpr2, tpr2, _ = roc_curve(y_test, y_proba_grid)

# Calculer l'AUC pour chaque modèle
auc1 = roc_auc_score(y_test, y_proba_cv)
auc2 = roc_auc_score(y_test, y_proba_grid)
```

4. Tracer les Courbes ROC

```python
import matplotlib.pyplot as plt
# Tracer les courbes ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr1, tpr1, label=f'Arbre CV (AUC = {auc1:.2f})', color='blue')
plt.plot(fpr2, tpr2, label=f'Arbre Grid (AUC = {auc2:.2f})', color='green')

# Ajouter la diagonale (aléatoire)
plt.plot([0, 1], [0, 1], 'k--', label='Modèle aléatoire')

# Ajouter des labels et un titre
plt.xlabel('Taux de Faux Positifs (FPR)')
plt.ylabel('Taux de Vrais Positifs (TPR)')
plt.title('Comparaison des courbes ROC entre deux modèles')
plt.legend(loc='lower right')

# Afficher le graphique
plt.show()
```

### D'autres méthodes
#### KNN

1. Modéliser avec la méthode des K plus proches voisins

```python
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

y_pred_knn = knn_model.predict(X_test)
print('f1_score : ' + str(f1_score(y_test,y_pred_knn,average='binary', pos_label=1)))
```

#### Régression logistique

1. Modéliser avec la méthode de régression logistique
   
```python
from sklearn.linear_model import LogisticRegression

reg_log = LogisticRegression()
reg_log_model = reg_log.fit(X_train, y_train)
y_pred_reg = reg_log_model.predict(X_test)

print('f1_score : ' + str(f1_score(y_test,y_pred_reg,average='binary', pos_label=1)))
```
2. Afficher les probabilités d'appartenance aux classes

```python
reg_log_model.predict_proba(X_test)[0:10]
```


3. Afficher les coefficients du modèle

```python
coef = pd.DataFrame(reg_log_model.coef_[0,] ,index = X_train.columns, columns=['Coef'])
coef.loc['Constante'] = reg_log_model.intercept_
coef
```


#### Random Forest

1. Modéliser avec la méthode des forêts aléatoires
   
```python
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(random_state=0)
rf_model = rfclf.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print('f1_score : ' + str(f1_score(y_test,y_pred_rf,average='binary', pos_label=1)))

pd.crosstab(y_test,y_pred, colnames=['pred'], rownames=['obs'], margins=True)
```

2. Analyser les variables les plus importantes

```python
pd.DataFrame(rf_model.feature_importances_,
             index=X_train.columns.tolist(),
             columns=['Importances']).sort_values(by = 'Importances', ascending=False)
```

#### Aller plus loin avec SMOTE pour ré équilibrer les classes

1. Avec la méthode `SMOTE`

```python
from imblearn.over_sampling import SMOTE

oversample = SMOTE()
X_train_smote, y_train_smote = oversample.fit_resample(X_train, y_train)
```

2. Avec la méthode `BorderlineSMOTE`

```python
from imblearn.over_sampling import BorderlineSMOTE

oversample = BorderlineSMOTE()
X_train_smote, y_train_smote = oversample.fit_resample(X_train, y_train)
```

#### Aller plus loin avec SHAP pour interpréter ses modèles


*Shapley Values* proviennent de la théorie des jeux coopératifs. Elles permettent de mesurer l'importance de chaque joueur dans un jeu. Cette méthode s'applique aux caractéristiques d'un modèle, en cherchant à comprendre comment chaque caractéristique contribue aux prédictions d'un modèle.

[SHAP](https://towardsdatascience.com/using-shap-values-to-explain-how-your-machine-learning-model-works-732b3f40e137) est particulièrement utile car il :

- Interprète localement : Il explique comment une prédiction spécifique est obtenue pour une observation donnée.
- Interprète globalement : En moyenne, il montre l'importance des caractéristiques pour l'ensemble des prédictions.
- Considère les interactions : SHAP tient compte des interactions entre les caractéristiques, ce qui rend ses explications robustes.

1. Installer la librairie

```sh
pip install shap
```

2. Calculer les valeurs SHAP


```python
import shap

# Créer un explainer basé sur le modèle Random Forest
explainer = shap.TreeExplainer(rf_model)

# Calculer les valeurs SHAP pour les données de test
shap_values = explainer.shap_values(X_test)
```

3. Visualiser les résultats

```python
# Expliquer la première prédiction dans le jeu de test
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])
```

4. Visualiser l'importance des variables pour tous le modèle

```python
# Importance globale des caractéristiques (summary plot)
shap.summary_plot(shap_values, X_test)
```

:warning: Calculer les valeurs SHAP pour des modèles très complexes ou avec de très grandes données peut être coûteux en termes de temps.

## Régression

### Variable cible

1. Analyse de la distribution de la variable cible (`Coût_total_5_usages`)

```python
import seaborn as sns
import matplotlib.pyplot as plt

target = "Coût_total_5_usages"

# Créer un boxplot pour une colonne spécifique
sns.boxplot(data=df, x = target, showfliers=False)

# Afficher le graphique
plt.title(f'Boxplot sur le {target}')
plt.show()
```

2. Analyse les déciles de la variable cible (`Coût_total_5_usages`)

```python
import pandas as pd
import numpy as np

# Créer une séquence de 0 à 1 avec un pas de 0.1
sequence = np.arange(0, 1.1, 0.1)

# Calculer les déciles (0.1, 0.2, ..., 0.9) en ajoutant les percentiles à describe()
resultat = df[target].describe(percentiles=sequence)
```

3. Modifier les paramètres d'affichage par défaut de pandas

```python
# Changer l'option pour afficher toutes les colonnes
pd.set_option('display.max_columns', None)

# Changer l'option pour afficher toutes les lignes
pd.set_option('display.max_rows', None)
```

### Variables explicatives

1. Selection des variables explicatives

```python
# Vérification des données manquantes
ls_variables_explicatives = ['Année_construction',
'Période_construction','Surface_habitable_logement','Type_énergie_n°1',
'Etiquette_DPE',
'N°_étage_appartement', 'Hauteur_sous-plafond',
'Logement_traversant_(0/1)',
'Présence_brasseur_air_(0/1)',
'Indicateur_confort_été',
'Isolation_toiture_(0/1)',
'Protection_solaire_exterieure_(0/1)',
'Inertie_lourde_(0/1)',
'Deperditions_baies_vitrées',
'Deperditions_enveloppe',
'Déperditions_murs',
'Deperditions_planchers_bas',
'Deperditions_planchers_hauts',
'Déperditions_ponts_thermiques',
'Déperditions_portes',
'Déperditions_renouvellement_air',
'Qualité_isolation_enveloppe',
'Qualité_isolation_menuiseries',
'Qualité_isolation_murs',
'Qualité_isolation_plancher_bas']
```

2. Statistiques des variables explicatives

```python
# Créer une séquence de 0 à 1 avec un pas de 0.1
sequence = np.arange(0, 1.1, 0.1)

# Calculer les déciles (0.1, 0.2, ..., 0.9) en ajoutant les percentiles à describe()
resultat = df.describe(percentiles=sequence)
```

3. Inspection des Données Manquantes
   
```python
# Vérification des données manquantes
df[ls_variables_explicatives].isnull().sum()
```

4. Imputation des Données Manquantes sur variable quantitatives

```python
from sklearn.impute import KNNImputer

# Sélectionner uniquement les colonnes quantitatives (numériques)
quant_cols = df[ls_variables_explicatives].select_dtypes(include=[np.number]).columns

# Afficher les colonnes quantitatives
print("Colonnes quantitatives :", quant_cols)

# Initialiser le KNNImputer
imputer = KNNImputer(n_neighbors=3)

# Appliquer l'imputation sur les colonnes quantitatives
df[quant_cols] = imputer.fit_transform(df[quant_cols])

# Vérification des données manquantes
df[quant_cols].isnull().sum()
```

5. Imputation des Données Manquantes sur variable qualitatives

```python
# Sélectionner toutes les colonnes non numériques (qualitatives)
categorical_cols = df[ls_variables_explicatives].select_dtypes(exclude=[np.number]).columns

# Appliquer l'imputation par la valeur la plus fréquente (mode) pour chaque colonne catégorielle
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Vérification des données manquantes
df[categorical_cols].isnull().sum()
```

6. Analyse des Corrélations entre les Variables Explicatives et la variable cible

```python
# Calcul de la matrice de corrélation
corr_matrix = round(df[list(quant_cols) + [target] ].corr(),2)
# Affichage de la matrice de corrélation sous forme de heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Matrice de Corrélation')
plt.show()
```

7. Encodage des variables catégorielles

```python
# Concaténer les deux listes : ls_variables_explicatives et target
df = df[list(ls_variables_explicatives) + [target]]
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
```

### Echantillonnage

1. Créer un objet `X` avec les variables explicatives

```python
# Utiliser set.difference() pour exclure la colonne cible de ls_variables_explicatives
X =df[df.columns.difference([target])]
```

2. Crée un objet `Y` avec la variable à expliquer

```python
Y = df[target]
```

3.  Scinder l'échantillon en train / test

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                    test_size = 0.30,
                                                    random_state = 42)
```

4. Afficher un extrait de  `X_train` et `X_test`
   
```python
print(X_train.shape)
print(X_test.shape)
X_test.head()
```

### Régression linéaire multiple

1. Calculer la régression linéaire multiple

```python
from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
lr_model = lr_model.fit(X_train,y_train)
```

2. Afficher les coefficients

```python
coef = pd.DataFrame(lr_model.coef_ ,index = X_train.columns, columns=['Coef'])
coef.loc['Constante'] = lr_model.intercept_
coef
```

### Evaluation de modèle

1. Prédire sur les données test

```python
y_pred = lr_model.predict(X_test)
y_pred
```

3. Afficher les prédictions 
   
```python
# Taille de la figure
plt.figure(figsize=(10, 6))

# Nuage de points pour les valeurs observées et prédites
plt.scatter(y_test, y_pred, color='black', label='Valeurs Observées vs Prédictions')

# Tracer la droite d'équation y = x
max_val = max(y_test.max(), y_pred.max())
plt.plot([0, max_val], [0, max_val], color='blue', linestyle='--', label='y = x (Droite de référence)')

# Définir les limites des axes
plt.xlim(0, max_val)
plt.ylim(0, max_val)

# Assurer que les axes ont la même échelle
plt.gca().set_aspect('equal', adjustable='box')

# Ajouter des labels et une légende
plt.xlabel('Valeurs Observées')
plt.ylabel('Valeurs Prédites')
plt.title('Nuage de Points: Valeurs Observées vs Prédictions')
plt.legend()

# Afficher le graphique
plt.show()
```

4. Calculer les métriques

```python
from sklearn.metrics import mean_squared_error, r2_score
print("RMSE : " + str(mean_squared_error(y_test, y_pred, squared= False)))
print("R² : " + str(r2_score(y_test, y_pred)))
```


### D'autres méthodes

#### Ridge

1. Modéliser avec Ridge
```python
from sklearn.linear_model import Ridge
ridge_model = Ridge(alpha=0)
ridge_model = ridge_model.fit(X_train_CR,y_train)

y_pred = ridge_model.predict(X_test_CR)

print("RMSE : " + str(mean_squared_error(y_test, y_pred, squared= False)))
print("R² : " + str(r2_score(y_test, y_pred)))
```

2. Analyser les coefficients

```python
coef = pd.DataFrame(ridge_model.coef_ ,index = X_train.columns, columns=['Coef'])
coef.loc['Constante'] = ridge_model.intercept_
coef
```


#### Lasso

1. Modéliser avec Lasso
   
```python
from sklearn.linear_model import Lasso
lasso_model = Lasso(alpha=10)
lasso_model = lasso_model.fit(X_train_CR,y_train)

y_pred = lasso_model.predict(X_test_CR)

print("RMSE : " + str(mean_squared_error(y_test, y_pred, squared= False)))
print("R² : " + str(r2_score(y_test, y_pred)))
```

2. Analyser les coefficients


```python
coef = pd.DataFrame(lasso_model.coef_ ,
                    index = X_train.columns, columns=['Coef'])
coef.loc['Constante'] = lasso_model.intercept_
coef
```



#### Elasticnet


```python

```

```python

```

```python

```

#### Arbre de régression


```python

```

```python

```

```python

```

## Liens utiles

Voici quelques liens utiles :

- [Introduction au machine learning](https://www.kaggle.com/learn/intro-to-machine-learning)
- [Machine learning avancé](https://www.kaggle.com/learn/intermediate-machine-learning)
- [Interprétation de modèle](https://www.kaggle.com/learn/machine-learning-explainability)
- [Bagging](https://en.wikipedia.org/wiki/Bootstrap_aggregating#/media/File:Ensemble_Bagging.svg) vs [Boosting](https://en.wikipedia.org/wiki/Boosting_(machine_learning)#/media/File:Ensemble_Boosting.svg)
- [Cas des classes déséquilibrées](https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/)
- [Interprter des modèles avec SHAP](https://towardsdatascience.com/using-shap-values-to-explain-how-your-machine-learning-model-works-732b3f40e137)