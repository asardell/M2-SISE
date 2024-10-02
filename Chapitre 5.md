# Chapitre 5 : Les bases du machine learning

## Objectifs

Voici les objectifs de ce module :
- [x] Différencier classification et régression
- [x] Evaluer un modèle
- [x] Optimiser les paramètres d'un modèle
- [x] Interpréter un modèle

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
df = df[ls_variables_explicatives]
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
from sklearn.model_selection import cross_val_predict
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
y_pred_proba_cv = model_arbre_cv.predict_proba(X_test)
y_pred_proba_grid = model_arbre_grid.predict_proba(X_test)
```

2. Calculer les Valeurs de la Courbe ROC

```python
from sklearn.metrics import roc_curve, roc_auc_score
fpr1, tpr1, _ = roc_curve(y_test, y_pred_proba_cv)
roc_auc1 = roc_auc_score(y_test, y_pred_proba_cv)

fpr2, tpr2, _ = roc_curve(y_test, y_pred_proba_grid)
roc_auc2 = roc_auc_score(y_test, y_pred_proba_grid)
```

4. Tracer les Courbes ROC

```python
import matplotlib.pyplot as plt
# Tracé des courbes ROC
plt.figure(figsize=(10, 6))
plt.plot(fpr1, tpr1, color='blue', label=f'Modèle 1 (AUC = {roc_auc1:.2f})')
plt.plot(fpr2, tpr2, color='red', label=f'Modèle 2 (AUC = {roc_auc2:.2f})')

# Tracé de la ligne de chance
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')

# Détails du graphique
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('Taux de Faux Positifs')
plt.ylabel('Taux de Vrais Positifs')
plt.title('Courbes ROC des Modèles')
plt.legend(loc='lower right')
plt.grid()

# Affichage du graphique
plt.show()
```

### D'autres méthodes
#### KNN
#### Régression logistique
#### Random Forest

## Régression
### Variable cible
### Echantillonnage
### Régression linéaire simple/multiple
### Evaluation de modèle
### D'autres méthodes
#### Ridge
#### Lasso
#### Elasticnet
#### Arbre de régression

## Liens utiles

Voici quelques liens utiles :

- [Introduction au machine learning](https://www.kaggle.com/learn/intro-to-machine-learning)
- [Machine learning avancé](https://www.kaggle.com/learn/intermediate-machine-learning)
- [Interprétation de modèle](https://www.kaggle.com/learn/machine-learning-explainability)
- [Bagging](https://en.wikipedia.org/wiki/Bootstrap_aggregating#/media/File:Ensemble_Bagging.svg) vs [Boosting](https://en.wikipedia.org/wiki/Boosting_(machine_learning)#/media/File:Ensemble_Boosting.svg)