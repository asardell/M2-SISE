# Chapitre 1 : Rappel du langage R

1. [Chapitre 1 : Rappel du langage R](#chapitre-1--rappel-du-langage-r)
   1. [Objectifs](#objectifs)
   2. [Importation des données :](#importation-des-données-)
   3. [Statistiques générales](#statistiques-générales)
   4. [Manipulation des données :](#manipulation-des-données-)
      1. [Découvrir la librairie `pandas`](#découvrir-la-librairie-pandas)
      2. [Filtre](#filtre)
      3. [Agrégation](#agrégation)
   5. [Création de graphique  :](#création-de-graphique--)
      1. [Graphiques élémentaires](#graphiques-élémentaires)
      2. [Régression linéaire simple](#régression-linéaire-simple)
      3. [Cartographie](#cartographie)
   6. [Liens utiles](#liens-utiles)

## Objectifs

Voici les objectifs de ce chapitre :
- [x] Rappel des bases du langage
- [x] Manipuler des vecteurs
- [x] Manipuler des dataframes
- [x] Calculer des statistiques
- [x] Construire des graphiques

Avec l’accélération du changement climatique et la hausse des prix de l’énergie, la sobriété énergétique est au cœur des préoccupations des Français. Aussi, afin d’éclairer et inspirer les acteurs de la transition écologique, Enedis propose des analyses et chiffres clés pour éclairer et orienter les décisions.

Dans ce contexte, le Diagnostic de Performance Energétique (DPE) permet d’évaluer la performance énergétique et climatique d’un bâtiment. Il consiste en une étiquette pouvant aller de A à G pour chaque logement ou bâtiment, qui évalue sa consommation d’énergie et son impact en terme d’émission de GES. Il sert notamment à identifier les passoires énergétiques (étiquettes F et G du DPE, c’est-à-dire les logements qui consomment le plus d’énergie et/ou émettent le plus de gaz à effet de serre). Il a pour objectif d’informer l’acquéreur ou le locataire sur la « valeur verte », de recommander des travaux à réaliser pour l’améliorer et d’estimer ses charges énergétiques. De plus, la mise en location de ces passoires thermiques sera progressivement interdite (interdiction pour les bâtiments notés G+ au 1er janvier 2023, qui sera étendue par la suite).

Les données ne seront pas représentatives de la France et ne peuvent donc pas être agrégées à la maille de la France entière. On suppose néanmoins que les données pourront permettre de donner déjà des informations intéressantes bien que non exhaustives. Dans ce chapitre, nous travaillerons sur un échantillon basé sur des logements du 8ème arrondissement de Lyon.

## Importation des données : 

1. Télécharger les jeux de données `dpe-v2-logements-existants.csv` et `dpe-v2-logements-neufs.csv` disponibles [ici](./data).
2. Importer les jeux de données
3. Afficher la dimension des 2 datasets
4. Créer une colonne nommée `Logement` dans les deux datasets avec la valeur `ancien` ou `neuf` selon la source.
5. La variable `Année_construction` n'apparaît pas dans les données des logements neufs. Créer cette colonne avec la valeur de l'année en cours.
6. Fusionner les deux dataframes avec uniquement les colonnes communes. Plus d'info dans le [dictionnaire de données](https://data.ademe.fr/data-fair/api/v1/datasets/dpe-v2-logements-existants/metadata-attachments/DPE_Dictionnaire%20de%20donn%C3%A9es_JDD_V3.pdf).
7. Créer une colonne avec uniquement l'année de la `Date de réception du DPE`
8. Créer une colonne qui vérifie si `Coût_total_5_usages` correspond bien à la somme du `Coût_chauffage` + `Coût_éclairage` + `Coût_ECS` + `Coût_refroidissement` +  `Coût_auxiliaires`.
9. Créer une colonne `Coût chauffage en %` qui est la part du coût du chauffage dans le coût total 5 usages.
10. Créer une colonne `Periode_construction` avec ces classes ci-dessous

| Période Construction | 
|------------------|
| Avant 1960 | 
| 1961 -  1970 |
| 1971 -  1980 |
| 1981 -  1990 |
| 1991 -  2000 |
| 2001 -  2010 |
| Après 2010 |

## Statistiques générales

1. Calculer la répartition des logements par `Etiquette_DPE`
2. Calculer la répartition des logements par `Date_réception_DPE`
3. Calculer la répartition des logements par type logement (ancien/neuf)
4. Calculer la répartition des logements par par type de Bâtiment
5. Calculer la répartition des logements par par type d'installation chauffage en pourcentage
6. Calculer la répartition des logements par période de construction
7. Calculer la surface habitable moyenne des logements
8. Calculer la moyenne du coût_chauffage
9. Calculer les quartiles puis déciles du Coût_ECS
10. Calculer le coefficient de corrélation entre la surface habitable du logement et le coût du chauffage.
11. Construire un corrélogramme sur ces variables (`Coût_total_5_usages`,`Coût_chauffage`,`Coût_éclairage`,`Coût_ECS`,`Coût_refroidissement`, `Coût_auxiliaires`, `Surface_habitable_logement` , `Emission_GES_5_usages`)

## Manipulation des données : 

### Découvrir la librairie `pandas`

| Nom de la commande                    | Description                                                     | Arguments Pertinents                                | Exemple                                              |
|---------------------------------------|-----------------------------------------------------------------|----------------------------------------------------|------------------------------------------------------|
| `df[['col1', 'col2']]`                | Sélectionne des colonnes d'un DataFrame.                         | `df` : le DataFrame, `[col1, col2]` : les colonnes à sélectionner | `df[['col1', 'col2']]`                               |
| `df.iloc[rows]`                       | Sélectionne des lignes spécifiques d'un DataFrame par leur index.| `df` : le DataFrame, `rows` : les indices des lignes à sélectionner | `df.iloc[[0, 2, 4]]`                                |
| `df[df['col1'] > 10]`                 | Filtre les lignes d'un DataFrame en fonction de conditions.       | `df` : le DataFrame, `condition` : condition sur une colonne | `df[df['col1'] > 10]`                                |
| `df.sort_values(by='col1')`           | Trie les lignes d'un DataFrame en fonction de colonnes.           | `df` : le DataFrame, `by` : colonnes de tri, `ascending` : ordre de tri | `df.sort_values(by='col1', ascending=True)`          |
| `df.groupby('col1')`                  | Crée des groupes de données en fonction de colonnes.              | `df` : le DataFrame, `col1` : colonne de regroupement | `df.groupby('col1')`                                 |
| `df.groupby('col1').agg({'col2': 'mean'})` | Résume les données groupées par des calculs (agrégations).     | `df` : le DataFrame, `col1` : colonne de regroupement, `agg_func` : fonction d'agrégation | `df.groupby('col1').agg({'col2': 'mean'})`           |
| `pd.read_csv('file.csv')`             | Charge un fichier CSV dans un DataFrame.                          | `file` : chemin vers le fichier CSV                | `df = pd.read_csv('file.csv')`                       |
| `df.merge(df2, on='col')`             | Effectue une jointure entre deux DataFrames sur des colonnes.     | `df` : DataFrame de gauche, `df2` : DataFrame de droite, `on` : colonne commune | `df.merge(df2, on='col')`                            |
| `df.rename(columns={'old_name': 'new_name'})` | Renomme des colonnes d'un DataFrame.                          | `df` : le DataFrame, `columns` : dictionnaire de renommage | `df.rename(columns={'old_name': 'new_name'})`        |
| `df['col1'].astype('type')`           | Change le type d'une colonne dans un DataFrame.                   | `df` : le DataFrame, `col1` : colonne à modifier, `type` : nouveau type | `df['col1'] = df['col1'].astype('float')`            |
| `df.dropna()`                         | Supprime les lignes contenant des valeurs manquantes.             | `df` : le DataFrame, `how` : critère de suppression | `df.dropna()`                                        |
| `df.fillna(value)`                    | Remplit les valeurs manquantes dans un DataFrame.                 | `df` : le DataFrame, `value` : valeur de remplacement | `df.fillna(0)`                                       |
| `df['new_col'] = df['col1'] + df['col2']` | Crée une nouvelle colonne à partir d'opérations sur des colonnes.| `df` : le DataFrame, `col1`, `col2` : colonnes à combiner | `df['new_col'] = df['col1'] + df['col2']`            |
| `df.drop(columns='col1')`             | Supprime une ou plusieurs colonnes d'un DataFrame.                | `df` : le DataFrame, `columns` : colonnes à supprimer | `df.drop(columns='col1')`                            |
| `df.nunique()`                        | Compte le nombre de valeurs uniques par colonne.                  | `df` : le DataFrame                                 | `df.nunique()`                                       |


### Filtre

1. Créer un dataframe avec uniquement les logements dont le type de batîment est un appartement
2. Créer un dataframe avec uniquement les logements dont l'étiquette DPE est D,E,F,G
3. Créer un dataframe avec les logements anciens dont la période de construction est *avant 1960*
4. Créer un dataframe avec les logements qui ont une surface habitable strictement supérieure à la surface habitable moyenne
5. Créer un dataframe en triant les logements qui consomme le plus d'énergie 5 usages en m² (`Conso_5_usages.m._é_finale`)à ceux qui consomme le moins
6. Créer un dataframe en triant par étiquette DPE, puis période de construction, puis par coût en chauffage décroissant.

### Agrégation

1. Calculer le coût moyen du chauffage  selon l'étiquette du DPE
2. Calculer la moyenne de la consommation annuelle  5 usages en energie en kWhef/an selon la période de construction
3. Calculer la moyenne de la consommation annuelle  5 usages en energie en kWhef/an selon le type de logement et d'étiquette DPE

## Création de graphique  : 

### Graphiques élémentaires 

1. Construire une histogramme de la distribution des surfaces habitables
2. Construire un boxplot de la distribution du la onsommation annuelle  5 usages en energie en kWhef/an
3. Construire un boxplot avec le coût du chauffage selon le type d'étiquette DPE
4. Construire un diagramme en barre du nombre de logements par période de construction
5. Construire un diagramme circulaire du principal type d'énergie (`Type_énergie_n.1`)

### Régression linéaire simple

1. Construire une nuage de point entre la surface habitable du logement et le coût du chauffage
2. Calculer le coefficient de corrélation entre la surface habitable du logement et le coût du chauffage.
3. Construire une régression linéaire simple pour modéliser le coût du chauffage en fonction de la surface habitable
4. Analyser les coefficients de la régression
5. Affichier la droite de régression sur le nuage de point

### Cartographie

1. Télécharger le jeu de données `adresses-69` disponibles [ici](./data).
2. Importer le jeu de données
3. Réaliser une jointure sur le champ `Identifiant__BAN` pour ajouter les coordonnées GPS (lattitude / longitude) au dataframe initial
4. Construire une carte des logements avec les indicateurs de votre choix


## Liens utiles

Voici quelques liens utiles :

- [Contexte du Challenge ENEDIS](https://defis.data.gouv.fr/defis/65b76f15d7874915c8e41298)
- [Base Adresse Nationale](https://adresse.data.gouv.fr/donnees-nationales)
- [Comprendre son DPE](https://www.ecologie.gouv.fr/sites/default/files/documents/comprendre_mon_dpe.pdf)
- [Dictionnaire de données](https://data.ademe.fr/data-fair/api/v1/datasets/dpe-v2-logements-existants/metadata-attachments/DPE_Dictionnaire%20de%20donn%C3%A9es_JDD_V3.pdf)