# Chapitre 6 : Développer une application intéractive


## Objectifs

Voici les objectifs de ce module :
- [x] Les bases de Dash
- [x] Les bases de shiny
- [x] Les bases de Streamlit
- [x] Les bases de la création d'API avec Flask

- [Chapitre 6 : Développer une application intéractive](#chapitre-6--développer-une-application-intéractive)
  - [Objectifs](#objectifs)
  - [Dash](#dash)
    - [Hello World](#hello-world)
    - [Se connecter à des données](#se-connecter-à-des-données)
    - [Merci ChatGPT](#merci-chatgpt)
  - [Shiny with Python](#shiny-with-python)
    - [Hello World](#hello-world-1)
    - [Se connecter à des données](#se-connecter-à-des-données-1)
  - [Streamlit](#streamlit)
    - [Hello World](#hello-world-2)
    - [Se connecter à des données](#se-connecter-à-des-données-2)
    - [Aller plus loin](#aller-plus-loin)
  - [Initiation à Docker](#initiation-à-docker)
  - [Créer une API avec différentes ROUTE](#créer-une-api-avec-différentes-route)
    - [Création](#création)
    - [Tester les routes](#tester-les-routes)
  - [Liens utiles](#liens-utiles)


## Dash

### Hello World

1. Créer un environnement virtuel

```sh
py -m venv test-dash
```

2. Installer la librairie  `dash`

```sh
pip install dash
```

3. Créer un script `.py` avec ce code et l'executer

```python
from dash import Dash, html

app = Dash()

app.layout = [html.Div(children='Hello World')]

if __name__ == '__main__':
    app.run(debug=True)
```

###  Se connecter à des données

```python
# Import packages
from dash import Dash, html, dash_table
import pandas as pd

# Incorporate data
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder2007.csv')

# Initialize the app
app = Dash()

# App layout
app.layout = [
    html.Div(children='My First App with Data'),
    dash_table.DataTable(data=df.to_dict('records'), page_size=10)
]

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
```

### Merci ChatGPT

1. Demander gentiment à ChatGPT : 

```sh
Une application python en dash avec des données issue d'un dataframe comme les données iris, code moi une application dash avec une page pour visualiser els données, une page avec qelques graphique, et une page pour tester des modeles de classification et analyser les métriques de point prédire. On veut une layout a gauche a chaque fois pour filtrer les donénes. pour la page 3 je veux plutot utiliser un arbre de decision et qu'un layout a gauche me permette de gerer les paramètres principaux du modele comme la profonfeur , la taille des feuilles et des branches 
```

2. Installer les librairies nécessaires

```sh
pip install dash scikit-learn pandas plotly dash-bootstrap-components
```

3. Copier coller.

```python
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris

# Charger le dataset Iris
iris = load_iris()

# Préparer le DataFrame Iris
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['target_name'] = df['target'].map({i: iris.target_names[i] for i in range(3)})

# Application Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout principal avec une barre latérale pour la navigation
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("Iris Dashboard", className="display-4"),
            html.Hr(),
            dbc.Nav(
                [
                    dbc.NavLink("Visualisation des données", href="/", active="exact"),
                    dbc.NavLink("Graphiques", href="/graphs", active="exact"),
                    dbc.NavLink("Modèles de classification", href="/models", active="exact"),
                ],
                vertical=True,
                pills=True,
            ),
        ], width=2),
        dbc.Col([
            dcc.Location(id='url'),
            html.Div(id='page-content')
        ], width=10)
    ])
], fluid=True)

# Page 1 : Visualisation des données
def render_data_table():
    return html.Div([
        html.H3('Visualisation des données Iris'),
        dcc.Dropdown(
            id='filter-species',
            options=[{'label': name, 'value': name} for name in iris.target_names],
            multi=True,
            placeholder="Filtrer par espèce..."
        ),
        html.Br(),
        html.Div(id='data-table', children=[]),
    ])

@app.callback(
    Output('data-table', 'children'),
    [Input('filter-species', 'value')]
)
def update_table(selected_species):
    if selected_species:
        filtered_df = df[df['target_name'].isin(selected_species)]
    else:
        filtered_df = df
    return dbc.Table.from_dataframe(filtered_df, striped=True, bordered=True, hover=True)

# Page 2 : Graphiques
def render_graphs_page():
    return html.Div([
        html.H3('Graphiques interactifs'),
        dcc.Dropdown(
            id='x-axis',
            options=[{'label': col, 'value': col} for col in iris.feature_names],
            value=iris.feature_names[0],
            placeholder="Choisir l'axe des X"
        ),
        dcc.Dropdown(
            id='y-axis',
            options=[{'label': col, 'value': col} for col in iris.feature_names],
            value=iris.feature_names[1],
            placeholder="Choisir l'axe des Y"
        ),
        dcc.Graph(id='scatter-plot')
    ])

@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('x-axis', 'value'), Input('y-axis', 'value')]
)
def update_scatter_plot(x_col, y_col):
    fig = px.scatter(df, x=x_col, y=y_col, color='target_name',
                     title=f"Scatter Plot ({x_col} vs {y_col})")
    return fig

# Page 3 : Modèles de classification avec Arbre de décision
def render_model_page():
    return html.Div([
        html.H3('Modèles de classification - Arbre de décision'),

        # Layout pour contrôler les paramètres
        html.Label("Profondeur maximale de l'arbre"),
        dcc.Slider(id='max-depth', min=1, max=10, step=1, value=3,
                   marks={i: str(i) for i in range(1, 11)}),

        html.Br(),

        html.Label("Nombre minimum d'échantillons par feuille"),
        dcc.Slider(id='min-samples-leaf', min=1, max=10, step=1, value=1,
                   marks={i: str(i) for i in range(1, 11)}),

        html.Br(),

        html.Label("Nombre minimum d'échantillons pour diviser un noeud"),
        dcc.Slider(id='min-samples-split', min=2, max=10, step=1, value=2,
                   marks={i: str(i) for i in range(2, 11)}),

        html.Br(),
        
        html.Button('Lancer le modèle', id='run-model', n_clicks=0),
        html.Div(id='model-output', children=[])
    ])

@app.callback(
    Output('model-output', 'children'),
    [Input('run-model', 'n_clicks')],
    [State('max-depth', 'value'), State('min-samples-leaf', 'value'), State('min-samples-split', 'value')]
)
def run_classification_model(n_clicks, max_depth, min_samples_leaf, min_samples_split):
    if n_clicks > 0:
        # Préparer les données
        X = df[iris.feature_names]
        y = df['target']

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Modèle d'arbre de décision avec les paramètres sélectionnés
        model = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calcul des métriques
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=iris.target_names)

        # Retourner les résultats du modèle
        return html.Div([
            html.H5(f"Accuracy : {accuracy:.2f}"),
            html.Pre(report)
        ])
    return ""

# Router pour naviguer entre les pages
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == "/graphs":
        return render_graphs_page()
    elif pathname == "/models":
        return render_model_page()
    else:
        return render_data_table()

# Lancer le serveur
if __name__ == '__main__':
    app.run_server(debug=True)
```

## Shiny with Python

### Hello World

1. Créer un environnement virtuel

```sh
py -m venv test-shiny
```

2. Installer les librairies  nécessaire

```sh
pip install shiny htmltools
```

3. Créer un script `app.py` avec ce code
   
```python
from shiny import App, render, ui

app_ui = ui.page_fluid(
    ui.h2("Hello World")
)

def server(input, output, session):
    pass

app = App(app_ui, server)

if __name__ == "__main__":
    app.run()
```

4. Executer l'application avec cette commande

```sh
shiny run --reload --launch-browser /folder1/folder2/app_dir/app.py
```

### Se connecter à des données

1. Installer la librairie  `shinywidgets`

```sh
pip install shinywidgets
```

2. Modifier le script `app.py` avec ce code et executer l'application

```python
from shiny import App, render, ui
import pandas as pd
from shinywidgets import output_widget, render_widget
import plotly.express as px

# Load data
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder2007.csv')

# App UI
app_ui = ui.page_fluid(
    ui.h2("My First App with Data"),
    ui.output_table("data_table")  # Output placeholder for the table
)

# Server logic
def server(input, output, session):
    # Render table with pagination
    @output
    @render.table
    def data_table():
        return df.head(10)  # Render first 10 rows as in Dash's page_size=10

# Create the app object
app = App(app_ui, server)

if __name__ == "__main__":
    app.run()

```


## Streamlit

### Hello World

1. Créer un environnement virtuel

```sh
py -m venv test-streamlit
```

2. Installer les librairies  nécessaire

```sh
pip install streamlit
```

3. Créer un script `app.py` avec ce code
   
```python
import streamlit as st

# Affichage du titre
st.header("Hello World")
```

4. Executer l'application avec cette commande

```sh
streamlit run /folder1/folder2/app_dir/app.py
```

### Se connecter à des données

1. Modifier le script `app.py` avec ce code et executer l'application

```python
# Import packages
import streamlit as st
import pandas as pd

# Incorporate data
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder2007.csv')

# Initialize the app
st.title('My First App with Data')

# Display the data in a table
st.dataframe(df)
```

### Aller plus loin

```sh
streamlit hello
```


## Initiation à Docker

### Qu’est-ce que Docker ?

Docker est une **plateforme de conteneurisation** qui permet d’exécuter des applications dans des environnements isolés appelés **conteneurs**.  
Un conteneur embarque tout ce dont une application a besoin pour fonctionner : le code, les dépendances, la configuration, le système de fichiers minimal, etc.

En d’autres termes, Docker garantit que ton application fonctionnera **de la même manière sur toutes les machines**, peu importe le système d’exploitation ou les dépendances locales.

### Pourquoi utiliser Docker ?

- **Portabilité** : ton application peut être exécutée sur n’importe quelle machine disposant de Docker.  
- **Isolation** : chaque conteneur est indépendant (pas de conflits entre environnements).  
- **Reproductibilité** : ton environnement de développement, de test et de production est identique.  
- **Déploiement simplifié** : un simple `docker run` permet de lancer ton application.  
- **Léger** : contrairement aux machines virtuelles, les conteneurs partagent le même noyau système et sont plus rapides à démarrer.

### Avantages et Inconvénients de Docker

#### Avantages
- **Rapidité** : les conteneurs démarrent en quelques secondes  
- **Légèreté** : moins de ressources nécessaires que des machines virtuelles  
- **Facilité de déploiement** : un même conteneur peut tourner sur Windows, macOS, Linux ou un serveur cloud  
- **Versionnement et contrôle** : les images sont versionnées et traçables
- **Écosystème riche** : grand nombre d’images disponibles sur Docker Hub (bases de données, frameworks, etc.)  
- **Intégration continue (CI/CD)** : parfait pour automatiser les tests et les déploiements

#### Inconvénients
- **Apprentissage** : nécessite une compréhension de base des concepts système (réseau, volumes, images)
- **Gestion du stockage** : les conteneurs et images peuvent occuper beaucoup d’espace disque
- **Non adapté à tout** : certaines applications nécessitant des interfaces graphiques complexes sont plus difficiles à conteneuriser
- **Sécurité** : si mal configuré, un conteneur peut exposer des vulnérabilités du système hôte


#### Docker et le Cloud

Docker n’est pas seulement utilisé en développement local : il est **largement adopté dans le cloud** pour **héberger, déployer et gérer des applications** de manière scalable et fiable.

##### Pourquoi Docker dans le Cloud ?

- **Portabilité totale** : une image Docker peut être déployée sur n’importe quel fournisseur cloud (AWS, Azure, Google Cloud, OVH, etc.) sans modification du code.  
- **Déploiement rapide** : les conteneurs se lancent en quelques secondes, ce qui facilite la mise à jour et la montée en charge des applications.  
- **Scalabilité automatique** : les plateformes cloud peuvent automatiquement ajouter ou supprimer des conteneurs selon la charge.  
- **Intégration avec les orchestrateurs** : Docker s’intègre parfaitement avec **Kubernetes**, **AWS ECS**, **Azure Container Apps**, ou **Google Kubernetes Engine (GKE)** pour le déploiement à grande échelle.  
- **Moins de maintenance** : les images étant reproductibles et isolées, les déploiements sont plus sûrs et stables.


:bulb: En résumé, Docker est devenu un **standard du déploiement cloud moderne**, car il garantit une compatibilité totale entre ton environnement local et la production.

:bulb: - **Heroku / Render / Fly.io** : acceptent les images Docker pour héberger des applications web comme Streamlit.

### Les commandes Docker principales

| Commande | Description |
|-----------|--------------|
| `docker --version` | Vérifie l’installation de Docker |
| `docker images` | Liste les images disponibles sur la machine |
| `docker ps` | Liste les conteneurs en cours d’exécution |
| `docker ps -a` | Liste tous les conteneurs (même arrêtés) |
| `docker run <image>` | Lance un conteneur à partir d’une image |
| `docker stop <id>` | Arrête un conteneur |
| `docker rm <id>` | Supprime un conteneur |
| `docker rmi <image>` | Supprime une image |
| `docker build -t <nom_image> .` | Construit une image Docker depuis un Dockerfile |
| `docker exec -it <id> bash` | Ouvre un terminal dans un conteneur en cours d’exécution |
| `docker pull <nom_image>` | Télécharger une image officielle |
| `docker login` | Se connecter à ton compte Docker Hub |
| `docker tag <nom_image> <user>/<nom_image>:<nom_tag>` | Taguer ton image avant publication |
| `docker push <user>/<nom_image>:<nom_tag>` | Publier ton image sur Docker Hub |

### Docker Hub

[Docker Hub](https://hub.docker.com/_/docker) est une plateforme en ligne qui sert de **registre d’images Docker**.  
C’est l’endroit où tu peux **télécharger, partager et publier** des images Docker.

#### Utilisations principales :
- **Télécharger des images officielles** (Python, Ubuntu, MySQL, Nginx, etc.)  
- **Partager ses propres images** avec l’équipe ou la communauté  
- **Automatiser les builds** depuis GitHub ou GitLab  
- **Stocker des images privées** pour des projets internes


### Exemple : Dockeriser une application Streamlit

#### Structure du projet

```
mon_app_streamlit
 ┣ app.py
 ┣ requirements.txt
 ┗ Dockerfile
```

#### Contenu du fichier `app.py`

```python
# Import packages
import streamlit as st
import pandas as pd

# Incorporate data
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder2007.csv')

# Initialize the app
st.title('My First App with Data')

# Display the data in a table
st.dataframe(df)
```

#### Contenu du fichier `requirements.txt`

```
streamlit
```

#### Contenu du fichier `Dockerfile`

```Dockerfile
# Étape 1 : Utiliser une image Python officielle
FROM python:3.11-slim

# Étape 2 : Définir le répertoire de travail
WORKDIR /app

# Étape 3 : Copier les fichiers de l’application
COPY . /app

# Étape 4 : Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Étape 5 : Exposer le port par défaut de Streamlit
EXPOSE 8501

# Étape 6 : Lancer l’application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### Construire l’image Docker

```bash
docker build -t mon_app_streamlit .
```

#### Lancer le conteneur

```bash
docker run -p 8501:8501 mon_app_streamlit
```

:bulb: L’application sera disponible sur [http://localhost:8501](http://localhost:8501)


## Créer une API avec différentes ROUTE

### Création

1. Installer la librairie `flask`

```sh
pip install flask
```

2. Créer un script `app.py`.

```python
from flask import Flask, jsonify, request

# Initialiser l'application Flask
app = Flask(__name__)

# Route 1 : Récupérer un message de bienvenue
@app.route('/api/welcome', methods=['GET'])
def welcome():
    return jsonify({"message": "Bienvenue sur l'API!"})

# Route 2 : Additionner deux nombres
@app.route('/api/add', methods=['GET'])
def add_numbers():
    # Récupérer les paramètres 'a' et 'b' de la requête
    a = request.args.get('a', type=int)
    b = request.args.get('b', type=int)
    
    if a is None or b is None:
        return jsonify({"error": "Les paramètres 'a' et 'b' sont requis!"}), 400

    result = a + b
    return jsonify({"result": result})

# Route 3 : Afficher un message personnalisé
@app.route('/api/greet', methods=['GET'])
def greet():
    name = request.args.get('name', default='Utilisateur', type=str)
    return jsonify({"message": f"Bonjour, {name}!"})

# Lancer l'API en local
if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

3. Executer votre script python

```sh
python folder1/folder2/app.py
```

### Tester les routes

1. Cette route répond par un message JSON de bienvenue.

```bash
http://127.0.0.1:5000/api/welcome
```

2. Cette route prend deux paramètres (a et b) dans l'URL et retourne leur somme.

```python
http://127.0.0.1:5000/api/add?a=5&b=10
```

3. Cette route permet de saluer un utilisateur en utilisant un paramètre name dans l'URL.

```python
http://127.0.0.1:5000/api/greet?name=Alice
```

4. Tester les routes avec Postman

## Liens utiles

Voici quelques liens utiles :

- [Tutoriel Dash](https://dash.plotly.com/tutorial)
- [Tutoriel Shiny](https://shiny.posit.co/py/docs/overview.html)
- [Tutoriel Streamlit](https://docs.streamlit.io/get-started/fundamentals/main-concepts)
- [Tutoriel Flask](https://flask.palletsprojects.com/en/3.0.x/quickstart/)
- [Postman](https://www.postman.com/downloads/)