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

Intro à Docker (en cours de rédaction)

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