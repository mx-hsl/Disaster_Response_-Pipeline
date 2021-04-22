import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    # Plot 1
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # Plot 2 / 3
    cat_list = ['ralated', 'request', 'offer', 'aid_related', 'medical_help',
   'medical_products', 'search_and_rescue', 'security', 'military',
   'child_alone', 'water', 'food', 'shelter', 'clothing', 'money',
   'missing_people', 'refugees', 'death', 'other_aid',
   'infrastructure_related', 'transport', 'buildings', 'electricity',
   'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
   'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
   'other_weather', 'direct_report']
    # 5 Most tagged categories
    tagged = df[cat_list]
    most_tagged_cats = tagged.sum().sort_values()[len(cat_list)-5:].index.tolist()
    df_most_tagged = tagged[most_tagged_cats]
    most_tagged_values = df_most_tagged.sum().tolist()

    
    # 5 least Tagged categories
    least_tagged_cats = tagged.sum().sort_values()[:5].index.tolist()
    df_least_tagged = tagged[least_tagged_cats]
    least_tagged_values = df_least_tagged.sum().tolist()

    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        
        
        {
            'data': [
                Bar(
                    x=most_tagged_cats,
                    y=most_tagged_values
                )
            ],

            'layout': {
                'title': 'Most tagged Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        
        
        {
            'data': [
                Bar(
                    x=least_tagged_cats,
                    y=least_tagged_values
                )
            ],

            'layout': {
                'title': 'Least tagged Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        
        
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()