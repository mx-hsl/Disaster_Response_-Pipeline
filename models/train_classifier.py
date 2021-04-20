# import libraries
import sys
# import libraries
import pandas as pd
import numpy
from sqlalchemy import create_engine
import pickle

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB

from nltk import word_tokenize
from nltk import WordNetLemmatizer
import nltk
nltk.download('punkt')
nltk.download('wordnet')


def load_data(database_filepath):
    ''' Load data from DBC
    '''
    # load data from database
    engine = create_engine("sqlite:///data/DisasterResponse.db")
    df = pd.read_sql_table('messages',engine)
    df = df.drop(['genre'],axis=1)
    #df = df.drop(['original'],axis=1)
    #df = df.drop(['id'],axis=1)
    X = df['message']
    X = X.astype(str)
    y = df.loc[:, df.columns != 'message']
    return X, y


def tokenize(text):
    ''' prepare and return clean tokens
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    ''' Building the model with pipeline and gridsearch
    '''
    X, y = make_multilabel_classification(n_classes=3, random_state=0)
    
    # define pipeline
    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(KNeighborsClassifier()))
    ])
    
    # define parameters
    parameters = {
    'tfidf__use_idf':[True,False]
    }
    
    # using gridsearch
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv
    

def evaluate_model(model, X_test, y_test):
    ''' Evaluationg the model
    Input: model from GridSearch, Testparameters for X and Y in and category_names
    Output: None
    '''
    
    # predict categories
    y_pred = model.predict(X_test)

    # get num of columns
    num = len(y_pred.T)

    for col in range(0, num):
        report = classification_report(y_pred.T[:][col], y_test.iloc[:,col].values)
        print(f"Report for Col{col}: {report}")



def save_model(model, model_filepath):
    ''' Save model as pickle file
    '''
    pickle.dump(model, open("classifier.pkl", 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X,y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()