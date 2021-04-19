# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

# Disaster_Response_Pipeline
Built a pipeline and model for classification of messages using sklearn


## Motivation
Within this project a webapp was build were message strings can be categoriezed in different categories. By using other classifier or adding more parameters to the GridSearch optimization workflow the results can be improved further. Furthermore, some other features including into the pipeline can be usefull to get the classifier better.
![Webapp](webapp.png)
## Installation
fork and clone the project to your local machine to run the project there.
git remote add origin {{the link you just copied}}
git clone
## ETL Pipeline
- reading data from csv files [pandas]
- cleaning data [pandas]
- storing in a SQLite database [SQLAlchemy]
## Machine Learning Pipeline
- splitting the data into train and test set
- creating and improving ML Pipeline with NLTK, GridSearchCV and Sklearn
- building a model and saving a pickle file
## Flask App
- Webapp for visualisation and using the model
