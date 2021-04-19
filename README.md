# Disaster Response Pipeline Project
Built a pipeline and model for classification of messages using sklearn

## Motivation
Within this project a webapp was build were message strings can be categoriezed in different categories. By using other classifier or adding more parameters to the GridSearch optimization workflow the results can be improved further. Furthermore, some other features including into the pipeline can be usefull to get the classifier better.
![Webapp](webapp.png)
## Installation
fork and clone the project to your local machine to run the project there.
```
  git clone https://github.com/mx-hsl/Disaster_Response_Pipeline.git
```
- To run ETL pipeline that cleans data and stores in database
        ```python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db```
- To run ML pipeline that trains classifier and saves
        ```python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl```
- Run the following command in the app's directory to run your web app.
    ```python run.py```
- Go to http://0.0.0.0:3001/

If the links isn´t working, then you need to find the workspace environmental variables with `env | grep WORK`, and you can open a new browser window and go to the address:
`http://WORKSPACESPACEID-3001.WORKSPACEDOMAIN` replacing WORKSPACEID and WORKSPACEDOMAIN with your values.

## Implemented steps in this project:
### ETL Pipeline
- reading data from csv files [pandas]
- cleaning data [pandas]
- storing in a SQLite database [SQLAlchemy]

### Machine Learning Pipeline
- splitting the data into train and test set
- creating and improving ML Pipeline with NLTK, GridSearchCV and Sklearn
- building a model and saving a pickle file

### Flask App
- Webapp for visualisation and using the model
