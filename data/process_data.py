# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''Loading all data from csv and prepar/return DataFrame
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = messages.merge(categories, left_on='id', right_on='id')
    return df

def clean_data(df):
    '''Cleaning the data by splitting and merging both dfs
    '''
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',expand=True)
    categories.columns = categories.iloc[0].apply(lambda x: x[:-2])
    
    # clean column names
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        
    # drop the original categories column from `df`
    df.drop(['categories'], axis=1, inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df1 = pd.concat([df,categories], axis=1)

    # drop duplicates
    df1 = df1.drop_duplicates()

    # drop 'original' & 'id' columns and check number of duplicates
    df1.drop(['original','id'],axis=1,inplace=True)
    df1.duplicated().value_counts()
    return df1
        
def save_data(df, database_filename):
     ''' Store data in database
    '''
    # create engine and store data
    print(database_filename)
    engine = create_engine(f"sqlite:///{database_filename}")
    df.to_sql('messages', engine, if_exists='replace', index=False)

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()