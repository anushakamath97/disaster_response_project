import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Reads and merges data from messages_filepath and categories_filepath into
    a pandas dataframe.

    Args:
     - messages_filepath: relative path to messages dataset csv
     - categories_filepath: relative path to categories dataset csv

    Returns: the merged (on "id" column) dataframe
    """

    # read data from csv files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # join dataframes on "id" column
    # "id" column of messages is matched with index of categories
    # therefore we need to call categories.set_index
    df = messages.join(categories.set_index('id'), on="id")
    return df


def clean_data(df):
    """
    Perform following cleaning steps on dataframe:
    1. Create binary categorical (0 or 1) column for all unique categories
    2. Remove duplicate rows in the dataframe

    Args:
     - df: dataframe with categories in string format

    Returns: dataframe after cleaning
    """
    # split the categories into separate columns
    categories = df["categories"].str.split(";", expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:-2])

    # rename the columns of categories
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.strip().str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # Remove columns with single value
    for col in categories.columns.tolist():
        if len(categories[col].unique()) == 1:
            print("Removing column {} as it has only 1 value".format(col))
            categories.drop(columns=[col, ], inplace=True)

    df.drop(columns=["categories", ], inplace=True)

    df = pd.concat([df, categories], axis=1)

    # check number of duplicates
    print("Number of duplicate rows in dataframe: ", df.duplicated().sum())

    # drop duplicates
    df.drop_duplicates(inplace=True)

    # check number of duplicates
    assert df.duplicated().sum() == 0, "All duplicate rows aren't removed"

    return df


def save_data(df, database_filename):
    """
    Saves pandas dataframe to sql database.

    Args:
     - df: dataframe
     - database_filename: Filename to save for database
    """
    # Remove .db extension if passed in filename
    if '.db' in database_filename:
        database_filename = database_filename[:database_filename.find(".db")]

    # Create SQLAlchemy engine and save data
    engine = create_engine('sqlite:///{}.db'.format(database_filename))
    df.to_sql('Disaster_Data', engine, index=False, if_exists='replace')


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
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
