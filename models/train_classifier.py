from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
# from sklearn.model_selection import GridSearchCV
import pickle
import re
import pandas as pd
import sys
import time


def load_data(database_filepath):
    """
    This methods loads data from a .db to a pandas dataframe.

    Args:
        - database_filepath: relative path to database file
    NOTE: the table name saved in db file is assumed Disaster_Data

    Returns:
        pandas dataframe
    """
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table("Disaster_Data", engine)
    X = df["message"]
    Y = df.drop(columns=['id', 'message', 'original', 'genre'])
    categories = Y.columns.tolist()
    return (X, Y, categories)


def tokenize(text):
    """
    This function converts the text into tokens by performing
    the following text processing operations:
    1. Normalization
    2. Lemmatization
    3. Removal of stop words

    Args:
        - text: String to perform the tokenization.

    Returns:
        - list of tokens in the string.
    """
    # Define the set of stop words in english language
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    # Normalize text: remove all punctuation marks
    # convert all text to lower case
    text = re.sub(r"[^a-z0-9]", " ", text.lower())
    # Tokenize text
    words = word_tokenize(text)
    # Remove stop words
    tokens = [lemmatizer.lemmatize(word.strip()) for word in words if word not in stop_words]
    return tokens


def build_model():
    """
    Build the machine learning pipeline model to train on the data.

    Returns:
        A ML model that has can call fit on training data and
        predict on testing data.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(SVC(tol=0.01, cache_size=500))),
    ])

    # training through all these parameters takes lots of time
    # As there is short of resources setting these parameters
    # to default values.
    """
    'vect__max_df': [0.8, 1.0]
    'vect__ngram_range': [(1,2), (1,1)]
    parameters = {
        'clf__estimator__C': [1, 10],
        'clf__estimator__kernel': ['linear', 'rbf'],
    }

    cv = GridSearchCV(pipeline, parameters, cv=3, n_jobs=-1)
    return cv
    """
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Predict the genre of text using the model and find the
    F1score, recall, precision and accuracy for each category

    Args:
        - model: ML model after training.
        - X_test: input test data
        - Y_test: actual output for test data
        - category_names: list of column names in output.
    """
    y_pred = model.predict(X_test)

    # convert predicted output to pandas dataframe from numpy ndarray
    # this helps in iterating through the columns
    y_pred = pd.DataFrame(y_pred, columns=Y_test.columns.tolist())
    for column in category_names:
        print("Pipeline classification report for column ", column)
        print(classification_report(Y_test[column].tolist(), y_pred[column].tolist()))
        print("*************************************************")


def save_model(model, model_filepath):
    """
    Save the trained model to a file as pickle

    Args:
        - model: trained model to save
        - model_filepath: relative path (with filename) to save the model.
    """
    # save the model to disk
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        start = time.time()
        model.fit(X_train, Y_train)
        print("Took {} seconds to train the model".format(time.time() - start))

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
