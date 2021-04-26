# Disaster Response Pipeline Project

This project provides a web app for classifying a message into 36 possible disaster response related categories. We use a multi class classifier algorithm to suggest all the possible categories a message can belong to. The visualizations on the home page show the distributions of messages across genres and categories in the dataset from [Figure Eight](https://www.figure-eight.com/).

### Installations

Your python environment would need the following packages installed. Specifying the versions used while working on the projects.

+ `Flask==1.1.2`
+ `numpy==1.20.2`
+ `nltk==3.6.1`
+ `pandas==1.2.4`
+ `plotly==4.14.3`
+ `requests==2.25.1`
+ `scikit-learn==0.24.1`
+ `SQLAlchemy==1.4.9`

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### File Structure and details

The files in this repository have the following structure:

```
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md
```
