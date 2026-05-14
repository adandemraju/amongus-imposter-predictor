import glob
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import joblib

def parse_time(s):
    if s == "-":
        return None
    parts = s.split("m ")
    minutes = int(parts[0])
    seconds = int(parts[1][:-1])
    total = minutes * 60 + seconds
    return total

def preprocess(df):
    df['Game Length (s)'] = df['Game Length'].apply(parse_time)

    df['Task Time (s)'] = df['Time to complete all tasks'].apply(parse_time)
    df['Task Time (s)'] = df['Task Time (s)'].fillna(0)

    #changing yes/no, win/lose values to 1/0 for later analysis
    df['All Tasks Completed'] = df['All Tasks Completed'].map({'Yes':1, 'No': 0, '-': 0})
    df['Ejected'] = df['Ejected'].map({'Yes':1, 'No': 0, '-': None})

    #fill in Nan - sabatagoes fixed
    df['Sabotages Fixed'] = pd.to_numeric(df['Sabotages Fixed'], errors='coerce')
    df['Sabotages Fixed'] = df['Sabotages Fixed'].fillna(0)

    #create is_imposter column as 1/0 column from team column
    df['is_imposter'] = df['Team'].map({'Imposter':1, 'Crewmate': 0})

    features = ['All Tasks Completed', 'Ejected', 'Sabotages Fixed', 'Game Length (s)', 'Task Time (s)']
    df_clean = df[features]

    return df_clean

def load_data(folder_path):
    files = glob.glob(os.path.join(folder_path, "*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df

def train(df):
    X = df[['All Tasks Completed','Ejected', 'Sabotages Fixed', 'Game Length (s)', 'Task Time (s)']]
    y = df["is_imposter"]

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42, stratify=y)

    rfm = RandomForestClassifier(class_weight='balanced')
    rfm.fit(X_train, y_train)

    y_pred = rfm.predict(X_test)

    accuracy = accuracy_score(y_pred, y_test)

    confusion = confusion_matrix(y_test, y_pred)

    # relying_attribute = rfm.feature_importances_
    # for feature, importance in zip(X.columns, relying_attribute):
    #      print(feature,importance)
        
    return rfm, accuracy, confusion

def save_model(model, path):
    joblib.dump(model, os.path.join(path, "model.joblib"))

def main():
    data = load_data("../data")
    cleaned_data = preprocess(data)
    trained_model, accuracy, confusion = train(cleaned_data)
    save_model(trained_model, "../model")

if __name__ == "__main__":
    main()