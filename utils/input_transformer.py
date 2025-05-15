# utils/input_transformer.py

import pandas as pd
from typing_extensions import Literal

def getColumnByTypes(dataframe, type: Literal["objects", "numeral"]):
    if type == "objects":
        return dataframe.select_dtypes(include=["object"]).columns        
    if type == "numeral":
        return dataframe.select_dtypes(include=[int, float]).columns


def getDummies(dataframe):
    if "customerID" in dataframe.columns:
        dataframe = dataframe.drop(columns=["customerID"])
    obj_columns = getColumnByTypes(dataframe,"objects").tolist()
    dummiedf = pd.get_dummies(dataframe, columns=obj_columns, drop_first=True)
    return dummiedf



def tenureToYears(tenure_value):
    years = tenure_value // 12
    return years

def createYearColumn(dataframe):
    dataframe["years"] = dataframe["tenure"].map(tenureToYears)
    return dataframe


def dataCleaning(dataframe):
    dataframe["TotalCharges"] = pd.to_numeric(dataframe["TotalCharges"], errors="coerce")
    dataframe["TotalCharges"] = dataframe["TotalCharges"].fillna(0)

    boolean_cols = dataframe.columns
    for col in boolean_cols:
        if set(dataframe[col]) <= {"Yes", "No"}:
            dataframe[col] = dataframe[col].map({"No": 0, "Yes": 1})

    createYearColumn(dataframe)

    dataframe = dataframe.drop(columns="Churn")

    return dataframe

def stepEnconding(dataframe, input_data):
    obj_columns = getColumnByTypes(dataframe, "objects")

    for categorie in obj_columns:
        if categorie in input_data.columns:
            uniques = dataframe[categorie].unique()
            input_data[categorie] = pd.Categorical(
                input_data[categorie], categories=uniques
            )

    dummies = pd.get_dummies(input_data, drop_first=False)

    dfd = dataCleaning(dataframe)
    dfd = getDummies(dfd)

    aligned_input = dummies.reindex(columns=dfd.columns, fill_value=0)

    return aligned_input


