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


def encondingInput(input_data, dataframe):
    obj_columns = getColumnByTypes(dataframe, "objects")

    for categorie in obj_columns:
        uniques = dataframe[categorie].unique()
        input_data[categorie] = pd.Categorical(
            input_data[categorie], categories=uniques
        )

    dummies = pd.get_dummies(input_data.drop(columns=["customerID"]), drop_first=False)

    dfd = dataframe
    dfd = getDummies(dfd)
    dummies_cols = dfd.drop(
        columns=["Churn"]
    )
    aligned_input = dummies.reindex(columns=dummies_cols.columns, fill_value=0)

    return aligned_input



def stepEnconding(dataframe, input_data):
    obj_columns = getColumnByTypes(dataframe, "objects")

    for categorie in obj_columns:
        if categorie in input_data.columns:
            uniques = dataframe[categorie].unique()
            input_data[categorie] = pd.Categorical(
                input_data[categorie], categories=uniques
            )

    dummies = pd.get_dummies(input_data, drop_first=False)

    dfd = dataframe
    dfd = getDummies(dfd)

    aligned_input = dummies.reindex(columns=dfd.columns, fill_value=0)

    return aligned_input


# def stepEnconding(dataframe,input_data):
#     obj_columns = getColumnByTypes(dataframe, "objects")

#     for categorie in obj_columns:
#         uniques = dataframe[categorie].unique()
#         input_data[categorie] = pd.Categorical(
#             input_data[categorie], categories=uniques
#         )

#     dummies = pd.get_dummies(input_data.drop(columns=["Churn","customerID"]), drop_first=False)

#     dfd = dataframe
#     dfd = getDummies(dfd)
#     dummies_cols = dfd.drop(
#         columns=["Churn"]
#     )

#     aligned_input = dummies.reindex(columns=dummies_cols.columns, fill_value=0)

#     return aligned_input