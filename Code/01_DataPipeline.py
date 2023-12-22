#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#######################################################################################
#                           Script: DataPipeline.py
#                           Author: Agudelo
#                           Date: May 5, 2022
#######################################################################################


"""
Description:
This script implements a data science pipeline that includes loading data from a CSV file,
classifying variables, transforming the 'destinated_area' column, assigning categorical
variables using One-Hot Encoding, calculating the Gower distance matrix, detecting
outliers using DBSCAN and IQR methods, and imputing missing data using MissForest.
The main function executes the entire process on a given dataset.

Note: Ensure that the required libraries are correctly imported and available in your
environment for the script to run successfully.
"""



import pandas as pd
import numpy as np
import sys
import math
import itertools
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import gower
import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from missingpy import MissForest

# Function to load data from a CSV file
def cargar_datos(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, delimiter=";")
    return df

# Function to classify variables and perform data type conversions
def clasificar_variables(path: str) -> pd.DataFrame:
    df = cargar_datos(path)
    df = df.astype({"city_code": "category", "product_type": "category", "product": "category", "destinated_area": "float"})
    df["year"] = pd.to_datetime(df["year"], dayfirst=True)
    Tipo_de_variables = df.info()
    print(Tipo_de_variables)
    return df

# Function to transform the 'destinated_area' column
def transformar_area(path: str) -> pd.DataFrame:
    df = clasificar_variables(path)
    df["destinated_area"] = df["destinated_area"].replace(0, 1)
    df["destinated_area"] = df["destinated_area"].replace(1, math.e)
    df["LN_destinated_area"] = np.log(df["destinated_area"])
    return df

# Function to assign categorical variables using One-Hot Encoding
def asginar_variables_cat(path: str) -> pd.DataFrame:
    enc = OneHotEncoder(handle_unknown='ignore')
    df = transformar_area(path)
    df['year_Cat'] = LabelEncoder().fit_transform(df['year'])
    Dummy = pd.DataFrame(enc.fit_transform(df[['city_code', "product_type", "product"]]).toarray())
    Columns_Dummy = list((enc.get_feature_names(['city_code', 'product_type', "product"])))
    Dummy.columns = Columns_Dummy
    df = df.join(Dummy)
    return df

# Function to calculate the Gower distance matrix
def distance_matrix_func() -> np.ndarray:
    df = asginar_variables_cat()
    gower_df = pd.DataFrame()
    gower_df["city_code"] = df["city_code"]
    gower_df["destinated_area"] = df["destinated_area"]
    gower_df = gower_df.groupby("city_code").agg({"destinated_area": "sum"}).reset_index()
    gower_df = gower_df.dropna()
    distance_matrix = gower.gower_matrix(gower_df, cat_features=[True, False])
    return distance_matrix

# Function to detect outliers using DBSCAN
def DBSCAN_Outliers() -> pd.DataFrame:
    distance_matrix = distance_matrix_func()
    df = asginar_variables_cat()
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs = neigh.fit(distance_matrix)
    distances, indices = nbrs.kneighbors(distance_matrix)
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    plt.plot(distances)
    dbscan_cluster = DBSCAN(eps=0.0007,
                            min_samples=2,
                            metric="precomputed")

    dbscan_cluster.fit(distance_matrix)
    gower_df = pd.DataFrame()
    gower_df["city_code"] = df["city_code"]
    gower_df["destinated_area"] = df["destinated_area"]
    gower_df = gower_df.groupby("city_code").agg({"destinated_area": "sum"}).reset_index()
    gower_df = gower_df.dropna()

    gower_df = pd.DataFrame(gower_df)
    gower_df["cluster"] = dbscan_cluster.labels_

    df_outliers = df[df["cluster"].isna()]
    return df_outliers

# Function to detect outliers using IQR
def IQR_Outliers() -> pd.DataFrame:
    df = transformar_area()
    Q1 = df.LN_destinated_area.quantile(0.25)
    Q3 = df.LN_destinated_area.quantile(0.75)
    IQR = Q3 - Q1
    df_final = df[~((df["LN_destinated_area"] < (Q1 - 1.5 * IQR)) | (df["LN_destinated_area"] > (Q3 + 1.5 * IQR)))]
    return df_final

# Function to impute missing data using MissForest
def inputar_datos(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.drop(["city_code", "product_type", "product", "year"], axis=1, inplace=False)
    np_imputed = MissForest().fit_transform(df_copy)
    df['destinated_area'] = np_imputed[:, 0]
    df['LN_destinated_area'] = np.log(df['destinated_area'])
    return df

# Main function to execute the entire process
def main(path: str):
    df = asginar_variables_cat(path)
    df_imputed = inputar_datos(df)
    print(df_imputed.head())
    print(df_imputed.isnull().sum())

# Entry point of the script
if __name__ == "__main__":
    path = '/Users/j.agudelo/git/PJ1/Input/historical-database.csv'
    main(path)
