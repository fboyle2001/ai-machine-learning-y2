import sklearn
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import json

def load_data():
    return pd.read_csv("./latestdata/latestdata.csv")

def reduce_data(df):
    # Why have I dropped these??
    # admin is kind of pointless as we have lat lon
    # applies for a few of them
    # mostly cause the data is very sparse for some of these columns
    # drop country new cause for all the data I wanted country == country_new
    reduced = df.drop(["symptoms", "lives_in_Wuhan", "travel_history_dates", "travel_history_location", "reported_market_exposure", "additional_information", "chronic_disease", "sequence_available", "date_death_or_discharge", "notes_for_discussion", "location", "data_moderator_initials", "admin1", "admin2", "admin3", "date_admission_hospital", "date_confirmation", "date_onset_symptoms", "source", "country_new"], axis=1)
    # core features that I wanted
    reduced = reduced[reduced["age"].notna()]
    reduced = reduced[reduced["sex"].notna()]
    reduced = reduced[reduced["outcome"].notna()]
    # country --> country_new difference?
    reduced = reduced[reduced["country"].notna()]
    reduced = reduced[reduced["city"].notna()]
    reduced = reduced[reduced["province"].notna()]
    reduced = reduced[reduced["travel_history_binary"].notna()]

    print(reduced.info())
    print(reduced.isna().sum())

    return reduced

def clean_data(df):
    # df["differing_country"] = np.where(df["country"] == df["country_new"], False, True)
    # df["country"] = np.where(df["country"] ==)
    #
    # df[df["differing_country"] == True].to_csv("dc.csv")

    # Could maybe fix this data rather than just drop them
    # missing_city = df[df["city"].isna()]
    # missing_city.to_csv("mc.csv")

    pass

def summarise_columns(df):
    for column in df.columns:
        print(df[column].value_counts())

def main():
    # raw_data = load_data()
    # df = reduce_data(raw_data)
    # df.to_csv("reduced.csv")
    # df = pd.read_csv("reduced.csv")
    #
    # print(df.info())
    # print(df.isna().sum())
    # summarise_columns(df)
    df = pd.read_csv("./latestdata/latestdata.csv", low_memory=False)
    print(df["chronic_disease_binary"].value_counts(dropna=False))


main()
