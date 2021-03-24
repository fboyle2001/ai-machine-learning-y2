import sklearn
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import json

def load_data():
    return pd.read_csv("./latestdata/latestdata.csv")

def clean_and_reduce(df):
    # First remove the columns that have virtually no data or aren't of any use to my analysis
    # Analysis of the data showed that country_new was not useful
    reduced = df.drop(["source", "notes_for_discussion", "sequence_available", "reported_market_exposure", "chronic_disease", "data_moderator_initials", "country_new", "additional_information"], axis=1)

    # Analysis of the data shows that there are 61 records without longitude, latitude, and admin_id
    # There was also 115 without a country so delete these too
    # So we can remove those rows that are missing these
    reduced = reduced[reduced["longitude"].notna()]
    reduced = reduced[reduced["latitude"].notna()]
    reduced = reduced[reduced["admin_id"].notna()]
    reduced = reduced[reduced["country"].notna()]

    reduced["travel_history_binary"] = reduced.apply(compute_travel_binary, axis=1)
    reduced = reduced[reduced["travel_history_binary"].notna()]
    # Now we can remove the travel_history_dates and travel_history_location
    reduced = reduced.drop(["travel_history_dates", "travel_history_location"], axis=1)
    return reduced

def basic_analysis(df):
    # look at the country and country_new and see if we can merge these together
    # df["country_diff"] = np.where((df["country"] != df["country_new"]), True, False)
    #
    # diff = df[df["country_diff"] == True]
    # print(diff["country"].value_counts(dropna=False))
    # print(diff["country_new"].value_counts(dropna=False))
    #
    # # Looking at this .csv in Excel shows that only country_new is blank in this scenario
    # # So if country isna then we may as well drop the row as we can't recover it
    # # Although there is one interesting case when the province is Taiwan
    # # xor_countries = df[(df["country"].isna()) ^ (df["country_new"].isna())]
    # # xor_countries.to_csv("xor_countries.csv")

    # There were 65543 records without a travel_history_binary
    # Lets see what is happening with these
    # no_travel_history_binary = df[df["travel_history_binary"].isna()]
    # no_travel_history_binary.to_csv("no_travel_history_binary.csv")
    # # Analysis suggested that some of these could be salvaged by looking at the other columns

    pass

def compute_travel_binary(row):
    if pd.notna(row["travel_history_binary"]):
        return row["travel_history_binary"]

    if pd.isna(row["travel_history_dates"]) and pd.isna(row["travel_history_location"]):
        return row["travel_history_binary"]

    #print(row["travel_history_binary"], row["travel_history_dates"], row["travel_history_location"])

    # If one of the columns has something then we can set it to true otherwise we don't know
    return True

def summarise(df):
    print("Rows", df.shape[0], "Columns", df.shape[1])
    print(df.info())
    print(df.isna().sum())

def after_analysis_clean(df):
    pass

def main():
    # df = load_data()
    # df = clean_and_reduce(df)
    # df.to_csv("reduced.csv")
    # summarise(df)

    df = pd.read_csv("reduced.csv")
    summarise(df)
    # basic_analysis(df)
    # after_analysis_clean(df)
    # summarise(df)
    # #summarise(df)

main()
