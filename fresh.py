import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import json
import datetime as dt
import math
import random
from urllib.parse import urlsplit

# def initial_analysis(df):
#     print(df.info())
#     print(df.isna().sum())
#
# def main():
#     # df = pd.read_csv("./latestdata/latestdata.csv")
#     # initial_analysis(df)
#     # # Immediately clear that we have a lot of missing data
#     # # A lot of them are also missing the outcome which we need
#     # # So lets start by removing all of the records without an outcome
#     # df = df[df["outcome"].notna()]
#     # df.to_csv("with_outcome.csv", index=False)
#     # # We will save it to a file so we can load it faster (2.6m vs 300k rows)
#     # This is much more managable
#     df = pd.read_csv("with_outcome.csv")
#     initial_analysis(df)
#     df.describe()

# The question I want to be able to answer is: "Given a case record are we able to predict if they will survive the illness or if they will succumb to the disease?"
# Why? Binary classifier which we've studied quite a lot
#

# We are interested in predicting outcomes
# So we need the outcomes to be labelled otherwise we can't do anything with them
df = pd.read_csv("./latestdata/latestdata.csv", low_memory=False)

df.info()
df.isna().sum()
df = df[df["outcome"].notna()]
df.to_csv("with_outcome.csv", index=False)

df = pd.read_csv("with_outcome.csv", low_memory=False)

df.info()
# We have a lot of missing data here
# I'll start by getting rid of columns that have too little data to try and fix
# So we'll drop "date_onset_symptoms", "date_admission_hospital", "symptoms", "lives_in_Wuhan", "reported_market_exposure", "sequence_available", "date_death_or_discharge", "notes_for_discussion", "admin3", "admin2"

df = df.drop(["date_onset_symptoms", "date_admission_hospital", "symptoms", "lives_in_Wuhan", "reported_market_exposure", "sequence_available", "date_death_or_discharge", "notes_for_discussion", "admin3", "admin2"], axis=1)

df.info()

# I've kept the travel_history_dates and travel_history_location as we may be able to increase the number of records with a binary value
# Currently have 307057 / 307382 records with a travel_history_binary but if either of ravel_history_dates or travel_history_location have a value we can set the binary to true

def clean_travel_binary(row):
    if pd.notna(row["travel_history_binary"]):
        return row["travel_history_binary"]

    if pd.isna(row["travel_history_dates"]) and pd.isna(row["travel_history_location"]):
        return row["travel_history_binary"]

    # If one of the columns has something then we can set it to true otherwise we don't know
    return True

df["travel_history_binary"] = df.apply(clean_travel_binary, axis=1)
df = df.drop(["travel_history_dates", "travel_history_location"], axis=1)

df.info()

# We now have 307150 / 307382 (increase of 93 records) with a travel_history_binary and I've dropped the travel_history_dates and travel_history_location columns
# So drop all of those without a travel_history_binary now as we can't recover this

df = df[df["travel_history_binary"].notna()]

pd.set_option("display.max_columns", None)
df.head()

df.info()

# It is probably safe to drop location as well since we have latitude and longitude encoding this information
# Also location is missing for ~97-98% of the records
# admin1 is also missing for a massive chunk ~88% of the records which is also encoded into the lat and lon
# The additional_information column isn't appropriate for the model either as it is free text which we can't process effectively
# The data_moderator_initials are also metadata about the collation of the dataset rather than relevant to the actual records so it can be dropped

df = df.drop(["location", "admin1", "additional_information", "data_moderator_initials"], axis=1)

df.info()
# 1 record is missing a country so lets see what the record is
missing_country = df[df["country"].isna()]
missing_country

# This record is in Taiwan and the country (and country_new) haven't been recorded for politically sensitive reasons
# So I check if there are any other records in with a province of Taiwan

taiwan = df[(df["province"].notna()) & (df["province"].str.lower() == "taiwan")]
taiwan

# It is the only record so I'm unsure what would be the correct thing to do about this
# Since it is only 1 record I believe it is sensible to drop this record

df = df[df["country"].notna()]
df.info()

# So we have ID, country, latitude, longitude, geo_resolution, chronic_disease_binary, outcome, admin_id, travel_history_binary for all of the 307149 records
# Next I want to look at the country_new and see if it differs from country
differing_countries = df[df["country_new"].notna()]
differing_countries = differing_countries[differing_countries["country"].str.lower() != differing_countries["country_new"].str.lower()]
differing_countries.info()
# So if we exclude the non-NA country_news and check for differences we have 0 records
# We can safely drop the country_new as it is redundant
df = df.drop(["country_new"], axis=1)
df.info()

(1 - 35 / 307149) * 100
# We have 35 records without a date_confirmation so we can drop those rows and still keep 99.9% of the records
df = df[df["date_confirmation"].notna()]
df.info()

# Now I'm going to look at all of the chronic_disease values (99)
chronic_diseases = df["chronic_disease"].unique()
len(chronic_diseases)
chronic_diseases

# I'm going to filter the dataset and see if all of those with a chronic_disease_binary=True have a chronic_disease free text filled out

df["chronic_disease_binary"].value_counts(dropna=False)

def clean_chronic_binary(row):
    if pd.notna(row["chronic_disease_binary"]):
        return row["chronic_disease_binary"]

    # If one of the columns has something then we can set it to true otherwise we don't know
    return pd.notna(row["chronic_disease_binary"])

df["chronic_disease_binary"] = df.apply(clean_chronic_binary, axis=1)
df["chronic_disease_binary"].value_counts(dropna=False)

(307013 - 101) / 307013 * 100

# Since chronic_disease_binary=True makes up < 0.01% of the dataset it doesn't make sense to deal with the chronic_disease field
# This is because it could massively skew the results for those with the chronic_disease that have been recorded since not enough records have
# the data for it create a significant effect and it may unintentionally skew those without a chronic_disease into the survival category
# So just drop the chronic_disease
# Will need to clarify this much more
df = df.drop(["chronic_disease"], axis=1)
df.info()
df["travel_history_binary"].value_counts(dropna=False)

df.info()

df = df.rename(columns = { "chronic_disease_binary": "has_chronic_disease", "travel_history_binary": "has_travel_history" })
df.info()

df["has_travel_history"].value_counts(dropna=False)

df["has_travel_history"] = df["has_travel_history"].astype("bool")
df["has_travel_history"].value_counts(dropna=False)
df.info()

# Less than 1000 records are missing a province, we will put an Unknown category in here instead
df["province"].value_counts(dropna=False)
df["province"].isna().sum()
df["province"] = df["province"].fillna("unknown")
df["province"].isna().sum()
df.info()

(307114 - 262845) / 307114 * 100
# we also want to do something similar for city as we are missing ~14-15% of the data for this column
# that would be a sizeable amount of the data to just drop the rows for
# while it could be doable, I'm going to replace them with an unknown placeholder instead

df["city"].isna().sum()
df["city"] = df["city"].fillna("unknown")
df["city"].isna().sum()
df.info()
(307114 - 279429) / 307114 * 100
# and likewise for source (missing for 9% of records)
df["source"].isna().sum()
df["source"] = df["source"].fillna("unknown")
df["source"].isna().sum()
df.info()
# Now we have to deal with age and sex, eliminating these records will severely reduce the size of the dataset
# Looking at the records with both age and sex
age_and_sex_records = df[(df["age"].notna()) & (df["sex"].notna())]
age_and_sex_records.isna().sum()
age_and_sex_records.info()
# Leaves us with only 33401 records from the original 2.6m that we had and down from the 307144 relatively clean records with outcomes
33401 / 307144
# That's a mere 10-11% records left which isn't really acceptable nor suitable

df.info()

df["sex"].value_counts(dropna=False)

# We need to impute some values into the NaN values instead for the sex
# I've thought of a couple of potential options:
# - Allocate uniform (in a systematic way rather than completely randomly) 50-50 male-female on those with NaN
# - Allocate according to the distribution of male-female in the 37914 other records
# - Use some external data source to determine the ratio
# I feel the best way may be to just introduce a new column to say we have imputed the value
# and then allocate systematically according to the distribution of sex already known in the data
# Considered using MICE but this requires assumptions that the data is randomly missing which isn't the case
df["sex_was_missing"] = df["sex"].isna()
df.info()
df["sex_was_missing"].value_counts(dropna=False)
# Now this is arguably where things get a bit murky with being somewhat correct but lets do it and see what happens

def allocate_sex(row):
    threshold = 24228 / (24228 + 13686)

    if pd.notna(row["sex"]):
        return row["sex"]

    rand = random.random()

    if rand > threshold:
        return "female"

    return "male"

df["sex"].value_counts(normalize=True)
df["sex"] = df.apply(allocate_sex, axis=1)
df["sex"].value_counts(normalize=True)
df.info()
df["sex"].value_counts(dropna=False)
# We now change the sex column to is_male so it is a bool
df["is_male"] = np.where(df["sex"].str.lower() == "male", True, False)
df["is_male"].value_counts(dropna=False)
df = df.drop("sex", axis=1)
df.info()
# Now need to do something about age
unique_ages = df["age"].unique()
print(len(unique_ages))
print(unique_ages)
# So we have fractional ages, ranges and int ages
# First lets make the ranges and fractionals into integers and average ages (ignoring NaN for now)

def fix_age_ranges_and_fractions(row):
    if pd.isna(row["age"]):
        return row["age"]

    str_age = str(row["age"])
    int_age = -1

    if "-" in str_age:
        splits = str_age.split("-")

        if len(splits) == 1:
            int_age = int(float(splits[0]))
        else:
            str_lower, str_upper = splits
            if str_upper != "":
                int_age = (int(float(splits[0])) + int(float(splits[1]))) // 2
            else:
                int_age = int(float(splits[0]))
    else:
        # floor the age if someone is 18.9 years we say they are 18 years not 19
        int_age = math.floor(float(str_age))

    return int_age

df["age"].value_counts(normalize=True)
df["age"] = df.apply(fix_age_ranges_and_fractions, axis=1)
df["age"].value_counts(normalize=True)
unique_ages = df["age"].unique()
print(len(unique_ages))
print(unique_ages)

df["age_was_missing"] = df["age"].isna()
df.info()
df["age_was_missing"].value_counts(dropna=False)
# Assign the mode to the age probably a bad idea
# really think this is going to skew the model a lot but we'll have to experiment
df["age"] = np.where(df["age"].isna(), df["age"].mean(), df["age"])
df["age"].value_counts(dropna=False)

df.info()

df.to_csv("preview.csv", index=False)

# Now have all of the records with every column having a non-N/A value
# We now need to look at standardising the city, province, country, date_confirmation, source and outcome
# date_confirmation is an ordinal attribute
df["date_confirmation"] = pd.to_datetime(df["date_confirmation"], format="%d.%m.%Y")
df.info()
df["date_confirmation"].value_counts()

# Lets look at the source URL
unique_sources = df["source"].str.lower().unique()
len(unique_sources)
df["source"].value_counts()

def simplify_sources(row):
    lowercase_url = row["source"].lower()
    if lowercase_url == "unknown":
        return "unknown"

    netloc = urlsplit(lowercase_url).netloc

    if netloc == "":
        return "unknown"

    return netloc

df["source"].value_counts(dropna=False)
df["source"] = df.apply(simplify_sources, axis=1)
df["source"].value_counts(dropna=False)
unique_sources = df["source"].unique()
df["source"] = df["source"].astype("category")
df.info()
df.head()
df["source"].value_counts()
df["source"] = df["source"].cat.codes
df["source"].value_counts()
df.info()

# Just city, province, country, geo_res, outcome to do

df["city"] = df["city"].str.lower()
df["city"].value_counts(dropna=False)
df["city"] = df["city"].astype("category")
df.info()
df.head()
df["city"].value_counts()
df["city"] = df["city"].cat.codes
df["city"].value_counts()
df.info()

df["province"] = df["province"].str.lower()
df["province"].value_counts(dropna=False)
df["province"] = df["province"].astype("category")
df.info()
df.head()
df["province"].value_counts()
df["province"] = df["province"].cat.codes
df["province"].value_counts()
df.info()

df["country"] = df["country"].str.lower()
df["country"].value_counts(dropna=False)
df["country"] = df["country"].astype("category")
df.info()
df.head()
df["country"].value_counts()
df["country"] = df["country"].cat.codes
df["country"].value_counts()
df.info()

df["geo_resolution"] = df["geo_resolution"].str.lower()
df["geo_resolution"].value_counts(dropna=False)
df["geo_resolution"] = df["geo_resolution"].astype("category")
df.info()
df.head()
df["geo_resolution"].value_counts()
df["geo_resolution"] = df["geo_resolution"].cat.codes
df["geo_resolution"].value_counts()
df.info()

# Outcome will need different mapping rather than category labelling
df["outcome"] = df["outcome"].str.lower()
unique_outcomes = df["outcome"].unique()
len(unique_outcomes)
print(unique_outcomes)

survive_map = {
    "critical condition, intubated as of 14.02.2020": True,
    "discharge": True,
    "discharged": True,
    'death': False,
    'recovered': True,
    'released from quarantine': True,
    'stable': True,
    'died': False,
    'symptoms only improved with cough. currently hospitalized for follow-up.': True,
    'alive': True,
    'dead': False,
    'deceased': False,
    'stable condition': True,
    'under treatment': True,
    'critical condition': True,
    'receiving treatment': True,
    'severe illness': True,
    'unstable': True,
    'hospitalized': True,
    'treated in an intensive care unit (14.02.2020)': True,
    'severe': True,
    'migrated': True,
    'migrated_other': True,
    'https://www.mspbs.gov.py/covid-19.php': None,
    'discharged from hospital': True,
    'not hospitalized': True,
    'recovering at home 03.03.2020': True
}

df["has_survived"] = df["outcome"].map(survive_map)
df["has_survived"].value_counts(dropna=False)
missing_survived = df[df["has_survived"].isna()]
missing_survived.head()
missing_survived_outcomes = missing_survived["outcome"].unique()
missing_survived_outcomes

# These are all the ones that mapped to None in the survive_map

df.info()
df = df[df["has_survived"].notna()]
# Gives a 2% death rate which is somewhere around what we expect
df["has_survived"].value_counts(dropna=False, normalize=True)
df.info()
df["has_survived"] = df["has_survived"].astype("bool")
df.info()

# Finally drop the outcome and the ID and save it to a CSV for easy importing
df = df.drop(["ID", "outcome"], axis=1)
df.info()
df.to_csv("ready.csv", index=False)

# Load the saved csv with our records
df = pd.read_csv("ready.csv")
df.info()
# We will still have to remap the date
df["date_confirmation"] = pd.to_datetime(df["date_confirmation"], format="%Y.%m.%d").map(dt.datetime.toordinal)
df.info()
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2, random_state=1828, shuffle=True)

train.shape[0], test.shape[0]

training_unlabelled = train.drop("has_survived", axis=1)
training_labels = train["has_survived"].copy()

test_unlabelled = test.drop("has_survived", axis=1)
test_labels = test["has_survived"].copy()

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
training_unlabelled = ss.fit_transform(training_unlabelled)
test_unlabelled = ss.transform(test_unlabelled)

from sklearn.linear_model import SGDClassifier

model = SGDClassifier()
model.fit(training_unlabelled, training_labels)
predicted = model.predict(test_unlabelled)

from sklearn.metrics import confusion_matrix, recall_score, precision_score, matthews_corrcoef, balanced_accuracy_score, f1_score, classification_report, plot_roc_curve
cmatrix = confusion_matrix(test_labels, predicted)
tn, fp, fn, tp = cmatrix.ravel()

tn, fp, fn, tp

rs = recall_score(test_labels, predicted)
rs

ps = precision_score(test_labels, predicted)
ps

mcc = matthews_corrcoef(test_labels, predicted)
mcc

bas = balanced_accuracy_score(test_labels, predicted)
bas

f1 = f1_score(test_labels, predicted)
f1

report = classification_report(test_labels, predicted)
report

## Getting terrible results think it is because of the age and sex predictions made in preprocessing
# Before deleting it might be worth getting some graphs for the experimental procedure part

%matplotlib inline
plot_roc_curve(model, test_unlabelled, test_labels, name="ROC Test")
plt.show()

#
