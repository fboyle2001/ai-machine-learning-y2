from sklearn import svm
import sklearn.model_selection
import sklearn.metrics
import sklearn.pipeline
import sklearn.preprocessing
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import json
import datetime as dt

def load_data():
    return pd.read_csv("./latestdata/latestdata.csv")

# Anything not in this list will be removed from the dataset
# True = alive, False = dead
outcome_lookup = {
    "hospitalized": True,
    "deceased": False,
    "recovered": True,
    "died": False,
    "under treatment": True,
    "receiving treatment": True,
    "alive": True,
    "stable condition": True,
    "stable": True,
    "discharge": True,
    "discharged": True,
    "death": False,
    "dead": False,
    "released from quarantine": True,
    "critical condition, intubated as of 14.False2.2False2False": True,
    "discharged from hospital": True,
    "symptoms only improved with cough. currently hospitalized for follow-up.": True,
    "critical condition": True,
    "severe": True,
    "not hospitalized": True,
    "treated in an intensive care unit (14.False2.2False2False)": True,
    "unstable": True,
    "recovering at home False3.False3.2False2False": True,
    "severe illness": True
}

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

    reduced["travel_history_binary"] = reduced.apply(clean_travel_binary, axis=1)
    reduced = reduced[reduced["travel_history_binary"].notna()]
    # Now we can remove the travel_history_dates and travel_history_location
    reduced = reduced.drop(["travel_history_dates", "travel_history_location"], axis=1)
    reduced = reduced[reduced["outcome"].notna()]
    reduced = reduced.drop(["date_death_or_discharge", "location", "admin3", "admin2", "admin1", "lives_in_Wuhan", "date_onset_symptoms", "date_admission_hospital"], axis=1)

    reduced = reduced[reduced["outcome"].str.lower().isin(outcome_lookup)]
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

def clean_travel_binary(row):
    if pd.notna(row["travel_history_binary"]):
        return row["travel_history_binary"]

    if pd.isna(row["travel_history_dates"]) and pd.isna(row["travel_history_location"]):
        return row["travel_history_binary"]

    #print(row["travel_history_binary"], row["travel_history_dates"], row["travel_history_location"])

    # If one of the columns has something then we can set it to true otherwise we don't know
    return True

def summarise(df):
    print(df.info())
    print(df.isna().sum())
    print("Rows", df.shape[0], "Columns", df.shape[1])

def summarise_columns(df):
    for column in df.columns:
        print(column)
        print(df[column].value_counts(normalize=True, dropna=False))
        print("------------")

def after_analysis_clean(df):
    pass

def clean_outcome(row):
    current = row["outcome"].lower()

    if current not in outcome_lookup:
        return np.nan

    return outcome_lookup[current]

def clean_outcomes(df):
    df["survived"] = df.apply(clean_outcome, axis=1)
    df = df[df["survived"].notna()]
    return df

def rushed_drops(df):
    reduced = df.drop(["symptoms", "city", "ID", "outcome"], axis=1)
    reduced = reduced[reduced["age"].notna()]
    reduced = reduced[reduced["sex"].notna()]
    reduced = reduced[reduced["province"].notna()]
    reduced = reduced[reduced["date_confirmation"].notna()]
    return reduced

def fix_age(row):
    current = row["age"]
    replace = np.nan

    if "-" in current:
        sp = current.split("-")
        lower = int(sp[0])
        higher = lower

        if len(sp) > 1:
            if len(sp[1]) != 0:
                higher = int(sp[1])

        avg = (lower + higher) // 2
        replace = avg
    else:
        replace = int(float(current))

    return replace

def fix_ages(df):
    df["age"] = df.apply(fix_age, axis=1)
    return df

def prep_ml():
    df = pd.read_csv("rushed.csv")
    # df = fix_ages(df)
    # df.to_csv("rushed.csv", index=False)
    # df["sex"] = df["sex"].astype("category")
    df["province"] = df["province"].astype("category")
    df["country"] = df["country"].astype("category")
    df["geo_resolution"] = df["geo_resolution"].astype("category")
    df["date_confirmation"] = pd.to_datetime(df["date_confirmation"]).map(dt.datetime.toordinal)
    # summarise(df)
    # summarise_columns(df)
    # ages = df["age"].unique()
    # print(ages)

    ohe_sex = pd.get_dummies(df.sex, prefix="sex")
    df = pd.concat([df, ohe_sex], axis=1)
    df = df.drop("sex", axis=1)

    ohe_country = pd.get_dummies(df.country, prefix="country")
    df = pd.concat([df, ohe_country], axis=1)
    df = df.drop("country", axis=1)

    df = df.drop("province", axis=1)
    # ohe_province = pd.get_dummies(df.province, prefix="province")
    # df = pd.concat([df, ohe_province], axis=1)
    # df = df.drop("province", axis=1)
    df = df.drop("geo_resolution", axis=1)

    # summarise(df)

    train, test = sklearn.model_selection.train_test_split(df, test_size=0.2, random_state=1828, shuffle=True)

    training_no_labels = train.drop("survived", axis=1)
    training_labels = train["survived"].copy()

    test_no_labels = test.drop("survived", axis=1)
    test_labels = test["survived"].copy()

    return training_no_labels, training_labels, test_no_labels, test_labels

def basic_ml():
    # df = load_data()
    # df = clean_and_reduce(df)
    # df.to_csv("reduced.csv", index=False)
    # summarise(df)

    # df = pd.read_csv("reduced.csv")
    # df = pd.read_csv("cleaned.csv")
    # summarise(df)
    # rushed = rushed_drops(df)
    # summarise(rushed)
    # rushed.to_csv("rushed.csv", index=False)
    # summarise_columns(df)
    # df = clean_outcomes(df)
    # summarise_columns(df)
    # df.to_csv("cleaned.csv", index=False)
    # summarise(df)
    # summarise_columns(df)
    # basic_analysis(df)
    # after_analysis_clean(df)
    # summarise(df)

    training_no_labels, training_labels, test_no_labels, test_labels = prep_ml()

    model = sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(), svm.SVC(class_weight='balanced'))
    print("Training")
    model.fit(training_no_labels, training_labels)
    print("Trained")

    print("Predicting")
    predicted = model.predict(test_no_labels)
    print("Predicted")

    correct = 0
    predicted_survived = 0

    for i in range(len(predicted)):
        match = predicted[i] == test_labels.iloc[i]
        correct += match
        predicted_survived += predicted[i]

    incorrect = len(predicted) - correct

    print("Metrics")
    print()
    print("Test Samples", len(predicted))
    print("Correct", correct)
    print("Incorrect", incorrect)
    print("Accuracy", (correct / (correct + incorrect)) * 100)
    print("Predicted Survive", predicted_survived)
    print("Predicted Died", len(predicted) - predicted_survived)
    print()

    confusion_matrix = sklearn.metrics.confusion_matrix(test_labels, predicted)
    tn, fp, fn, tp = confusion_matrix.ravel()

    print("True Positives", tp)
    print("False Positives", fp)
    print("True Negatives", tn)
    print("False Negatives", fn)
    print()

    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)

    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)

    tnr = tn / (tn + fp)
    fnr = fn / (tp + fn)
    fdr = fp / (tp + fp)

    print("True Positive Rate (Sens)", tpr)
    print("False Positive Rate (Spec)", fpr)
    # ps = sklearn.metrics.precision_score(test_labels, predicted)
    # print("Precision Score", ps)
    print("Positive Predictive Value (PPV / Precision)", ppv)
    print("Negative Predictive Value (NPV)", npv)
    print("True Negative Rate", tnr)
    print("False Negative Rate", fnr)
    print("False Discovery Rate", fdr)
    print()

    f1_score = sklearn.metrics.f1_score(test_labels, predicted)
    fhalf_score = sklearn.metrics.fbeta_score(test_labels, predicted, beta=0.5)
    f2_score = sklearn.metrics.fbeta_score(test_labels, predicted, beta=2)
    print("F1 Score", f1_score)
    print("F0.5 Score", fhalf_score)
    print("F2 Score", f2_score)
    print()

    print("Plotting ROC...")
    sklearn.metrics.plot_roc_curve(model, test_no_labels, test_labels)
    plt.show()

def main():
    basic_ml()

main()
