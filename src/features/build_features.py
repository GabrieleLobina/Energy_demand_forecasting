# System
# =================================================================
import os

# Data import and manipulation
# =================================================================
import pandas as pd
import re


# Feature extraction
# =================================================================
def dummy_extractor(data, final_data_name):
    """ Estrazione variabili dummy che identificano i giorni della settimana, ora e mese """
    data["time"] = pd.to_datetime(data["time"])

    # Creazione variabili dummy per l'ora del giorno
    data["hour"] = data["time"].dt.hour
    data = pd.get_dummies(data, columns=["hour"])

    # Creazione variabili dummy per il giorno della settimana
    data["week_day"] = data["time"].dt.day_of_week
    data = pd.get_dummies(data, columns=["week_day"])

    # Creazione variabili dummy per il mese dell'anno
    data["month"] = data["time"].dt.month
    data = pd.get_dummies(data, columns=["month"])

    # Salvataggio
    data.to_csv(final_data_name)


if __name__ == "__main__":
    # Identificazione path della directory del file di lavoro
    py_path = re.sub(r"\\", "/", os.environ["PYTHONPATH"]).split(";")[2]
    path_processed_data = py_path + "/Forecasting_Repository/data/processed/processed_data.csv"

    df = pd.read_csv(path_processed_data)
    dummy_extractor(df, "../../data/interim/DF_SeasonalDummies.csv")
