# -*- coding: utf-8 -*-
import os
import re

import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd


# Non riuscivo a importarla quindi l'ho copiata quì
def dummy_extractor(data, final_data_name):
    """ Estrazione variabili dummy che identificano i giorni della settimana, ora e mese """

    # Creazione variabili dummy per l'ora del giorno
    data["hour"] = data.index.hour
    data = pd.get_dummies(data, columns=["hour"])

    # Creazione variabili dummy per il giorno della settimana
    data["week_day"] = data.index.day_of_week
    data = pd.get_dummies(data, columns=["week_day"])

    # Creazione variabili dummy per il mese dell'anno
    data["month"] = data.index.month
    data = pd.get_dummies(data, columns=["month"])

    # Salvataggio
    data.to_csv(final_data_name)


# TODO: capire se sia meglio spostare questa funzione su un altro file
def adjust_datetime(df, col_name, form):
    df[col_name] = df[col_name].str.split("+", expand=True)[0]  # elimina utc se presente
    df[col_name] = pd.to_datetime(df[col_name].str.strip(), format=form, yearfirst=True)
    return df


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).

        L' ordine d' importazione dei file è importante:
            - first_input_filepath -> energy_datasetSpain.csv
            - second_input_filepath -> ninja_weather_country_ES_merra_2_land_area_weighted.csv
    """
    logger = logging.getLogger(__name__)
    logger.info('Making final data set from raw data')

    # Import datasets
    logger.info('Importing raw datasets...')
    weather_dataset = pd.read_csv(input_filepath, sep=";")

    # Cleaning
    logger.info('Cleaning...')
    weather_dataset = adjust_datetime(weather_dataset, "time", form='%d/%m/%Y %H:%M')

    weather_dataset["precipitation"] = [float(re.sub(",", ".", i)) for i in weather_dataset["precipitation"]]
    weather_dataset["snowfall"] = [float(re.sub(",", ".", i)) for i in weather_dataset["snowfall"]]

    for i in weather_dataset.columns[1:]:
        if i != "cloud_cover":
            weather_dataset[i] = [float(value) / 1000 for value in weather_dataset[i]]
        else:
            weather_dataset[i] = [float(value) / 10000 for value in weather_dataset[i]]
    logger.info('   + scale OK', len(weather_dataset))

    # Set the "time" column as index
    weather_dataset = weather_dataset.set_index("time")
    weather_dataset = weather_dataset.loc["2019-01-01 00:00:00":]


    # Sistemazione dataframe esterni energy
    data_path = os.listdir("../../data/raw/annual_load")

    df_cleaned = []

    for df in data_path:
        m_df = pd.read_csv("../../data/raw/annual_load/" + df)

        # Sistemazione colonna time
        new_time_column = [t.split("-")[0] for t in m_df["Time (CET/CEST)"]]
        m_df["time"] = new_time_column
        m_df = adjust_datetime(df=m_df, col_name="time", form='%d.%m.%Y %H:%M')

        # Riassegnazione indice ed eliminazione colonna Time (CET/CEST)
        m_df = m_df.set_index(m_df["time"])
        m_df.index = pd.to_datetime(m_df.index)
        m_df = m_df.drop(columns=["Time (CET/CEST)", "time"])

        # Ridenominazione colonne
        m_df = m_df.rename(columns={"Day-ahead Total Load Forecast [MW] - BZN|ES": "total load forecast",
                                    "Actual Total Load [MW] - BZN|ES": "total load actual"})

        # Eliminazione righe con orari non tondi (Es. 2022-12-31 22:45:00)
        for t in m_df.index:
            if t.minute != 0:
                m_df = m_df.drop(m_df[m_df.index == t].index)

        df_cleaned.append(m_df)

    # Concatenazione dataset esterni
    final_df = pd.concat([df_cleaned[0], df_cleaned[1], df_cleaned[2], df_cleaned[3]])

    # Merge con i dataset esterni
    df = weather_dataset.merge(final_df, how="left", on="time")

    # Add holiday variable
    festivity = ["2015-01-01", "2016-01-01", "2017-01-01", "2018-01-01", "2019-01-01", "2020-01-01", "2021-01-01", "2022-01-01"
                 "2015-04-15", "2016-04-15", "2017-04-15", "2018-04-15", "2019-04-15", "2020-04-15", "2021-04-15", "2022-04-15"
                 "2015-05-01", "2016-05-01", "2017-05-01", "2018-05-01", "2019-05-01", "2020-05-01", "2021-05-01", "2022-05-01"
                 "2015-08-15", "2016-08-15", "2017-08-15", "2018-08-15", "2019-08-15", "2020-08-15", "2021-08-15", "2022-08-15"
                 "2015-10-12", "2016-10-12", "2017-10-12", "2018-10-12", "2019-10-12", "2020-10-12", "2021-10-12", "2022-10-12"
                 "2015-11-01", "2016-11-01", "2017-11-01", "2018-11-01", "2019-11-01", "2020-11-01", "2021-11-01", "2022-11-01"
                 "2015-12-06", "2016-12-06", "2017-12-06", "2018-12-06", "2019-12-06", "2020-12-06", "2021-12-06", "2022-12-06"
                 "2015-12-08", "2016-12-08", "2017-12-08", "2018-12-08", "2019-12-08", "2020-12-08", "2021-12-08", "2022-12-08"
                 "2015-12-25", "2016-12-25", "2017-12-25", "2018-12-25", "2019-12-25", "2020-12-25", "2021-12-25", "2022-12-25"
                 "2015-12-26", "2016-12-26", "2017-12-26", "2018-12-26", "2019-12-26", "2020-12-26", "2021-12-26", "2022-12-26"]

    holidays = []

    for i in df.index:
        if str(i).split()[0] in festivity:
            holidays.append(1)
        elif i.dayofweek == 6 or i.dayofweek == 5:
            holidays.append(1)
        else:
            holidays.append(0)

    df["Holidays"] = holidays

    # Add dummy variables and save the file in data/interim
    dummy_extractor(data=df, final_data_name=output_filepath)

    return logger.info('Done')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

    # TODO: SISTEMARE
