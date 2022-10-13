# -*- coding: utf-8 -*-
import re

import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd


def adjust_datetime(df, col_name, form):
    df[col_name] = df[col_name].str.split("+", expand=True)[0]  # elimina utc se presente
    df[col_name] = pd.to_datetime(df[col_name].str.strip(), format=form, yearfirst=True)
    return df


@click.command()
@click.argument("first_input_filepath", type=click.Path(exists=True))
@click.argument("second_input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(first_input_filepath, second_input_filepath, output_filepath):
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
    raw_energy_dataset, weather_dataset = pd.read_csv(first_input_filepath), pd.read_csv(second_input_filepath, sep=";")

    # Cleaning
    logger.info('Cleaning...')
    raw_energy_dataset.drop(["generation fossil coal-derived gas",
                             "generation fossil oil shale",
                             "generation fossil peat",
                             "generation geothermal",
                             "generation hydro pumped storage aggregated",
                             "generation marine",
                             "generation wind offshore",
                             "forecast wind offshore eday ahead"], axis=1, inplace=True)

    raw_energy_dataset = raw_energy_dataset.interpolate(method="linear")

    raw_energy_dataset = adjust_datetime(raw_energy_dataset, "time", form='%Y/%m/%d %H:%M')
    weather_dataset = adjust_datetime(weather_dataset, "time", form='%d/%m/%Y %H:%M')

    weather_dataset["precipitation"] = [float(re.sub(",", ".", i)) for i in weather_dataset["precipitation"]]
    weather_dataset["snowfall"] = [float(re.sub(",", ".", i)) for i in weather_dataset["snowfall"]]

    for i in weather_dataset.columns[1:]:
        if i != "cloud_cover":
            weather_dataset[i] = [float(value) / 1000 for value in weather_dataset[i]]
        else:
            weather_dataset[i] = [float(value) / 10000 for value in weather_dataset[i]]

    # Merge su time
    logger.info('Merging datasets')
    new_df = raw_energy_dataset.merge(weather_dataset, how="left", on="time")

    # Set the "time" column as index
    new_df = new_df.set_index("time")

    # Add holiday variable
    festivity = ["2015-01-01", "2016-01-01", "2017-01-01", "2018-01-01",
                 "2015-04-15", "2016-04-15", "2017-04-15", "2018-04-15",
                 "2015-05-01", "2016-05-01", "2017-05-01", "2018-05-01",
                 "2015-08-15", "2016-08-15", "2017-08-15", "2018-08-15",
                 "2015-10-12", "2016-10-12", "2017-10-12", "2018-10-12",
                 "2015-11-01", "2016-11-01", "2017-11-01", "2018-11-01",
                 "2015-12-06", "2016-12-06", "2017-12-06", "2018-12-06",
                 "2015-12-08", "2016-12-08", "2017-12-08", "2018-12-08",
                 "2015-12-25", "2016-12-25", "2017-12-25", "2018-12-25",
                 "2015-12-26", "2016-12-26", "2017-12-26", "2018-12-26"]
    holidays = []

    for i in new_df.index:
        if str(i).split()[0] in festivity:
            holidays.append(1)
        elif i.dayofweek == 6 or i.dayofweek == 5:
            holidays.append(1)
        else:
            holidays.append(0)

    new_df["Holidays"] = holidays

    return new_df.to_csv(output_filepath, index=True), logger.info('Done')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
