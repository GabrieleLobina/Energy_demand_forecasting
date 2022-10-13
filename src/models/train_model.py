# Data import and manipulation
# ==============================================================================
import os
import pickle
import re

import numpy as np

# Modelling and Forecasting
# ==============================================================================
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import grid_search_forecaster
from skforecast.model_selection import backtesting_forecaster

# Warnings configuration
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')


class BacktestPipeline(object):
    """
    Classe per la creazione dei modelli finali.

    La classe importa il dataset, lo suddivide in training, validation e test set. Crea il modello di base che verrà
    poi passato a una grid search per selezionare i parametri più adatti. Il modello più performante ottenuto viene
    testato tramite un' operazione di Backtesting.
    I risultati ottenuti vengono rappresentati su un grafico che comprende gli ultimi quattro mesi del dataset
    (il training set). Il modello viene salvato in formato pickle nella cartella "models".
    
    Params:
     - save_as: Nome da assegnare al modello durante il salvataggio
     - model_name: nome dell' algoritmo cher si vuole utilizzare (Ridge o LGBM)

    """
    def __init__(self, save_as, model_name):

        models_dict = {"Ridge": Ridge(),
                       "LGBM": LGBMRegressor(),
                       }  # TODO: Aggiungere Facebook Prophet

        self.model_name = model_name  # nome modello
        self.algorithm = models_dict[model_name]  # algoritmo

        # Split the data in training, validation and test
        self.dataframe = self.import_data()
        self.training_set = self.dataframe.loc[:"2017-12-31 23:00:00"]  # tre anni
        self.validation_set = self.dataframe.loc["2018-01-01 00:00:00":"2018-08-31 23:00:00"]  # otto mesi
        self.test_set = self.dataframe.loc["2018-09-01 00:00:00":]  # quattro mese

        self.forecast_model = self.make_forecast()
        self.best_param = self.search_best_parameters()

        self.mae, self.predictions = self.make_backtest()

        print(self.mae, self.plot_result())

        # Save the model
        with open(f"../../models/{save_as}", "wb") as file:
            pickle.dump(self.forecast_model, file)

    def import_data(self):
        """ Importazione dataframe_for_prediction """
        py_path = re.sub(r"\\", "/", os.environ["PYTHONPATH"]).split(";")[2]
        dataframe = pd.read_csv(py_path + "/Forecasting_Repository/data/interim/DF_SeasonalDummies.csv",
                                index_col="time")
        dataframe.index = pd.to_datetime(dataframe.index)
        return dataframe

    def make_forecast(self):  # FIXME: implementare il Facebook prophet
        """ Crea il modello di base che verrà passato alla grid search """

        forecaster = ForecasterAutoreg(regressor=make_pipeline(StandardScaler(),
                                                               self.algorithm),
                                       lags=24  # verrà sostituito nella grid search
                                       )
        forecaster.fit(y=self.dataframe.loc[:"2018-08-31 23:00:00", 'total load actual'])
        return forecaster

    def search_best_parameters(self):
        """ Ricerca dei parametri migliori per il modello sfruttando il training e il validation set"""

        # Parametri da testare se il modello è una Ridge
        lags_grid = [1, 2, 3, 24, 25, 48, 49]
        param_grid = {'ridge__alpha': np.logspace(-3, 5, 10)}

        # Parametri da testare se il modello è un LGMB
        if self.model_name == "LGBM":
            lags_grid = [24, 25, 48, 49, 2190]  # lag 2190 = numero di ore in quattro mesi
            param_grid = {}

        results_grid = grid_search_forecaster(forecaster=self.forecast_model,

                                              # training + validation
                                              y=self.dataframe.loc[:"2018-08-31 23:00:00", 'total load actual'],

                                              # Tutte le variabili esogene #
                                              # 32134 = riga dataframe_for_prediction al 2018-08-31 23:00:00
                                              # 20: = tutte le variabili meteo + le dummies
                                              exog=self.dataframe.iloc[:32135, 21:],

                                              param_grid=param_grid,
                                              lags_grid=lags_grid,
                                              steps=24,
                                              metric='mean_absolute_error',
                                              refit=False,
                                              initial_train_size=len(self.training_set),
                                              fixed_train_size=False,
                                              return_best=True,
                                              verbose=False)
        return results_grid

    def make_backtest(self):
        """ Testa il modello ottenuto dalla grid search tramite il backtesting """
        mae_metric, predictions = backtesting_forecaster(forecaster=self.forecast_model,

                                                         # dataframe_for_prediction completo
                                                         y=self.dataframe["total load actual"],

                                                         # training + validation
                                                         initial_train_size=len(self.dataframe[:"2018-08-31 23:00:00"]),

                                                         # Tutte le variabili esogene #
                                                         exog=self.dataframe.iloc[:, 21:],

                                                         steps=24,
                                                         metric='mean_absolute_error',
                                                         interval=[10, 90],
                                                         refit=False,
                                                         verbose=True)
        # Fix dataframe_for_prediction con le previsioni
        predictions = predictions.assign(time=self.dataframe.index[len(self.dataframe) - len(predictions):])
        predictions.set_index("time", inplace=True)
        return mae_metric, predictions

    def plot_result(self):
        """ Plot dei risultati """
        fig, ax = plt.subplots(figsize=(40, 8))
        self.dataframe['total load actual'][min(self.predictions.index):max(self.predictions.index)].plot(linewidth=2,
                                                                                                          ax=ax)
        self.predictions["pred"].plot(linewidth=1,
                                      label='predictions',
                                      ax=ax).legend(loc='upper right')
        ax.set_title('Prediction vs real demand')
        ax.fill_between(
            self.predictions.index,
            self.predictions.iloc[:, 1],
            self.predictions.iloc[:, 2],
            alpha=0.3,
            color='grey',
            label='prediction interval'
        )
        return plt.show()


if __name__ == "__main__":
    # AR_ridge = BacktestPipeline(save_as="AR_model.pkl", model_name="Ridge")
    # LGBM_model = BacktestPipeline(save_as="LGBM_model.pkl", model_name="LGBM")
    # LGBM_model_noHex = BacktestPipeline(save_as="LGBM_model_noHex.pkl", model_name="LGBM")

    # Test con le variabili esogene più importanti a livello di correlazione
    AR_ridge = BacktestPipeline(save_as="AR_model_OnlyBestHex.pkl", model_name="Ridge")
    LGBM_model = BacktestPipeline(save_as="LGBM_model_OnlyBestHex.pkl", model_name="LGBM")

