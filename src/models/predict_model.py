# TODO: aggiungere salvataggio risultati previsioni dato che questa classe servirà per la pipeline di re-train
import pickle

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error


class Predictor(object):
    def __init__(self, model_path, steps, data, hexog=True):
        self.model_path, self.steps = model_path, steps

        self.data = data
        self.hexog = hexog

        # Modificare il path
        dataframe_for_prediction = pd.read_csv(
            "C:/Users/KY662AT/PycharmProjects/Energy_demand_forecastingT/Forecasting_Repository/data/interim/DF_for_predictions.csv"
        )
        self.dataframe_for_prediction = dataframe_for_prediction.iloc[338952:]

        self.predictions = self.make_predictions()
        self.mae, self.mape = self.calculate_errors()

    def make_predictions(self):
        """ Utilizza il modello passato dall'esterno per fare le previsioni """
        with open(self.model_path, "rb") as model:
            loaded_model = pickle.load(model)
            if self.hexog:  # con variabili esogene
                predictions = pd.DataFrame(loaded_model.predict(self.steps, exog=self.dataframe_for_prediction))
            else:  # senza variabili esogene
                predictions = pd.DataFrame(loaded_model.predict(self.steps, self.data["total load actual"]))
                predictions.index = pd.date_range(self.data.tail(1).index[0], periods=self.steps, freq='H')
        return predictions

    def plot_results(self):
        """ Plot dei risultati """
        fig, ax = plt.subplots(figsize=(40, 8))

        self.data['total load actual'].plot()
        self.predictions["pred"].plot(linewidth=2,
                                      label='predictions',
                                      ax=ax).legend(loc='upper right')
        ax.set_title('Prediction vs real demand')
        return plt.show()

    def calculate_errors(self):  # FIXME: deve calcolare solo se abbiamo i dati veri
        mape = mean_absolute_percentage_error(y_true=self.data['total load actual'][:self.steps],
                                              y_pred=self.predictions["pred"])

        mae = mean_absolute_error(y_true=self.data['total load actual'][:self.steps],
                                  y_pred=self.predictions["pred"])
        return mae, mape


if __name__ == "__main__":
    external_data = pd.read_csv("C:/Users/KY662AT/PycharmProjects/Energy_demand_forecastingT/Forecasting_Repository/data/processed"
                  "/DF_only_load.csv", index_col="time")
    data = external_data.set_index(external_data.index)
    data = data.interpolate("linear")

    model = Predictor(
        "C:/Users/KY662AT/PycharmProjects/Energy_demand_forecastingT/Forecasting_Repository/models/LGBM_model_noHex.pkl",
        steps=10,
        data=data, hexog=False)

    print(model.predictions)
    # model.plot_results()

    # print(model.mae)
    # print(model.mape)



