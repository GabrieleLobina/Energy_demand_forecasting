"""
Per far girare il file aprire il terminale e incollare:
py -m streamlit run C:/Users/KY662AT/PycharmProjects/Energy_demand_forecastingT/Forecasting_Repository/reports/Streamlit_dashboard.py
"""

import streamlit as st
from Forecasting_Repository.src.models.predict_model import Predictor

# Data exploration
# ================================
from statsmodels.tsa._stl import STL

# Plots
# ================================
from streamlit import altair_chart
import altair as alt
import seaborn as sb
from matplotlib import pyplot as plt
from PIL import Image

# Data manipulation
# ================================
import pandas as pd
import numpy as np

st.set_option('deprecation.showPyplotGlobalUse', False)

# Importazione dataset
# ======================================================================================================================
df = pd.read_csv("C:/Users/KY662AT/PycharmProjects/Energy_demand_forecastingT/Forecasting_Repository/data/processed/DF_only_load.csv",
                 index_col="time")

df.index = pd.to_datetime(df.index)

this_week_h = df.tail(24).index.weekday*24
this_week_h = this_week_h[0]

# df.tail(24).index.weekday*24 <-- questo serve per poter visualizzare i valori della settimana corrente

# df.tail(24).index --> indice delle ultime 24 ore
# df.tail(24).index.weekday --> giorno della settimana delle ultime 24 ore
# df.tail(24).index.weekday * 24  --> x 24 ore: quindi se il giorno della settimana di ieri era 2 (martedì) allora
# prenderemo 24 x 2 quindi gli ultimi due giorni

# ======================================================================================================================
#                                                    Introduzione                                                      #
# ======================================================================================================================
"# Energy Demand Forecasting"

col1, col2, col3, col4 = st.columns(4)

col2.metric("This week mean", int(df["total load actual"].tail(this_week_h).mean()))
col3.metric("This week Standard deviation", int(df["total load actual"].tail(this_week_h).std()))
col1.metric("This week Min", int(df["total load actual"].tail(this_week_h).min()))
col4.metric("This week mean Max", int(df["total load actual"].tail(this_week_h).max()))

# ======================================================================================================================
#                                        Grafici andamento Total load actual:                                          #
# ======================================================================================================================

# Bottone #
# ======================================================================================================================
# Widget per decidere quanti giorni prevedere
step = int(st.number_input('Day ahead prediction',
                           step=1))

# LGBM (con variabili esogene)
# ======================================================================================================================
predictions_LGBM = Predictor(model_path=
                             "Forecasting_Repository/models/LGBM_model.pkl",
                             steps=step * 24, data=df)

pred = predictions_LGBM.predictions

# Aggiustamento indice previsioni
pred["new index"] = pd.date_range(df.tail(1).index[0],
                                  periods=(step * 24),  # + 1 perché sennò la prima data delle previsioni coincide con
                                  # l'ultima del dataset con i dati reali
                                  freq='H')

pred = pred.set_index(pred["new index"])
pred = pred.drop(columns="new index")

pred = pd.concat([df.tail(48), pred])  # ultime 8 ore di dati reali + le previsioni
# print(pred)

col1, col2 = st.columns(2)
col1.line_chart(pred[["total load actual", "pred"]])
col2.dataframe(pred["pred"].tail(step * 24))
# ======================================================================================================================
#                                                                                                                      #
# ======================================================================================================================


# # annuale
# # ======================================================================================================================
# df['year'] = df.index.year
# sb.boxplot(data=df, y='total load actual', x='year', ax=ax[1][1], color="#64b7e7")
# y_means = [y for y in df.groupby('year')["total load actual"].median()]
# sb.pointplot(data=df, y=y_means, x=df.year.unique(), color="#494949", ax=ax[1][1])
#
# st.pyplot(fig)

# ======================================================================================================================
#                                                      MODELLI                                                         #
# ======================================================================================================================

# Decidere se tenere la funzione di sotto --> in caso è da cambiare
# def adjust_predictions_for_plot(predicted_values):
#     """ Crea un nuovo dataframe con le previsioni e i valori reali utilizzabile in line_chart """
#
#     # Valori reali
#     real_values_df = pd.DataFrame(predicted_values.data["total load actual"])
#
#     pred_df = pd.DataFrame()
#
#     if len(predicted_values.predictions) >= len(real_values_df):
#         # Creazione nuovo indice con un numero di righe pari ai valori predetti
#         new_index = real_values_df.index.append(
#             predicted_values.predictions.iloc[len(real_values_df):].index)
#         # print(new_index)
#
#         new_col_total = []
#         for position, i in enumerate(
#                 new_index):  # aggiunge una nuova riga con Na dove i valori reali non sono disponibili
#             try:
#                 new_col_total.append(real_values_df["total load actual"][position])
#             except:
#                 new_col_total.append(None)
#
#         # Creazione dataframe completo
#         pred_df = pd.DataFrame({"new_total_load_actual": new_col_total,
#                                 "predictions": predicted_values.predictions["pred"]}, index=new_index)
#     else:
#         pred_df["Real Values"] = real_values_df.iloc[:len(predicted_values.predictions)]
#         pred_df["Predicted Values"] = predicted_values.predictions
#     return pred_df


# with column1LGBM.expander("Metriche"):
#     st.metric("MAE", predictions_LGBM.calculate_errors()[0])
#     st.metric("MAPE", predictions_LGBM.calculate_errors()[1])

#
#
# # ======================================================================================================================
# #                                                      ERRORI                                                          #
# # ======================================================================================================================
#
# # Importazione dataset con la distribuzione di MAPE e MAE nel tempo
# # ======================================================================================================================
# errori = pd.read_csv("Forecasting_Repository/data/interim/DF_errori_modelli.csv")
#
# errori = errori.drop(columns="Unnamed: 0")
#
# mape_cols = ["MAPE_AR_HEXOG", "MAPE_LGBM_HEXOG", "MAPE_AR_NonHEXOG", "MAPE_LGBM_NonHEXOG"]
# mae_cols = ["MAE_AR_HEXOG", "MAE_LGBM_HEXOG", "MAE_AR_NonHEXOG", "MAE_LGBM_NonHEXOG"]
#
# mape_df = errori.loc[:, mape_cols]
# mape_df = mape_df.rename(columns={"MAPE_AR_HEXOG": "AR HEXOG",
#                                   "MAPE_LGBM_HEXOG": "LGBM HEXOG",
#                                   "MAPE_AR_NonHEXOG": "AR Non HEXOG",
#                                   "MAPE_LGBM_NonHEXOG": "LGBM Non HEXOG"})
#
# mae_df = errori.loc[:, mae_cols]
# mae_df = mae_df.rename(columns={"MAE_AR_HEXOG": "AR HEXOG",
#                                 "MAE_LGBM_HEXOG": "LGBM HEXOG",
#                                 "MAE_AR_NonHEXOG": "AR Non HEXOG",
#                                 "MAE_LGBM_NonHEXOG": "LGBM Non HEXOG"})
#
#
# # Grafico MAE
# # ======================================================================================================================
# "### Distribuzione degli errori "
# st.write("I grafici sottostanti mostrano l'andamento degli errori di previsione (misurati con MAE e MAPE) in un perioso"
#          "di 4 mesi.")
#
# "##### Distribuzione MAE per giorni previsti"
# st.line_chart(mae_df)
#
#
# # Grafico MAPE
# # ======================================================================================================================
# "##### Distribuzione MAPE per giorni previsti"
# st.line_chart(mape_df)
#
# st.write("Come possiamo vedere dai due grafici, nel breve periodo, le performance predittive più alte le hanno i "
#          "modelli più *semplici* che non sfruttano variabili esogene (AR e LGBM)")
# st.write("Con l'aumentare dei giorni da prevedere i metodi non parametrici (LGBM con e senza variabili esogene) "
#          "risultano essere i migliori insieme al AR con variabili esogene")
# st.write("A farla da padrona in tutta la distribuzione però è l' LGBM senza variabili esogene, il quale mantiene sempre"
#          "dei bassi valori di MAE e MAPE (con una leggera perdita intorno al quarto mese)")
#
# # TODO: tradurre in inglese
