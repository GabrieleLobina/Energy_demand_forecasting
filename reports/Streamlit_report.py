"""
Per far girare il file aprire il terminale e incollare:
py -m streamlit run C:/Users/KY662AT/PycharmProjects/Energy_demand_forecastingT/Forecasting_Repository/reports/Streamlit_report.py
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
df = pd.read_csv(
    "Forecasting_Repository/data/processed/processed_data.csv",
    index_col="time")

df.index = pd.to_datetime(df.index)


# ======================================================================================================================
#                                                    Introduzione                                                      #
# ======================================================================================================================
"# Energy Demand Forecasting"
st.write("Il progetto ha il fine di prevedere la domanda energetica di uno stato.")
st.write("In questo caso sono stati utilizzati i dati della produzione e della domanda energetica della Spagna, dal "
         "primo Gennaio 2015 al 31 Dicembre 2018, con una precisione di un' ora.")
st.write("E' stato utilizzato anche un dataset relativo alle condizioni meteorologiche della medesima nazione "
         "(precisione di un ora) in modo da aggiungere informazioni utili alle previsioni.")


# Dataframe
# ======================================================================================================================
"#### Dataframe utilizzato per le analisi"
st.dataframe(df)


# ======================================================================================================================
#                                                Analisi descrittiva                                                   #
# ======================================================================================================================
"## Analisi domanda energetica"
st.write("Si è partiti con una prima analisi della distribuzione della domanda energetica in modo da comprenderne le "
         "caratteristiche, e quindi quali fossero i modelli più adatti da utilizzare per fare delle previsioni su di "
         "essa.")


# Valori summary
# ======================================================================================================================

col1, col2, col3, col4 = st.columns(4)
col2.metric("Mean", int(df["total load actual"].mean()))
col3.metric("Standard deviation", int(df["total load actual"].std()))
col1.metric("Min", int(df["total load actual"].min()))
col4.metric("Max", int(df["total load actual"].max()))


# ======================================================================================================================
#                                        Grafici andamento Total load actual:                                          #
# ======================================================================================================================

# Bottone #
# ======================================================================================================================
# Widget per fare un grafico con il numero di giorni che specifichiamo
if st.checkbox('Show demand'):
    number = int(st.number_input('Numero giorni da visualizzare',
                                 min_value=1,
                                 max_value=1461,
                                 step=1))
    st.line_chart(df["total load actual"][:number * 24])

fig, ax = plt.subplots(2, 2, figsize=(20, 12))
plt.suptitle("DISTRIBUZIONE DOMANDA ENERGETICA")


# giornaliero
# ======================================================================================================================
df['hour'] = df.index.hour + 1
sb.boxplot(data=df, y='total load actual', x='hour', ax=ax[0][0], color="#64b7e7")
h_means = [h for h in df.groupby('hour')["total load actual"].median()]
sb.pointplot(data=df, y=h_means, x=df.hour.unique(), color="#494949", ax=ax[0][0])


# settimanale
# ======================================================================================================================
df['week_day'] = df.index.day_of_week + 1
sb.boxplot(data=df, y='total load actual', x='week_day', ax=ax[0][1], color="#64b7e7")
w_means = [w for w in df.groupby('week_day')["total load actual"].median()]
sb.pointplot(data=df, y=w_means, x=np.sort(df["week_day"].unique()), color="#494949", ax=ax[0][1])


# mensile
# ======================================================================================================================
df['month'] = df.index.month
sb.boxplot(data=df, y='total load actual', x='month', ax=ax[1][0], color="#64b7e7")
m_means = [m for m in df.groupby('month')["total load actual"].median()]
sb.pointplot(data=df, y=m_means, x=df.month.unique(), color="#494949", ax=ax[1][0])


# annuale
# ======================================================================================================================
df['year'] = df.index.year
sb.boxplot(data=df, y='total load actual', x='year', ax=ax[1][1], color="#64b7e7")
y_means = [y for y in df.groupby('year')["total load actual"].median()]
sb.pointplot(data=df, y=y_means, x=df.year.unique(), color="#494949", ax=ax[1][1])

st.pyplot(fig)


# ======================================================================================================================
#                                                   Test sui dati:                                                     #
# ======================================================================================================================

# Descrizione #
# ======================================================================================================================
"### Test specifici per serie storiche"


# Augmented Dickey-Fuller test (Stazionarietà)
# ======================================================================================================================
with st.expander("ADF"):
    col1, col2 = st.columns(2)
    col1.metric("ADF Test Statistic: ", -21.42)
    col2.metric("P-Value: ", 0.0)
    st.write("La variabile relativa alla domanda energetica si può considerare stazionaria")


# STL Decomposition (Stazionarietà - Stagionalità - Rumore)
# ======================================================================================================================
with st.expander("STL Decomposition (Stazionarietà - Stagionalità - Rumore)"):
    col1, col2 = st.columns(2)
    image = Image.open(
        'Forecasting_Repository/src/visualization/STL_4anni_in_giorni.png')
    col1.image(image, caption='Precisione annuale')

    image = Image.open(
        'Forecasting_Repository/src/visualization/STL_3mesi_in_settimane.png')
    col2.image(image, caption='Precisione settimanale')

    image = Image.open(
        'Forecasting_Repository/src/visualization/STL_1mese_in_giorni.png')
    col1.image(image, caption='Precisione giornaliera')


# PACF e ACF
# ======================================================================================================================
with st.expander("PACF e ACF"):
    image = Image.open(
        'Forecasting_Repository/src/visualization/PACF_ACF.png')
    st.image(image, caption='PACF e ACF')


# ======================================================================================================================
#                                                      MODELLI                                                         #
# ======================================================================================================================

def adjust_predictions_for_plot(predicted_values):
    """ Crea un nuovo dataframe con le previsioni e i valori reali utilizzabile in line_chart """

    # Valori reali dal 2018-09-01 00:00:00 (riga 32135) al 2018-12-31 23:00:00 (dati utilizzati nel test set)
    real_values_df = pd.DataFrame(predicted_values.data["total load actual"][32135:])

    pred_df = pd.DataFrame()

    if len(predicted_values.predictions) >= len(real_values_df):
        # Creazione nuovo indice con un numero di righe pari ai valori predetti
        new_index = real_values_df.index.append(
            predicted_values.predictions.iloc[len(real_values_df):].index)
        # print(new_index)

        new_col_total = []
        for position, i in enumerate(
                new_index):  # aggiunge una nuova riga con Na dove i valori reali non sono disponibili
            try:
                new_col_total.append(real_values_df["total load actual"][position])
            except:
                new_col_total.append(None)

        # Creazione dataframe completo
        pred_df = pd.DataFrame({"new_total_load_actual": new_col_total,
                                "predictions": predicted_values.predictions["pred"]}, index=new_index)
    else:
        pred_df["Real Values"] = real_values_df.iloc[:len(predicted_values.predictions)]
        pred_df["Predicted Values"] = predicted_values.predictions
    return pred_df

"## Modelli"
st.write("I modelli utilizzati per prevedere l'andamento della domanda energetica sono:")
st.write("- Auto-regressive (con e senza variabili esogene)")
st.write("- Light Gradient Boosting Machine (con e senza variabili esogene)")
st.write("Le variabili esogene comprendono il giorno, il mese, l'ora, le festività e alcune variabili meteorologiche")

# Contatore per decidere il numero di giorni da prevedere
# ======================================================================================================================
step = st.number_input('Inserisci il numero di giorni da prevedere', min_value=1, step=1, max_value=487)


# AR con variabili esogene #
# ======================================================================================================================
"#### Autoregressive model con variabili esogene (sinistra) e senza (destra)"

column1AR, column2AR = st.columns(2)

predictions_AR = Predictor(model_path=
                           "Forecasting_Repository/models/AR_model.pkl",
                           steps=step * 24)

pred = adjust_predictions_for_plot(predictions_AR)
column1AR.line_chart(pred)
with column1AR.expander("Metriche"):
    st.metric("MAE (hexog)", predictions_AR.calculate_errors()[0])
    st.metric("MAPE (hexog)", predictions_AR.calculate_errors()[1])


# AR senza variabili esogene
# ======================================================================================================================
predictions_AR_NonHexog = Predictor(model_path=
                                    "Forecasting_Repository/models/AR_model_NonHex.pkl",
                                    steps=step * 24, hexog=False)

pred = adjust_predictions_for_plot(predictions_AR_NonHexog)
column2AR.line_chart(pred)

with column2AR.expander("Metriche"):
    st.metric("MAE (no hexog)", predictions_AR_NonHexog.calculate_errors()[0])
    st.metric("MAPE (no hexog)", predictions_AR_NonHexog.calculate_errors()[1])


# LGBM (con variabili esogene)
# ======================================================================================================================
"#### LGBM con variabili esogene (sinistra) e senza (destra)"

column1LGBM, column2LGBM = st.columns(2)

predictions_LGBM = Predictor(model_path=
                             "Forecasting_Repository/models/LGBM_model.pkl",
                             steps=step * 24)

pred = adjust_predictions_for_plot(predictions_LGBM)
column1LGBM.line_chart(pred)
with column1LGBM.expander("Metriche"):
    st.metric("MAE", predictions_LGBM.calculate_errors()[0])
    st.metric("MAPE", predictions_LGBM.calculate_errors()[1])

# LGBM (no hexog)
# ======================================================================================================================
predictions_LGBM_NonHexog = Predictor(model_path=
                                      "Forecasting_Repository/models/LGBM_model_noHex.pkl",
                                      steps=step * 24, hexog=False)

pred = adjust_predictions_for_plot(predictions_LGBM_NonHexog)
column2LGBM.line_chart(pred)
with column2LGBM.expander("Metriche"):
    st.metric("MAE", predictions_LGBM_NonHexog.calculate_errors()[0])
    st.metric("MAPE", predictions_LGBM_NonHexog.calculate_errors()[1])


# ======================================================================================================================
#                                                      ERRORI                                                          # 
# ======================================================================================================================

# Importazione dataset con la distribuzione di MAPE e MAE nel tempo
# ======================================================================================================================
errori = pd.read_csv("Forecasting_Repository/data/interim/DF_errori_modelli.csv")

errori = errori.drop(columns="Unnamed: 0")

mape_cols = ["MAPE_AR_HEXOG", "MAPE_LGBM_HEXOG", "MAPE_AR_NonHEXOG", "MAPE_LGBM_NonHEXOG"]
mae_cols = ["MAE_AR_HEXOG", "MAE_LGBM_HEXOG", "MAE_AR_NonHEXOG", "MAE_LGBM_NonHEXOG"]

mape_df = errori.loc[:, mape_cols]
mape_df = mape_df.rename(columns={"MAPE_AR_HEXOG": "AR HEXOG",
                                  "MAPE_LGBM_HEXOG": "LGBM HEXOG",
                                  "MAPE_AR_NonHEXOG": "AR Non HEXOG",
                                  "MAPE_LGBM_NonHEXOG": "LGBM Non HEXOG"})

mae_df = errori.loc[:, mae_cols]
mae_df = mae_df.rename(columns={"MAE_AR_HEXOG": "AR HEXOG",
                                "MAE_LGBM_HEXOG": "LGBM HEXOG",
                                "MAE_AR_NonHEXOG": "AR Non HEXOG",
                                "MAE_LGBM_NonHEXOG": "LGBM Non HEXOG"})


# Grafico MAE
# ======================================================================================================================
"### Distribuzione degli errori "
st.write("I grafici sottostanti mostrano l'andamento degli errori di previsione (misurati con MAE e MAPE) in un perioso"
         "di 4 mesi.")

"##### Distribuzione MAE per giorni previsti"
st.line_chart(mae_df)


# Grafico MAPE
# ======================================================================================================================
"##### Distribuzione MAPE per giorni previsti"
st.line_chart(mape_df)

st.write("Come possiamo vedere dai due grafici, nel breve periodo, le performance predittive più alte le hanno i "
         "modelli più *semplici* che non sfruttano variabili esogene (AR e LGBM)")
st.write("Con l'aumentare dei giorni da prevedere i metodi non parametrici (LGBM con e senza variabili esogene) "
         "risultano essere i migliori insieme al AR con variabili esogene")
st.write("A farla da padrona in tutta la distribuzione però è l' LGBM senza variabili esogene, il quale mantiene sempre"
         "dei bassi valori di MAE e MAPE (con una leggera perdita intorno al quarto mese)")

# TODO: tradurre in inglese
