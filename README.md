Energy_demand_forecasting
==============================

Energy demand forecasting of spanish data for thesis project

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>







----

# Datasets:

## 2) *ninja_weather_country_ES_merra-2_land_area_weighted.csv*

### Renewables.ninja Weather (hourly data, 1980-2019) - Spain

- Version: 1.3
- License: https://science.nasa.gov/earth-science/earth-science-data/data-information-policy
- Reference: https://renewables.ninja and https://doi.org/10.1175/JCLI-D-16-0758.1

#

Units: time in UTC, other columns are average weather variables, land area weighted.

- **precipitation** in mm/hour (PRECTOTLAND in MERRA-2).
- **temperature** at 2 metres above ground in degrees C (T2M in MERRA-2).
- **irradiance** at ground level (_surface) and top of atmosphere (_toa) in W/m^2 (SWGDN and SWTDN in MERRA-2).
- **snowfall** in mm/hour (PRECSNOLAND in MERRA-2).
- **snow_mass** in kg/m^2 (SNOMAS in MERRA-2).
- **cloud_cover** as a fraction [0,1] (CLDTOT in MERRA-2).
- **air_density** at ground level in kg/m^3 (RHOA in MERRA-2).

## Tips:

io ho anche i dati meteo successivi ai dati energetici quindi si possono fare delle previsioni
future più ampie utilizzando anche quelli

## Domande:

- ...

# Cose da fare:

1. [x] Fare altri modelli in un nuovo file jupyter
2. [x] Trovare un modo per analizzare i residui dei modelli
3. [ ] Fare feature extraction --> guardare link mandati da Simone
4. [x] Sistemare la pipeline in modo che accetti parametri di dataset, modello da utilizzare, parametri da testare
5. [x] Creare la pipeline di forecasting che richiami il modello pickle

6. [ ] Fare pipeline di re-train
    - Gestire il dataframe con le previsioni meteo in modo che le osservazioni partano da dicembre 2018 e si
      aggiornino tramite il download dei dati dalla pagina dalla quale ho scaricato il dataframe –> se fa.
    - Creare un codice che sistemi risultati delle previsioni in base alle date, le quali queste ultime saranno
      prese dal dataset di cui sopra in modo da avere sempre un orizzonte temporale in crescita e che dipende dalla
      dimensione del dataset dal quale si possono fare appunto le previsioni.
7. [x] Fare interfaccia grafica
8. [x] Sistemare codice e markdown del file jupyter 2.1
9. [ ] Sistemare README.md
   

   