import pytest
import pandas as pd

@pytest.fixture
def data_insee():
    # Charger les données depuis l'URL JSON
    df = pd.read_json('https://fireprojectbislead.s3.us-east-1.amazonaws.com/dataset/corse_insee.json', orient='records')
    return df

@pytest.fixture
def data_feu():
    df_feu = pd.read_csv('https://fireprojectbislead.s3.us-east-1.amazonaws.com/dataset/historique_incendies_avec_coordonnees.csv', sep=';', encoding='utf-8')   
    return df_feu