import pytest
import pandas as pd

@pytest.fixture
def data_insee():
    # Charger les données depuis l'URL JSON
    df = pd.read_json('https://fireprojectbislead.s3.us-east-1.amazonaws.com/dataset/corse_insee.json', orient='records')
    return df