import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def data_insee():
    return df = pd.read_json('https://fireprojectbislead.s3.us-east-1.amazonaws.com/dataset/corse_insee.json', orient='records')
    

