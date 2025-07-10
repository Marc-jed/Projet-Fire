import pandas as pd
import pytest



def test_load_data():

    df = pd.read_json('https://fireprojectbislead.s3.us-east-1.amazonaws.com/dataset/corse_insee_toto.json', orient='records')
    assert not df.empty, "df est vide" 

    