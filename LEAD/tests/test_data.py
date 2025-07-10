import pandas as pd
import pytest



def test_load_data():

    df = pd.read_json('https://fireprojectbislead.s3.us-east-1.amazonaws.com/dataset/corse_insee.json', orient='records')
    assert not df.empty 
    assert df.shape[1] == 3
    assert df.shape[0] == 417



def test_cleaner_data():

    df = pd.read_json('https://fireprojectbislead.s3.us-east-1.amazonaws.com/dataset/corse_insee.json', orient='records')
    assert df.isnull().sum().sum() == 0