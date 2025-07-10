import pandas as pd
import pytest

def test_load_data(data_insee):
    assert not data_insee.empty
    assert data_insee.shape[1] == 3
    assert data_insee.shape[0] == 417



def test_cleaner_data(data_insee):
    assert data_insee.isnull().sum().sum() == 0
   