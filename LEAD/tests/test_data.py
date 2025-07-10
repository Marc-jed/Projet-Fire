import pandas as pd
import pytest

def test_load_data(data_insee):
    assert not data_insee.empty
    assert data_insee.shape[1] == 3
    assert data_insee.shape[0] == 417

def test_cleaner_data(data_insee):
    assert data_insee.isnull().sum().sum() == 0

def test_load_data_feu(data_feu):
    assert not data_feu.empty
    assert data_feu.shape[1] == 19

def test_cleaner_data_feu(data_feu):
    assert data_feu.isnull().sum()[data_feu.columns[0]]==0



