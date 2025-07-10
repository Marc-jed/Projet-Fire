import pandas as pd
import pytest


@pytest.fixture
def test_load_data(data_insee):
    assert data_insee().empty



  