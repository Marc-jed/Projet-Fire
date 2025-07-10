import sys
import os
import pytest
import pandas as pd

# Pour pouvoir importer depuis Dags/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Dags')))
from Dags.meteo_requete_final import compile_meteo_data

class DummyTI:
    """Objet simulé pour imiter le behavior de ti (TaskInstance) dans Airflow."""
    def __init__(self, paths):
        self.paths = paths
        self.pushed = {}

    def xcom_pull(self, task_ids, key):
        assert task_ids == 'get_meteo' and key == 'meteo_paths'
        return self.paths

    def xcom_push(self, key, value):
        self.pushed[key] = value

def create_csv_file(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content, encoding='utf-8')
    return str(path)

def test_compile_meteo_data(tmp_path):
    # Crée 2 fichiers CSV temporaires
    csv1 = "A;B;C\n1;2;3\n4;5;6"
    csv2 = "A;B;C\n7;8;9\n10;11;12"
    f1 = create_csv_file(tmp_path, "file1.csv", csv1)
    f2 = create_csv_file(tmp_path, "file2.csv", csv2)

    ti = DummyTI(paths=[f1, f2])

    compile_meteo_data(ti)

    # Vérifie le chemin du fichier de sortie
    out_key = "meteo-compile_csv_path"
    assert out_key in ti.pushed
    out_path = ti.pushed[out_key]
    assert os.path.isfile(out_path)

    # Vérifie le contenu
    df = pd.read_csv(out_path)
    assert len(df) == 4
    assert set(['A', 'B', 'C']).issubset(df.columns)

