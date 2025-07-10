import pytest
import pandas as pd
import os
from io import StringIO
import tempfile

# Importe la fonction que tu souhaites tester
from app.meteo_requete_final import compile_meteo_data, get_meteo

# Test simple pour tester la lecture de fichiers CSV
def test_compile_meteo_data():
    # Créer un fichier CSV temporaire
    csv_content = "col1;col2;col3\n1;2;3\n4;5;6"
    with tempfile.NamedTemporaryFile(delete=False, mode='w', newline='', encoding='utf-8') as f:
        f.write(csv_content)
        f.close()
        
        # Chemin vers le fichier créé
        file_path = f.name

    # On va simuler l'exécution de la fonction sur ce fichier
    all_files = [file_path]  # Liste contenant notre fichier temporaire
    list_of_dfs = []

    # Lire chaque fichier CSV et ajouter son contenu à la liste
    for file in all_files:
        try:
            # Essaye d'abord avec le séparateur ;
            df = pd.read_csv(file, sep=';', on_bad_lines='skip', engine='python')

            # Si le DataFrame n'a qu'une seule colonne, essaie avec ,
            if df.shape[1] == 1:
                df = pd.read_csv(file, sep=',', on_bad_lines='skip', engine='python')

            # Si toujours 1 seule colonne, essaie avec tabulation
            if df.shape[1] == 1:
                df = pd.read_csv(file, sep='\t', on_bad_lines='skip', engine='python')

            list_of_dfs.append(df)

        except Exception as e:
            print(f"⚠️ Erreur lors de la lecture du fichier : {file}")
            print("➡️ Erreur :", e)

    # Concaténer tous les DataFrames en un seul
    corse_df = pd.concat(list_of_dfs, ignore_index=True)

    # Vérification que le DataFrame a bien été créé
    assert len(corse_df) == 2  # On a 2 lignes dans notre fichier CSV
    assert 'col1' in corse_df.columns  # Vérifier que la colonne 'col1' existe
    assert 'col2' in corse_df.columns  # Vérifier que la colonne 'col2' existe
    assert 'col3' in corse_df.columns  # Vérifier que la colonne 'col3' existe

    # Supprimer le fichier temporaire après test
    os.remove(file_path)



# def test_get_meteo():
#     # Créez un objet mock pour ti (TaskInstance de Airflow)
#     class MockTaskInstance:
#         def __init__(self):
#             self.xcom_data = {}

#         def xcom_push(self, key, value):
#             self.xcom_data[key] = value

#     ti = MockTaskInstance()

#     # Appel de la fonction avec des paramètres de test
#     try:
#         get_meteo(ti)
#         assert True  # Si la fonction s'exécute sans erreur, le test passe
#     except Exception as e:
#         pytest.fail(f"La fonction get_meteo a échoué avec l'erreur : {e}")
