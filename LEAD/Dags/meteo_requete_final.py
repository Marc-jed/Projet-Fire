import pandas as pd
import requests
import time
import tqdm
import os
import io
import boto3
from sqlalchemy import create_engine
from airflow import DAG
from airflow.hooks.base import BaseHook
from airflow.operators.python import PythonOperator
from airflow.hooks.S3_hook import S3Hook
from airflow.models import Variable
from datetime import date, timedelta
import datetime
import glob

default_args = {
    "owner": "airflow",
    "start_date": datetime.datetime(2024, 1, 1),
    "retries": 1
}
all_paths = []
# appelle du département par code postal
def get_meteo(ti, **kwargs):
    
    api = Variable.get("meteoapi")
    # 'eyJ4NXQiOiJZV0kxTTJZNE1qWTNOemsyTkRZeU5XTTRPV014TXpjek1UVmhNbU14T1RSa09ETXlOVEE0Tnc9PSIsImtpZCI6ImdhdGV3YXlfY2VydGlmaWNhdGVfYWxpYXMiLCJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJnZGxlZHMzMUBjYXJib24uc3VwZXIiLCJhcHBsaWNhdGlvbiI6eyJvd25lciI6ImdkbGVkczMxIiwidGllclF1b3RhVHlwZSI6bnVsbCwidGllciI6IlVubGltaXRlZCIsIm5hbWUiOiJEZWZhdWx0QXBwbGljYXRpb24iLCJpZCI6MjgxOTcsInV1aWQiOiJjMDA0ZmQ1NC0xMTY3LTQ3MTEtOWQ3MC04M2ExZWI0YmI0MGYifSwiaXNzIjoiaHR0cHM6XC9cL3BvcnRhaWwtYXBpLm1ldGVvZnJhbmNlLmZyOjQ0M1wvb2F1dGgyXC90b2tlbiIsInRpZXJJbmZvIjp7IjUwUGVyTWluIjp7InRpZXJRdW90YVR5cGUiOiJyZXF1ZXN0Q291bnQiLCJncmFwaFFMTWF4Q29tcGxleGl0eSI6MCwiZ3JhcGhRTE1heERlcHRoIjowLCJzdG9wT25RdW90YVJlYWNoIjp0cnVlLCJzcGlrZUFycmVzdExpbWl0IjowLCJzcGlrZUFycmVzdFVuaXQiOiJzZWMifX0sImtleXR5cGUiOiJQUk9EVUNUSU9OIiwic3Vic2NyaWJlZEFQSXMiOlt7InN1YnNjcmliZXJUZW5hbnREb21haW4iOiJjYXJib24uc3VwZXIiLCJuYW1lIjoiRG9ubmVlc1B1YmxpcXVlc0NsaW1hdG9sb2dpZSIsImNvbnRleHQiOiJcL3B1YmxpY1wvRFBDbGltXC92MSIsInB1Ymxpc2hlciI6ImFkbWluX21mIiwidmVyc2lvbiI6InYxIiwic3Vic2NyaXB0aW9uVGllciI6IjUwUGVyTWluIn1dLCJleHAiOjE3NTIzOTM0MzIsInRva2VuX3R5cGUiOiJhcGlLZXkiLCJpYXQiOjE3NTE3ODg2MzIsImp0aSI6IjJhODY3ODM1LTM4ZGYtNDYzZS04NjllLWMwM2YzZTNjMTk4NyJ9.t6Vl9r3L9smJ1XfPqRGGa_kxvLL-q-WTszgAgXxAWdP5M6TJaA_pPUeDT-CVtDhipxm-3HfHDV0t2DpRyT35F_fVLfq8sZsS1UzBIYR1fA9lgKjQAGHedsOwflva-nqNJTGLtEwHGbxjQjH6fw5uBxyvAydAbkg8NzPiXkREQO0ur6hiA7tWP1QhfSLLb6NwhF6ec4zdQqkfjH9jtus9tq78baEIct7z84RqAZoq7zkiVaGo22l1A0KSLb6Krf0pBE-AfMm_A6OLz1Z8KG0CTtiVpHToKukzIun1uCpynVKC1T8gN1rZZfhL_keOVbuPOaUn-1hd6TBwZCAPLth6pA=='/n
    dep = '20'
    url = 'https://public-api.meteofrance.fr/public/DPClim/v1/liste-stations/quotidienne'

    params = {
        'id-departement': dep,
        'parametre': 'temperature'
    }

    headers = {
        'accept': '*/*',
        'apikey': api
    }

    corse = requests.get(url, headers=headers, params=params)

    print('erreur corse', corse.status_code)
    corse.json()

    # on applique un mask pour ne prendre que les stations ouvertes
    corse_df = pd.DataFrame(corse.json())
    corse_df['posteOuvert'] = corse_df['posteOuvert'].astype(bool)
    mask = corse_df['posteOuvert'] == True
    corse_df = corse_df[mask]

    # appelle les information par station sur plusieurs années
    # all_data = pd.DataFrame()

    id = corse_df['id']

    for i in id:
        # pour récupérer les données météo de la veille, on peut utiliser la date du jour moins un jour
        # d=datetime.date.today()
        # d=d-timedelta(days=1)
        # date_debut = f'{d}T00:00:00Z'
        # date_fin = f'{d}T23:59:59Z'

        for années in range(2025, 2026):
            date_debut = f'{années}-01-01T00:00:00Z'
            date_fin = f'{années}-06-05T23:59:59Z'

            url = "https://public-api.meteofrance.fr/public/DPClim/v1/commande-station/quotidienne"
            params = {
                "id-station": i,
                "date-deb-periode": date_debut,
                "date-fin-periode": date_fin
            }
            headers = {
                "accept": "*/*",
                "apikey": api
            }

            corse1 = requests.get(url, headers=headers, params=params)
            print('erreur corse1', corse1.status_code)
            corse1_json = corse1.json()

            # Wrap in list if it's a dict of scalars
            corse1_df = pd.DataFrame(corse1_json).reset_index()

            name = dep + '_' + corse_df.loc[corse_df['id'] == i, 'nom'].values[0] + '_' + str(années)
            # name = dep + '_' + corse_df.loc[corse_df['id'] == i, 'nom'].values[0] + '_' + str(d)
        
            # Extract the 'return' value if the response is a dict
            id_cmde = corse1_df.iloc[0,1]
        

            url = "https://public-api.meteofrance.fr/public/DPClim/v1/commande/fichier"
            params = {
                "id-cmde": id_cmde
            }
            headers = {
                "accept": "*/*",
                "apikey": api
            }
            corse2 = requests.get(url, headers=headers, params=params)
            
            # print('erreur corse2', corse2.status_code)
            #print(corse2.text)
            
            # output_file = name +'.csv'
            # pd.DataFrame(corse2).to_csv(output_file, index=False)

            nom_station = corse_df.loc[corse_df['id'] == i, 'nom'].values[0].replace(' ', '_')
            path = f"tmp/{dep}_{nom_station}_{années}.csv"
            # path = f"tmp/{dep}/{dep}_{nom_station}_{d}.csv"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # Enregistrement du fichier
            with open(path, 'w', encoding='utf-8') as f:
                f.write(corse2.text)
            ti.xcom_push(key='meteo_paths', value=all_paths)
            all_paths.append(path)

            time.sleep(60 / 25)  # 60 seconds divided by 25 requests
    # output_file = dep + '.csv'
    # all_data.to_csv(output_file, index=False)

def upload_csv_to_s3(ti, **kwargs):
    paths = ti.xcom_pull(task_ids='get_meteo', key='meteo_paths')
    bucket = Variable.get("S3BucketName")
    folder = Variable.get("S3FolderName")
    s3 = S3Hook(aws_conn_id="aws_default")
    for path in paths:
        filename = os.path.basename(path)
        s3.load_file(filename=path, key=f"{folder}/{filename}", bucket_name=bucket, replace=True)


def compile_meteo_data(ti, **kwargs):
    # Chemin vers les fichiers CSV
    s3 = S3Hook(aws_conn_id="aws_default")
    bucket_name = Variable.get("S3BucketName")
    folder_name = Variable.get("S3FolderName")
    # Lire les fichiers CSV depuis S3
    all_files = s3.list_keys(bucket_name=bucket_name, prefix=folder_name)
    all_files = [f"s3://{bucket_name}/{file}" for file in all_files if file.endswith('.csv')]
     
    # all_files = glob.glob(path + "/*.csv")

    # Liste pour stocker les DataFrames
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
    # Écrire le DataFrame combiné dans un nouveau fichier CSV
    csv_path = "/tmp/compile-meteo-corse.csv"
    corse_df.to_csv(csv_path, index=False)
    # Push le chemin du fichier vers XCom
    ti.xcom_push(key="meteo-compile_csv_path", value=csv_path)

def upload_compile_csv_to_s3(ti, **kwargs):
    path = ti.xcom_pull(task_ids='compile_meteo_data', key='meteo-compile_csv_path')
    bucket = Variable.get("S3BucketName")
    compile_folder = Variable.get("S3Compile")
    s3 = S3Hook(aws_conn_id="aws_default")
    filename = os.path.basename(path)
    s3.load_file(filename=path, key=f"{compile_folder}/{filename}", bucket_name=bucket, replace=True)

 
with DAG(
    dag_id="meteo_requete_final",
    default_args=default_args,
    schedule_interval="@daily",
    catchup=False,
    description="Flux meteo de la corse"
) as dag:

    fetch_weather = PythonOperator(
        task_id="get_meteo",
        python_callable=get_meteo,
    )

    upload_csv = PythonOperator(
        task_id="upload_csv_to_s3",
        python_callable=upload_csv_to_s3
    )
    compile_meteo = PythonOperator(
        task_id="compile_meteo_data",
        python_callable=compile_meteo_data
    )
    upload_compile_csv = PythonOperator(
        task_id="upload_compile_csv_to_s3",
        python_callable=upload_compile_csv_to_s3
    )
fetch_weather >> upload_csv >> compile_meteo >> upload_compile_csv