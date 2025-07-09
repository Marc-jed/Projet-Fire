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
import numpy as np

default_args = {
    "owner": "airflow",
    "start_date": datetime.datetime(2024, 1, 1),
    "retries": 1
}

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
    # corse.json()

    # on applique un mask pour ne prendre que les stations ouvertes
    corse_df = pd.DataFrame(corse.json())
    corse_df['posteOuvert'] = corse_df['posteOuvert'].astype(bool)
    mask = corse_df['posteOuvert'] == True
    corse_df = corse_df[mask]

    # appelle les information par station sur plusieurs années
    # all_data = pd.DataFrame()

    id = corse_df['id']
    all_paths = []

    for i in id:
        # pour récupérer les données météo de la veille, on peut utiliser la date du jour moins un jour
        # d=datetime.date.today()
        # d=d-timedelta(days=1)
        # date_debut = f'{d}T00:00:00Z'
        # date_fin = f'{d}T23:59:59Z'

        for années in range(2025, 2026):
            date_debut = f'{années}-06-03T00:00:00Z'
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
            time.sleep(60 / 25)  # 60 seconds divided by 25 requests
            all_paths.append(path)
    ti.xcom_push(key='meteo_paths', value=all_paths)
            

            
    # output_file = dep + '.csv'
    # all_data.to_csv(output_file, index=False)

# def upload_csv_to_s3(ti, **kwargs):
#     paths = ti.xcom_pull(task_ids='get_meteo', key='meteo_paths')
#     bucket = Variable.get("S3BucketName")
#     folder = Variable.get("S3FolderName")
#     s3 = S3Hook(aws_conn_id="aws_default")
#     for path in paths:
#         filename = os.path.basename(path)
#         s3.load_file(filename=path, key=f"{folder}/{filename}", bucket_name=bucket, replace=True)


def compile_meteo_data(ti, **kwargs):
    # # Chemin vers les fichiers CSV
    # s3 = S3Hook(aws_conn_id="aws_default")
    # bucket_name = Variable.get("S3BucketName")
    # folder_name = Variable.get("S3FolderName")
    # # Lire les fichiers CSV depuis S3
    # all_files = s3.list_keys(bucket_name=bucket_name, prefix=folder_name)
    # all_files = [f"s3://{bucket_name}/{file}" for file in all_files if file.endswith('.csv')]
     
    # all_files = glob.glob(path + "/*.csv")
    all_files = ti.xcom_pull(task_ids='get_meteo', key='meteo_paths')

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


def cleaner_data(ti,**kwargs):
    path = ti.xcom_pull(task_ids='compile_meteo_data', key='meteo-compile_csv_path')
    # Chemin vers les fichiers CSV
    s3 = S3Hook(aws_conn_id="aws_default")
    bucket_name = Variable.get("S3BucketName")
    compile_name = Variable.get("S3Compile")
    df = pd.read_csv(path)
    df['DATE'] = pd.to_datetime(df['DATE'], format='%Y%m%d')
    # Traitements de colonnes innutiles
    drop_cols = list(set([
        'PMERM','PMERMIN','QPMERMIN','FF2M','QFF2M','FXI2','QFXI2','DXI2','QDXI2','HXI2','QHXI2','DXI3S','QDXI3S','DHUMEC','QDHUMEC','INST','QINST','GLOT','QGLOT','DIFT','QDIFT','DIRT','QDIRT','SIGMA','QSIGMA','INFRART','QINFRART','UV_INDICEX','QUV_INDICEX','NB300','QNB300','BA300','QBA300','NEIG','QNEIG','BROU','QBROU','GRESIL','GRELE','QGRELE','ROSEE','QROSEE','VERGLAS','QVERGLAS','SOLNEIGE','QSOLNEIGE','GELEE','QGELEE','FUMEE','QFUMEE','UV','QUV','TMERMAX','QTMERMAX','TMERMIN','QTMERMIN','HNEIGEF','QHNEIGEF','NEIGETOTX','QNEIGETOTX','NEIGETOT06','QNEIGETOT06','QRR','QDRR','QTN','QHTN','QTX','QHTX','QTM','QTMNX','QTNSOL','QTN50','DG','QDG','QTAMPLI','QTNTXM','QPMERM','QFFM','QFXI','QDXI','QHXI','QFXY','QDXY','QHXY','QFXI3S','QHXI3S','QUN','QHUN','QUX','QHUX','QDHUMI40','QDHUMI80','QTSVM','QUM','QORAG','QGRESIL','QBRUME','ECLAIR','QECLAIR','QETPMON','QETPGRILLE'
    ]))
    df = df.drop(columns=drop_cols, axis=1)

    # convertion des colonnes object en float64
    for column in df.columns:
        if df[column].dtype == 'object':
            # Replace comma with dot for float conversion
            df[column] = df[column].str.replace(',', '.', regex=False)
            df[column] = df[column].astype('Float64')
    # Convertir la colonne post en code postal à 5 chiffres
    A = 5
    df['Code INSEE'] = df['POSTE'].astype(str).str[:A].astype(str)

    # appelle du fichier code insee
    df2 = pd.read_json('https://fireprojectbislead.s3.us-east-1.amazonaws.com/dataset/corse_insee.json', orient='records')
    df2.rename(columns={'code_insee': 'Code INSEE'}, inplace=True)
    df2.rename(columns={'code_postale': 'Code Postal'}, inplace=True)
    df2.rename(columns={'nom_de_la_commune': 'ville'}, inplace=True)
    df2['Code INSEE'] = df2['Code INSEE'].astype(str)
    # merge des 2 fichiers
    df_corse = pd.merge(df, df2, on='Code INSEE', how='left')
    ti.xcom_push(key="cleaner_data_csv_path", value=df_corse)

def features_data(ti, **kwargs):
    df = ti.xcom_pull(task_ids="cleaner_data", key="cleaner_data_csv_path")
    # fonction de moyenne lissante avec np.convolve
    def moving_average(x, w):
        # Remplir le tableau d'entrée avec 'w//2' éléments de chaque côté en utilisant les valeurs de bord
        padded_x = np.pad(x, (w//2, w//2), mode='edge')
        # Effectuer la convolution avec le mode 'valid'
        return np.convolve(padded_x, np.ones(w), 'valid') / w
    # ajout de colonne sur les précispitation moyenne par an et mois
    df['moyenne precipitations année'] = moving_average(df['RR'], 365).astype('float64').round(2)
    df['moyenne precipitations mois'] = moving_average(df['RR'], 31).astype('float64').round(2)
    # moyenne ecapotranspiration par mois et année
    df['moyenne evapotranspiration année'] = moving_average(df['ETPMON'], 365).astype('float64').round(2)
    df['moyenne evapotranspiration mois'] = moving_average(df['ETPMON'], 31).astype('float64').round(2)
    # moyenne vitesse de vent par mois et année
    df['moyenne vitesse vent année'] = moving_average(df['FFM'], 365).astype('float64').round(2)
    df['moyenne vitesse vent mois'] = moving_average(df['FFM'], 31).astype('float64').round(2)
    # moyenne température par mois et année
    df['moyenne temperature année'] = moving_average(df['TN'], 365).astype('float64').round(2)
    df['moyenne temperature mois'] = moving_average(df['TN'], 31).astype('float64').round(2)
   
    ti.xcom_push(key="cleaner_data_csv_path", value=df)

def fusion_data(ti, **kwargs):
    df=ti.xcom_pull(task_ids="features_data", key="cleaner_data_csv_path")
    
    # appelle du dataset insee
    df_insee = pd.read_csv('https://fireprojectbislead.s3.us-east-1.amazonaws.com/dataset/correspondance-code-insee-code-postal.csv', sep=';',encoding='utf-8')
    # suppression des colonnes inutiles sur le dataset insee
    df_insee = df.drop(columns=['Département','Région','Statut','Altitude Moyenne','Superficie','Population','geo_shape','ID Geofla','Code Commune','Code Canton','Code Arrondissement','Code Département','Code Région'], axis=1)
    # appelle du dataset feu
    feux = pd.read_csv('https://fireprojectbislead.s3.us-east-1.amazonaws.com/dataset/historique_incendies_avec_coordonnees.csv', sep=';', encoding='utf-8')
    # merge du dataset feu et insee
    df_feux = pd.merge(feux, df_insee, on=['Code INSEE'], how='left')
    # modification des colonnes date
    df.rename({'DATE': 'Date'}, axis=1, inplace=True)
    df['Date'] = pd.to_datetime(df_meteo['Date']).dt.normalize()
    df_feux['Date'] = pd.to_datetime(df_feux['Date']).dt.normalize()
    # dans le fichier df_feux on filtre les departement corse on supprimme des colonnes et on renomme une colonne
    feux_corse = df_feux[df_feux['Département'].isin(['2A', '2B', 2])]
    feux_corse = feux_corse.drop(feux_corse.columns[[12, 13, 14, 21]], axis=1)
    feux_corse = feux_corse.rename(columns={'Nom de la commune': 'ville'})
    # fusion du météo et feu
    df_fusion= pd.merge(df, feux_corse, on=['Date', 'ville'], how='outer')
    # traitement des doublons
    df_clean = df.groupby(['ville', 'Date'], as_index=False).agg(lambda x: x.dropna().iloc[0] if not x.dropna().empty else None)
    # on met 0 dans la colonne feux si pas de données 
    df_clean['Feux'] = df_clean['Feux'].fillna(0).astype(int)
    df=df_clean

    # Mise en place de la colonne décompte avant le feu suivant

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['ville', 'Date'])

    def days_until_next_fire(group):
        # Dates où il y a un feu, sinon NaT
        feu_dates = group['Date'].where(group['Feux'] == 1)

        # On inverse la série pour faire un forward fill à rebours (pour chaque date, la prochaine date feu)
        next_feu_dates = feu_dates[::-1].ffill()[::-1]

        # Calcul du delta en jours entre la prochaine date feu et la date actuelle
        delta_days = (next_feu_dates - group['Date']).dt.days

        # Pour les lignes où Feux==1, mettre 0 (par sécurité)
        delta_days[group['Feux'] == 1] = 0

        return delta_days
    # on créé une colonne décompte jusqu'au prochain feu
    df['décompte'] = df.groupby('ville').apply(days_until_next_fire).reset_index(level=0, drop=True)
    # on merge avec le fichiers coordonnées lat/long corse csv 
    gps=pd.read_csv('https://fireprojectbislead.s3.us-east-1.amazonaws.com/dataset/coordonnees_corses.csv')
    df_merge = df.merge(gps,on="ville", how="left")

    # Création de la colonne évènement pour indiquer si un feu a eu lieu
    df_merge['évènement'] = df_merge['Feux'] == 1
    # on encore des probleme avec la lat et long donc on merge avec un autre fichier
    news_gps = pd.read_csv('https://fireprojectbislead.s3.us-east-1.amazonaws.com/dataset/corse_gps.csv', sep=';', encoding='utf-8')
    # on renomme la colonne qui va servir au merge
    news_gps = news_gps.rename(columns={'properties.name':'ville'})
    # on supprime les colonnes inutiles
    news_gps = news_gps.drop(news_gps.columns[[0,1,3]], axis=1)
    # Fusionner les deux DataFrames sur la colonne 'ville'
    df_combined = df_merge.merge(news_gps, on='ville', how='left', suffixes=('', '_y'))
    # Remplacer les valeurs manquantes dans df1 par celles de df2
    df_combined['latitude'] = df_combined['latitude_y'].combine_first(df_combined['latitude'])
    df_combined['longitude'] = df_combined['longitude_y'].combine_first(df_combined['longitude'])
    # Supprimer les colonnes supplémentaires créées par la fusion
    df_combined = df_combined.drop(columns=['latitude_y', 'longitude_y'])
    df_merge = df_combined
    # il restait 256 lignes sans localisation gps que l'on supprime
    df_merge = df_merge.dropna(subset=['latitude', 'longitude'])
    # S'assurer que la date est bien au bon format
    df_merge["Date"] = pd.to_datetime(df_merge["Date"])

    # Trier le DataFrame par ville et date
    df_merge = df_merge.sort_values(by=["ville", "Date"]).reset_index(drop=True)

    # Nouvelle colonne initialisée à NaN
    df_merge["compteur jours vers prochain feu"] = pd.NA

    # Traitement par ville
    for ville, groupe in df_merge.groupby("ville"):
        groupe = groupe.sort_values("Date")
        indices_feux = groupe[groupe["évènement"] == True].index.tolist()
        
        for i in range(len(indices_feux) - 1):
            debut = indices_feux[i]
            fin = indices_feux[i + 1]
            
            # Remplir les jours entre les deux feux avec un compteur croissant
            for j, idx in enumerate(range(debut, fin)):
                df_merge.loc[idx, "compteur jours vers prochain feu"] = j
    
    # # nombre de jour sans feu + log et carré
    df_merge['compteur feu log'] = df_merge['compteur jours vers prochain feu'].apply(lambda x: np.log1p(x) if pd.notnull(x) else np.nan)
    df_merge['compteur feu carré'] = df_merge['compteur jours vers prochain feu'].apply(lambda x: x**2 if pd.notnull(x) else np.nan)
    # # Calcule le nombre de feux par an et mois pour chaque ville
    df_merge['Année'] = df_merge['Date'].dt.year
    df_merge['Mois'] = df_merge['Date'].dt.month
    df_merge['Nombre de feu par an'] = df_merge.groupby(['ville', 'Année'])['Feux'].transform('sum')
    df_merge['Nombre de feu par mois'] = df_merge.groupby(['ville', 'Année', 'Mois'])['Feux'].transform('sum')

        # Trier par ville et par date
    df_merge = df_merge.sort_values(['ville', 'Date'])

    # Fonction pour compter les jours consécutifs sans pluie
    def compter_jours_sans_pluie(groupe):
        compteur = 0
        jours_sans_pluie = []
        for rr in groupe['RR']:
            if pd.isna(rr):
                jours_sans_pluie.append(np.nan)
            elif rr == 0:
                compteur += 1
                jours_sans_pluie.append(compteur)
            else:
                compteur = 0
                jours_sans_pluie.append(compteur)
        return jours_sans_pluie

    # Appliquer par ville
    df_merge['jours_sans_pluie'] = df_merge.groupby('ville').apply(compter_jours_sans_pluie).explode().astype(float).values
    
    
    # Fonction pour compter les jours consécutifs avec TX > 30
    def compter_jours_chauds(groupe):
        compteur = 0
        jours_chauds = []
        for tx in groupe['TX']:
            if pd.isna(tx):
                jours_chauds.append(np.nan)
            elif tx > 30:
                compteur += 1
                jours_chauds.append(compteur)
            else:
                compteur = 0
                jours_chauds.append(compteur)
        return jours_chauds

    # Appliquer la fonction par ville
    df_merge= df_merge.sort_values(['ville', 'Date'])  # Assurer l'ordre temporel
    df_merge['jours_TX_sup_30'] = df_merge.groupby('ville').apply(compter_jours_chauds).explode().astype(float).values

    df_merge["ETPGRILLE_7j"] = df_merge.groupby("ville")["ETPGRILLE"].transform(lambda x: x.rolling(7, min_periods=1).mean())


    # Chargement du fichier CSV
    # df_merge = pd.read_csv("https://fireprojectbislead.s3.us-east-1.amazonaws.com/dataset/dataset_modele_decompte2.csv", sep=';', low_memory=False)

    # Colonnes météo à compléter
    colonnes_meteo = [
        'RR', 'DRR', 'TN', 'HTN', 'TX', 'HTX', 'TM', 'TMNX', 'TNSOL', 'TN50',
        'TAMPLI', 'TNTXM', 'FFM', 'FXI', 'DXI', 'HXI', 'FXY', 'DXY', 'HXY',
        'FXI3S', 'HXI3S', 'UN', 'HUN', 'UX', 'HUX', 'DHUMI40', 'DHUMI80',
        'TSVM', 'UM', 'ORAG', 'BRUME', 'ETPMON', 'ETPGRILLE'
    ]

    # Séparer les lignes avec et sans données météo
    df_manquantes = df_merge[df_merge[colonnes_meteo].isnull().any(axis=1)].copy()
    df_completes = df_merge.dropna(subset=colonnes_meteo).copy()

    # Fonction pour trouver la ville la plus proche avec données météo
    def trouver_ville_proche(row, ref_df):
        if pd.isna(row['latitude']) or pd.isna(row['longitude']):
            return None

        ville_ref = ref_df[['ville', 'latitude', 'longitude']].dropna().drop_duplicates()
        coord = (row['latitude'], row['longitude'])

        ville_ref['distance'] = ville_ref.apply(
            lambda x: geodesic(coord, (x['latitude'], x['longitude'])).km, axis=1
        )

        plus_proche = ville_ref.loc[ville_ref['distance'].idxmin()]
        return plus_proche['ville']

    # Associer une ville de référence à chaque ligne manquante
    df_manquantes['ville_proche'] = df_manquantes.apply(
        lambda x: trouver_ville_proche(x, df_completes), axis=1
    )

    # Copier les valeurs météo depuis la ville proche
    # Fonction robuste de récupération des données météo
    def recuperer_donnees_meteo(row, df_source, max_villes=5):
        if pd.isna(row['latitude']) or pd.isna(row['longitude']):
            return pd.Series([None] * len(colonnes_meteo), index=colonnes_meteo)

        # Calcul des distances vers toutes les villes avec données météo
        coord = (row['latitude'], row['longitude'])
        villes_ref = df_source[['ville', 'latitude', 'longitude']].dropna().drop_duplicates().copy()

        villes_ref['distance'] = villes_ref.apply(
            lambda x: geodesic(coord, (x['latitude'], x['longitude'])).km, axis=1
        )

        # Trier par proximité
        villes_proches = villes_ref.sort_values('distance').head(max_villes)

        # Chercher une ville avec données pour cette date
        for _, ville_row in villes_proches.iterrows():
            ville = ville_row['ville']
            meme_jour = df_source[
                (df_source['ville'] == ville) & (df_source['Date'] == row['Date'])
            ]
            if not meme_jour.empty:
                return meme_jour[colonnes_meteo].iloc[0]

        # Si aucune ville ne convient
        return pd.Series([None] * len(colonnes_meteo), index=colonnes_meteo)



    # Appliquer proprement les remplacements
    for idx, row in df_manquantes.iterrows():
        valeurs_remplacement = recuperer_donnees_meteo(row, df_completes)
        for col in colonnes_meteo:
            if pd.isna(df_manquantes.at[idx, col]) and pd.notna(valeurs_remplacement[col]):
                df_manquantes.at[idx, col] = valeurs_remplacement[col]


    # Fusion des deux ensembles pour un dataframe complet
    df_final = pd.concat([
        df_completes,
        df_manquantes
    ]).sort_index()

    # Export possible si besoin
    path=df_final.to_csv("dataset_complet_meteo.csv", sep=';', index=False)
    ti.xcom_push(key="dataset_complet_csv_path", value=path)

def upload_fusion_csv_to_s3(ti, **kwargs):
    path = ti.xcom_pull(task_ids='fusion_data', key='dataset_complet_csv_path')
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
    compile_meteo = PythonOperator(
        task_id="compile_meteo_data",
        python_callable=compile_meteo_data
    )
    upload_compile_csv = PythonOperator(
        task_id="upload_compile_csv_to_s3",
        python_callable=upload_compile_csv_to_s3
    )
    cleaner_data=PythonOperator(
        task_id="cleaner_data",
        python_callable=cleaner_data
    )
    features_data=PythonOperator(
        task_id="features_data",
        python_callable=features_data
    )
    fusion_data=PythonOperator(
        task_id="fusion_data",
        python_callable=fusion_data
    )
    upload_fusion_csv_to_s3=PythonOperator(
        task_id="upload_fusion_csv_to_s3",
        python_callable=upload_fusion_csv_to_s3
    )

fetch_weather >> compile_meteo >> upload_compile_csv >> cleaner_data >> features_data >> fusion_data >> upload_fusion_csv_to_s3