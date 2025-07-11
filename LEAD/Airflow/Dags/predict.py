from airflow.decorators import task
import mlflow.sklearn
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
import datetime

default_args = {
    "owner": "airflow",
    "start_date": datetime.datetime(2024, 1, 1),
    "retries": 1
}

def predict_survival(ti):
    df = pd.read_csv('https://fireprojectbislead.s3.us-east-1.amazonaws.com/compile/dataset_complet_meteo.csv')

    # 🔹 Nettoyage & filtrage
    df['Feu prévu'] = df['Feu prévu'].astype(bool)
    df["décompte"] = df["décompte"].fillna(0)
    mask = df['Année'] == 2025
    df_clean = df[mask].copy()

    # 🔹 Définition des features disponibles
    features = [
        'moyenne precipitations mois', 'moyenne temperature mois',
        'moyenne evapotranspiration mois', 'moyenne vitesse vent année',
        'moyenne vitesse vent mois', 'moyenne temperature année',
        'RR', 'UM', 'ETPMON', 'TN', 'TX', 'Nombre de feu par an',
        'Nombre de feu par mois', 'jours_sans_pluie', 'jours_TX_sup_30',
        'ETPGRILLE_7j', 'compteur jours vers prochain feu', 'compteur feu log',
        'Année', 'Mois', 'moyenne precipitations année', 'moyenne evapotranspiration année'
    ]
    features = [f for f in features if f in df_clean.columns]

    # 🔹 Préparation des données
    df_clean = df_clean.rename(columns={"Feu prévu": "event", "décompte": "duration"})
    X = df_clean[features]
    duration = df_clean["duration"]
    event = df_clean["event"]

    # 🔹 Chargement du modèle depuis MLflow
    mlflow.set_tracking_uri("http://mlflow:5000")
    model_uri = "runs:/<RUN_ID>/model"  # 🧠 À remplacer dynamiquement ou par variable stockée
    pipeline = mlflow.sklearn.load_model(model_uri)

    # 🔹 Prédiction log(HR)
    log_hr = pipeline.predict(X)

    # 🔹 Estimation fonction de survie
    df_risque = pd.DataFrame({
        "duration": duration,
        "event": event,
        "log_risque": log_hr
    })
    df_risque["log_risque"] += np.random.normal(0, 1e-4, size=len(df_risque))

    cph = CoxPHFitter()
    cph.fit(df_risque, duration_col="duration", event_col="event", show_progress=False)

    # 🔹 Prédiction aux horizons futurs
    df_pred = pd.DataFrame({"log_risque": log_hr})
    times = [7, 30, 60, 90, 180]
    surv_funcs = cph.predict_survival_function(df_pred, times=times)
    probas_feu = 1 - surv_funcs.T
    probas_feu.columns = [f"proba_{t}j" for t in times]

    # 🔹 Fusion avec les coordonnées
    df_output = pd.concat([
        df_clean[["latitude", "longitude", "ville"]].reset_index(drop=True),
        probas_feu.reset_index(drop=True)
    ], axis=1)

    # 🔹 Export CSV pour Airbyte ou autre usage
    output_path = "/tmp/predictions.csv"
    df_output.to_csv(output_path, index=False)

    ti.xcom_push(key="prediction_path", value=output_path)

def upload_to_s3(ti):
    # 🔹 Récupération du chemin du fichier prédictions
    path = ti.xcom_pull(task_ids="predict_survival", key="prediction_path")
    bucket = Variable.get("S3BucketName")
    compile_folder = Variable.get("S3Compile")
    s3 = S3Hook(aws_conn_id="aws_default")
    filename = os.path.basename(path)
    s3.load_file(filename=path, key=f"{compile_folder}/{filename}", bucket_name=bucket, replace=True)

with DAG(
    dag_id="predict",
    default_args=default_args,
    schedule_interval="@daily",
    catchup=False,
    description="prédiction"
) as dag:

    predict_survival= PythonOperator(
        task_id="predict_survival",
        python_callable=predict_survival,
    )
    upload_to_s3 = PythonOperator(
        task_id="upload_to_s3",
        python_callable=upload_to_s3
    )

predict_survival >> upload_to_s3