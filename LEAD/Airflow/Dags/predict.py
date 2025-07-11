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

    mask = df_clean["Année"] == 2025
    df_clean = df_clean[~mask]
    df_clean.head()

################## code si paramètre du modele chargé sur mlflow ######################""""
    # # 🔹 Préparation des données
    # df_clean = df_clean.rename(columns={"Feu prévu": "event", "décompte": "duration"})
    # X = df_clean[features]
    # duration = df_clean["duration"]
    # event = df_clean["event"]

    # # 🔹 Chargement du modèle depuis MLflow
    # mlflow.set_tracking_uri("http://mlflow:5000")
    # model_uri = "runs:/<RUN_ID>/model"  # 🧠 À remplacer dynamiquement ou par variable stockée
    # pipeline = mlflow.sklearn.load_model(model_uri)

    # # 🔹 Prédiction log(HR)
    # log_hr = pipeline.predict(X)

    # # 🔹 Estimation fonction de survie
    # df_risque = pd.DataFrame({
    #     "duration": duration,
    #     "event": event,
    #     "log_risque": log_hr
    # })
    # df_risque["log_risque"] += np.random.normal(0, 1e-4, size=len(df_risque))

    # cph = CoxPHFitter()
    # cph.fit(df_risque, duration_col="duration", event_col="event", show_progress=False)

    # # 🔹 Prédiction aux horizons futurs
    # df_pred = pd.DataFrame({"log_risque": log_hr})
    # times = [7, 30, 60, 90, 180]
    # surv_funcs = cph.predict_survival_function(df_pred, times=times)
    # probas_feu = 1 - surv_funcs.T
    # probas_feu.columns = [f"proba_{t}j" for t in times]

    # # 🔹 Fusion avec les coordonnées
    # df_output = pd.concat([
    #     df_clean[["latitude", "longitude", "ville"]].reset_index(drop=True),
    #     probas_feu.reset_index(drop=True)
    # ], axis=1)

    # # 🔹 Export CSV pour Airbyte ou autre usage
    # output_path = "/tmp/predictions.csv"
    # df_output.to_csv(output_path, index=False)

    # ti.xcom_push(key="prediction_path", value=output_path)

#///////////////////////////////////////////////////////////////////////////////////////////////#

######################## code si pas charge dans mlflow###############################

################ Entrainement du modèle pour caler ces coef #################
    # on filtre pour ne pas prendre les données de 2025 dans l'entrainement 
    mask = df_clean["Année"] == 2025
    df_clean = df_clean[~mask]
    df_clean.head()

    # 🔹 Préparation des données réelles
    df_clean = df_clean.rename(columns={"Feu prévu": "event", "décompte": "duration"})
    y_structured = Surv.from_dataframe("event", "duration", df_clean)

    X = df_clean[features]
    y = y_structured

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    event_train = y_train["event"]
    duration_train = y_train["duration"]
    event_test = y_test["event"]
    duration_test = y_test["duration"]

    # 🔹 Pipeline XGBoost survie avec StandardScaler
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("xgb", XGBRegressor(
            objective="survival:cox",
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            tree_method="hist",
            device="cuda",
            random_state=42
        ))
    ])
    pipeline.fit(X_train, duration_train, xgb__sample_weight=event_train)

    # 🔹 Prédictions réelles (log(HR)) sur données test
    log_hr_test = pipeline.predict(X_test)

    # 🔹 Jeu factice pour estimer le modèle de Cox
    df_fake = pd.DataFrame({
        "duration": duration_train,
        "event": event_train,
        "const": 1
    })
    dtrain_fake = DMatrix(df_fake[["const"]])
    dtrain_fake.set_float_info("label", df_fake["duration"])
    dtrain_fake.set_float_info("label_lower_bound", df_fake["duration"])
    dtrain_fake.set_float_info("label_upper_bound", df_fake["duration"])
    dtrain_fake.set_float_info("weight", df_fake["event"])

    params = {
        "objective": "survival:cox",
        "eval_metric": "cox-nloglik",
        "learning_rate": 0.1,
        "max_depth": 1,
        "verbosity": 0
    }
    bst_fake = train(params, dtrain_fake, num_boost_round=100)

    log_hr_fake = bst_fake.predict(dtrain_fake)
    df_risque = pd.DataFrame({
        "duration": duration_train,
        "event": event_train,
        "log_risque": log_hr_fake
    })
    # insertion de bruit pour aider le modèle à converger
    df_risque["log_risque"] += np.random.normal(0, 1e-4, size=len(df_risque))

    # 🔹 Modèle de Cox factice
    cph = CoxPHFitter()
    cph.fit(df_risque, duration_col="duration", event_col="event", show_progress=False)

    # 🔹 Évaluation avec le c-index
    c_index = concordance_index_censored(event_test, duration_test, log_hr_test)[0]
    print(f"\nC-index (test) : {c_index:.3f}")

############################ entrainement fini maintenant on traite les données de 2025 pour faire la prédiction ################
    df_clean = df.copy()
    mask = df_clean["Année"] == 2025
    df_clean = df_clean[mask]

    X_new = df_clean[features]
    log_hr_new = pipeline.predict(X_new)

    # 🔹 Prédiction des probabilités de survie
    df_preds_new = pd.DataFrame({"log_risque": log_hr_new})
    times = [7, 30, 60, 90, 180]
    surv_funcs_new = cph.predict_survival_function(df_preds_new, times=times)
    probas_feu_new = 1 - surv_funcs_new.T
    probas_feu_new.columns = [f"proba_{t}j" for t in times]

    # 🔹 Fusion avec les coordonnées
    df_map_new = pd.concat([
        df_clean[["latitude", "longitude", "ville"]].reset_index(drop=True),
        probas_feu_new.reset_index(drop=True)
    ], axis=1)

    # 🔹 Sauvegarde dans un fichier CSV
    output_path = "/tmp/predictions.csv"
    df_map_new.to_csv(output_path, index=False)

# 🔹 Envoi du chemin via XCom
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