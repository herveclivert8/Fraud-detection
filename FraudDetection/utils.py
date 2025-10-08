import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import random
import lightgbm as lgbm
import streamlit as st
import time
import datetime

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_curve, auc, roc_auc_score
from sklearn.cluster import KMeans
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timedelta
import plotly.graph_objects as go

# FONCTION POUR PREPARER LE DATASET à être transformé pour l'Autoencoder
def preprocessing(df):
    df = df.copy()
    
    # feature extraction
    df['Hour'] = df['Timestamp'].dt.hour
    df['Day'] = df['Timestamp'].dt.day
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek # 0 = lundi, 6 = dimanche
    
    # transformation log
    df['Log_Amount_Paid'] = np.log1p(df['Amount Paid'])
    df['Log_Amount_Diff'] = np.log1p(np.abs(df['Amount Paid'] - df['Amount Received']))
    
    # à supprimer
    drop_cols = ['From Account', 'To Account','Amount Paid', 'Amount Received', 'Timestamp']
    df.drop(columns=drop_cols, inplace=True)
    
    # From/To Bank : encodage par fréquence
    for col in ['From Bank', 'To Bank'] :
        freq = df[col].value_counts()
        df[col] = df[col].map(freq).fillna(0)
        df[col] = np.log1p(df[col])
    
    num_cols = df.select_dtypes(exclude='object').columns.tolist()
    cat_cols = ['Receiving Currency', 'Payment Currency', 'Payment Format']
    
    return df, num_cols, cat_cols



# pour le lightgbm
def preprocessing_lgbm(df):
    df = df.copy()
    df, num_cols, cat_cols = preprocessing(df)
    X = df.drop(columns='Pseudo_Labels')
    y = df['Pseudo_Labels']
    return X, y, cat_cols



# FONCTION DE CREATION DES VARIABLES RFM (utile pour les signaux d'alarmes) après clustering
def rfm_features(df):
    df=df.copy()

    #------ RECENCE (R) -------

    df['Hour'] = df['Timestamp'].dt.hour
    df['Day'] = df['Timestamp'].dt.day
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek # 0 = lundi, 6 = dimanche

        # récence compte émetteur : nombre de jours écoulés depuis la dernière transaction (From Account)
    last_tx = df.groupby('From Account')['Timestamp'].transform('max')
    df['Recency_Days'] = (df['Timestamp'].max() - last_tx).dt.days # récence de réference (From Account)
    
    df['isNight'] = df['Hour'].apply(lambda x: 1 if (x < 6 or x > 22) else 0)
    df['isWeekend'] = df['DayOfWeek'].isin([5,6]).astype(int)


    #------- FREQUENCE (F) ---------

    # nombre de transactions par compte
    df['Freq_Tx'] = df.groupby('From Account')['Timestamp'].transform('count') # transactions effectuées (réference : From Account)

    # nombre de destinataires (To Account) uniques par compte (From Account)
    df['Unique_To_per_From'] = df.groupby('From Account')['To Account'].transform('nunique')

    # Brust : nombre de transactions dans une petite intervalle de temps (ex: < 5min)
    Time_diff_Min = df.sort_values(['From Account', 'Timestamp']).groupby('From Account')['Timestamp'].diff().dt.total_seconds() / 60
    df['isBrust'] = Time_diff_Min.apply(lambda x: int(x <=3 if pd.notnull(x) else 0))


    #------ MONETAIRE (M) ------

    # différence des montants payés et reçus (alternative stable pour ne garder qu'une seule des variables de base)
    df['Amount_Diff'] = df['Amount Paid'] - df['Amount Received']

    # moyenne et max des montants envoyées (From account)
    df['Amount_Mean'] = df.groupby('From Account')['Amount Paid'].transform('mean')
    df['Amount_Max'] = df.groupby('From Account')['Amount Paid'].transform('max')

    # montants petits fréquents (smurfing)
    df['Small_Amount'] = (df['Amount Paid'] < 200).astype(int)
    df['Nb_Small_Tx'] = (df.groupby('From Account')['Small_Amount'].transform('sum'))

    # transformation logarithmique pour stabiliser la distribution
    df['Log_Amount_Paid'] = np.log1p(df['Amount Paid'])
    df['Log_Amount_Diff'] = np.log1p(np.abs(df['Amount_Diff']))
    df['Log_Amount_Mean'] = np.log1p(df['Amount_Mean'])
    df['Log_Amount_Max'] = np.log1p(df['Amount_Max'])


    #------- AUTRES -------
    # pour garder le minimum d'information de From bank et To Bank
    df['Same_Bank_Transfer'] = (df['From Bank'] == df['To Bank']).astype(int)

    return df



# SIGNAUX D'ALARMES
def signals_frauds(df, cluster_col='Cluster', confidence_threshold=0.05):
    df = df.copy()

    # SIGNAUX
    df['High_Amount'] = ((df['Log_Amount_Mean'] > df['Log_Amount_Mean'].quantile(0.95)) |
                         (df['Log_Amount_Max'] > df['Log_Amount_Max'].quantile(0.95))).astype(int)
    df['isInternational'] = (df['Receiving Currency'] != df['Payment Currency']).astype(int)
    df['Freq_Small_Tx'] = (df['Nb_Small_Tx'] > df['Nb_Small_Tx'].quantile(0.9)).astype(int)
    df['Many_Dests'] = (df['Unique_To_per_From'] > df['Unique_To_per_From'].quantile(0.95)).astype(int)
    df['High_Freq_From'] = (df['Freq_Tx'] > df['Freq_Tx'].quantile(0.95)).astype(int)
    df['Similary'] = ((df['Receiving Currency'] == df['Payment Currency']) |
                      (df.get('Same_Bank_Transfer', 0) == 1)).astype(int)
    df['Wire_ACH_Bitcoin_Format'] = df['Payment Format'].isin(['Wire', 'Bitcoin', 'ACH']).astype(int)

    df['Very_Recent'] = (df['Recency_Days'] < 1).astype(int)
    df['Reactivation_Suspect'] = ((df['Recency_Days'] > 10) & (df['Freq_Tx'] > 2)).astype(int)
    df['Cash_Bitcoin_Format'] = df['Payment Format'].isin(['Cash', 'Bitcoin']).astype(int)

    df['Credit_Format'] = (df['Payment Format'] == 'Credit Card').astype(int)

    frauds = {
        'blanchiment': ['High_Amount','isInternational','Freq_Small_Tx','Many_Dests','High_Freq_From','Similary','Wire_ACH_Bitcoin_Format'],
        'fraude par carte': ['Very_Recent','isNight','isWeekend', 'isBrust', 'Credit_Format'],
        'fraude par compte mule': ['Very_Recent','Reactivation_Suspect','Many_Dests','High_Freq_From','Cash_Bitcoin_Format']
    }

    all_signals = list(set(sig for sigs in frauds.values() for sig in sigs))
    existing_signals = [s for s in all_signals if s in df.columns]
    cluster_profiles = df.groupby(cluster_col)[existing_signals].mean()

    cluster_scores = []

    # Calcul des scores possibles selon formats de paiement par cluster
    for cluster_id, row in cluster_profiles.iterrows():
        formats_present = set(df[df[cluster_col] == cluster_id]['Payment Format'].unique())

        possible_types = set()
        if any(f in formats_present for f in ['Wire', 'ACH', 'Bitcoin']):
            possible_types.add('blanchiment')
        if 'Credit Card' in formats_present:
            possible_types.add('fraude par carte')
        if any(f in formats_present for f in ['Cash', 'Bitcoin']):
            possible_types.add('fraude par compte mule')

        for fraud_type in possible_types:
            signals = frauds[fraud_type]
            present_signals = [s for s in signals if s in row.index]
            if present_signals:
                score = row[present_signals].mean()
                cluster_scores.append((cluster_id, fraud_type, score))

    # Trier par score décroissant
    cluster_scores.sort(key=lambda x: x[2], reverse=True)

    cluster_to_fraud = {}
    assigned_types = set()
    assigned_clusters = set()

    # Attribution unique pour chaque type, au cluster avec meilleur score
    for cluster_id, fraud_type, score in cluster_scores:
        if fraud_type not in assigned_types and cluster_id not in assigned_clusters and score >= confidence_threshold:
            cluster_to_fraud[cluster_id] = fraud_type
            assigned_types.add(fraud_type)
            assigned_clusters.add(cluster_id)

    # Les clusters non attribués sont assignés au dernier type de fraude non attribué,
    # sauf si leur score max est trop bas, alors "normale"
    remaining_types = set(frauds.keys()) - assigned_types

    for cluster_id in cluster_profiles.index:
        if cluster_id not in assigned_clusters:
            # Score max possible pour ce cluster
            scores_for_cluster = [score for cid, _, score in cluster_scores if cid == cluster_id]
            max_score = max(scores_for_cluster) if scores_for_cluster else 0

            if max_score < confidence_threshold or not remaining_types:
                cluster_to_fraud[cluster_id] = "légitime"
            else:
                # Attribuer un type restant (au hasard, ou mieux à celui avec meilleur score)
                # Ici on prend le premier de remaining_types
                fraud_type = remaining_types.pop()
                cluster_to_fraud[cluster_id] = fraud_type
                assigned_types.add(fraud_type)
                assigned_clusters.add(cluster_id)

    df['Pseudo_Labels'] = df[cluster_col].map(cluster_to_fraud)

    # Affichage résumé
    for cluster_id, fraud_type in cluster_to_fraud.items():
        print(f"Cluster {cluster_id} attribué à : {fraud_type}")

    return df, cluster_to_fraud


####### AUTOENCODER #######

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(16,32),
            nn.ReLU(),
            nn.Linear(32,input_dim)
        )

    def forward(self,x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon
    
def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
class AutoEncoderWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, epochs=20, batch_size=256, lr=1e-3, verbose=1, device=None, seed=0):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.verbose = verbose
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.seed = seed

    def fit(self, X, y=None):
        set_seed(self.seed)
        X = np.array(X, dtype=np.float32)
        self.input_dim = X.shape[1]
        self.model = Autoencoder(self.input_dim).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        dataset = torch.utils.data.TensorDataset(torch.from_numpy(X))
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_x, in loader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_x)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_x.size(0)
            epoch_loss /= len(loader.dataset)
            if self.verbose:
                print(f"Epoch {epoch+1}/{self.epochs} - Loss: {epoch_loss:.6f}")
        return self

    # retourne les erreurs de reconstruction (= score d’anomalie)
    def score_samples(self, X):
        X = np.array(X, dtype=np.float32)
        with torch.no_grad():
            inputs = torch.from_numpy(X).to(self.device)
            outputs = self.model(inputs).cpu().numpy()
        errors = np.mean((X - outputs) ** 2, axis=1)
        return errors

    # pour compatibilité sklearn — retourne les scores
    def transform(self, X):
        return self.score_samples(X)

    # Renvoie des labels binaires selon un seuil
    def predict(self, X, threshold=None):
        scores = self.score_samples(X)
        if threshold is None:
            threshold = np.percentile(scores, 99)
        return (scores > threshold).astype(int)


