from utils import *



#################################

# Configuration de la page
st.set_page_config(
    page_title="DEMO DASHBOARD",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data(n):
    df = pd.read_csv('data/demo_dataset.csv')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    for col in df.select_dtypes(include='int'):
        df[col] = df[col].astype('object')
    # df = df.sort_values('Timestamp', ascending=True)
    X_AE, num_cols, cat_cols = preprocessing(df)
    X_AE['Anomaly_Score'] = ae_pipeline.score_samples(X_AE) 
    X_AE['isAnomaly'] = ae_pipeline.predict(X_AE, threshold = np.percentile(X_AE['Anomaly_Score'], 99))
    df_anomalies = df.copy()
    df_anomalies.loc[X_AE.index, 'isAnomaly'] = X_AE['isAnomaly']
    anomalies = df_anomalies[df_anomalies['isAnomaly'] == 1].drop(columns='isAnomaly')
    X_KMEANS, num_cols, cat_cols = preprocessing(anomalies)
    X_KMEANS['Cluster'] = kmeans_pipeline.predict(X_KMEANS)
    df_cluster = anomalies.copy()
    df_cluster.loc[X_KMEANS.index, 'Cluster'] = X_KMEANS['Cluster']
    RFM = rfm_features(df_cluster)
    df_frauds, mapping = signals_frauds(RFM)
    df_labels = df_cluster.copy()
    df_labels.loc[df_frauds.index, 'Pseudo_Labels'] = df_frauds['Pseudo_Labels']
    X, y, cat_cols = preprocessing_lgbm(df_labels)
    y_encoded = label_encoder.transform(y)
    y_pred = lgbm_pipeline.predict(X)
    y_labels = label_encoder.inverse_transform(y_pred)
    
    df['Anomaly_Score'] = X_AE['Anomaly_Score']
    df['isAnomaly'] = X_AE['isAnomaly'] 
    df.loc[df_labels.index, 'Cluster'] = df_labels['Cluster']
    df.loc[df_labels.index, 'Pseudo_Labels'] = df_frauds['Pseudo_Labels']
    df['Pseudo_Labels'] = df['Pseudo_Labels'].fillna('légitime')
    df.loc[df_labels.index, 'Prediction'] = y_labels
    df['Prediction'] = df['Prediction'].fillna('légitime')
    
    print(classification_report(y_encoded,y_pred))
    print(confusion_matrix(y_encoded,y_pred))
    
    return df.sample(n, random_state=0)



ae_pipeline = joblib.load('model/ae_pipeline.pkl')
kmeans_pipeline = joblib.load('model/kmeans_pipeline.pkl')
lgbm_pipeline = joblib.load('model/lgbm_pipeline.pkl')
label_encoder = joblib.load('model/label_encoder.pkl')


def preprocessing_test(df):
    df= df.copy()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    for col in df.select_dtypes(include='int'):
        df[col] = df[col].astype('object')
    X_AE, num_cols, cat_cols = preprocessing(df)
    X_AE['Anomaly_Score'] = ae_pipeline.score_samples(X_AE)
    X_AE['isAnomaly'] = ae_pipeline.predict(X_AE, threshold = np.percentile(X_AE['Anomaly_Score'], 99))
    df_anomalies = df.copy()
    df_anomalies.loc[X_AE.index, 'isAnomaly'] = X_AE['isAnomaly']
    # anomalies = df_anomalies[df_anomalies['isAnomaly'] == 1].drop(columns='isAnomaly')
    X_KMEANS, num_cols, cat_cols = preprocessing(df_anomalies)
    X_KMEANS['Cluster'] = kmeans_pipeline.predict(X_KMEANS)
    df_cluster = df_anomalies.copy()
    df_cluster.loc[X_KMEANS.index, 'Cluster'] = X_KMEANS['Cluster']
    RFM = rfm_features(df_cluster)
    df_frauds, mapping = signals_frauds(RFM)
    df_labels = df_cluster.copy()
    df_labels.loc[df_frauds.index, 'Pseudo_Labels'] = df_frauds['Pseudo_Labels']
    X_test, y_test, cat_cols = preprocessing_lgbm(df_labels)
    return X_test, y_test, cat_cols



###############################

# -------------- SIDEBAR ------------- #
st.sidebar.title("Paramètres")
check = st.sidebar.checkbox('Données brutes')
speed = st.sidebar.slider("Vitesse (sec)", 0.1, 5.0, 0.4, 0.1)
transactions_per_interval = st.sidebar.slider("Nombre de transactions :", 1000, 830886, 5000)
type_filter = st.sidebar.multiselect("Filtre par type de fraude :", ["blanchiment", "fraude par carte", "fraude par compte mule"],)
                                    #  default=["blanchiment", "fraude par carte", "fraude par compte mule"])

df_demo = load_data(transactions_per_interval) 

if "data" not in st.session_state:
    st.session_state.data = load_data(500)
if 'stop_simulation' not in st.session_state:
    st.session_state.stop_simulation = False
    
st.session_state.data = df_demo

latest_data = st.session_state.data.copy()
if type_filter:
    latest_data = latest_data[latest_data["Pseudo_Labels"].isin(type_filter)]
    
FRAUD_COLOR_MAP = {
    'blanchiment': 'white',
    'fraude par carte':'orangered',
    'fraude par compte mule': 'purple',
    'légitime': 'green'
}    
    
    
tabs = st.tabs(["Vue d'ensemble", "Détection", "Prédiction", "Détails techniques"])



##############################

with tabs[0]:

    # ---------- KPI TOP ---------- #

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        n_anomalies = (latest_data["isAnomaly"] == 1).sum()
        st.metric("🟠 Anomalies détectées", n_anomalies)
    with col2:
        n_frauds = latest_data[latest_data["Pseudo_Labels"] != "légitime"].shape[0]
        st.metric("🔴 Fraudes présumées", n_frauds)
    with col3:
        n_worsts = latest_data[latest_data["Pseudo_Labels"] != latest_data["Prediction"]].shape[0]
        st.metric("🔀 Confusion", n_worsts)
    with col4:
        n_clusters = latest_data["Cluster"].dropna().nunique()
        st.metric("🟢 Clusters actifs", n_clusters)
    with col5:
        n_wrongs = n_anomalies - n_frauds
        st.metric("🟡 Fausses alertes", n_wrongs)
        
        
    if check:
        # st.dataframe(latest_data.drop(columns=['Payment Format']))
        st.dataframe(latest_data.drop(columns=['Anomaly_Score','isAnomaly','Cluster','Pseudo_Labels','Prediction']))      

    # ---------- GRAPHIQUES BAS ---------- #
    
    col_left, col_right = st.columns(2)

    with col_left:
        latest_data["isAnomaly"] = latest_data["isAnomaly"].map({0:"normale", 1:"anomalie"})
        fig_scatter = px.scatter(
            latest_data, x="Amount Paid", y="Amount Received",
            color="isAnomaly", hover_data=["From Bank", "Payment Currency", "To Bank", "Receiving Currency"],
            color_discrete_map={"normale": "blue", "anomalie": 'red'},
            title="Montants payés et reçus"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_right:
        st.plotly_chart(px.pie(latest_data, names='Pseudo_Labels', color='Pseudo_Labels', hole=0.5, title="Répartition des types de fraudes", color_discrete_map=FRAUD_COLOR_MAP))
    
    
    
##############################    
    
with tabs[2]:
    
    last_predict_proba = [0.0,0.0,0.0]
    
    # KPI placeholders
    kpi5, kpi6, kpi7, kpi8 = st.columns(4)
    ph_kpi5 = kpi5.empty()
    ph_kpi6 = kpi6.empty()
    ph_kpi7 = kpi7.empty()
    ph_kpi8 = kpi8.empty()

    # Graph placeholders
    col3, col4 = st.columns(2)
    ph3 = col3.empty()
    ph4 = col4.empty()



##############################  

with tabs[3] :
    
    kpi9, kpi10 = st.columns(2)
    ph_auc_macro = kpi9.empty()
    ph_auc_weighted = kpi10.empty()
    
    ph5 = st.empty()         
    
    prev_auc_macro = None
    prev_auc_weighted = None
        
    
##############################        

with tabs[1]:
           
    # KPI placeholders
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    ph_kpi1 = kpi1.empty()
    ph_kpi2 = kpi2.empty()
    ph_kpi3 = kpi3.empty()
    ph_kpi4 = kpi4.empty()

    # Graph placeholders
    col1, col2= st.columns(2, vertical_alignment='top')
    ph1 = col1.empty()
    ph2 = col2.empty()

    # Historique dynamique
    history = []
    cumulative_volume = 0
    anomaly_count = 0
    previous_score = None

    # Simulation        
    for step in range(len(latest_data)):
        
        row = latest_data.iloc[step]
        
        history.append(row)

        hist_df = pd.DataFrame(history)

        # === KPI ===
        current_score = row['Anomaly_Score']
        cumulative_volume += row['Amount Paid']
        if row['isAnomaly'] == 'anomalie':
            anomaly_count += 1

        # Déterminer le delta
        if previous_score is not None:
            delta_score = current_score - previous_score
        else:
            delta_score = 0.0  # Premier point

        # Affichage avec delta
        ph_kpi1.metric(
            label="📈 Score d’anomalie actuel",
            value=f"{current_score:.4f}",
            delta=f"{delta_score:+.4f}"
        )

        # Mettre à jour le score précédent
        previous_score = current_score
        
        # ph_kpi1.metric("📈 Score d’anomalie actuel", f"{current_score:.4f}")
        ph_kpi2.metric("💲 Volume des transactions", f"{cumulative_volume:,.0f}")
        ph_kpi3.metric("🔴 Anomalies détectées", f"{anomaly_count}")
        ph_kpi4.metric("📊 Total transactions simulées", f"{len(history)}")

        # === Graphique 1: Area chart score d’anomalie ===
        # fig1 = px.area(hist_df.tail(10), y="Anomaly_Score", title="Score d’anomalie (progression)")
        # ph1.plotly_chart(fig1, use_container_width=True, key=f'fig1_{step}')
        
        # Graphique en défilement continu comme un chart boursier
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            y=hist_df["Anomaly_Score"],
            mode="lines",
            line=dict(color="royalblue", width=2),
            name="Anomaly Score"
        ))

        # Configuration de l'axe x dynamique
        fig1.update_layout(
            title="Score d’anomalie (progression)",
            xaxis=dict(
                range=[max(0, len(hist_df)-30), len(hist_df)],
                title="Index",
                showgrid=False
            ),
            yaxis=dict(title="Score", showgrid=True),
           
        )

        ph1.plotly_chart(fig1, use_container_width=True, key=f'scroll_fig1_{step}')
                        
        # === Graphique 2: Scatter - Amount Paid vs Received ===
        fig2 = px.scatter(
            hist_df, x="Amount Paid", y="Amount Received",
            color="isAnomaly", hover_data=["From Bank", "Payment Currency", "To Bank", "Receiving Currency"],
            color_discrete_map={"normale": "blue", "anomalie": 'red'},
            title="Montants normaux et anormaux"
            
        )
        ph2.plotly_chart(fig2, use_container_width=True,key=f'fig2_{step}')     


        with tabs[2]:
            
            ph_kpi5.metric("Blanchiment  d'argent", f"{hist_df[hist_df['Prediction'] == 'blanchiment'].shape[0]}")
            ph_kpi6.metric("Fraude par carte de crédit", f"{hist_df[hist_df['Prediction'] == 'fraude par carte'].shape[0]}")
            ph_kpi7.metric("Fraude par compte mule", f"{hist_df[hist_df['Prediction'] == 'fraude par compte mule'].shape[0]}")
            ph_kpi8.metric("Fausses alertes", f"{(hist_df['isAnomaly'] == 'anomalie').sum() -(hist_df[hist_df['Prediction'] != 'légitime'].shape[0])}")
            
            # === Graphique 3: Scatter - Amount Paid vs Received ===
            fig3 = px.scatter(
                hist_df,
                x="Amount Paid", y="Amount Received",
                color="Prediction",
                size='Anomaly_Score',
                title="Montants par type de fraude",
                opacity=0.7,
                color_discrete_map=FRAUD_COLOR_MAP
            )
            ph3.plotly_chart(fig3, use_container_width=True, key=f'fig3_{step}')
            
            # === Graphique 4: bar - predict proba ===  
            # if 'last_predict_proba' not in st.session_state :
            #     st.session_state.last_predict_proba = [0.0,0.0,0.0] 
            
            if row['isAnomaly'] == 'anomalie':
                X_test, y_test, cat_cols = preprocessing_test(hist_df.tail(1))
                # st.session_state.last_predict_proba = lgbm_pipeline.predict_proba(X_test)[0]     
                last_predict_proba = lgbm_pipeline.predict_proba(X_test)[0]        
                 
            classes_encoded = lgbm_pipeline.classes_ 
            classes = label_encoder.inverse_transform(classes_encoded)
            
                
            df_proba = pd.DataFrame({
                'Type de Fraude': classes,
                'Probabilité': last_predict_proba
            })

            # Création du graphique
            fig4 = px.bar(df_proba,
                        x='Type de Fraude',
                        y='Probabilité',
                        color='Type de Fraude',
                        title="Probabilité d'appartenance aux classes",
                        color_discrete_map = FRAUD_COLOR_MAP
                        )
                        

            # Personnalisation
            fig4.update_layout(
                showlegend=False,
                yaxis=dict(range=[0, 1]),
                xaxis_title=None,
                yaxis_title="Probabilité prédite"
            )

            ph4.plotly_chart(fig4, use_container_width=True,key=f'fig4_{step}')
                  
            # fig4 = px.bar(
            #     hist_df['Payment Currency'].value_counts(),
            #     x=hist_df['Payment Currency'].value_counts().index, y=hist_df['Payment Currency'].value_counts().values,
            #     labels={'x': 'Devise', 'y': 'Nombre'},
            #     title="Devises les plus utilisées"
            # )
            # ph4.plotly_chart(fig4, use_container_width=True,key=f'fig4_{step}')
            
        with tabs[3]:
            if 'Prediction' in hist_df.columns and 'Pseudo_Labels' in hist_df.columns:
                try:
                    y_true_hist = label_encoder.transform(hist_df["Pseudo_Labels"])
                    y_pred_hist = label_encoder.transform(hist_df["Prediction"])
                    X_hist, y_hist, cat_cols = preprocessing_test(hist_df)
                    y_proba = lgbm_pipeline.predict_proba(X_hist)

                    # AUC macro et pondérée
                    auc_macro = roc_auc_score(y_true_hist, y_proba, multi_class='ovr', average='macro')
                    auc_weighted = roc_auc_score(y_true_hist, y_proba, multi_class='ovr', average='weighted')
                    
                    # Déterminer le delta
                    if prev_auc_macro is not None:
                        delta_auc_macro = auc_macro - prev_auc_macro
                    else:
                        delta_auc_macro = 0.0  
                        
                    if prev_auc_weighted is not None:
                        delta_auc_weighted = auc_weighted - prev_auc_weighted
                    else:
                        delta_auc_weighted = 0.0  

                    ph_auc_macro.metric("AUC macro", f"{auc_macro:.4f}", f"{delta_auc_macro:+.4f}")
                    ph_auc_weighted.metric("AUC pondérée", f"{auc_weighted:.4f}", f"{delta_auc_weighted:+.4f}")
                    
                    prev_auc_macro = auc_macro
                    prev_auc_weighted = auc_weighted

                    # ROC Curve
                    fpr = dict()
                    tpr = dict()
                    n_classes = len(label_encoder.classes_)
                    for i in range(n_classes):
                        fpr[i], tpr[i], _ = roc_curve((y_true_hist == i).astype(int), y_proba[:, i])

                    fig_roc = go.Figure()
                    for i in range(n_classes):
                        fig_roc.add_trace(go.Scatter(
                            x=fpr[i], y=tpr[i],
                            mode='lines',
                            name=f"{label_encoder.classes_[i]}",
                            line=dict(color = FRAUD_COLOR_MAP.get(label_encoder.classes_[i], 'blue') )                           
                        ))

                    fig_roc.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1],
                        mode='lines',
                        name="Hasard",
                        line=dict(dash='dash', color='gray')
                    ))

                    fig_roc.update_layout(
                        title="Courbe ROC multi-classe",
                        xaxis_title="Taux de faux positifs (FPR)",
                        yaxis_title="Taux de vrais positifs (TPR)", 
                    )

                    ph5.plotly_chart(fig_roc, use_container_width=True, key=f'roc_{step}')

                except Exception as e:
                    
                    print(f"⚠️ Erreur ROC/AUC à l'étape {step} : {e}")
                                    
        
        # Pause entre chaque itération
        time.sleep(speed)
  
        






