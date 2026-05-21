# ################################################################
# PROJETO FINAL
#
# Universidade Federal de Sao Carlos (UFSCAR)
# Departamento de Computacao - Sorocaba (DComp-So)
# Disciplina: Aprendizado de Maquina
# Prof. Tiago A. Almeida
#
#
# Nome: Juan Pedro
# RA: 823164
# ################################################################

# Arquivo com todas as funcoes e codigos referentes ao preprocessamento
import pandas as pd
import os
import numpy as np
from pandas.errors import EmptyDataError
from tqdm import tqdm 
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

def load_sensor_data(folder_path, file_name, column_names):
    """
    Carrega dados de um sensor específico.
    """
    file_path = os.path.join(folder_path, file_name)
    
    if not os.path.exists(file_path):
        return pd.DataFrame()
        
    try:
        # Esta linha pode falhar se o arquivo estiver vazio
        df = pd.read_csv(file_path, header=None)
        
    except EmptyDataError:
        # Se o arquivo estiver vazio, captura o erro e retorne um DataFrame vazio.
        return pd.DataFrame()

    # Evitar erro se o arquivo estiver vazio (verificação dupla)
    if df.empty:
        return pd.DataFrame()
        
    # --- Linha 1: Processamento do Timestamp Inicial ---
    raw_timestamp = df.iloc[0, 0]
    start_datetime = pd.NaT 
    start_datetime = pd.to_datetime(raw_timestamp)
            
    # === Início da lógica IBI vs. Outros ===
    
    if file_name == 'IBI.csv':
        data = df.iloc[1:].copy()
        if data.empty:
             return pd.DataFrame()
        data.columns = ['Offset_sec', 'IBI']
        data['Offset_sec'] = pd.to_numeric(data['Offset_sec'], errors='coerce')
        data['IBI'] = pd.to_numeric(data['IBI'], errors='coerce')
        data['datetime'] = start_datetime + pd.to_timedelta(data['Offset_sec'], unit='s')
        
    else:
        # === Lógica para EDA, ACC, HR, BVP, TEMP, etc. ===
        if len(df) < 3: 
            return pd.DataFrame()
            
        raw_freq = df.iloc[1, 0]
        sampling_freq = pd.to_numeric(raw_freq, errors='coerce') 
        data = df.iloc[2:].copy()
        if data.empty:
            return pd.DataFrame()
            
        data.columns = column_names
        
        if pd.isna(sampling_freq) or sampling_freq <= 0:
             data['datetime'] = pd.NaT
        else:
             num_samples = len(data)
             time_seconds = [i / sampling_freq for i in range(num_samples)]
             data['datetime'] = start_datetime + pd.to_timedelta(time_seconds, unit='s')

        for col in column_names:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    data = data.dropna().reset_index(drop=True)
            
    return data


def extract_features(user_id, base_folder="wearables/"):
    """
    Carrega todos os dados de um sujeito e calcula um dicionário de features.
    """
    user_folder = os.path.join(base_folder, user_id)
    
    # 1. Carregar todos os dados do sensor
    df_hr = load_sensor_data(user_folder, 'HR.csv', ['HR'])
    df_eda = load_sensor_data(user_folder, 'EDA.csv', ['EDA'])
    df_temp = load_sensor_data(user_folder, 'TEMP.csv', ['TEMP'])
    df_acc = load_sensor_data(user_folder, 'ACC.csv', ['X', 'Y', 'Z'])
    df_ibi = load_sensor_data(user_folder, 'IBI.csv', ['IBI'])
    df_bvp = load_sensor_data(user_folder, 'BVP.csv', ['BVP'])
    
    # 2. Pré-cálculos (ex: magnitude do Acelerômetro)
    if not df_acc.empty:
        df_acc['Mag'] = np.sqrt(df_acc['X']**2 + df_acc['Y']**2 + df_acc['Z']**2)
    
    # 3. Dicionário de Features
    features = {'Id': user_id}

    # 4. Função Auxiliar para calcular estatísticas
    def get_stats(data, col_name):
        """Calcula estatísticas para uma série temporal e retorna como um dicionário."""
        # Se o DataFrame estiver vazio (arquivo não existia), retorna NaN
        if data.empty or col_name not in data.columns:
            stats = {
                f'{col_name}_mean': np.nan, f'{col_name}_std': np.nan,
                f'{col_name}_min': np.nan, f'{col_name}_max': np.nan,
                f'{col_name}_q25': np.nan, f'{col_name}_q75': np.nan
            }
            return stats
        
        series = data[col_name]
        q = series.quantile([0.25, 0.75])
        
        stats = {
            f'{col_name}_mean': series.mean(),
            f'{col_name}_std': series.std(),
            f'{col_name}_min': series.min(),
            f'{col_name}_max': series.max(),
            f'{col_name}_q25': q.get(0.25, np.nan),
            f'{col_name}_q75': q.get(0.75, np.nan),
        }
        return stats

    # 5. Extrair features de cada sinal
    features.update(get_stats(df_hr, 'HR'))
    features.update(get_stats(df_eda, 'EDA'))
    features.update(get_stats(df_temp, 'TEMP'))
    features.update(get_stats(df_acc, 'Mag'))
    features.update(get_stats(df_ibi, 'IBI'))
    features.update(get_stats(df_bvp, 'BVP'))

    # Mede a variação "batimento a batimento"
    if not df_ibi.empty and len(df_ibi['IBI']) > 1:
        features['IBI_rmssd'] = np.sqrt(np.mean(np.diff(df_ibi['IBI'])**2))
    else:
        features['IBI_rmssd'] = np.nan
    
    return features


def create_feature_matrices(df_train_labels, df_test_ids, base_folder):
    """Itera sobre os IDs de treino e teste para construir as matrizes de features."""
    
    # Processar Treino
    all_features_train = []
    print("Processando dados de TREINO...")
    for user_id in tqdm(df_train_labels['Id']):
        all_features_train.append(extract_features(user_id, base_folder=base_folder))
    
    # Processar Teste
    all_features_test = []
    print("Processando dados de TESTE...")
    for user_id in tqdm(df_test_ids['Id']):
        all_features_test.append(extract_features(user_id, base_folder=base_folder))

    # Criar DataFrames
    X_raw = pd.DataFrame(all_features_train).set_index('Id')
    X_test_raw = pd.DataFrame(all_features_test).set_index('Id')
    
    # Alinhar rótulos
    y_labels = df_train_labels.set_index('Id')['Label']
    
    return X_raw, y_labels, X_test_raw


def plot_feature_outliers_by_class(X_raw_imputed_df, y_labels_series):
    """
    Plota boxplots das principais features, agrupadas por classe, para identificar outliers.
    Usa o X_raw (imputado) e os y_labels (antes do LabelEncoder).
    """
    
    # Juntar features e labels para plotting
    data = X_raw_imputed_df.copy()
    data['Label'] = y_labels_series.values
    
    # Vamos focar nas features de média e no RMSSD para a visualização
    features_to_plot = [
        'HR_mean', 'EDA_mean', 'TEMP_mean', 
        'Mag_mean', 'BVP_mean', 'IBI_mean', 'IBI_rmssd'
    ]
    
    print("\n--- Análise Visual de Outliers (Features vs. Classe) ---")
    
    n_features = len(features_to_plot)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols # Arredonda para cima
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten() # Facilita a iteração

    for i, feature in enumerate(features_to_plot):
        if feature not in data.columns:
            print(f"Aviso: Feature '{feature}' não encontrada. Pulando.")
            continue
        
        # Plota o boxplot agrupado por classe
        sns.boxplot(x='Label', y=feature, data=data, ax=axes[i], 
                    order=['STRESS', 'AEROBIC', 'ANAEROBIC']) # Usa a ordem original das classes
        axes[i].set_title(f'Distribuição de {feature} por Classe')
    
    # Esconder eixos não utilizados
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
        
    plt.tight_layout()
    plt.show()


def apply_sklearn_pipeline(X_raw, y_labels, X_test_raw):
    """
    Aplica o pipeline de LabelEncoding, Imputação e Escalonamento Robusto.
    Retorna os dados imputados (para plot) e os dados escalonados (para treino).
    """
    
    # 1. Codificar o Alvo (y)
    le = LabelEncoder()
    y = le.fit_transform(y_labels)
    print("\nClasses do LabelEncoder:", le.classes_)
    print("Shape de y (alvo):", y.shape)

    # 2. Lidar com Valores Ausentes (NaN)
    imputer = SimpleImputer(strategy='median')
    
    # Salvar colunas e índices para recriar o DataFrame
    X_raw_cols = X_raw.columns
    X_raw_index = X_raw.index
    X_test_raw_cols = X_test_raw.columns
    X_test_raw_index = X_test_raw.index

    X_train_imputed = imputer.fit_transform(X_raw)
    X_test_imputed = imputer.transform(X_test_raw)
    
    # Recriar os DataFrames imputados (para visualização de outliers)
    X_train_imputed_df = pd.DataFrame(X_train_imputed, columns=X_raw_cols, index=X_raw_index)
    X_test_imputed_df = pd.DataFrame(X_test_imputed, columns=X_test_raw_cols, index=X_test_raw_index)
    print("\nTratamento de Valores Ausentes (Imputação) Concluído.")

    # 3. Escalonamento Robusto (Melhor para outliers)
    scaler = RobustScaler() 
    X_train = scaler.fit_transform(X_train_imputed_df)
    X_test = scaler.transform(X_test_imputed_df)
    
    print("Escalonamento (RobustScaler) Concluído!")
    print("Shape final de X_train:", X_train.shape)
    print("Shape final de X_test:", X_test.shape)

    # Retornamos os dados processados e o LabelEncoder e o DF imputado
    return X_train, y, X_test, le, X_train_imputed_df