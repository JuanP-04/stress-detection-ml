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

# Arquivo com todas as funcoes e codigos referentes a analise exploratoria

import pandas as pd
import os
from pandas.errors import EmptyDataError
import matplotlib.pyplot as plt
import seaborn as sns

def plot_demographics(df_users):
    """Gera os 4 gráficos de análise demográfica."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Análise Demográfica dos Sujeitos', fontsize=16)

    # Histograma de Idade
    sns.histplot(df_users['Age'].dropna(), kde=True, ax=axes[0, 0], bins=10)
    axes[0, 0].set_title('Distribuição de Idade')

    # Gráfico de Gênero
    sns.countplot(x='Gender', data=df_users, ax=axes[0, 1])
    axes[0, 1].set_title('Distribuição de Gênero')

    # Gráfico de Prática de Atividade Física
    sns.countplot(x='Does physical activity regularly?', data=df_users, ax=axes[1, 0])
    axes[1, 0].set_title('Pratica Atividade Física Regularmente?')

    # Gráfico de Protocolo
    sns.countplot(x='Protocol', data=df_users, ax=axes[1, 1])
    axes[1, 1].set_title('Protocolo do Experimento')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


    # Em analise_exploratoria.py

def plot_sample_signals(df_hr, df_eda, df_temp, df_acc, df_bvp, df_ibi):
    """Plota os gráficos dos diferentes sinais de um usuário de exemplo."""
    
    # --- Gráfico 1: HR, EDA, TEMP (Combinados) ---
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    fig.suptitle('Sinais Fisiológicos (Exemplo U_79201)', fontsize=16)

    if not df_hr.empty:
        axes[0].plot(df_hr['datetime'], df_hr['HR'], color='red')
        axes[0].set_title('Frequência Cardíaca (HR)')
        axes[0].set_ylabel('BPM')
    else:
        axes[0].set_title('Frequência Cardíaca (HR) - Dados N/A')

    if not df_eda.empty:
        axes[1].plot(df_eda['datetime'], df_eda['EDA'], color='blue')
        axes[1].set_title('Atividade Eletrodérmica (EDA)')
        axes[1].set_ylabel('microsiemens (uS)')
    else:
        axes[1].set_title('Atividade Eletrodérmica (EDA) - Dados N/A')

    if not df_temp.empty:
        axes[2].plot(df_temp['datetime'], df_temp['TEMP'], color='green')
        axes[2].set_title('Temperatura da Pele (TEMP)')
        axes[2].set_ylabel('Celsius (°C)')
    else:
        axes[2].set_title('Temperatura da Pele (TEMP) - Dados N/A')
    
    plt.xlabel('Tempo')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # --- Gráfico 2: Acelerômetro ---
    if not df_acc.empty:
        fig_acc, ax_acc = plt.subplots(figsize=(15, 5))
        plot_data_acc = df_acc.head(19200) # 10 min
        ax_acc.plot(plot_data_acc['datetime'], plot_data_acc['X'], label='X')
        ax_acc.plot(plot_data_acc['datetime'], plot_data_acc['Y'], label='Y')
        ax_acc.plot(plot_data_acc['datetime'], plot_data_acc['Z'], label='Z')
        ax_acc.set_title('Acelerômetro (ACC) - (Primeiros 10 min)')
        ax_acc.set_ylabel('1/64g')
        ax_acc.legend()
        plt.xlabel('Tempo')
        plt.show()

    # --- Gráfico 3: BVP ---
    if not df_bvp.empty:
        fig_bvp, ax_bvp = plt.subplots(figsize=(15, 5))
        plot_data_bvp = df_bvp.head(640) # 10 seg
        ax_bvp.plot(plot_data_bvp['datetime'], plot_data_bvp['BVP'], color='purple')
        ax_bvp.set_title('Pulso de Volume Sanguíneo (BVP) - (Primeiros 10 seg)')
        ax_bvp.set_ylabel('Amplitude')
        plt.xlabel('Tempo')
        plt.show()

    # --- Gráfico 4: IBI ---
    if not df_ibi.empty:
        fig_ibi, ax_ibi = plt.subplots(figsize=(15, 5))
        ax_ibi.plot(df_ibi['datetime'], df_ibi['IBI'], 
                    linestyle='none', marker='o', markersize=2, color='black')
        ax_ibi.set_title('Intervalo Inter-Batimento (IBI)')
        ax_ibi.set_ylabel('Segundos (s)')
        plt.xlabel('Tempo')
        plt.show()


def plot_signals_with_tags(df_hr, df_eda, USER_ID, USER_FOLDER):
    """
    Plota os sinais de HR e EDA sobrepostos com os marcadores de evento (tags).
    """
    tags_path = os.path.join(USER_FOLDER, 'tags.csv')
    
    if not os.path.exists(tags_path):
        print(f"Arquivo tags.csv não encontrado em {USER_FOLDER}")
        return

    try:
        df_tags = pd.read_csv(tags_path, header=None, names=['timestamp'])
        if df_tags.empty:
             print(f"Arquivo tags.csv está vazio em {USER_FOLDER}")
             return
    except EmptyDataError:
        print(f"Arquivo tags.csv está vazio em {USER_FOLDER}")
        return

    tag_datetimes = pd.to_datetime(df_tags['timestamp'])
    
    print(f"\n--- {len(tag_datetimes)} Marcadores de Evento encontrados para {USER_ID} ---")
    
    # --- Plotar HR e EDA com os Marcadores de Evento ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # Plotar HR
    if not df_hr.empty:
        ax1.plot(df_hr['datetime'], df_hr['HR'], color='red', label='HR (Freq. Cardíaca)')
        ax1.set_ylabel('BPM')
    
    # Plotar EDA
    if not df_eda.empty:
        ax2.plot(df_eda['datetime'], df_eda['EDA'], color='blue', label='EDA (Ativ. Eletrodérmica)')
        ax2.set_ylabel('microsiemens (uS)')

    # Adicionar linhas verticais para cada tag
    for tag_time in tag_datetimes:
        ax1.axvline(tag_time, color='black', linestyle='--', linewidth=1, label='Marcador de Evento')
        ax2.axvline(tag_time, color='black', linestyle='--', linewidth=1, label='Marcador de Evento')
    
    ax1.set_title(f'Sinais Fisiológicos com Marcadores de Evento ({USER_ID})')
    plt.xlabel('Tempo')
    
    # Lidar com legendas duplicadas
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys())
    
    handles, labels = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys())
    
    plt.tight_layout()
    plt.show()