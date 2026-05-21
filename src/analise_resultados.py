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

# Arquivo com todas as funcoes e codigos referentes a analise de resultados

# ################################################################
# PROJETO FINAL
# (Seus dados de cabeçalho)
# Nome: Juan Pedro
# RA: 823164
# ################################################################

# Arquivo com todas as funcoes e codigos referentes à analise de resultados

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, wilcoxon
import itertools
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.inspection import permutation_importance

def plot_statistical_comparison(cv_results_dict, tuned_scores_series):
    """
    Realiza o teste de Friedman e testes post-hoc de Wilcoxon
    para comparar estatisticamente os modelos.
    
    Recebe o dicionário de scores (cv_results) e a série de médias (tuned_scores_series).
    """
    print("\n--- Análise Estatística dos Modelos (Teste de Friedman) ---")
    
    df_scores = pd.DataFrame(cv_results_dict)
    
    # 1. Teste de Friedman (Verifica se há alguma diferença)
    stat, p_friedman = friedmanchisquare(*[df_scores[col] for col in df_scores.columns])
    
    print(f"Estatística de Friedman: {stat:.3f}, p-valor: {p_friedman:.3f}")
    
    if p_friedman >= 0.05:
        print("Resultado (Friedman): p-valor >= 0.05. NÃO há diferença estatística significativa entre os modelos.")
        return

    print("Resultado (Friedman): p-valor < 0.05. HÁ diferença estatística significativa.")
    print("\n--- Teste Post-hoc (Wilcoxon Signed-Rank) ---")
    print("Comparando os 3 principais modelos (do CV Local):")
    print("(p < 0.05 indica diferença estatisticamente significativa)")

    # 2. Teste Post-hoc de Wilcoxon (Compara os pares)
    # Pega os 3 melhores modelos (baseado nas médias do tuned_scores_series)
    top_models = tuned_scores_series.index[:3].tolist()
    
    # Cria combinações de pares (ex: (LR, SVM), (LR, MLP), (SVM, MLP))
    pairs = list(itertools.combinations(top_models, 2))
    
    for model1, model2 in pairs:
        scores1 = df_scores[model1]
        scores2 = df_scores[model2]
        
        try:
            # O Wilcoxon precisa que os scores não sejam idênticos
            if (scores1 == scores2).all():
                print(f"  {model1} vs {model2}: Scores são idênticos (p=1.0)")
                continue
                
            stat_wilcoxon, p_wilcoxon = wilcoxon(scores1, scores2)
            
            print(f"  {model1} vs {model2}: p-valor = {p_wilcoxon:.4f}", 
                  "(Significativo)" if p_wilcoxon < 0.05 else "(Não significativo)")

        except ValueError as e:
            # Captura erro se o Wilcoxon não puder ser calculado
            print(f"  Não foi possível comparar {model1} vs {model2}: {e}")

    print("\nInterpretação: Se o p-valor (Wilcoxon) for < 0.05, os dois modelos têm desempenho estatisticamente diferente.")


def plot_confusion_matrix(model, X_train, y, display_labels):
    """
    Gera a Matriz de Confusão usando predições do cross_val_predict.
    """
    print("\n--- Análise Individual (Matriz de Confusão) ---")
    print(f"Gerando previsões K-Fold para o modelo: {model.__class__.__name__}")

    # 1. Gerar predições "out-of-fold"
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X_train, y, cv=kf)
    
    # 2. Gerar a Matriz de Confusão
    print("Matriz de Confusão (Linhas=Verdadeiro, Colunas=Previsto):")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true=y, 
        y_pred=y_pred, 
        ax=ax, 
        display_labels=display_labels,
        cmap='Blues',
        normalize='true' # Mostra em percentual
    )
    ax.set_title(f'Matriz de Confusão Normalizada\n{model.__class__.__name__}')
    plt.show()


def plot_feature_importance(model, X_train, y, feature_names, n_top_features=20):
    """
    Calcula e plota a importância das features usando Permutation Importance.
    """
    print("\n--- Análise de Importância das Features (Permutation Importance) ---")
    
    # 1. Calcular a importância
    result = permutation_importance(
        model, X_train, y, n_repeats=5, random_state=42, n_jobs=-1,
        scoring='roc_auc_ovr'
    )
    
    # 2. Organizar os resultados
    perm_sorted_idx = result.importances_mean.argsort()
    # Pega as N features mais importantes
    n_top_features = min(n_top_features, len(feature_names))
    top_idx = perm_sorted_idx[-n_top_features:] 
    
    top_features_names = [feature_names[i] for i in top_idx]
    top_importances = result.importances_mean[top_idx]
    
    # 3. Plotar
    fig, ax = plt.subplots(figsize=(10, n_top_features / 2.5))
    ax.barh(top_features_names, top_importances, 
            xerr=result.importances_std[top_idx], align='center')
    
    ax.set_title(f'Importância das Features (Permutation)\nModelo: {model.__class__.__name__}')
    ax.set_xlabel('Redução Média na AUC (ROC OVR)')
    plt.tight_layout()
    plt.show()