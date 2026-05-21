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

# Arquivo com todas as funcoes e codigos referentes aos experimentos

import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import GridSearchCV

def get_models():
    """Retorna um dicionário dos modelos base a serem avaliados."""
    models = {
        'Regressão Logística': LogisticRegression(max_iter=1000, random_state=42),
        'k-Vizinhos (KNN)': KNeighborsClassifier(),
        'Naive Bayes (Gauss.)': GaussianNB(),
        'SVM (RBF Kernel)': SVC(kernel='rbf', probability=True, random_state=42),
        'Rede Neural (MLP)': MLPClassifier(max_iter=500, random_state=42),
        'XGBoost': xgb.XGBClassifier(objective='multi:softprob', 
                                      eval_metric='mlogloss', 
                                      random_state=42)
    }
    return models


def get_param_grids():
    """Retorna as grades de parâmetros para o GridSearchCV."""
    param_grids = {
        'Regressão Logística': {
            'C': [0.1, 1.0, 10, 100],
            'solver': ['liblinear']
        },
        'k-Vizinhos (KNN)': {
            'n_neighbors': [3, 5, 7, 9, 11]
        },
        'Naive Bayes (Gauss.)': {
            'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
        },
        'SVM (RBF Kernel)': {
            'C': [0.1, 1, 10, 100],
            'gamma': [0.1, 0.01, 0.001, 'scale']
        },
        'Rede Neural (MLP)': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'alpha': [0.0001, 0.001, 0.01]
        },
        'XGBoost': {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1]
        }
    }
    return param_grids



def tune_all_models(X_train, y, models, param_grids):
    """
    Realiza o GridSearchCV para todos os modelos fornecidos.
    Retorna um dict de estimadores já treinados e um dict de scores médios.
    """
    print("\n--- Iniciando Ajuste Fino de Hiperparâmetros (GridSearchCV) ---")
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    tuned_models = {}
    tuned_scores = {}

    for name, model in models.items():
        if name not in param_grids:
            print(f"Aviso: Modelo {name} não possui grade de parâmetros. Pulando.")
            # Para modelos sem parâmetros (como Naive Bayes padrão), 
            # podemos simplesmente treiná-lo.
            if name == 'Naive Bayes (Gauss.)':
                 print(f"Treinando: {name} (sem grade)...")
                 model.fit(X_train, y)
                 tuned_models[name] = model
                 tuned_scores[name] = np.mean(cross_val_score(model, X_train, y, cv=kf, scoring='roc_auc_ovr'))
            continue
            
        print(f"Ajustando: {name}...")
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grids[name],
            cv=kf,
            scoring='roc_auc_ovr',
            n_jobs=-1,
            refit=True
        )
        grid_search.fit(X_train, y)
        
        print(f"  Melhores Parâmetros: {grid_search.best_params_}")

        tuned_models[name] = grid_search.best_estimator_
        tuned_scores[name] = grid_search.best_score_

    print(f"\n--- Ajuste Fino Concluído ---")
    
    return tuned_models, tuned_scores


def run_cross_validation(X_train, y, tuned_models):
    """
    Executa K-Fold nos modelos já otimizados para gerar scores para o boxplot.
    """
    print("\nIniciando Validação Cruzada (K-Fold) nos modelos OTIMIZADOS...")

    # Usamos o mesmo KFold para garantir uma comparação justa
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_results = {}
    cv_results_mean = {}

    for model_name, model in tuned_models.items():
        print(f"Avaliando: {model_name} (Otimizado)...")
        scores = cross_val_score(model, X_train, y, 
                                 cv=kf, 
                                 scoring='roc_auc_ovr', 
                                 n_jobs=-1)
        cv_results[model_name] = scores
        cv_results_mean[model_name] = scores.mean()

    print("\nValidação Cruzada Concluída.")
    
    return cv_results_mean, cv_results


def plot_cv_results(cv_results, cv_results_mean):
    """Plota o boxplot dos resultados da validação cruzada."""
    
    # Ordena os resultados pela média para o plot
    mean_scores_sorted = pd.Series(cv_results_mean).sort_values(ascending=False)
    
    print("\n--- Média da AUC (ROC One-vs-Rest) de cada Modelo Otimizado (Ordenado) ---")
    print(mean_scores_sorted)
    
    # Cria o DataFrame para o boxplot na ordem correta
    df_cv_results = pd.DataFrame(cv_results)
    df_cv_results_sorted = df_cv_results[mean_scores_sorted.index]

    plt.figure(figsize=(12, 7))
    sns.boxplot(data=df_cv_results_sorted)
    plt.title('Comparação da AUC (ROC OVR) entre Modelos Otimizados (CV=5)')
    plt.ylabel('AUC Score')
    plt.xticks(rotation=45, ha='right')
    plt.show()
    
    # Retorna o nome do melhor modelo
    return mean_scores_sorted, mean_scores_sorted.idxmax()


def generate_submission_file(best_model_object, best_model_name, X_test, 
                                le, df_test_ids, target_order, submission_filename):
    """
    Usa o modelo já treinado (best_model_object) para gerar o arquivo de submissão.
    """
    
    # O modelo (best_model_object) já foi treinado pelo GridSearchCV
    print(f"\n--- Usando o Modelo Vencedor Otimizado ({best_model_name}) ---")

    # Gerar probabilidades
    y_pred_proba = best_model_object.predict_proba(X_test)
    print("Previsões (probabilidades) geradas para o conjunto de teste.")

    # Reordenar colunas
    model_order = list(le.classes_) 
    print(f"Ordem do Modelo: {model_order}")
    print(f"Ordem Alvo: {target_order}")
    
    try:
        col_indices = [model_order.index(target) for target in target_order]
        y_pred_ordered = y_pred_proba[:, col_indices]
        
        # Criar DataFrame
        df_submission = pd.DataFrame(
            y_pred_ordered,
            columns=[f'Predicted_{i}' for i in range(len(target_order))]
        )
        df_submission['Id'] = df_test_ids['Id'].values
        df_submission = df_submission[['Id'] + [col for col in df_submission if col != 'Id']]

        # Salvar
        df_submission.to_csv(submission_filename, index=False)
        print(f"\nArquivo de submissão '{submission_filename}' salvo com sucesso!")
        print(df_submission.head())
        
    except ValueError as e:
        print(f"\nERRO: Uma das classes em {target_order} não foi encontrada em {model_order}.")
        print(e)