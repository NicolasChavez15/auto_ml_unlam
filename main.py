"""
===============================================================================
PROYECTO FINAL INTEGRADOR: APRENDIZAJE AUTOMÁTICO (MACHINE LEARNING)
===============================================================================
Materia: Machine Learning / Especialización en Ciencia de Datos
Título: Implementación End-to-End de AutoML para Predicción de Churn
Dominio: Telecomunicaciones / Retención de Clientes
===============================================================================
"""

import pandas as pd
import numpy as np
import warnings
import os

# Desactivar advertencias innecesarias en la consola de Salida de VS Code
warnings.filterwarnings('ignore')

def main():
    print("=" * 75)
    print("INICIANDO EJECUCIÓN DEL PIPELINE END-TO-END DE MACHINE LEARNING")
    print("=" * 75)

    # -------------------------------------------------------------------------
    # FASE 1: PLANIFICACIÓN Y OBTENCIÓN DE DATOS (ETAPAS 1 a 5)
    # -------------------------------------------------------------------------
    print("\n[ETAPA 3-5] Extracción y Carga de Datos desde Repositorio de Telecomunicaciones...")
    
    # URL pública del conjunto de datos estandarizado de Telecomunicaciones (IBM Telco Churn)
    url_dataset = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"    
    try:
        df = pd.read_csv(url_dataset)
        print(f"-> Datos cargados exitosamente: {df.shape[0]} filas y {df.shape[1]} columnas.")
    except Exception as e:
        print(f"Error al descargar los datos: {e}")
        return

    # -------------------------------------------------------------------------
    # FASE 2: DESARROLLO DEL MODELO (ETAPAS 6 a 10)
    # -------------------------------------------------------------------------
    print("\n[ETAPA 7] Limpieza y Preprocesamiento de Datos...")
    
    # Imputación y conversión de tipo de datos para TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].str.strip(), errors='coerce')
    monto_mediana = df['TotalCharges'].median()
    df['TotalCharges'].fillna(monto_mediana, inplace=True)
    
    # Mapeo de la variable objetivo (Churn) a formato binario (1: Sí, 0: No)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Eliminación de identificador primario no correlacionado
    df_model = df.drop(columns=['customerID'])

    print("\n[ETAPA 8] Ingeniería de Características (Feature Engineering)...")
    # 1. Ratio de facturación acumulada por mes de antigüedad
    df_model['Charges_per_Tenure'] = df_model['MonthlyCharges'] / (df_model['tenure'] + 1)
    
    # 2. Conteo total de servicios adicionales contratados por el cliente
    servicios_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df_model['Service_Count'] = (df_model[servicios_cols] == 'Yes').sum(axis=1)

    print(f"-> Dataset procesado: {df_model.shape[1]} variables listas para AutoML.")

    # -------------------------------------------------------------------------
    # ETAPAS 9 y 10: SELECCIÓN DE ALGORITMOS Y MÉTRICAS CON AUTOML (PYCARET)
    # -------------------------------------------------------------------------
    print("\n[ETAPA 9] Configuración e Invocación del Motor de AutoML (PyCaret)...")
    
    from pycaret.classification import (setup, compare_models, blend_models, 
                                       tune_model, finalize_model, predict_model, 
                                       save_model)

    # Inicialización del entorno de AutoML con validación cruzada y preprocesamiento automático
    exp_setup = setup(
        data=df_model, 
        target='Churn', 
        session_id=42,
        train_size=0.8,
        verbose=False
    )
    print("-> Entorno PyCaret configurado correctamente.")

    print("\n[ETAPA 9.1] Evaluación Competitiva de Algoritmos...")
    # Entrena y evalúa múltiples clasificadores ordenados por métrica ROC-AUC
    top3_modelos = compare_models(n_select=3, sort='AUC')

    print("\n[ETAPA 9.2] Construcción del Ensamble (Blending)...")
    # Creación de un modelo ensamblado combinando los 3 mejores algoritmos
    modelo_ensamblado = blend_models(estimator_list=top3_modelos, optimize='AUC')

    print("\n[ETapay 9.3] Optimización de Hiperparámetros (AutoHPO)...")
    # Búsqueda bayesiana / aleatoria de hiperparámetros óptimos
    modelo_optimizado = tune_model(modelo_ensamblado, optimize='AUC', n_iter=10)

    print("\n[ETAPA 10] Entrenamiento Final del Modelo Registrado...")
    # Entrena el pipeline optimizado utilizando la totalidad del conjunto de entrenamiento
    pipeline_final = finalize_model(modelo_optimizado)

 # -------------------------------------------------------------------------
    # FASE 3: DESPLIEGUE, MLOPS Y MONITOREO (ETAPAS 11 a 15)
    # -------------------------------------------------------------------------
    print("\n[ETAPA 10.1] Evaluación en Datos Hold-Out de Prueba...")
    
    # Obtenemos el conjunto de prueba (X_test e y_test) directamente desde el entorno de PyCaret
    from pycaret.classification import get_config
    X_test_holdout = get_config('X_test')
    y_test_holdout = get_config('y_test')

    # Predicciones utilizando el pipeline final de PyCaret
    y_pred = pipeline_final.predict(X_test_holdout)
    y_prob = pipeline_final.predict_proba(X_test_holdout)[:, 1] # Probabilidad real de Churn = 1

    from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

    print("\n" + "=" * 50)
    print("RESULTADOS DE PERFORMANCE EN PRODUCCIÓN SIMULADA")
    print("=" * 50)
    print("Matriz de Confusión:")
    print(confusion_matrix(y_test_holdout, y_pred))
    
    print("\nReporte de Clasificación (Precision / Recall / F1):")
    print(classification_report(y_test_holdout, y_pred))

    auc_valor = roc_auc_score(y_test_holdout, y_prob)
    print(f"ROC-AUC Score Obtenido: {auc_valor:.4f}")

    print("\n[ETAPA 12] MLOps: Serialización y Guardado del Artefacto...")
    nombre_modelo = 'modelo_churn_telecom_automl'
    save_model(pipeline_final, nombre_modelo)
    print(f"-> Artefacto exportado exitosamente como '{nombre_modelo}.pkl'.")

    print("\n[ETAPA 14 & 15] Evaluación de Deriva (Drift) y Políticas de Reentrenamiento...")
    # Simulación del control de calidad continuo del modelo en producción
    umbral_minimo_auc = 0.75
    if auc_valor < umbral_minimo_auc:
        print(f"[ALERTA DRIFT] El ROC-AUC ({auc_valor:.4f}) cayó por debajo del límite permitido ({umbral_minimo_auc}). Disparando Reentrenamiento...")
    else:
        print(f"[ESTADO OK] El ROC-AUC ({auc_valor:.4f}) cumple con los estándares operativos (> {umbral_minimo_auc}). ¡Modelo aprobado para despliegue!")

    print("\n" + "=" * 75)
    print("EJECUCIÓN DEL PROYECTO FINALIZADA CON ÉXITO")
    print("=" * 75)

if __name__ == '__main__':
    main()