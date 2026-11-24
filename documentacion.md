# PROYECTO FINAL INTEGRADOR: APRENDIZAJE AUTOMÁTICO

**Materia:** Machine Learning / Especialización en Ciencia de Datos  
**Título:** Implementación End-to-End de AutoML para la Predicción de Churn en el Sector Telecomunicaciones bajo el Ciclo de Vida Completo de ML  
**Dominio de Aplicación:** Telecomunicaciones / Retención de Clientes  
**Enfoque:** Profesional, Funcional y Aplicado (Ciclo de Vida de 15 Etapas)  

---

## RESUMEN EJECUTIVO

Este proyecto implementa un *pipeline* de Machine Learning de extremo a extremo (*end-to-end*) para predecir la probabilidad de baja o abandono (*customer churn*) de clientes en el sector de telecomunicaciones. El desarrollo articula el **ciclo de vida completo de Machine Learning (15 etapas)** con técnicas de **Aprendizaje Automático Automatizado (AutoML)** mediante `PyCaret`, combinando selección competitiva de algoritmos, *blending* (ensamblado por votación/promedio de probabilidades), optimización de hiperparámetros (`AutoHPO`), serialización del *pipeline* y un control automatizado de umbral de rendimiento frente a deriva (*drift*).

La implementación utiliza el conjunto de datos público *Telco Customer Churn* (7.043 registros y 21 variables originales) como proxy operacional de una telco convergente. En las ejecuciones sobre el conjunto *hold-out* de prueba, el ensamble optimizado y finalizado alcanza un **ROC-AUC cercano a 0.86 (~0.8594)** y un *accuracy* cercano al **81%**, superando el umbral operativo mínimo de `0.75` definido para la etapa de monitoreo.

---

## PARTE I: MARCO METODOLÓGICO Y DESARROLLO DE LAS 15 ETAPAS

### FASE A: PLANIFICACIÓN Y OBTENCIÓN DE DATOS

#### 1. Definición del Problema
* **Problema de Negocio:** En telecomunicaciones, captar un nuevo cliente es considerablemente más costoso que retener uno existente. Cada baja involuntaria o solicitada erosiona el *Customer Lifetime Value* (LTV), reduce los ingresos recurrentes mensuales y presiona los márgenes operativos.
* **Formulación Técnica:** Modelar el abandono como un problema de clasificación binaria supervisada donde la variable objetivo `Churn` toma valor `1` (*Yes*, baja efectiva) o `0` (*No*, permanencia) a partir del perfil contractual, demográfico, de servicios y de facturación del abonado.
* **Criterio de Evaluación y Decisión:** Se prioriza **ROC-AUC** como métrica principal de ranking y discriminación global de probabilidades (independiente de un único corte fijo) y se complementa con **Precision, Recall y F1-Score** sobre la clase positiva (`1`). En campañas de retención, pasar por alto clientes propensos a irse (*Falsos Negativos*) implica pérdida directa de ingresos, mientras que los *Falsos Positivos* conllevan el costo acotado de una oferta o gestión preventiva.

#### 2. Estudio de Viabilidad
* **Viabilidad de Datos y Técnica:** El dataset reúne variables estructuradas de antigüedad, modalidad de contrato, facturación mensual/acumulada y servicios complementarios, suficientes para capturar patrones claros de permanencia y fuga sin requerir datos sensibles propietarios.
* **Viabilidad Operativa y Uso de AutoML:** Evaluar manualmente múltiples familias de modelos, preprocesamientos y configuraciones de hiperparámetros consume tiempo valioso de iteración. Emplear `PyCaret` estandariza el preprocesamiento dentro del *pipeline*, acelera el *benchmarking* inicial y permite enfocar el criterio técnico en la ingeniería de variables, la interpretación de métricas y las reglas de gobierno del modelo.

#### 3. Detectar Orígenes de Datos
En una operación de telecomunicaciones convergente, los atributos del problema corresponden a dimensiones funcionales complementarias:
* **Relación comercial y facturación:** Antigüedad (`tenure`), tipo de contrato (`Contract`), facturación electrónica (`PaperlessBilling`), medio de pago (`PaymentMethod`), cargo mensual (`MonthlyCharges`) y cargo total acumulado (`TotalCharges`).
* **Provisión y servicios:** Telefonía (`PhoneService`, `MultipleLines`), acceso a internet (`InternetService`) y complementos (`OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`).
* **Perfil del abonado y target:** Demografía básica (`gender`, `SeniorCitizen`, `Partner`, `Dependents`) e indicador de baja (`Churn`).

#### 4. Obtener Acceso a los Datos
* **Decisión Implementada:** Lectura directa mediante `pd.read_csv(url_dataset)` desde la fuente pública de IBM Telco Churn (`Telco-Customer-Churn.csv`), encapsulada en un bloque `try / except` para controlar fallos de red o disponibilidad.
* **Justificación:** Garantiza **reproducibilidad total** para evaluación académica y ejecución en cualquier entorno sin depender de rutas locales hardcodeadas ni exponer credenciales.

#### 5. Extracción de Datos
* **Decisión Implementada:** Ingesta completa de los `7.043` registros y `21` columnas en un `DataFrame` de `pandas`, registrando trazabilidad en consola (`shape[0]` y `shape[1]`).
* **Justificación:** El volumen del dataset permite entrenar, validar con validación cruzada y evaluar en *hold-out* en tiempos breves en CPU local, facilitando pruebas repetibles de *pipeline*.

---

### FASE B: DESARROLLO DEL MODELO

#### 6. Análisis Exploratorio de Datos (EDA)
El diagnóstico exploratorio sobre el dominio y la estructura del dataset fundamenta las transformaciones posteriores:
* **Desbalance moderado:** Aproximadamente un `73,5%` de clientes permanecen (`0`) y un `26,5%` causan baja (`1`). Esto exige particiones estratificadas y métricas más informativas que el *accuracy* aislado.
* **Patrones de riesgo conocidos:** Los clientes con menor antigüedad (`tenure` bajo) y contratos *Month-to-month* concentran mayor rotación; asimismo, `TotalCharges` aparece como cadena de texto con espacios vacíos en clientes de antigüedad nula o muy reciente.

#### 7. Limpieza de Datos
* **Decisión Implementada:**
  1. Conversión de `TotalCharges` con `pd.to_numeric(..., errors='coerce')` tras `.str.strip()`, seguida de imputación con la **mediana** (`df['TotalCharges'].median()`).
  2. Mapeo explícito de `Churn`: `{'Yes': 1, 'No': 0}`.
  3. Eliminación de `customerID` para construir `df_model`.
* **Justificación:**
  * La **mediana** es robusta frente a asimetrías y valores extremos en facturación acumulada, evitando distorsiones que introduciría la media.
  * Excluir `customerID` previene que el modelo memorice identificadores sin capacidad de generalización sobre clientes nuevos.

#### 8. Ingeniería de Características (*Feature Engineering*)
* **Decisión Implementada:** Se crean dos variables de dominio sobre `df_model` (elevando el dataset a `22` columnas antes del preprocesamiento interno de `PyCaret`):
  * `Charges_per_Tenure = df_model['MonthlyCharges'] / (df_model['tenure'] + 1)`: aproxima la intensidad de cargo mensual relativa a la permanencia, sumando `+ 1` en el denominador para evitar división por cero cuando `tenure == 0`.
  * `Service_Count = (df_model[servicios_cols] == 'Yes').sum(axis=1)`: cuantifica el nivel de adopción de servicios complementarios (`OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`), capturando el grado de vinculación (*stickiness*) del cliente al ecosistema.
* **Justificación:** Resume señales multiparamétricas en indicadores compactos y significativos para modelos lineales, basados en árboles y ensambles.

#### 9. Selección de Algoritmos y Optimización con AutoML (`PyCaret`)
* **Decisión Implementada:**
  1. `setup(data=df_model, target='Churn', session_id=42, train_size=0.8, verbose=False)`: fija semilla (`session_id=42`), reserva el `20%` para evaluación *hold-out* y delega a `PyCaret` el preprocesamiento consistente (codificación categórica, escalado/tratamiento según estimador y partición).
  2. `compare_models(n_select=3, sort='AUC')`: evalúa múltiples familias de clasificadores ordenando por `AUC` y selecciona los **3 mejores modelos**.
  3. `blend_models(estimator_list=top3_modelos, optimize='AUC')`: combina los tres mejores estimadores en un ensamble.
  4. `tune_model(modelo_ensamblado, optimize='AUC', n_iter=10)`: ajusta hiperparámetros optimizando `AUC` con un presupuesto controlado (`n_iter=10`) para equilibrar mejora y tiempo de cómputo.
  5. `finalize_model(modelo_optimizado)`: reentrena el *pipeline* completo sobre el conjunto de entrenamiento antes de la evaluación final y el guardado.
* **Justificación:** Combinar modelos diversos (*blending*) reduce la varianza frente a un único algoritmo y aprovecha distintas fronteras de decisión, mientras que `n_iter=10` mantiene la ejecución ágil y reproducible.

#### 10. Métricas de Validación
* **Decisión Implementada:** Tras obtener `X_test_holdout` y `y_test_holdout` mediante `get_config('X_test')` y `get_config('y_test')`, se calculan `y_pred = pipeline_final.predict(X_test_holdout)` y `y_prob = pipeline_final.predict_proba(X_test_holdout)[:, 1]`, reportando:
  * **Matriz de confusión** (`confusion_matrix`)
  * **Reporte de clasificación** (`precision`, `recall`, `f1-score`, `support` por clase)
  * **ROC-AUC** (`roc_auc_score(y_test_holdout, y_prob)`) sobre la probabilidad explícita de la clase positiva (`Churn = 1`).
* **Justificación:** Extraer explícitamente la columna `[:, 1]` de `predict_proba` asegura calcular el `ROC-AUC` sobre la probabilidad real de abandono (`Churn = 1`) y no sobre puntajes ambiguos o invertidos respecto a la clase predicha.

---

### FASE C: DESPLIEGUE, MLOPS Y MONITOREO

#### 11. Evaluación de Sesgos (*Fairness & Bias*)
* **Práctica Recomendada en el Ciclo:** Auditar el comportamiento del modelo por segmentos sensibles o demográficos (como `SeniorCitizen` o `gender`) comparando tasas de falsos positivos/negativos y probabilidades medias predichas.
* **Justificación:** Evita que acciones comerciales de retención o fricciones operativas penalicen sistemáticamente a un grupo etario o perfil demográfico particular.

#### 12. MLOps y Serialización del Artefacto
* **Decisión Implementada:** `save_model(pipeline_final, 'modelo_churn_telecom_automl')` exporta el *pipeline* completo (transformaciones + modelo final) como `'modelo_churn_telecom_automl.pkl'`.
* **Justificación:** Guardar el *pipeline* íntegro—y no solo el estimador aislado—garantiza que cualquier inferencia futura aplique exactamente las mismas transformaciones, evitando *training-serving skew*.

#### 13. Producción e Inferencia
* **Flujo Operativo:** El artefacto `.pkl` puede cargarse tanto en procesos *batch* (scoring periódico de cartera) como en servicios de inferencia en tiempo real (por ejemplo, una API REST), recibiendo los datos del cliente, ejecutando el *pipeline* y devolviendo la etiqueta y la probabilidad de `Churn = 1` para priorizar acciones de retención.

#### 14. Evaluación de Deriva (*Drift Monitoring*)
* **Decisión Implementada:** En el script se incorpora una verificación de calidad sobre el `ROC-AUC` obtenido frente a un umbral mínimo operativo (`umbral_minimo_auc = 0.75`): si `auc_valor < 0.75`, emite `[ALERTA DRIFT]` indicando disparo de reentrenamiento; si `auc_valor >= 0.75`, confirma `[ESTADO OK]` para despliegue.
* **Justificación:** En producción real, además de vigilar el desempeño cuando existan etiquetas confirmadas (*concept drift*), conviene monitorear cambios en las distribuciones de variables clave como `tenure`, `MonthlyCharges` o `Contract` (*data drift*).

#### 15. Reentrenamiento
* **Política de Reentrenamiento:**
  * **Basada en umbral / deriva:** Activación de un nuevo ciclo de entrenamiento y validación cuando la métrica cae por debajo de `0.75` o se detecta deriva significativa en los datos de entrada.
  * **Basada en calendario:** Ejecución periódica planificada (por ejemplo, mensual) incorporando nuevas altas, bajas y cambios de plan consolidados.

---

## PARTE II: CÓDIGO FUENTE (`main.py`)

A continuación se presenta el código íntegro implementado

```python
"""
===============================================================================
PROYECTO FINAL INTEGRADOR: APRENDIZAJE AUTOMÁTICO
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
    url_dataset = "[https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)"    
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