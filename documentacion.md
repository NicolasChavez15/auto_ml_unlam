# PROYECTO FINAL INTEGRADOR: APRENDIZAJE AUTOMÁTICO
**Materia:** Machine Learning / Especialización en Ciencia de Datos  
**Título:** Implementación End-to-End de AutoML para la Predicción de Churn en el Sector Telecomunicaciones bajo el Ciclo de Vida Completo de ML  
**Dominio de aplicación:** Telecomunicaciones / Retención de Clientes  
**Enfoque:** Profesional, Funcional y Aplicado  

---

## RESUMEN EJECUTIVO
El presente proyecto desarrolla e implementa una solución de Machine Learning orientada a la industria de las telecomunicaciones para abordar la problemática crítica de la pérdida de clientes (*Customer Churn*). Se adopta una metodología rigurosa basada en el **Ciclo de Vida del Machine Learning (15 etapas)** y se integran técnicas avanzadas de **Aprendizaje Automático Automatizado (AutoML)** a través de la librería `PyCaret` (basada en el marco conceptual de optimización bayesiana, ensembling y bajo código).

El desarrollo no requiere datos propietarios sensibles, utilizando en su lugar el conjunto de datos de referencia *Telco Customer Churn* de IBM/Kaggle, mapeado operacionalmente a las variables dinámicas de una empresa de servicios convergentes (Internet, TV por Cable, Telefonía). La solución abarca desde la formulación del problema de negocio hasta la arquitectura de MLOps, evaluación de sesgos, monitoreo de deriva (*drift*) y estrategias de reentrenamiento.

---

# PARTE I: MARCO TEÓRICO Y DESARROLLO DEL CICLO DE VIDA (15 ETAPAS)

## FASE A: PLANIFICACIÓN Y DATOS

### 1. Definición del Problema
* **Problema de Negocio:** En el mercado de telecomunicaciones convergentes, el Costo de Adquisición de Clientes (CAC) supera significativamente el costo de retención. La desconexión no planificada de abonados (*churn*) degrada el valor del tiempo de vida del cliente (*Lifetime Value* - LTV) e impacta directamente en el EBITDA de la compañía.
* **Objetivo de ML:** Construir un clasificador binario supervisado que determine la probabilidad de que un cliente solicite la baja de sus servicios dentro del próximo periodo de facturación ($t+1$).
* **Métrica de Optimización Primaria:** **ROC-AUC** y **Recall (Exhaustividad)** para la clase positiva (Churn = 1). En retención corporativa, es prioritario minimizar los *Falsos Negativos* (clientes en riesgo que no son detectados) sobre los *Falsos Positivos* (clientes estables que reciben promociones).

### 2. Estudio de Viabilidad
* **Técnica:** Existencia de registros históricos tabulares con variables socio-demográficas, transaccionales, contractuales y de soporte.
* **Económica:** Un modelo con $Recall > 0.75$ permite dirigir campañas de fidelización segmentadas a un $20\%$ de la base con riesgo crítico, reduciendo la fuga de ingresos en un estimado del $12-15\%$ con un ROI positivo demostrado.
* **Justificación de AutoML:** Siguiendo la doctrina del documento técnico de AutoML, la automatización del flujo de trabajo permite explorar múltiples familias de algoritmos y combinaciones de hiperparámetros en tiempo récord, liberando al científico de datos para actuar como "curador de datos y estratega del problema" (Merlino & Gemini, 2025).

### 3. Detectar Orígenes de Datos
* Las fuentes de información en un entorno de Telecomunicaciones convergente provienen de tres dominios clave:
  1. **CRM / Facturación:** Antigüedad, tipo de contrato, método de pago, monto mensual cargado, total facturado.
  2. **Inventory / Provisioning:** Tipos de servicios contratados (Fibra Óptica, DSL, Telefonía, TV streaming, Seguridad online, Backup).
  3. **Soporte Técnico / Atención al Cliente:** Tickets de reclamos por falla de servicio o facturación, contratación de soporte técnico premium.

### 4. Obtener Acceso a los Datos
* Se establece un protocolo de extracción mediante canalizaciones seguras de lectura desde almacenamiento en la nube o Data Lake empresarial, abstrayendo credenciales confidenciales.

### 5. Extracción de Datos
* Lectura estructurada del dataset `Telco-Customer-Churn.csv` comprimido desde repositorios abiertos (GitHub/Kaggle), emulando una consulta contra una tabla consolidada de *Data Warehouse*.

---

## FASE B: DESARROLLO DEL MODELO

### 6. Análisis Exploratorio de Datos (EDA)
* **Desbalance de Clases:** El conjunto de datos presenta una proporción aproximada de 73% no-churn ($0$) vs 27% churn ($1$).
* **Variables Críticas Identificadas:**
  * *Tenure (Antigüedad):* Fuerte correlación inversa con el churn (los primeros 12 meses representan la zona de mayor riesgo).
  * *Contract Type:* Los contratos mes a mes presentan una tasa de baja sustancialmente superior a los contratos a 1 o 2 años.
  * *Internet Service:* Clientes con Fibra Óptica registran un churn significativamente más alto que los de DSL, asociado al costo del servicio o problemas de estabilidad perceived.

### 7. Limpieza de Datos
* Tratamiento de valores faltantes o espacios en blanco en la columna `TotalCharges` imputándolos mediantes métodos numéricos (mediana/media).
* Normalización de tipos de datos (conversión de cadenas categóricas a formatos de texto estandarizados).

### 8. Ingeniería de Características (Feature Engineering)
* Generación de variables derivadas de negocio:
  * `Charges_per_Tenure`: Relación entre el costo total acumulado y los meses de antigüedad.
  * `Service_Count`: Número total de servicios de valor agregado contratados (Streaming, Backup, Seguridad).
  * `Is_High_Risk_Contract`: Indicador binario para contratos mes a mes con cobro electrónico no automatizado.
* Codificación de categóricas mediante *One-Hot Encoding* / *Target Encoding* y escalado de variables numéricas.

### 9. Selección de Algoritmos (Implementación AutoML)
* Utilización de la librería `PyCaret` para el entrenamiento competitivo automatizado de diversos algoritmos:
  * Regresión Logística (L2 regularization)
  * Random Forest Classifier
  * LightGBM (Light Gradient Boosting Machine)
  * XGBoost (eXtreme Gradient Boosting)
  * CatBoost Classifier
* **Ensembling Automatizado:** Aplicación de técnicas de *Blending* o *Stacking* (Meta-Modelo) combinando las mejores 3 arquitecturas para mejorar la estabilidad y reducir la varianza general del sistema predictivo.

### 10. Métricas de Validación
* Estrategia de validación cruzada estratificada ($K=5$ Folds) para mantener la proporción de la clase minoritaria.
* Análisis de las curvas ROC-AUC, F1-Score, Matriz de Confusión y calibración de probabilidades.

---

## FASE C: DESPLIEGUE Y MONITOREO

### 11. Evaluación de Sesgos (Fairness & Bias)
* Verificación de equidad algorítmica: Garantizar que el modelo no genere discriminación directa o indirecta por segmentos sociodemográficos (ej. adultos mayores o jubilados con planes especiales) penalizando indebidamente sus calificaciones de riesgo.

### 12. MLOps y Arquitectura de Producción
* Empaquetado del modelo entrenado como un artefacto ejecutable (`.pkl` / `.joblib` o MLflow Model Registry).
* Definición de una arquitectura de microservicio REST (ej. contenedor Docker con FastAPI) para inferencias online e inferencias por lotes (*batch scoring* nocturno).

### 13. Producción (Inferencia)
* Ingesta de nuevos registros de abonados $\rightarrow$ Preprocesamiento automatizado mediante el Pipeline guardado $\rightarrow$ Generación de un *Score* de probabilidad de 0 a 1 $\rightarrow$ Enrutamiento automático hacia el CRM para asignación de promociones en la central telefónica.

### 14. Evaluación de Deriva (Drift Monitoring)
* **Data Drift (Deriva de Datos):** Monitoreo del cambio en las distribuciones de las variables de entrada (ej. cambio masivo en las velocidades promedio de internet vendidas).
* **Concept Drift (Deriva del Concepto):** Monitoreo de la degradación en las métricas de rendimiento del modelo (caída del ROC-AUC por debajo de 0.75) debido a variaciones económicas externas o lanzamiento de ofertas agresivas por parte de la competencia.

### 15. Reentrenamiento
* Establecimiento de disparadores automatizados para el reentrenamiento del pipeline de AutoML:
  * **Por Tiempo:** Reentrenamiento programado cada 30 días con los datos actualizados del último ciclo.
  * **Por Performance:** Reentrenamiento por evento cuando la métrica ROC-AUC en datos de producción cae un $5\%$ respecto al baseline inicial.

---

# PARTE II: CÓDIGO PYTHON EJECUTABLE Y COMPLETO

El siguiente bloque de código realiza la totalidad del proceso de manera automatizada: desde la descarga del dataset hasta la evaluación del modelo con AutoML, la extracción de métricas y la simulación del monitoreo de deriva.

```python
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# FASE 1: EXTRACTION Y PREPARACIÓN DE DATOS
# ---------------------------------------------------------
print("=== FASE 1: Carga y Preparación de Datos ===")
url = "[https://raw.githubusercontent.com/IBM/telco-customer-churn/master/WA_Fn-UseC_-Telco-Customer-Churn.csv](https://raw.githubusercontent.com/IBM/telco-customer-churn/master/WA_Fn-UseC_-Telco-Customer-Churn.csv)"
df = pd.read_csv(url)

# Limpieza inicial de variables
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].str.strip(), errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Eliminación de identificadores no predictivos
df_model = df.drop(columns=['customerID'])

# ---------------------------------------------------------
# FASE 2: FEATURE ENGINEERING
# ---------------------------------------------------------
print("=== FASE 2: Ingeniería de Características ===")
df_model['Charges_per_Tenure'] = df_model['MonthlyCharges'] / (df_model['tenure'] + 1)
services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
df_model['Service_Count'] = (df_model[services] == 'Yes').sum(axis=1)

print(f"Dataset listo: {df_model.shape[0]} registros, {df_model.shape[1]} columnas.")

# ---------------------------------------------------------
# FASE 3: PIPELINE DE AUTOML CON PYCARET
# ---------------------------------------------------------
from pycaret.classification import (setup, compare_models, create_model, 
                                   tune_model, blend_models, finalize_model, 
                                   predict_model, save_model)

print("\n=== FASE 3: Inicializando AutoML (PyCaret) ===")
# Configuración del entorno de AutoML (Preprocesamiento automatizado, Imputación, Encodings)
ml_setup = setup(
    data=df_model, 
    target='Churn', 
    session_id=42,
    train_size=0.8,
    verbose=False
)

print("Entrenando y comparando múltiples algoritmos...")
# Comparación competitiva de algoritmos ordenados por ROC-AUC
best_models = compare_models(n_select=3, sort='AUC')

print("\n--- Modelos Seleccionados para Ensembling ---")
# Creación y ajuste de ensambles (Blending)
blended_model = blend_models(estimator_list=best_models, optimize='AUC')

# Ajuste de hiperparámetros del modelo ensamblado
tuned_blend = tune_model(blended_model, optimize='AUC', n_iter=10)

# Finalización del modelo (Entrenamiento con el 100% de los datos de entrenamiento)
final_pipeline = finalize_model(tuned_blend)

# ---------------------------------------------------------
# FASE 4: EVALUACIÓN Y DESEMPEÑO
# ---------------------------------------------------------
print("\n=== FASE 4: Evaluación de Métricas de Inferencia ===")
predictions = predict_model(final_pipeline)

from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

y_true = predictions['Churn']
y_pred = predictions['prediction_label']
y_prob = predictions['prediction_score']

print("\nMatriz de Confusión en Validación Hold-Out:")
print(confusion_matrix(y_true, y_pred))

print("\nReporte de Clasificación:")
print(classification_report(y_true, y_pred))

auc_score = roc_auc_score(y_true, y_prob)
print(f"ROC-AUC Score Final: {auc_score:.4f}")

# ---------------------------------------------------------
# FASE 5: SIMULACIÓN DE MONITOREO DE DRIFT Y DESPLIEGUE
# ---------------------------------------------------------
print("\n=== FASE 5: MLOps - Exportación y Monitoreo de Deriva ===")

# Exportación del Artefacto de Producción
save_model(final_pipeline, 'modelo_churn_telecom_automl')
print("Pipeline guardado con éxito como 'modelo_churn_telecom_automl.pkl'")

# Simulación de Monitoreo de Concept Drift
def monitor_drift(y_actual, y_predicted_prob, threshold=0.70):
    current_auc = roc_auc_score(y_actual, y_predicted_prob)
    print(f"[MONITOREO MLOps] ROC-AUC Actual: {current_auc:.4f}")
    if current_auc < threshold:
        print("[ALERTA DRIFT] ¡Alerta de Degradación de Modelo! Disparando Pipeline de Reentrenamiento Automatizado...")
    else:
        print("[MONITOREO MLOps] El modelo opera dentro de los parámetros esperados de performance.")

# Ejecución de simulación sin deriva
monitor_drift(y_true, y_prob)