import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

df = pd.read_csv(r'J:\Personal\Maestría Ciencia de Datos\Clases\2. Fundamentos de Ciencias de Datos\Proyecto\dataset\detenidos2019_2024.csv', sep=';')

print("\nInformacion del dataset:")
print(df.info())

print("\nValores nulos por columna:")
print(df.isnull().sum())

print("\nNumero de filas y columnas:")
print(df.shape)

print("\nPrimeros datos:")
print(df.head(20))

conteo_estado_civil = df['ESTADO_CIVIL'].value_counts(dropna=False)
print(conteo_estado_civil)

conteo_edad = df['EDAD_OK'].value_counts(dropna=False)
print(conteo_edad)

conteo_sexo = df['SEXO'].value_counts(dropna=False)
print(conteo_sexo)

conteo_genero = df['GENERO'].value_counts(dropna=False)
print(conteo_genero)

conteo_instruccion = df['NIVEL_DE_INSTRUCCION'].value_counts(dropna=False)
print(conteo_instruccion)

df['NIVEL_DE_INSTRUCCION'] = df['NIVEL_DE_INSTRUCCION'].replace('NO APLICA', 'SIN ESTUDIOS')

conteo_nacionalidad = df['NACIONALIDAD'].value_counts(dropna=False)
print(conteo_nacionalidad)

conteo_lugar = df['LUGAR'].value_counts(dropna=False)
print(conteo_lugar)

conteo_condicion = df['CONDICION'].value_counts(dropna=False)
print(conteo_condicion)

conteo_nivel_instruccion = df['MOVILIZACION'].value_counts(dropna=False)
print(conteo_nivel_instruccion)

conteo_tipo_arma = df['TIPO_ARMA'].value_counts(dropna=False)
print(conteo_tipo_arma)

conteo_arma = df['ARMA'].value_counts(dropna=False)
print(conteo_arma)

conteo_provincia = df['NOMBRE_PROVINCIA'].value_counts(dropna=False)
print(conteo_provincia)

conteo_autoidentificacion = df['AUTOIDENTIFICACION_ETNICA'].value_counts(dropna=False)
print(conteo_autoidentificacion)

reemplazos = {
    'SHUAR': 'INDIGENA',
    'KICHWA': 'INDIGENA',
    'MANTA': 'INDIGENA',
    'ACHUAR': 'INDIGENA',
    'NATABUELA': 'INDIGENA',
    'OTAVALO': 'INDIGENA',
    'SALASACA': 'INDIGENA',
    'SARAGURO': 'INDIGENA',
    'WAORANI': 'INDIGENA',
    'CHACHI': 'INDIGENA',
    'KAYAMBI': 'INDIGENA',
    'EPERA': 'INDIGENA',
    'KANARI': 'INDIGENA',
    'SECOYA': 'INDIGENA',
    'KITU KARA': 'INDIGENA',
    'COFAN': 'INDIGENA',
    'PASTO': 'INDIGENA',
    'ANDOA': 'INDIGENA',
    'TSACHILA': 'INDIGENA',
    'CHIBULEO': 'INDIGENA',
    'PANZALEO': 'INDIGENA',
    'KARANKI': 'INDIGENA',
    'SHIWIAR': 'INDIGENA',
    'PURUHA': 'INDIGENA',
    'QUISAPINCHA': 'INDIGENA',
    'PALTAS': 'INDIGENA',
    'SIONA': 'INDIGENA',
    'ZAPARA': 'INDIGENA',
    'TOMABELA': 'INDIGENA',
    'WARANKA': 'INDIGENA'
}


df['AUTOIDENTIFICACION'] = df['AUTOIDENTIFICACION_ETNICA'].replace(reemplazos)

conteo_autoidentificacionN = df['AUTOIDENTIFICACION'].value_counts(dropna=False)
print(conteo_autoidentificacionN)

conteo_grupo_edad = df['GRUPO_EDAD'].value_counts(dropna=False)
print(conteo_grupo_edad)

conteo_presunta_infraccion = df['PRESUNTA_INFRACCION'].value_counts(dropna=False)
print(conteo_presunta_infraccion)

conteo_tipo_lugar = df['TIPO_LUGAR'].value_counts(dropna=False)
print(conteo_tipo_lugar)

edad_a_eliminar = ['SIN_DATO']
df = df[~df['EDAD_OK'].astype(str).isin(edad_a_eliminar)]

nacionalidad_eliminar = ['SIN_DATO']
df = df[~df['NACIONALIDAD'].astype(str).isin(nacionalidad_eliminar)]

delito_eliminar = ['SIN_DATO']
df = df[~df['PRESUNTA_INFRACCION'].astype(str).isin(delito_eliminar)]

provincia_eliminar = ['MAR TERRITORIAL']
df = df[~df['NOMBRE_PROVINCIA'].astype(str).isin(provincia_eliminar)]

columnas_a_eliminar = ['CODIGO_DISTRITO', 'CODIGO_CIRCUITO', 'CODIGO_SUBCIRCUITO', 'NOMBRE_DISTRITO','NOMBRE_CIRCUITO', 'NOMBRE_SUBCIRCUITO','CODIGO_PARROQUIA','CODIGO_CANTON','NOMBRE_CANTON','NOMBRE_PARROQUIA' ]
df = df.drop(columns=columnas_a_eliminar)
df['ESTADO_CIVIL'] = df['ESTADO_CIVIL'].replace('SE DESCONOCE', 'SIN_DATO')
df['NIVEL_DE_INSTRUCCION'] = df['NIVEL_DE_INSTRUCCION'].replace('SE DESCONOCE', 'SIN_DATO')

# Normalizar las columnas

df['CODIGO_ICCS'] = df['CODIGO_ICCS'].astype('string')
df['TIPO'] = df['TIPO'].astype('string')
df['EDAD_OK'] = pd.to_numeric(df['EDAD_OK'], errors='coerce')
df['AUTOIDENTIFICACION_ETNICA'] = df['AUTOIDENTIFICACION_ETNICA'].astype('string')
df['FECHA_DETENCION_APREHENSION'] = pd.to_datetime(df['FECHA_DETENCION_APREHENSION'])
df['FECHA_DETENCION_APREHENSION'] = df['FECHA_DETENCION_APREHENSION'].dt.date
df['HORA_DETENCION_APREHENSION'] = pd.to_datetime(df['HORA_DETENCION_APREHENSION'], errors='coerce')
df['HORA_DETENCION_APREHENSION'] = df['HORA_DETENCION_APREHENSION'].dt.strftime('%H:%M')
df['NOMBRE_ZONA'] = df['NOMBRE_ZONA'].astype('string')
df['NOMBRE_SUBZONA'] = df['NOMBRE_SUBZONA'].astype('string')
df['CODIGO_PROVINCIA'] = pd.to_numeric(df['CODIGO_PROVINCIA'], errors='coerce')
df['PRESUNTA_INFRACCION'] = df['PRESUNTA_INFRACCION'].astype('string')

pd.set_option('display.max_rows', None)

conteo_estado_civil = df['ESTADO_CIVIL'].value_counts(dropna=False)
print(conteo_estado_civil)

conteo_edad = df['EDAD_OK'].value_counts(dropna=False)
print(conteo_edad)

conteo_sexo = df['SEXO'].value_counts(dropna=False)
print(conteo_sexo)

conteo_genero = df['GENERO'].value_counts(dropna=False)
print(conteo_genero)

conteo_instruccion = df['NIVEL_DE_INSTRUCCION'].value_counts(dropna=False)
print(conteo_instruccion)

conteo_nacionalidad = df['NACIONALIDAD'].value_counts(dropna=False)
print(conteo_nacionalidad)

conteo_lugar = df['LUGAR'].value_counts(dropna=False)
print(conteo_lugar)

conteo_condicion = df['CONDICION'].value_counts(dropna=False)
print(conteo_condicion)

conteo_nivel_instruccion = df['MOVILIZACION'].value_counts(dropna=False)
print(conteo_nivel_instruccion)

conteo_tipo_arma = df['TIPO_ARMA'].value_counts(dropna=False)
print(conteo_tipo_arma)

conteo_arma = df['ARMA'].value_counts(dropna=False)
print(conteo_arma)

conteo_provincia = df['NOMBRE_PROVINCIA'].value_counts(dropna=False)
print(conteo_provincia)

conteo_autoidentificacionN = df['AUTOIDENTIFICACION'].value_counts(dropna=False)
print(conteo_autoidentificacionN)

conteo_año = df['ANIO'].value_counts(dropna=False)
print(conteo_año)

conteo_grupo_edad = df['GRUPO_EDAD'].value_counts(dropna=False)
print(conteo_grupo_edad)


conteo_presunta_infraccion = df['PRESUNTA_INFRACCION'].value_counts(dropna=False)
print(conteo_presunta_infraccion)

conteo_tipo_lugar = df['TIPO_LUGAR'].value_counts(dropna=False)
print(conteo_tipo_lugar)

# ENTRENAMIENTO MODELO XGBOOSTCLASSIFIER GRUPO_EDAD

#  Variables predictoras
features = [
    'SEXO', 'GENERO', 'NACIONALIDAD', 'AUTOIDENTIFICACION', 
    'ESTADO_CIVIL',
    'ESTATUS_MIGRATORIO',
    'NIVEL_DE_INSTRUCCION', 'CONDICION', 'MOVILIZACION',
    'TIPO_ARMA', 'ARMA', 
    'TIPO_LUGAR', 
    'LUGAR', 
    'ANIO', 
    'CODIGO_PROVINCIA',
    'PRESUNTA_INFRACCION'
]
target = 'GRUPO_EDAD'

# Eliminar registros con valores faltantes
df = df.dropna(subset=features + [target])
X = df[features].copy()
y = df[target].astype(str).copy()

# Codificar la variable objetivo
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

# Detectar columnas categóricas y numéricas
cat_cols = X.select_dtypes(include='object').columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.difference(cat_cols).tolist()

# Preprocesamiento
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ('num', StandardScaler(), num_cols)
])

# Pipeline con XGBoostClassifier
pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('classifier', XGBClassifier(class_weight='balanced', objective='multi:softprob', num_class=3, eval_metric='mlogloss', use_label_encoder=False, random_state=42))
])

# División de los datos
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Entrenamiento
pipeline.fit(X_train, y_train)

# Extraer el modelo entrenado de XGBoost
modelo_xgb = pipeline.named_steps['classifier']

# Obtener nombres de columnas tras OneHotEncoder
onehot_cols = pipeline.named_steps['preprocess'].named_transformers_['cat'].get_feature_names_out(cat_cols)
all_features = list(onehot_cols) + num_cols

# Obtener importancia
importancias = modelo_xgb.feature_importances_
importancias_df = pd.DataFrame({'feature': all_features, 'importance': importancias})
importancias_df = importancias_df.sort_values(by='importance', ascending=False).head(20)

# Mostrar
print("\n Top 20 variables más importantes:")
print(importancias_df)

# Gráfico
plt.figure(figsize=(10, 6))
sns.barplot(data=importancias_df, x='importance', y='feature', palette='viridis')
plt.title("Top 20 Variables Más Importantes - XGBoost")
plt.xlabel("Importancia")
plt.ylabel("Variable")
plt.tight_layout()
plt.savefig('Top 20 Variables Más Importantes - GRUPO_EDAD XGBoost.png')
plt.show()

# Predicción
y_pred = pipeline.predict(X_test)

# Reporte de clasificación
print("\n Reporte de clasificación GRUPO_EDAD:")
print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=target_encoder.classes_, columns=target_encoder.classes_)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.title("Matriz de Confusión - GRUPO_EDAD (XGBoostClassifier)")
plt.xlabel("Predicho")
plt.ylabel("Verdadero")
plt.tight_layout()
plt.savefig('Matriz de Confusión - GRUPO_EDAD.png')
plt.show()
