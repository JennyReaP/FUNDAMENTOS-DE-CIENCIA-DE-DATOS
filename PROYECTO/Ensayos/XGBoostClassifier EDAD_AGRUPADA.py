import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

df = pd.read_csv(r'J:\Personal\Maestría Ciencia de Datos\Clases\2. Fundamentos de Ciencias de Datos\Proyecto\dataset\detenidos2019_2024.csv', sep=';')
pd.set_option('display.max_rows', None)

print("\nInformacion del dataset:")
print(df.info())

print("\nValores nulos por columna:")
print(df.isnull().sum())

print("\nNumero de filas y columnas:")
print(df.shape)

print("\nPrimeros datos:")

conteo_estado_civil = df['ESTADO_CIVIL'].value_counts(dropna=False)
print(conteo_estado_civil)

conteo_edad = df['EDAD_OK'].value_counts(dropna=False)
print(conteo_edad)

conteo_sexo = df['SEXO'].value_counts(dropna=False)
print(conteo_sexo)

conteo_nacionalidad = df['NACIONALIDAD'].value_counts(dropna=False)
print(conteo_nacionalidad)

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

conteo_año = df['ANIO'].value_counts(dropna=False)
print(conteo_año)

edad_a_eliminar = ['SIN_DATO']
df = df[~df['EDAD_OK'].astype(str).isin(edad_a_eliminar)]

nacionalidad_eliminar = ['SIN_DATO']
df = df[~df['NACIONALIDAD'].astype(str).isin(nacionalidad_eliminar)]

delito_eliminar = ['SIN_DATO']
df = df[~df['PRESUNTA_INFRACCION'].astype(str).isin(delito_eliminar)]

provincia_eliminar = ['MAR TERRITORIAL']
df = df[~df['NOMBRE_PROVINCIA'].astype(str).isin(provincia_eliminar)]

columnas_a_eliminar = ['CODIGO_DISTRITO', 'CODIGO_CIRCUITO', 'CODIGO_SUBCIRCUITO', 'NOMBRE_DISTRITO','NOMBRE_CIRCUITO', 'NOMBRE_SUBCIRCUITO']
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
df['CODIGO_CANTON'] = df['CODIGO_CANTON'].astype('string')
df['CODIGO_PARROQUIA'] = df['CODIGO_PARROQUIA'].astype('string')
df['PRESUNTA_INFRACCION'] = df['PRESUNTA_INFRACCION'].astype('string')

conteo_grupo_edad = df['GRUPO_EDAD'].value_counts(dropna=False)
print(conteo_grupo_edad)

#Gráfica 

conteo_infracciones = df.groupby(['ANIO', 'PRESUNTA_INFRACCION']).size().reset_index(name='Cantidad')
top_infracciones_por_anio = conteo_infracciones.sort_values(['ANIO', 'Cantidad'], ascending=[True, False]).drop_duplicates(subset=['ANIO'])
print(top_infracciones_por_anio)

# ENTRENAMIENTO XGBOOSTCLASSIFIER

# Reagrupar GRUPO_EDAD en tres clases
df['GRUPO_EDAD_AGRUPADA'] = df['GRUPO_EDAD'].replace({
    '(12-17)': 'JOVEN',
    '(18-24)': 'JOVEN',
    '(25-34)': 'ADULTO',
    '(35-44)': 'ADULTO',
    '(45-64)': 'ADULTO MAYOR',
    '(65+)': 'ADULTO MAYOR'
})

# Definir variables predictoras
features = [
    'SEXO', 'GENERO', 'NACIONALIDAD', 'AUTOIDENTIFICACION', 'ESTADO_CIVIL',
    'ESTATUS_MIGRATORIO', 'NIVEL_DE_INSTRUCCION', 'CONDICION', 'MOVILIZACION',
    'TIPO_ARMA', 'ARMA', 'TIPO_LUGAR', 'LUGAR', 'ANIO', 'CODIGO_PROVINCIA','PRESUNTA_INFRACCION'
]
target = 'GRUPO_EDAD_AGRUPADA'

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
    ('classifier', XGBClassifier(objective='multi:softmax', num_class=3, eval_metric='mlogloss', use_label_encoder=False, random_state=42))
])

# División de los datos
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Entrenamiento
pipeline.fit(X_train, y_train)

# Predicción
y_pred = pipeline.predict(X_test)

# Reporte de clasificación
print("\n Reporte de clasificación GRUPO_EDAD_AGRUPADA XGBoost:")
print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=target_encoder.classes_, columns=target_encoder.classes_)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.title("Matriz de Confusión - GRUPO_EDAD_AGRUPADA (XGBoostClassifier)")
plt.xlabel("Predicho")
plt.ylabel("Verdadero")
plt.tight_layout()
plt.savefig('GRUPO_EDAD_AGRUPADA (XGBoostClassifier).png')
plt.show()