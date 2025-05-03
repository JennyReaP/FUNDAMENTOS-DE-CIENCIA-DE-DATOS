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

#       LIMPIEZA Y PREPROCESAMIENTO
df = pd.read_csv(r'J:\Personal\Maestría Ciencia de Datos\Clases\2. Fundamentos de Ciencias de Datos\Proyecto\dataset\detenidos2019_2024.csv', sep=';')
print("\nInformacion del dataset:")
print(df.info())

print("\nValores nulos por columna:")
print(df.isnull().sum())

print("\nNumero de filas y columnas:")
print(df.shape)

print("\nPrimeros datos:")
print(df.head(20))

# Información de variables relevantes

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

# Agrupación de nacionalidades Indígenas
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

# Eliminar columnas SIN_DATO 
edad_a_eliminar = ['SIN_DATO']
df = df[~df['EDAD_OK'].astype(str).isin(edad_a_eliminar)]

nacionalidad_eliminar = ['SIN_DATO']
df = df[~df['NACIONALIDAD'].astype(str).isin(nacionalidad_eliminar)]

delito_eliminar = ['SIN_DATO']
df = df[~df['PRESUNTA_INFRACCION'].astype(str).isin(delito_eliminar)]

# Eliminar dato inconsistente
provincia_eliminar = ['MAR TERRITORIAL']
df = df[~df['NOMBRE_PROVINCIA'].astype(str).isin(provincia_eliminar)]

# Eliminar variables no relevantes
columnas_a_eliminar = ['CODIGO_DISTRITO', 'CODIGO_CIRCUITO', 'CODIGO_SUBCIRCUITO', 'NOMBRE_DISTRITO','NOMBRE_CIRCUITO', 'NOMBRE_SUBCIRCUITO','CODIGO_PARROQUIA','CODIGO_CANTON','NOMBRE_CANTON','NOMBRE_PARROQUIA' ]
df = df.drop(columns=columnas_a_eliminar)
df['ESTADO_CIVIL'] = df['ESTADO_CIVIL'].replace('SE DESCONOCE', 'SIN_DATO')
df['NIVEL_DE_INSTRUCCION'] = df['NIVEL_DE_INSTRUCCION'].replace('SE DESCONOCE', 'SIN_DATO')

# Normalizar las columnas

df['CODIGO_ICCS'] = df['CODIGO_ICCS'].astype('string')
df['TIPO'] = df['TIPO'].astype('string')
df['EDAD_OK'] = pd.to_numeric(df['EDAD_OK'], errors='coerce')
df['FECHA_DETENCION_APREHENSION'] = pd.to_datetime(df['FECHA_DETENCION_APREHENSION'])
df['FECHA_DETENCION_APREHENSION'] = df['FECHA_DETENCION_APREHENSION'].dt.date
df['HORA_DETENCION_APREHENSION'] = pd.to_datetime(df['HORA_DETENCION_APREHENSION'], errors='coerce')
df['HORA_DETENCION_APREHENSION'] = df['HORA_DETENCION_APREHENSION'].dt.strftime('%H:%M')
df['NOMBRE_ZONA'] = df['NOMBRE_ZONA'].astype('string')
df['NOMBRE_SUBZONA'] = df['NOMBRE_SUBZONA'].astype('string')
df['CODIGO_PROVINCIA'] = pd.to_numeric(df['CODIGO_PROVINCIA'], errors='coerce')
df['PRESUNTA_INFRACCION'] = df['PRESUNTA_INFRACCION'].astype('string')

#       AGRUPACIÓN Y CONTEO DE VARIABLES DESPUÉS DE LIMPIEZA

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

#df.to_excel('dataset.xlsx', index=False)

#       VISUALIZACIÓN DE DATOS

# Top 10 presuntas infracciones
top_infracciones = df['PRESUNTA_INFRACCION'].value_counts().head(10)
top_infracciones.plot(kind='barh', figsize=(8,6), edgecolor='black')
plt.title('Top 10 Presuntas Infracciones')
plt.xlabel('Cantidad')
plt.ylabel('Presunta Infracción')
plt.gca().invert_yaxis()  
plt.grid(axis='x')
plt.savefig('Top 10 presuntas infracciones.png')
plt.show()

# Top 5 Provincias con Más Delitos por Año
top_provincias = df['NOMBRE_PROVINCIA'].value_counts().head(5).index.tolist()
df_top = df[df['NOMBRE_PROVINCIA'].isin(top_provincias)]
delitos_por_anio = df_top.groupby(['ANIO', 'NOMBRE_PROVINCIA']).size().reset_index(name='Total_Delitos')
plt.figure(figsize=(12, 6))
sns.barplot(data=delitos_por_anio, x='ANIO', y='Total_Delitos', hue='NOMBRE_PROVINCIA')
plt.title("Top 5 Provincias con Más Delitos por Año")
plt.xlabel("Año")
plt.ylabel("Cantidad de Delitos")
plt.legend(title="Provincia", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('Top 5 Provincias con Más Delitos por Año.png')
plt.show()

# Distribución por autoidentificación
valores = df['AUTOIDENTIFICACION'].value_counts(normalize=True) * 100  # porcentajes
fig, ax = plt.subplots(figsize=(8,6))
wedges, texts = ax.pie(
    valores,
    startangle=90 
    )
etiquetas = [f"{categoria} ({porcentaje:.1f}%)" for categoria, porcentaje in zip(valores.index, valores)]
ax.legend(
    wedges,
    etiquetas,
    title="Autoidentificación",
    loc="center left",
    bbox_to_anchor=(1, 0, 0.5, 1)
    )
plt.title('Distribución por Autoidentificación')
plt.tight_layout()
plt.savefig('Distribución por autoidentificación.png')
plt.show()

# Mapa de Calor de Infracciones por Autoidentificación Étnica (sin MESTIZOS)
df['FECHA_DETENCION_APREHENSION'] = pd.to_datetime(df['FECHA_DETENCION_APREHENSION'], errors='coerce')
df['ANIO'] = df['FECHA_DETENCION_APREHENSION'].dt.year
df_ecuador = df[
    (df['NACIONALIDAD'].str.upper() == 'ECUADOR') &
    (df['AUTOIDENTIFICACION_ETNICA'].str.upper() != 'MESTIZO/A') &
    (~df['AUTOIDENTIFICACION'].str.upper().isin(['NO APLICA', 'SIN_DATO']))
]
tabla = df_ecuador.groupby(['AUTOIDENTIFICACION', 'ANIO'])['PRESUNTA_INFRACCION'].count().reset_index(name='Total_Infracciones')
pivot = tabla.pivot(index='AUTOIDENTIFICACION', columns='ANIO', values='Total_Infracciones').fillna(0)
plt.figure(figsize=(12, 6))
sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrBr', linewidths=0.5)
plt.title("Mapa de Calor de Infracciones por Autoidentificación Étnica (Ecuatorianos sin MESTIZO/A)")
plt.xlabel("Año")
plt.ylabel("Autoidentificación Étnica")
plt.tight_layout()
plt.savefig('Infracciones por Autoidentificación Étnica.png')
plt.show()

# Delitos por Grupo Etario en las 10 Provincias con más Delitos
df['EDAD_OK'] = pd.to_numeric(df['EDAD_OK'], errors='coerce')
df['GRUPO_EDAD_AGRUPADA'] = pd.cut(df['EDAD_OK'], 
                                   bins=[11, 24, 44, 120], 
                                   labels=['JOVEN', 'ADULTO', 'ADULTO MAYOR'])
top_provincias = (
    df['NOMBRE_PROVINCIA'].value_counts()
    .head(10)
    .sort_values(ascending=False)
    .index.tolist()
)
df_top = df[df['NOMBRE_PROVINCIA'].isin(top_provincias)]
conteo = df_top.groupby(['NOMBRE_PROVINCIA', 'GRUPO_EDAD_AGRUPADA'])['PRESUNTA_INFRACCION'].count().reset_index()
conteo.rename(columns={'PRESUNTA_INFRACCION': 'Cantidad'}, inplace=True)
orden_provincias = (
    conteo.groupby('NOMBRE_PROVINCIA')['Cantidad']
    .sum()
    .sort_values(ascending=False)
    .index.tolist()
)
conteo['NOMBRE_PROVINCIA'] = pd.Categorical(conteo['NOMBRE_PROVINCIA'], categories=orden_provincias, ordered=True)
plt.figure(figsize=(14, 6))
sns.barplot(data=conteo, x='NOMBRE_PROVINCIA', y='Cantidad', hue='GRUPO_EDAD_AGRUPADA')
plt.title("Delitos por Grupo Etario en las 10 Provincias con más Delitos")
plt.xlabel("Provincia")
plt.ylabel("Cantidad de Delitos")
plt.legend(title="Grupo Edad")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('Delitos por Grupo Etario en las 10 Prov.png')
plt.show()

# Mapa de Calor de Presuntas Infracciones por Nivel de Instrucción (Ecuatorianos)
df['FECHA_DETENCION_APREHENSION'] = pd.to_datetime(df['FECHA_DETENCION_APREHENSION'], errors='coerce')
df['ANIO'] = df['FECHA_DETENCION_APREHENSION'].dt.year
df_ecuatoriano = df[df['NACIONALIDAD'].str.upper() == 'ECUADOR']
df_ecuatoriano = df_ecuatoriano[
    ~(df_ecuatoriano['NIVEL_DE_INSTRUCCION'].str.upper()=='SIN_DATO')
]
tabla = df_ecuatoriano.groupby(['NIVEL_DE_INSTRUCCION', 'ANIO'])['PRESUNTA_INFRACCION'].count().reset_index(name='Total_Infracciones')
pivot = tabla.pivot(index='NIVEL_DE_INSTRUCCION', columns='ANIO', values='Total_Infracciones').fillna(0)
plt.figure(figsize=(12, 6))
sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlGnBu', linewidths=0.5)
plt.title("Mapa de Calor de Presuntas Infracciones por Nivel de Instrucción (Ecuatorianos)")
plt.xlabel("Año")
plt.ylabel("Nivel de Instrucción")
plt.tight_layout()
plt.savefig('Presuntas Infracciones por Nivel de Instrucción.png')
plt.show()

# Top 5 Nacionalidades Extranjeras con Más Registros de Delitos por Año
df['FECHA_DETENCION_APREHENSION'] = pd.to_datetime(df['FECHA_DETENCION_APREHENSION'], errors='coerce')
df['ANIO'] = df['FECHA_DETENCION_APREHENSION'].dt.year
df_extranjeros = df[df['NACIONALIDAD'].str.upper() != 'ECUADOR']
top_nacionalidades = df_extranjeros['NACIONALIDAD'].value_counts().head(5).index.tolist()
df_top = df_extranjeros[df_extranjeros['NACIONALIDAD'].isin(top_nacionalidades)]
agrupado = df_top.groupby(['NACIONALIDAD', 'ANIO']).size().reset_index(name='Total_Casos')
orden_nacionalidades = (
    agrupado.groupby('NACIONALIDAD')['Total_Casos']
    .sum()
    .sort_values(ascending=False)
    .index.tolist()
)
plt.figure(figsize=(12, 6))
sns.barplot(
    data=agrupado,
    x='Total_Casos',
    y='NACIONALIDAD',
    hue='ANIO',
    order=orden_nacionalidades,
    orient='h'
)
plt.title("Top 5 Nacionalidades Extranjeras con Más Registros de Delitos por Año")
plt.xlabel("Cantidad de Delitos")
plt.ylabel("Nacionalidad")
plt.legend(title="Año", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('Nacionalidades Extranjeras con Más Delitos.png')
plt.show()

# Crecimiento Porcentual de Delitos por Género a Través de los Años
df['FECHA_DETENCION_APREHENSION'] = pd.to_datetime(df['FECHA_DETENCION_APREHENSION'], errors='coerce')
df['ANIO'] = df['FECHA_DETENCION_APREHENSION'].dt.year
df_genero = df[df['GENERO'].notna()]
df_genero = df_genero[~df_genero['GENERO'].str.upper().isin(['SIN_DATO', 'NO APLICA'])]
conteo = df_genero.groupby(['GENERO', 'ANIO'])['PRESUNTA_INFRACCION'].count().reset_index(name='Cantidad')
conteo['Crecimiento_%'] = conteo.groupby('GENERO')['Cantidad'].pct_change() * 100
plt.figure(figsize=(12, 6))
sns.lineplot(data=conteo, x='ANIO', y='Crecimiento_%', hue='GENERO', marker='o')
plt.title("Crecimiento Porcentual de Delitos por Género a Través de los Años")
plt.xlabel("Año")
plt.ylabel("Crecimiento (%)")
plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.legend(title="Género")
plt.grid(True)
plt.tight_layout()
plt.savefig('Crecimiento Porcentual de Delitos por Género.png')
plt.show()

# Crecimiento Porcentual de mayor presunto delito por Género
df['FECHA_DETENCION_APREHENSION'] = pd.to_datetime(df['FECHA_DETENCION_APREHENSION'], errors='coerce')
df['ANIO'] = df['FECHA_DETENCION_APREHENSION'].dt.year
delito_top1 = df['PRESUNTA_INFRACCION'].value_counts().idxmax()
df_top_delito = df[df['PRESUNTA_INFRACCION'] == delito_top1]
df_top_delito = df_top_delito[df_top_delito['GENERO'].notna()]
df_top_delito = df_top_delito[~df_top_delito['GENERO'].str.upper().isin(['SIN_DATO', 'NO APLICA'])]
conteo = df_top_delito.groupby(['GENERO', 'ANIO'])['PRESUNTA_INFRACCION'].count().reset_index(name='Cantidad')
conteo['Crecimiento_%'] = conteo.groupby('GENERO')['Cantidad'].pct_change() * 100
plt.figure(figsize=(12, 6))
sns.lineplot(data=conteo, x='ANIO', y='Crecimiento_%', hue='GENERO', marker='o')
plt.title(f"Crecimiento Porcentual de '{delito_top1}' por Género")
plt.xlabel("Año")
plt.ylabel("Crecimiento (%)")
plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.legend(title="Género")
plt.grid(True)
plt.tight_layout()
plt.savefig('Crecimiento Porcentual de mayor presunto delito por Género.png')
plt.show()


#       ENTRENAMIENTO MODELO XGBOOSTCLASSIFIER (MEJOR MÉTRICA)

# Reagrupar GRUPO_EDAD en tres clases
df['GRUPO_EDAD_AGRUPADA'] = df['GRUPO_EDAD'].replace({
    '(12-17)': 'JOVEN',
    '(18-24)': 'JOVEN',
    '(25-34)': 'ADULTO',
    '(35-44)': 'ADULTO',
    '(45-64)': 'ADULTO MAYOR',
    '(65+)': 'ADULTO MAYOR'
})

# Reducir 120.000 registros para balancear clases dominantes
filtro_desequilibrante = (
    (df['GRUPO_EDAD'].isin(['(25-34)', '(35-44)'])) &
    (df['NOMBRE_PROVINCIA'].isin(['BOLIVAR', 'MORONA SANTIAGO', 'NAPO', 'ORELLANA', 'PASTAZA','SANTA ELENA', 'SUCUMBIOS', 'ZAMORA CHINCHIPE','CANAR','CARCHI','PICHINCHA','GUAYAS','IMBABURA','EL ORO'])) &
    (df['ESTADO_CIVIL']==('SOLTERO/A')) 
)

# Tomar solo 20,000 filas de los que cumplen el filtro
indices_a_eliminar = df[filtro_desequilibrante].sample(n=120000, random_state=42).index

# Eliminar del DataFrame
df = df.drop(index=indices_a_eliminar)
conteo_grupo_edadA = df['GRUPO_EDAD_AGRUPADA'].value_counts(dropna=False)
print(conteo_grupo_edadA)

# Definir variables predictoras
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
plt.savefig('Top 20 Variables Más Importantes - XGBoost.png')
plt.show()

# Predicción
y_pred = pipeline.predict(X_test)

# Reporte de clasificación
print("\n Reporte de clasificación:")
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
plt.savefig('Matriz de Confusión - GRUPO_EDAD_AGRUPADA.png')
plt.show()
