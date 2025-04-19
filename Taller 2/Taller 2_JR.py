import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans, MeanShift
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
import joblib

#Función para homologar columnas
def obtener_valor_homologado(series):
    modas = series.mode()
    if not modas.empty:
        return modas.iloc[0]
    else:
        return series.iloc[0]

#---------Paso O: Preparación-----------
# Cargar los dos archivos del dataset Online Retail II
archivo= r'J:\Personal\Maestría Ciencia de Datos\Clases\2. Fundamentos de Ciencias de Datos\Semana 2\Taller 2\dataset\online_retail_II.xlsx'

df_2009 = pd.read_excel(archivo, sheet_name='Year 2009-2010')
df_2010 = pd.read_excel(archivo, sheet_name='Year 2010-2011')


#Información de la hoja de 2009_2010
print("\nInformación del dataset de la hoja 2009:")
print(df_2009.info())

print("\nValores nulos por columna:")
print(df_2009.isnull().sum())

print("\nNúmero de filas y columnas:")
print(df_2009.shape)

print("\nValores negativos")
print((df_2009[['Quantity', 'Price']] < 0).sum())

#Información de la hoja de 2010_2011

print("\nInformación del dataset de la hoja 2010:")
print(df_2010.info())

print("\nValores nulos por columna:")
print(df_2010.isnull().sum())

print("\nNúmero de filas y columnas:")
print(df_2010.shape)

print("\nValores negativos")
print((df_2010[['Quantity', 'Price']] < 0).sum())


# Combinar las dos hojas en un solo DataFrame
df = pd.concat([df_2009, df_2010], ignore_index=True)
#Información de hojas combinadas
print("\nInformación del dataset combinado:")
print(df.info())

print("Datos del DataSet:")
print(df.head())

print("\nValores nulos por columna:")
print(df.isnull().sum())

print("\nNúmero de filas y columnas:")
print(df.shape)

df.columns = ['InvoiceNo', 'StockCode', 'Description', 'Quantity','InvoiceDate','UnitPrice','CustomerID','Country']
print("\nInformación del dataset combinado:")
print(df.info())

#Cambiar de tipo de dato a las variables CustomerID
df['CustomerID'] = pd.to_numeric(df['CustomerID'], errors='coerce')
df['CustomerID'] = df['CustomerID'].astype('Int64')

df['UnitPrice'] = df['UnitPrice'].astype(str).str.strip()
df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce')
 
df['StockCode'] = df['StockCode'].astype('string')

df['InvoiceNo'] = df['InvoiceNo'].astype('string')
df['Description'] = df['Description'].astype('string')
df['Country'] = df['Country'].astype('string')
df['InvoiceNo'] = df['InvoiceNo'].astype('string')
 
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['InvoiceDay'] = df['InvoiceDate'].dt.date
df['InvoiceDay'] = pd.to_datetime(df['InvoiceDay'])

#-------------------
print("\nInformación del dataset combinado:")
print(df.info())


# Homologar la Descripción por StockCode
description_homologado = df.groupby('StockCode')['Description'].apply(obtener_valor_homologado)
df['Description'] = df['StockCode'].map(description_homologado)
df['Description'] = df['Description'].fillna(df['StockCode'].map(description_homologado))

# Homologar la Descripción por InvoiceNo
invoiceNo_homologado = df.groupby('InvoiceNo')['CustomerID'].apply(obtener_valor_homologado)
df['CustomerID'] = df['InvoiceNo'].map(invoiceNo_homologado)
df['CustomerID'] = df['CustomerID'].fillna(df['InvoiceNo'].map(invoiceNo_homologado))

# Normalizar descripciones de productos y paises
df['StockCode'] = df['StockCode'].str.upper().str.strip()
df['Description'] = df['Description'].str.upper().str.strip()
df['Country'] = df['Country'].str.upper().str.strip()

# Transacciones que están en ambos conjuntos
registros_comunes = pd.merge(df_2010, df_2009, how='inner')
print(f"Número de registros comunes en ambas hojas: {len(registros_comunes)}")

#Limpieza de datos
subset=None
keep='first'
df = df.drop_duplicates(subset=subset, keep=keep)

# Eliminar registros con InvoiceNo que empieza con 'C' 
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]

df['UnitPrice'] = df['UnitPrice'].round(2)

# Eliminar cantidades y precios negativas
df = df[df['Quantity'] > 0]
df = df[df['UnitPrice'] > 0]

# Eliminar registros donde CustomerID es nulo
df = df.dropna(subset=['CustomerID'])
df['CustomerID'] = df['CustomerID'].astype('Int64')

print("Valores ≤ 0 en UnitPrice:", (df['UnitPrice'] <= 0).sum()) 

# cuántos clientes quedaron
print(f"CustomerIDs únicos: {df['CustomerID'].nunique()}")

# Productos con variación de precios entre los años 2009_2010 y 2010_2011
df['Year'] = df['InvoiceDate'].dt.year
df_años = df[df['Year'].isin([2009, 2010])]
precios_por_año = df_años.groupby(['StockCode', 'Year'])['UnitPrice'].median().unstack()
precios_por_año = precios_por_año.dropna()
precios_por_año['Variacion'] = precios_por_año[2010] - precios_por_año[2009]
productos_con_cambio = precios_por_año[precios_por_año['Variacion'] != 0]
productos_con_cambio_top = productos_con_cambio.reindex(productos_con_cambio['Variacion'].abs().sort_values(ascending=False).index)

print("\n Productos con variación de precio:")
print(productos_con_cambio)


#Nuevo campo con el valor total para cada registro
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

df['UnitPrice'] = df['UnitPrice'].round(2)
df['TotalPrice'] = df['TotalPrice'].round(2)
      
df.to_excel('datos_limpios.xlsx', index=False)  
print(f"Proceso de limpieza completado.")


#Análisis Exploratorio de Datos
print("\nValores nulos después de limpieza:")
print(df.isnull().sum())

print("\nNúmero de filas y columnas luego de limpieza:")
print(df.shape)


sns.set(style="whitegrid")

# Gráfico 1. Histograma de Quantity
plt.figure(figsize=(14, 6))
# Distribución general
plt.subplot(1, 2, 1)  
sns.histplot(df['Quantity'], bins=100, kde=True, color='skyblue')
plt.title("Distribución General de Cantidades")
plt.xlabel("Cantidad")
plt.ylabel("Frecuencia")
plt.xlim(0, 850)

# Distribución detallada para cantidades < 100
plt.subplot(1, 2, 2)  # 1 fila, 2 columnas, posición 2
sns.histplot(df[df['Quantity'] < 100]['Quantity'], bins=50, kde=True, color='skyblue')
plt.title("Distribución Detallada (Cantidad < 100)")
plt.xlabel("Cantidad")
plt.ylabel("Frecuencia")

plt.tight_layout()
plt.savefig('DistribucionQuantity.png')
plt.show()


# Gráfico 2. Boxplot de UnitPrice
sns.boxplot(x=df['UnitPrice'], color='tomato')
plt.title("Boxplot de Precios")
plt.xlim(0, 35)
plt.savefig('DistribucionUnitPrice.png')
plt.show()


# Gráfico 3. Compras por fecha
df['YearMonth'] = df['InvoiceDate'].dt.to_period('M').astype(str)
ventas_mensuales = df.groupby('YearMonth')['Quantity'].sum().reset_index()

plt.figure(figsize=(14, 6))
sns.lineplot(data=ventas_mensuales, x='YearMonth', y='Quantity', marker='o')
plt.title("Total de Cantidades Vendidas por Mes")
plt.xlabel("Mes")
plt.ylabel("Cantidad Total Vendida")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig('TotalCantidadesVendidasMes.png')
plt.show()


# Top 10 de los productos más vendidos (por cantidad total)
top_productos = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
print("Top 10 productos más vendidos:")
print(top_productos)

#Gráfica 4. Tendencia de ventas y precio promedio mensual del producto más vendido
producto_top = top_productos.index[0]
df_top = df[df['Description'] == producto_top]

df_top['YearMonth'] = df_top['InvoiceDate'].dt.to_period('M').astype(str)

ventas_mensuales = df_top.groupby('YearMonth')['Quantity'].sum()
precio_promedio_mensual = df_top.groupby('YearMonth')['UnitPrice'].mean()

fig, ax1 = plt.subplots(figsize=(14, 6))

ax1.set_title(f"Tendencia de ventas y precio promedio mensual: {producto_top}")
ax1.set_xlabel("Mes")
ax1.set_ylabel("Cantidad vendida", color='tab:blue')
ax1.plot(ventas_mensuales.index, ventas_mensuales.values, marker='o', color='tab:blue', label='Cantidad')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.tick_params(axis='x', rotation=45)

ax2 = ax1.twinx()
ax2.set_ylabel("Precio promedio", color='tab:red')
ax2.plot(precio_promedio_mensual.index, precio_promedio_mensual.values, marker='s', color='tab:red', label='Precio')
ax2.tick_params(axis='y', labelcolor='tab:red')

fig.tight_layout()
plt.grid(True)
plt.savefig('TendenciaVentaProductoTop.png')
plt.show()

# Gráfica 5. Patrones estacionales y máximas ventas
ventas_mensuales = df.groupby(df['InvoiceDate'].dt.to_period('M'))['TotalPrice'].sum()
ventas_mensuales.plot(kind='line', marker='o')
plt.title("Ingresos totales por mes")
plt.ylabel("Ventas")
plt.xlabel("Mes")
plt.xticks(rotation=45)
plt.grid(True)
plt.savefig('PatronesEstacionales.png')
plt.show()

mes_top = ventas_mensuales.idxmax()
monto_max = ventas_mensuales.max()

print(f"\n Mes con mayores ventas: {mes_top}, Total vendido: {monto_max:,.2f}")




#------Paso 1: Clasificación de clientes------


df_clientes = df.groupby('CustomerID')['TotalPrice'].sum().reset_index()

# Cliente Premium (percentil 75)
umbral_premium = df_clientes['TotalPrice'].quantile(0.75)

# Nueva columna de cliente con etiqueta
df_clientes['Categoria'] = df_clientes['TotalPrice'].apply(lambda x: 'Premium' if x >= umbral_premium else 'Normal')

print(df_clientes['Categoria'].value_counts())

# Datos para el modelamiento
df_features = df.groupby('CustomerID').agg({
    'Quantity': 'sum',
    'TotalPrice': 'sum',
    'InvoiceNo': 'nunique',
    'UnitPrice': 'mean'
}).reset_index()

df_features = df_features.rename(columns={
    'Quantity': 'CantidadTotal',
    'TotalPrice': 'GastoTotal',
    'InvoiceNo': 'NumTransacciones',
    'UnitPrice': 'PrecioPromedio'
})


df_modelo = pd.merge(df_features, df_clientes[['CustomerID', 'Categoria']], on='CustomerID')

# Variable objetivo cambiado a números
df_modelo['Categoria'] = df_modelo['Categoria'].map({'Normal': 0, 'Premium': 1})

# Variables predictoras y objetivo
X = df_modelo[['CantidadTotal', 'GastoTotal', 'NumTransacciones', 'PrecioPromedio']]
y = df_modelo['Categoria']

# División en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Escalado
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenar el modelo
modelo_log = LogisticRegression()
modelo_log.fit(X_train_scaled, y_train)

# Predicción
if 'y_pred' not in globals():
    y_pred = modelo_log.predict(X_test_scaled)

# Evaluación con recall para la clase "Premium" (1)
recall = recall_score(y_test, y_pred)
print(f"\n Recall para clase 'Premium': {recall:.4f}")

# Reporte completo
print("\n Reporte de Clasificación:")
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
print("\n Matriz de Confusión:")
print(conf_matrix)
print("Forma de la matriz:", conf_matrix.shape)

plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Normal", "Premium"],
            yticklabels=["Normal", "Premium"])
plt.title("Matriz de Confusión")
plt.xlabel("Predicha")
plt.ylabel("Real")
plt.tight_layout()
plt.savefig('MatrizConfusion.png')
plt.show()

#Validación Cruzada
X_scaled_total = scaler.fit_transform(X)
cv_recall = cross_val_score(modelo_log, X_scaled_total, y, cv=5, scoring='recall')
print("\n Recall por fold (Validación Cruzada 5-Folds):")
print(cv_recall)
print(f"Promedio de Recall: {cv_recall.mean():.4f}")

#Ajustes Hiperparámetros con GridSearchCV
# Definir grilla de hiperparámetros
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'lbfgs']
}

# Grid Search con scoring basado en Recall
grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, scoring='recall')
grid.fit(X_scaled_total, y)

# Mostrar los mejores hiperparámetros
print(f"\n Mejores hiperparámetros: {grid.best_params_}")
print(f"Mejor Recall obtenido: {grid.best_score_:.4f}")


# Crear el pipeline con los mejores hiperparámetros
pipeline_final = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(C=10, solver='liblinear', max_iter=1000))
])

# Entrenar el pipeline
pipeline_final.fit(X_train, y_train)

# Predecir
y_pred = pipeline_final.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

conf_matrix = confusion_matrix(y_test, y_pred)
print("\n Matriz de Confusión después del ajuste:")
print(conf_matrix)

plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Normal", "Premium"],
            yticklabels=["Normal", "Premium"])
plt.title("Matriz de Confusión después del ajuste")
plt.xlabel("Predicha")
plt.ylabel("Real")
plt.tight_layout()
plt.savefig('MatrizConfusionAjuste.png')
plt.show()

print(f"\n Accuracy final del modelo: {accuracy:.4f}")
print(f" Recall final (clientes Premium): {recall:.4f}")

# Guardar el pipeline completo
joblib.dump(pipeline_final, 'modelo_pipeline_logistico.pkl')
print("\n Modelo guardado como 'modelo_pipeline_logistico.pkl'")

# Cargar el modelo guardado
modelo_cargado = joblib.load('modelo_pipeline_logistico.pkl')

#-----Parte 2: Segmentación de Clientes-----

# Crear RFM
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
}).reset_index()

rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

#Escalar los datos
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

#Agrupamiento K-Means
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm['Cluster_KMeans'] = kmeans.fit_predict(rfm_scaled)

#Agrupamiento Mean Shift
meanshift = MeanShift()
rfm['Cluster_MeanShift'] = meanshift.fit_predict(rfm_scaled)

#Visualización de los resultados
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.scatterplot(data=rfm, x='Recency', y='Monetary', hue='Cluster_KMeans', palette='tab10')
plt.title("K-Means: Recency vs Monetary")

plt.subplot(1, 2, 2)
sns.scatterplot(data=rfm, x='Recency', y='Monetary', hue='Cluster_MeanShift', palette='tab20')
plt.title("Mean Shift: Recency vs Monetary")

plt.tight_layout()
plt.savefig('ComparacionAgrupamiento.png')
plt.show()

#Resumen por clúster
resumen_kmeans = rfm.groupby('Cluster_KMeans')[['Recency', 'Frequency', 'Monetary']].mean()
resumen_kmeans['Clientes'] = rfm['Cluster_KMeans'].value_counts().sort_index()

resumen_meanshift = rfm.groupby('Cluster_MeanShift')[['Recency', 'Frequency', 'Monetary']].mean()
resumen_meanshift['Clientes'] = rfm['Cluster_MeanShift'].value_counts().sort_index()


print("\n Resumen por clúster - KMeans:\n", resumen_kmeans)
print("\n Resumen por clúster - Mean Shift:\n", resumen_meanshift)

#-------Parte 3: Predicción de Ventas ------

# Características temporales
df['Month'] = df['InvoiceDate'].dt.month
df['Week'] = df['InvoiceDate'].dt.isocalendar().week
df['Year'] = df['InvoiceDate'].dt.year

def obtener_estacion(mes):
    if mes in [12, 1, 2]:
        return 'Invierno'
    elif mes in [3, 4, 5]:
        return 'Primavera'
    elif mes in [6, 7, 8]:
        return 'Verano'
    else:
        return 'Otoño'

df['Season'] = df['Month'].apply(obtener_estacion)

# Agregar ventas por semana (target: TotalPrice)
ventas_por_semana = df.groupby(['Year', 'Week', 'Season'])['TotalPrice'].sum().reset_index()

# Codificación de estación (Season)
ventas_por_semana['Season'] = ventas_por_semana['Season'].astype('category')
ventas_por_semana['Season_encoded'] = ventas_por_semana['Season'].cat.codes

# Variables independientes y variable objetivo
X = ventas_por_semana[['Year', 'Week', 'Season_encoded']]
y = ventas_por_semana['TotalPrice']

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Entrenar modelo Random Forest
modelo = RandomForestRegressor(n_estimators=600, random_state=42)
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

# Evaluación del modelo
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n--- EVALUACIÓN DEL MODELO ---")
print(f"R² Score: {r2:.2f}")
print(f"RMSE (Raíz del error cuadrático medio): {rmse:.2f}")
print(f"MAE (Error absoluto medio): {mae:.2f}")



# Visualizar ventas reales vs predichas
plt.figure(figsize=(10, 5))
plt.plot(range(len(y_test)), y_test.values, label='Ventas Reales', marker='o')
plt.plot(range(len(y_pred)), y_pred, label='Ventas Predichas', marker='x')
plt.title('Comparación de Ventas Reales vs Predichas (por semana)')
plt.xlabel('Semana')
plt.ylabel('Ventas (£)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('VentasRealesVsPredichas.png')
plt.show()


#---------
q_high = df['Quantity'].quantile(0.991)
#print(q_high)
print(f" Percentil Quantity: {q_high}")
df = df[(df['Quantity'] < q_high)]



# Extraer características temporales
df['Month'] = df['InvoiceDate'].dt.month
df['Week'] = df['InvoiceDate'].dt.isocalendar().week
df['Year'] = df['InvoiceDate'].dt.year

def obtener_estacion(mes):
    if mes in [12, 1, 2]:
        return 'Invierno'
    elif mes in [3, 4, 5]:
        return 'Primavera'
    elif mes in [6, 7, 8]:
        return 'Verano'
    else:
        return 'Otoño'

df['Season'] = df['Month'].apply(obtener_estacion)

# Agregar ventas por semana (target: TotalPrice)
ventas_por_semana = df.groupby(['Year', 'Week', 'Season'])['TotalPrice'].sum().reset_index()

# Codificación de estación (Season)
ventas_por_semana['Season'] = ventas_por_semana['Season'].astype('category')
ventas_por_semana['Season_encoded'] = ventas_por_semana['Season'].cat.codes

# Variables independientes y variable objetivo
X = ventas_por_semana[['Year', 'Week', 'Season_encoded']]
y = ventas_por_semana['TotalPrice']

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Entrenar modelo Random Forest
modelo = RandomForestRegressor(n_estimators=600, random_state=42)
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

# Evaluación del modelo
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n--- EVALUACIÓN DEL MODELO SIN OUTLIERS---")
print(f"R² Score: {r2:.2f}")
print(f"RMSE (Raíz del error cuadrático medio): {rmse:.2f}")
print(f"MAE (Error absoluto medio): {mae:.2f}")



# Visualizar ventas reales vs predichas
plt.figure(figsize=(10, 5))
plt.plot(range(len(y_test)), y_test.values, label='Ventas Reales', marker='o')
plt.plot(range(len(y_pred)), y_pred, label='Ventas Predichas', marker='x')
plt.title('Comparación de Ventas Reales vs Predichas (por semana) sin Outliers')
plt.xlabel('Semana')
plt.ylabel('Ventas (£)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('VentasRealesVsPredichasSO.png')
plt.show()

















