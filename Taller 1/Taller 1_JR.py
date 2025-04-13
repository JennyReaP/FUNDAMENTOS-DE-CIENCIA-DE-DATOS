import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns
import plotly.express as px
from sqlalchemy import create_engine
from sqlalchemy import text
import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'notebook'

def obtener_valor_homologado(series):
    # Calcula el modo; si hay más de uno, toma el primero
    modas = series.mode()
    if not modas.empty:
        return modas.iloc[0]
    else:
        return series.iloc[0]

def limpiar_datos(archivo_excel):
    
    try:
        df = pd.read_excel(archivo_excel)
    except Exception as e:
        print(f"Error al cargar el archivo: {e}")
        return None, 0
    
    # Parte 1.1 Leer inicialmente los datos
    print("Información del DataSet")
    print(df.info())

    print("Información de los datos")
    print(df.describe())
    
    print("Primeros datos del DataSet:")
    print(df.head())
    
    # Fechas de inicio y fin de compras

    fecha_inicio = df['InvoiceDate'].min()
    fecha_fin = df['InvoiceDate'].max()

    print(f" Fecha de la primera compra: {fecha_inicio.date()}")
    print(f" Fecha de la última compra: {fecha_fin.date()}")

    registros_iniciales = len(df)
        
    # 1.2 Limpieza de datos
    # 1. Eliminar datos duplicados y dejar un registro único
    subset=None
    keep='first'
    df = df.drop_duplicates(subset=subset, keep=keep)
    
    # 2. Eliminar registros con: CustomerID vacío, UnitPrice <= 0, Quantity negativos
    df = df[(df['CustomerID'].notna()) & 
            (df['UnitPrice'] > 0) & 
            (df['Quantity'] > 0)]
        
    # 3. Eliminar cantidades y precios negativas
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]

    # 4. Eliminar registros con InvoiceNo que empieza con 'A' 
    df = df[~df['InvoiceNo'].astype(str).str.startswith('A')]
    
    # 5. Eliminar productos específicos en StockCode: D, S, M, PADS
    productos_a_eliminar = ['D', 'S', 'M','PADS']
    df = df[~df['StockCode'].astype(str).isin(productos_a_eliminar)]

    # 6. Calcular percentiles para identificar outliers y quitarlos
    q_high = df['Quantity'].quantile(0.9991)
    #print(q_high)
    print(f" Percentil Quantity: {q_high}")
    p_high = df['UnitPrice'].quantile(0.9991)
    print(f" Percentil UnitPrice: {p_high}")
    #print(p_high)
    df = df[(df['Quantity'] < q_high) & (df['UnitPrice'] < p_high)]    

    # 7. Homologar el UnitPrice por StockCode

    unitprice_homologado = df.groupby('StockCode')['UnitPrice'].apply(obtener_valor_homologado)
    df['UnitPrice'] = df['StockCode'].map(unitprice_homologado)
    df['UnitPrice'] = df['UnitPrice'].fillna(df['StockCode'].map(unitprice_homologado))

    # 8. Homologar la Descripción por StockCode
    description_homologado = df.groupby('StockCode')['Description'].apply(obtener_valor_homologado)
    df['Description'] = df['StockCode'].map(description_homologado)
    df['Description'] = df['Description'].fillna(df['StockCode'].map(description_homologado))

    # Resetear índice para que sea consecutivo tras las eliminaciones
    df.reset_index(drop=True, inplace=True)


    # 9. Identificar productos de modelos diferentes que tienen 5 números y una letra (longitud 6) y cambiar el StockCode entre los números 00001 hasta 10000
    pattern = re.compile(r'^\d{5}[A-Za-z]$')
    mask = df['StockCode'].apply(lambda x: bool(pattern.match(str(x))))

    special_stockcodes = df[mask]['StockCode'].unique()

    # Asignar un nuevo código numérico único a cada uno de esos códigos

    stockcode_mapping = {}
    code_counter = 1

    for code in sorted(special_stockcodes):
        if code not in stockcode_mapping:
            # Asignar nuevo código como string con ceros a la izquierda
            new_code = str(code_counter).zfill(5)
            stockcode_mapping[code] = new_code
            code_counter += 1

    #9.2: Reemplazar en el DataFrame original
#    df['StockCode_Original'] = df['StockCode']  # Por si queremos conservar el original
    df['StockCode'] = df['StockCode'].apply(lambda x: stockcode_mapping.get(x, x))

   # df[['StockCode_Original', 'StockCode', 'Description']]
    df[['StockCode', 'Description']]


    # 10. Para CustomerID faltantes marcarlas como 99999
    df['CustomerID'] = df['CustomerID'].fillna(99999)


    # 11. Normalizar descripciones de productos y paises
    df['Description'] = df['Description'].str.upper().str.strip()
    df['Country'] = df['Country'].str.upper().str.strip()
    
    # 12. Reemplazar códigos personalizados por valores numéricos
    reemplazos = {
        'C2': 99988,
        'DOT': 99977,
        'POST': 99966,
        '15056BL': 99911
        }

    df['StockCode'] = df['StockCode'].replace(reemplazos)
    
    #Cambiar de tipo de dato a las variables CustomerID
    df['CustomerID'] = pd.to_numeric(df['CustomerID'], errors='coerce')
    df['CustomerID'] = df['CustomerID'].astype('Int64')
    
    df['StockCode'] = pd.to_numeric(df['StockCode'], errors='coerce')
    df['StockCode'] = df['StockCode'].astype('Int64')
   
    df['InvoiceNo'] = df['InvoiceNo'].astype('string')
    df['Description'] = df['Description'].astype('string')
    df['Country'] = df['Country'].astype('string')
    df['InvoiceNo'] = df['InvoiceNo'].astype('string')
    
    # Convertir a datetime y extraer componentes útiles
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['InvoiceDay'] = df['InvoiceDate'].dt.date
    df['InvoiceDay'] = pd.to_datetime(df['InvoiceDay'])
    
    #Crear un campo con el valor total para cada registro
    df['TotalValue'] = df['Quantity'] * df['UnitPrice']
    #Para redondear el precio y el total
    df['UnitPrice'] = df['UnitPrice'].round(2)
    df['TotalValue'] = df['TotalValue'].round(2)
    
    # Tipos de datos cambiados
    print("Tipos de datos cambiados del DataSet")
    print(df.info())
    
    #1.3 Cargar la información ya depurada en la BDD de SqLite
    
    engine = create_engine('sqlite:///online_retail.db')
    df.to_sql('online_retail_limpia', con=engine, if_exists='replace', index=False)

    print("Datos cargados correctamente en la base de datos SQLite.")
    
    #Consulta de prueba de los registros en la bdd
    with engine.connect() as conn:

     query = """
       SELECT Description, SUM(Quantity) as TotalVendido
       FROM online_retail_limpia
       GROUP BY Description
       ORDER BY TotalVendido DESC
       LIMIT 10
      """
     result = conn.execute(text(query))

     print("Productos menos vendidos en SqLite:")
     for row in result:
             print(row)
    
#------------------------------------------
    #Parte 3. Gráficas
    
    #3.1 Gráfica de los 7 países con mayor venta
     
    top_countries = df.groupby('Country')['Quantity'].sum().sort_values(ascending=False).head(7)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_countries.values, y=top_countries.index, palette="crest")
    plt.title('Top 7 países con más ventas')
    plt.xlabel('Cantidad vendida')
    plt.ylabel('País')
    plt.savefig('top7Paises.png')
    plt.show()
    
   
    #3.2. Gráfica del producto más vendido en los 5 países con más compras

    top_paises = df.groupby('Country')['Quantity'].sum().sort_values(ascending=False).head(5).index.tolist()
    df_top = df[df['Country'].isin(top_paises)]
    ventas_top = df_top.groupby(['Country', 'Description'])['Quantity'].sum().reset_index()
    producto_top = ventas_top.sort_values('Quantity', ascending=False).groupby('Country').first().reset_index()

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=producto_top,
        x='Country',
        y='Quantity',
        hue='Description',
        dodge=False,
        palette='Set2'
    )


    for index, row in producto_top.iterrows():
        plt.text(
            x=index,
            y=row['Quantity'] + 50,
            s=row['Description'],
            ha='center',
            fontsize=9,
            rotation=45
        )

    plt.title('Producto más vendido en los 5 países con más compras')
    plt.ylabel('Unidades Vendidas')
    plt.xlabel('País')
    plt.legend().remove()
    plt.tight_layout()
    plt.savefig('TopProductoPais.png')
    plt.show()
    
   
    
    #3.3- Gráfica de las ventas realizadas en cada mes
    ventas_por_mes=df.copy()
    ventas_por_mes['Mes'] = ventas_por_mes['InvoiceDate'].dt.to_period('M')
    ventas_por_mes = ventas_por_mes.groupby('Mes')['Quantity'].sum().reset_index()
    ventas_por_mes['Mes'] = ventas_por_mes['Mes'].astype(str)  # Convertir a string para mejor visualización

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=ventas_por_mes, x='Mes', y='Quantity', marker='o', color='seagreen', linewidth=2)
    plt.title('Transacciones por mes', fontsize=14)
    plt.xlabel('Mes')
    plt.ylabel('Cantidad vendida')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('ventasMes.png')
    plt.show()
    
    

    #3.4. Gráfica de los 10 clientes que más compran
    top_customers = df.groupby('CustomerID')['Quantity'].sum().sort_values(ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_customers.values, y=top_customers.index.astype(int), palette="mako")
    plt.title('Top 10 clientes que más compran')
    plt.xlabel('Customer ID')
    plt.ylabel('Cantidad total comprada')
    plt.savefig('Top10Clientes.png')
    plt.show()
    
    
    #3.5. Gráfica de Postage por país

    postage_df = df[df['Description'].str.contains("POSTAGE", case=False, na=False)]
    postage_by_country = postage_df.groupby('Country')['Quantity'].sum().sort_values(ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(x=postage_by_country.values, y=postage_by_country.index, palette="coolwarm")
    plt.title('Cantidad de POSTAGE por país')
    plt.xlabel('Cantidad de POSTAGE')
    plt.ylabel('País')
    plt.savefig('PostagePais.png')
    plt.show()
    
    
    # Calcular registros eliminados
    registros_eliminados = registros_iniciales - len(df)
    return df, registros_eliminados
    
    # Uso de la función
if __name__ == "__main__":
    archivo = r"J:\Personal\Maestría Ciencia de Datos\Clases\2. Fundamentos de Ciencias de Datos\Semana 1\Taller 1\dataset\OnlineRetail_1.xlsx"  # Reemplazar con la ruta real
    datos_limpios, eliminados = limpiar_datos(archivo)
    
    print(f"Proceso de limpieza completado.")
    print(f"Registros eliminados: {eliminados}")
    print(f"Registros restantes: {len(datos_limpios)}")
    
    # Guardar los datos limpios
    datos_limpios.to_excel('datos_limpios.xlsx', index=False)
    
    
    
    
    