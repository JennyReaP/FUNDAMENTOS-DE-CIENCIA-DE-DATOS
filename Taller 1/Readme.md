# Fundamentos de Ciencia de Datos

## Taller 1: Adquisición, procesamiento y visualización de datos

**Dataset:** [Online Retail Data Set](https://archive.ics.uci.edu/dataset/352/online+retail)

---

## Parte 1: Adquisición y limpieza de datos

Se revisó el archivo `OnlineRetail.xlsx` con Excel para identificar características especiales. El dataset contiene **541,909 registros** de compras realizadas entre el **1 de diciembre de 2010 y el 9 de diciembre de 2011**.

### Decisiones de limpieza:
- Eliminación de registros duplicados.
- Eliminación de registros con valores nulos/negativos en `CustomerID`, `UnitPrice` y `Quantity` (transacciones no válidas).
- Filtrado de outliers usando percentiles (1%) en `Quantity` y `UnitPrice`.
- Reasignación de `StockCode` a códigos de 5 dígitos (00001-10000).
- Homologación de formatos:
  - `UnitPrice` y `Description` basados en `StockCode`.
  - `InvoiceDate` convertido a formato `datetime`.
  - Normalización a mayúsculas en `Description` y `Country`.
  - Reemplazo de valores vacíos en `CustomerID` (ej: 99999 para vacíos).
- Creación de columnas adicionales:
  - **TotalValue**: `Quantity * UnitPrice`.
  - **InvoiceDay**: Fecha en formato `aaaa-mm-dd`.

---

## Parte 2: EDA (Análisis Exploratorio de Datos)

**Registros finales tras limpieza:** 391,551  
**Descripción del dataset:** Tienda en línea de regalos ubicada en Reino Unido.

### Estructura del dataset:
| Columna       | Tipo de dato      | Tamaño requerido   | Descripción                                                                 |
|---------------|-------------------|--------------------|-----------------------------------------------------------------------------|
| `InvoiceNo`   | Object            | 5 caracteres       | Número de factura (si empieza con "C" es cancelada).                        |
| `StockID`     | Object            | 5 dígitos          | Código único del producto.                                                 |
| `Description` | Object            | N/A                | Nombre del producto.                                                       |
| `Quantity`    | int64             | 5 dígitos          | Cantidad comprada.                                                         |
| `InvoiceDate` | datetime64[ns]    | N/A                | Fecha y hora de la compra.                                                 |
| `UnitPrice`   | float64           | 2 decimales        | Precio unitario (en libras).                                               |
| `CustomerID`  | float64           | 5 dígitos          | ID del cliente.                                                            |
| `Country`     | Object            | N/A                | País de destino.                                                           |
| `InvoiceDay`  | datetime64[ns]    | N/A                | Fecha de compra (para agrupación).                                         |
| `TotalValue`  | float64           | 2 decimales        | Valor total (`Quantity * UnitPrice`).                                      |

### Ideas de análisis:
- **Cancelaciones**: Relación entre `InvoiceNo` y `CustomerID`.
- **Productos destacados**: `StockID` vs `Quantity` (transacciones especiales como descuentos).
- **Ventas por temporada**: `Description`, `TotalValue`, `InvoiceDay`, `Country`.
- **Clientes VIP**: Frecuencia de compras por `CustomerID`.
- **Logística**: Envíos por `Country`.

---

## Parte 3: Visualización

### Gráfico 1: Ventas por país
- Reino Unido lidera las ventas, seguido por Países Bajos, Irlanda, Alemania, Francia, Australia y Suiza.

### Gráfico 2: Productos más vendidos por país (Top 5)
- **Reino Unido**: Producto más vendido: *JUMBO BAG RED RETROSPOT* (~40,000 unidades).

### Gráfico 3: Ventas mensuales (2011)
- Pico de ventas en **noviembre** (época navideña).  
- *Nota*: Datos de diciembre 2011 incompletos.

### Gráfico 4: Clientes frecuentes
- Cliente **57656** es el comprador más activo (oportunidad para programas de fidelización).

### Gráfico 5: Envíos internacionales
- **Alemania** destaca con +1,000 productos comprados (oportunidad para convenios logísticos).

---

> **Nota**: Los gráficos referidos se generaron con Python (librerías: Pandas, Matplotlib/Seaborn).
