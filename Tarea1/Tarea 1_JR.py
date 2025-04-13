import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import matplotlib.pyplot as plt

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://api.open-meteo.com/v1/forecast"
params = {
	"latitude": 0.362418,
	"longitude": -78.155142,

	"hourly": ["soil_temperature_0cm", "soil_temperature_6cm", "soil_moisture_0_to_1cm", "soil_moisture_1_to_3cm", "precipitation_probability"],
	"timezone": "auto"
}
responses = openmeteo.weather_api(url, params=params)

# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]
print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation {response.Elevation()} m asl")
print(f"Timezone {response.Timezone()}{response.TimezoneAbbreviation()}")
print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

# Process hourly data. The order of variables needs to be the same as requested.
hourly = response.Hourly()
hourly_soil_temperature_0cm = hourly.Variables(0).ValuesAsNumpy()
hourly_soil_temperature_6cm = hourly.Variables(1).ValuesAsNumpy()
hourly_soil_moisture_0_to_1cm = hourly.Variables(2).ValuesAsNumpy()
hourly_soil_moisture_1_to_3cm = hourly.Variables(3).ValuesAsNumpy()
hourly_precipitation_probability = hourly.Variables(4).ValuesAsNumpy()

hourly_data = {"date": pd.date_range(
	start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
	end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
	freq = pd.Timedelta(seconds = hourly.Interval()),
	inclusive = "left"
)}

hourly_data["soil_temperature_0cm"] = hourly_soil_temperature_0cm
hourly_data["soil_temperature_6cm"] = hourly_soil_temperature_6cm
hourly_data["soil_moisture_0_to_1cm"] = hourly_soil_moisture_0_to_1cm
hourly_data["soil_moisture_1_to_3cm"] = hourly_soil_moisture_1_to_3cm
hourly_data["precipitation_probability"] = hourly_precipitation_probability


hourly_dataframe = pd.DataFrame(data = hourly_data)
hourly_dataframe.set_index('date', inplace=True)

# Gráfica con subplot
plt.figure(figsize=(15, 12))

# Gráfico 1: Temperaturas
plt.subplot(3, 1, 1)
plt.plot(hourly_dataframe.index, hourly_dataframe['soil_temperature_0cm'], label='Temperatura suelo 0cm', color='orange')
plt.plot(hourly_dataframe.index, hourly_dataframe['soil_temperature_6cm'], label='Temperatura suelo 6cm', color='brown')
plt.ylabel('Temperatura (°C)')
plt.title('Datos Meteorológicos Horarios')
plt.legend()
plt.grid(True)

# Gráfico 2: Humedad del suelo
plt.subplot(3, 1, 2)
plt.plot(hourly_dataframe.index, hourly_dataframe['soil_moisture_0_to_1cm'], label='Humedad 0-1cm', color='blue')
plt.plot(hourly_dataframe.index, hourly_dataframe['soil_moisture_1_to_3cm'], label='Humedad 1-3cm', color='lightblue')
plt.ylabel('Humedad del suelo')
plt.legend()
plt.grid(True)

# Gráfica de posible lluvia
plt.subplot(3, 1, 3)
plt.plot(hourly_dataframe.index, hourly_dataframe['precipitation_probability'], label='Probabilidad (%)', color='limegreen', linestyle='--', linewidth=1.5)

plt.ylabel('Lluvia mm / %')
plt.legend()
plt.grid(True)

plt.tight_layout()  
plt.savefig('tarea1.png')
plt.show()

