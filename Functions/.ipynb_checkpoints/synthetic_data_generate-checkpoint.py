import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to generate seasonal weather data with realistic seasonal changes
def seasonal_variation(day_of_year, min_value, max_value, noise_std=0):
    # Use a sine wave to simulate seasonality
    base_value = min_value + (max_value - min_value) * (np.sin(2 * np.pi * day_of_year / 365))
    noise = np.random.normal(loc=0, scale=noise_std, size=len(day_of_year))
    return base_value + noise

# Function to determine appropriate min/max values for temperature, humidity, and wind speed
def get_seasonal_ranges(day_of_year):
    temperature_min = -5 + 10 * np.sin(2 * np.pi * day_of_year / 365)
    temperature_max = 5 + 20 * np.sin(2 * np.pi * day_of_year / 365)

    humidity_min = 30 + 20 * np.sin(2 * np.pi * day_of_year / 365)
    humidity_max = 60 + 20 * np.sin(2 * np.pi * day_of_year / 365)

    wind_speed_min = 0 + 5 * np.sin(2 * np.pi * day_of_year / 365)
    wind_speed_max = 5 + 5 * np.sin(2 * np.pi * day_of_year / 365)

    return (temperature_min, temperature_max), (humidity_min, humidity_max), (wind_speed_min, wind_speed_max)

# Function to generate daily energy consumption patterns
def daily_energy_pattern(hour, noise_std=0):
    morning_peak = (7 <= hour) & (hour < 9)
    daytime = (9 <= hour) & (hour < 17)
    evening_peak = (17 <= hour) & (hour < 21)
    night = ((21 <= hour) & (hour < 24)) | ((0 <= hour) & (hour < 7))

    base_energy_pattern = (
        morning_peak * np.random.uniform(1.1, 1.2, size=len(hour)) +
        daytime * np.random.uniform(1.0, 1.1, size=len(hour)) +
        evening_peak * np.random.uniform(1.2, 1.3, size=len(hour)) +
        night * np.random.uniform(0.8, 0.9, size=len(hour))
    )

    noise = np.random.normal(loc=0, scale=noise_std, size=len(hour))
    return base_energy_pattern + noise

# Function to generate synthetic weather and energy data
def generate_synthetic_data(start_date, end_date, freq='h', weather_noise_std=0.5, energy_noise_std=0.5, plot_percentage=0):
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    num_samples = len(date_range)

    np.random.seed(0)

    day_of_year = date_range.dayofyear
    hour_of_day = date_range.hour
    weekday = date_range.weekday

    # Get the seasonal min/max ranges
    (temperature_min, temperature_max), (humidity_min, humidity_max), (wind_speed_min, wind_speed_max) = get_seasonal_ranges(day_of_year)

    # Generate weather data with seasonal variation
    temperature = seasonal_variation(day_of_year, temperature_min, temperature_max, noise_std=weather_noise_std)
    humidity = seasonal_variation(day_of_year, humidity_min, humidity_max, noise_std=weather_noise_std)
    wind_speed = seasonal_variation(day_of_year, wind_speed_min, wind_speed_max, noise_std=weather_noise_std)

    # Create base energy consumption model
    base_energy_consumption = 100 + temperature * 3.5 - humidity * 1.2 + wind_speed * 0.1

    # Apply daily pattern with noise
    daily_pattern = daily_energy_pattern(hour_of_day, noise_std=energy_noise_std)
    energy_consumption = base_energy_consumption * daily_pattern

    # Smoothing the energy consumption by making it partially dependent on the previous hour's value
    energy_consumption = energy_consumption.to_numpy()
    for i in range(1, num_samples):
        energy_consumption[i] = 0.7 * energy_consumption[i-1] + 0.3 * energy_consumption[i]

    data = pd.DataFrame({
        'timestamp': date_range,
        'temperature': temperature,
        'humidity': humidity,
        'wind_speed': wind_speed,
        'energy_consumption': energy_consumption
    })

    # Plotting
    if plot_percentage > 0:
        plot_samples = int(len(data) * (plot_percentage / 100))
        plot_data = data.head(plot_samples)

        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 12), sharex=True)

        plot_data.plot(x='timestamp', y='temperature', ax=axes[0], title='Temperature')
        plot_data.plot(x='timestamp', y='humidity', ax=axes[1], title='Humidity')
        plot_data.plot(x='timestamp', y='wind_speed', ax=axes[2], title='Wind Speed')
        plot_data.plot(x='timestamp', y='energy_consumption', ax=axes[3], title='Energy Consumption')

        plt.tight_layout()
        plt.show()

    # Correlation analysis
    correlation = data[['temperature', 'energy_consumption']].corr().iloc[0, 1]
    print(f"Correlation between temperature and energy consumption: {correlation:.2f}")

    return data