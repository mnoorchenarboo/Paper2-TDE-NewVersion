import pandas as pd

import pandas as pd

def clean_missing_values(df):
    threshold = 0.2  # 20% threshold for missing values
    
    # Iterate through each column in the DataFrame
    for column in df.columns:
        # Calculate the percentage of missing values in the column
        missing_ratio = df[column].isna().mean()
        
        if missing_ratio > threshold:
            # Drop the column if more than 20% of values are missing
            print(f"Dropping column '{column}' with {missing_ratio*100:.2f}% missing values.")
            df.drop(columns=[column], inplace=True)
        else:
            # Fill missing values with the average of ten nearest non-missing neighbors
            print(f"Filling missing values in column '{column}' with {missing_ratio*100:.2f}% missing values.")
            
            # Define a function to calculate the mean of the ten nearest neighbors
            def fill_with_neighbors_avg(series):
                filled_series = series.copy()
                for i, value in enumerate(series):
                    if pd.isna(value):
                        # Get ten nearest non-missing values around the current missing value
                        neighbors = series[max(i-5, 0):i].dropna().append(series[i+1:i+6].dropna())
                        if len(neighbors) > 0:
                            # Fill with the average of these neighbors
                            filled_series[i] = neighbors.mean()
                return filled_series
            
            # Apply the function to the column to fill missing values
            df[column] = fill_with_neighbors_avg(df[column])

    return df

def remove_leading_trailing_zeros(df, column_name):
    while len(df) > 0 and df[column_name].iloc[0] == 0:
        df = df.iloc[1:]
    while len(df) > 0 and df[column_name].iloc[-1] == 0:
        df = df.iloc[:-1]
    return df

def load_and_fill_weather_data(weather_df, site_id):

    try:
        # Convert 'timestamp' to datetime
        weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'])
    
        # Set 'timestamp' as the index
        weather_df.set_index('timestamp', inplace=True)
    except:
        pass
        
    # Filter the DataFrame by 'site_id'
    filtered_weather = weather_df[weather_df['site_id'] == site_id]

    # Select relevant columns
    # columns_of_interest = ['airTemperature', 'dewTemperature', 'seaLvlPressure', 'windDirection', 'windSpeed']
    columns_of_interest = ['airTemperature', 'dewTemperature', 'seaLvlPressure', 'windSpeed']
    filtered_weather = filtered_weather[columns_of_interest]

    for col in columns_of_interest:
        filtered_weather = remove_leading_trailing_zeros(filtered_weather, column_name=col)
    # Fill missing values forward and backward without limit
    filled_weather = filtered_weather.ffill().bfill()
    
    return filled_weather
    
def extract_unique_components(df):
    # Initialize sets to store unique components
    site_ids = set()
    primary_uses = set()
    building_names = set()
    
    # Iterate over each column name
    for col in df.columns:
        # Split the column name into components
        parts = col.split('_')
        if len(parts) == 3:
            site_id, primary_use, building_name = parts
            site_ids.add(site_id)
            primary_uses.add(primary_use)
            building_names.add(building_name)
        # else:
        #     print(f"Column '{col}' does not conform to the expected naming convention.")
    
    # Convert sets to sorted lists
    site_ids = sorted(site_ids)
    primary_uses = sorted(primary_uses)
    building_names = sorted(building_names)
    
    # # Display the unique values in list format
    # print("Unique SiteIDs:")
    # print(site_ids)
    # print("\nUnique PrimaryUses:")
    # print(primary_uses)
    # print("\nUnique BuildingNames:")
    # print(building_names)
    return site_ids, primary_uses, building_names
    



import pandas as pd
import numpy as np

def get_datetime_features(df, cos_sin=False):
    # Ensure the index is in datetime format
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors='coerce')

    # Extracting date-time features from the index
    df['Hour'] = df.index.hour
    df['DayOfWeek'] = df.index.dayofweek
    df['DayOfMonth'] = df.index.day
    df['Month'] = df.index.month
    df['DayOfYear'] = df.index.dayofyear

    # Binary representation for weekends
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)

    if cos_sin:
        # Apply cosine and sine transformations for periodic features
        df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
        df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)

    return df

import pandas as pd
import numpy as np

def get_unique_values(df):
    # Sets to store unique values for each part
    site_ids = set()
    primary_uses = set()
    building_names = set()
    
    # Iterate through the column names
    for column in df.columns:
        # Split the column name by underscores
        parts = column.split('_')
        
        # Check if the column has exactly three parts
        if len(parts) == 3:
            site_ids.add(parts[0])         # First part is Site ID
            primary_uses.add(parts[1])      # Middle part is Primary Use
            building_names.add(parts[2])    # Last part is Building Name
    
    # Convert sets to lists for easier use
    return list(site_ids), list(primary_uses), list(building_names)

# # Example usage
# # Assuming `df` is your DataFrame
# site_ids, primary_uses, building_names = get_unique_values(df)

# print("Unique Site IDs:", site_ids)
# print("Unique Primary Uses:", primary_uses)
# print("Unique Building Names:", building_names)


# Assume predefined lists for SiteIDs and building_names
# Unique Site IDs: ['Wolf', 'Robin', 'Moose', 'Shrew', 'Peacock', 'Rat', 'Hog', 'Lamb', 'Swan', 'Bear', 'Bull', 'Crow', 'Panther', 'Mouse', 'Fox', 'Gator', 'Bobcat', 'Eagle', 'Cockatoo']
# Unique Primary Uses: ['services', 'health', 'education', 'lodging', 'industrial', 'unknown', 'office', 'retail', 'food', 'other', 'religion', 'utility', 'public', 'science', 'warehouse', 'parking', 'assembly']

def get_columns_with_term(df, term):
    # List to store column names that match the criteria
    matching_columns = []
    
    # Iterate through the column names
    for column in df.columns:
        # Split the column name by underscores
        parts = column.split('_')
        
        # Check if the term is in the middle part
        if len(parts) >= 3 and parts[1] == term:
            matching_columns.append(column)
    
    return matching_columns

import pandas as pd

# Load weather data with 'timestamp' as an index
weather_df = pd.read_csv('./Data/building-data-genome-project-2/weather.txt')


def get_column_by_criteria(df, primary_use, option_number=0):
    # Determine columns with the term
    columns_with_term = get_columns_with_term(df, primary_use)
    
    # Check if the option number is within the range
    if option_number >= len(columns_with_term):
        print("Option number out of range.")
        return None
    
    # Get the column name based on option number
    column_name = columns_with_term[option_number]
    
    # Check if the column exists in the DataFrame
    if column_name in df.columns:
        # Ensure 'timestamp' in df is set as index if available
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df.set_index('timestamp', inplace=True)
        
        # Extract the specified column as a DataFrame
        column_data = df[[column_name]]
        
        # Find first and last valid indices to slice around non-NaN data
        first_valid = column_data[column_name].first_valid_index()
        last_valid = column_data[column_name].last_valid_index()
        
        # Slice the DataFrame to remove leading and trailing NaNs
        if first_valid is not None and last_valid is not None:
            column_data = column_data.loc[first_valid:last_valid]
        
        # Print the number of missing values before filling
        missing_count = column_data[column_name].isna().sum()
        print(f"Number of missing values before filling: {missing_count}")
        
        # Fill missing values with one non-missing neighbor before and after
        column_data[column_name] = column_data[column_name].ffill().bfill()
        
        # Print the number of missing values after filling
        missing_count_after = column_data[column_name].isna().sum()
        print(f"Number of missing values after filling: {missing_count_after}")

        # Rename column for consistency
        column_data = column_data.rename(columns={column_name: 'energy_consumption'})

        # Apply datetime feature extraction on the index
        energy = get_datetime_features(column_data, cos_sin=False)

        # Clean and merge weather data with energy data
        site_id = column_name.split('_')[0]  # Extract site_id from column name
        weather_df_cleaned = load_and_fill_weather_data(weather_df, site_id=site_id)
        merged_df = pd.merge(energy, weather_df_cleaned, left_index=True, right_index=True, how='inner')

        # Define feature columns and filter merged DataFrame
        feature_names = ['energy_consumption', 'Hour', 'DayOfWeek', 'DayOfMonth', 'Month', 'DayOfYear', 'IsWeekend', 
                         'airTemperature', 'dewTemperature', 'seaLvlPressure', 'windSpeed']
        merged_df = merged_df[feature_names]

        # Print column details
        print(
            f"Primary Use: '{column_name.split('_')[1]}'\n"
            f"Building Name: '{column_name.split('_')[2]}'\n"
            f"SiteID: '{site_id}'"
        )
        more_info = output_string = f"Primary Use: {column_name.split('_')[1]}, Building Name: {column_name.split('_')[2]}, SiteID: {site_id}"

        # Clean missing values in the merged DataFrame
        cleaned_df = clean_missing_values(merged_df)

        return cleaned_df, more_info
    else:
        print(f"Column '{column_name}' not found in the DataFrame.")
        return None

# # Example call to the function
# primary_use = "industrial"  # Replace with the actual primary use you want to search for
# cleaned_df = get_column_by_criteria(df, primary_use, weather_df, option_number=0)


