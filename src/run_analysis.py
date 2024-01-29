import os
import re
import csv
import json
import requests
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from fredapi import Fred
from datetime import datetime
from pmdarima import auto_arima
from plotly.subplots import make_subplots
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA

def normalize_variables(df, columns):
    normalized_df = df.copy()
    for column in columns:
        normalized_df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
    return normalized_df

def calculate_aggregate_score(row, normalized_df, weights):
        score_sum = 0
        weight_sum = 0
        for var in weights:
            if pd.notna(row[var]):
                score_sum += normalized_df[var][row.name] * weights[var]
                weight_sum += weights[var]
        return score_sum / weight_sum if weight_sum > 0 else np.nan
    
def assign_score_tier(score):
    if np.isnan(score) or score <= 0 or score > 1:
        return 0
    elif score < 0.2:
        return 1
    elif score < 0.4:
        return 2
    elif score < 0.6:
        return 3
    elif score < 0.8:
        return 4
    else:
        return 5

def main():
    # Import data
    years = range(2003, 2023)
    output_directory = 'data/processed/collected-dataframes'
    merged_dataframes = {}
    for year in years:
        country_csv_path = f'{output_directory}/df_{year}_country.csv'
        company_csv_path = f'{output_directory}/df_{year}_company.csv'
        country_df = pd.read_csv(country_csv_path)
        company_df = pd.read_csv(company_csv_path)
        repeated_country_df = pd.concat([country_df]*len(company_df), ignore_index=True)
        merged_df = pd.concat([company_df, repeated_country_df], axis=1)
        merged_dataframes[f'df_{year}'] = merged_df
        
    # Calculate aggregate score
    weights = {
        'Market Cap': 0.15,
        'EPS': 0.15,
        'P/E Ratio': 0.10,
        'Revenue Growth': 0.10,
        'Net Profit Margin': 0.10,
        'Stock Price': 0.05,
        'Revenue': 0.10,
        'NetIncomeLoss': 0.15,
        'GDP Growth (Annual Mean)': 0.05,
        'Inflation Rate (Annual Mean)': 0.03,
        'Unemployment Rate (Annual Mean)': 0.02
    }
    variables_to_normalize = ['Market Cap', 'EPS', 'P/E Ratio', 'Revenue Growth', 'Net Profit Margin', 
                            'Stock Price', 'Revenue', 'NetIncomeLoss', 'GDP Growth (Annual Mean)', 
                            'Inflation Rate (Annual Mean)', 'Unemployment Rate (Annual Mean)']
    for year_key, df in merged_dataframes.items():
        normalized_df = normalize_variables(df, variables_to_normalize)
        df['Aggregate Score'] = normalized_df.apply(calculate_aggregate_score, args=(normalized_df, weights), axis=1)
        merged_dataframes[year_key] = df
        
    # Assign tiers and combine data
    for year, df in merged_dataframes.items():
        df['Final Score'] = df['Aggregate Score'].apply(assign_score_tier)
        merged_dataframes[year] = df
        
    # Data organization
    combined_data = pd.concat(merged_dataframes.values(), ignore_index=True)
    data_organized_by_company = {}
    symbols = list(combined_data['Symbol'].unique())
    for symbol in symbols:
        symbol_df = combined_data[combined_data['Symbol'] == symbol]
        data_organized_by_company[symbol] = symbol_df
        
    # Building the model to predict the final score
    model_data = combined_data[combined_data['Final Score'] != 0]
    model_data = pd.get_dummies(model_data, columns=['Company Name', 'Symbol'])
    model_data = model_data.fillna(model_data.mean())
    model_data = model_data.drop('Aggregate Score', axis=1)
    x = model_data.drop('Final Score', axis=1)
    y = model_data['Final Score']
    column_names = x.columns
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=6)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    classifier = RandomForestClassifier(n_estimators=100, random_state=6)
    classifier.fit(x_train_scaled, y_train)
    y_pred = classifier.predict(x_test_scaled)
    print(f"Accuracy: {round(accuracy_score(y_test, y_pred), 2)}\n\n{classification_report(y_test, y_pred)}")
    """
    This code is designed to forecast financial and economic features for a specific company,
    in this case, Amazon, for the upcoming year. These forecasts will later serve as inputs 
    to the RandomForestClassifier predictive model we previously built, 
    aiming to estimate the target variable, which is the final score for the forthcoming year.
    """
    company_name = "Microsoft"
    company_symbol = "MSFT"
    year = "2023"
    features_list = []
    for feature in combined_data.columns:
        features_list.append(feature)
    features_list.remove('Year')
    features_list.remove('Final Score')
    features_list.remove('Company Name')
    features_list.remove('Symbol')
    warnings.filterwarnings("ignore")
    forecasts_dict = {'Year':year, 'Company Name': company_name, 'Symbol':company_symbol }
    for feature in features_list:
        company_data = combined_data[combined_data['Company Name'] == company_name]
        time_series = company_data.sort_values('Year').set_index('Year')[feature]
        uni_model = ARIMA(time_series, order=(1,0,0))
        uni_model_fit = uni_model.fit()
        forecast = uni_model_fit.forecast(steps=1)
        forecasts_dict[feature] = forecast.iloc[0]
    forecast_df = pd.DataFrame([forecasts_dict])
    if 'Company Name' in forecast_df.columns and 'Symbol' in forecast_df.columns:
        forecast_df = pd.get_dummies(forecast_df, columns=['Company Name', 'Symbol'])
    for column in column_names:
        if column not in forecast_df.columns:
            forecast_df[column] = 0 
    forecast_df = forecast_df.reindex(columns=column_names)
    forecast_df_scaled = scaler.transform(forecast_df)
    future_prediction = classifier.predict(forecast_df_scaled)
    print(f"Predicted Future Final Score for {company_name} for 2023 is:", future_prediction[0])
    print("\n\n")