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

    # Correlation Map
    corr_matrix = combined_data.corr()
    plt.figure(figsize=(16, 10))
    plt.title("Correlation Matrix of Financial and Economic Indicators")
    heatmap = sns.heatmap(corr_matrix, cmap='coolwarm', fmt=".2f", linewidths=1)

    for y in range(corr_matrix.shape[0]):
        for x in range(corr_matrix.shape[1]):
            plt.text(x + 0.5, y + 0.5, '%.2f' % corr_matrix.iloc[y, x],
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontsize=10,
                     color='black')
    plt.show()
    print("\n\n")
    # Graph
    axis_font = dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgrey',
        linecolor='black',
        linewidth=2,
        mirror=True,
        ticks='outside',
        tickfont=dict(
            family='sans-serif',
            size=12,
            color='black'
        ),
    )
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for company in list(data_organized_by_company.keys()):
        company_data = data_organized_by_company[company]
        mask = company_data['Final Score'] != 0
        fig.add_trace(go.Scatter(x=company_data['Year'][mask], y=company_data['Final Score'][mask], 
                                 mode='lines', name=company))
    fig.update_layout(
        title='Final Score of 20 Companies from 2003 to 2022',
        xaxis_title='Year',
        yaxis_title='Final Score',
        legend_title='Companies',
        font=dict(
            family="sans-serif",
            size=12,
            color="#000"
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=axis_font,
        yaxis=axis_font,
        legend=dict(
            font=dict(
                family='sans-serif',
                size=12,
                color='black'
            )
        )
    )
    fig.show()