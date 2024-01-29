import os
import re
import csv
import json
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from fredapi import Fred
from bs4 import BeautifulSoup
from datetime import datetime

def retrieve_company_tickers(headers):
    try:
        response = requests.get("https://www.sec.gov/files/company_tickers.json", headers=headers)
        response.raise_for_status()
        return pd.DataFrame.from_dict(response.json(), orient='index')
    except requests.exceptions.RequestException as err:
        raise(f"Request Error: {err}")
        
def get_CIK(ticker):
    cikl = topCompanies.loc[topCompanies['Ticker'] == ticker, 'CIK'].values[0]
    cik = str(cikl)
    return cik

def retrieve_financial_data(cik, user, variable):
    try:
        url = f'https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{variable}.json'
        response = requests.get(url, headers=user)
        response.raise_for_status()
        
        companyFacts = requests.get(f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json', headers=user)
        unit = next(iter(companyFacts.json()['facts']['us-gaap'][variable]['units'].keys()))
        data = pd.DataFrame.from_dict(response.json()['units'][unit])
        
        data['start'] = pd.to_datetime(data['start'])
        data['end'] = pd.to_datetime(data['end'])
        
        data = data[data['form'] == '10-K']
        data = data.sort_values(by='end', ascending=False).groupby(data['end'].dt.year).head(1)

        data['filing'] = data['end'] + pd.DateOffset(years=1)
        data['frame'] = np.nan
        data.rename(columns={'start': 'Start', 'end': 'End', 'val': 'Value'}, inplace=True)
        return data
    
    except requests.exceptions.RequestException as err:
        print(f"Request Error: {err}")
        return pd.DataFrame()
    
def month_diff(start_date, end_date):
    return (end_date.year - start_date.year) * 12 + end_date.month - start_date.month

def determine_year(start_date, end_date):
    if start_date.month < 7:
        return start_date.year
    else:
        return end_date.year
    
def get_fred_data(series_id, start, end):
    '''
    Retrieves time series data from the FRED database based on the given series id,
    where each series_id corresponds to a specific economic indicator series.
    :param series_id: The FRED series ID.
    :param start: The start date for the data retrieval (YYYY-MM-DD format).
    :param end: The end date for the data retrieval (YYYY-MM-DD format).
    :return: A Pandas DataFrame containing the retrieved data.
    '''
    fred = Fred(api_key='3dba2f16e182a60610bea249ca0e9580')
    data_series = fred.get_series(series_id, observation_start=start, observation_end=end)
    return data_series

def main():
    output_directory = 'data/processed/collected-dataframes'
    os.makedirs(output_directory, exist_ok=True)
    
    # Get the current Top 20 Companies by Market Capitalization
    data = pd.read_csv('marketcap.csv', index_col='Rank')
    data_sorted = data.sort_index()
    data_sorted = data_sorted.rename(columns={
        'marketcap': 'Market Cap (USD)',
        'price (USD)': 'Price (USD)',
        'country': 'Country'})
    data_df = data_sorted[data_sorted['Market Cap (USD)'] > 0]
    top20 = data_df[data_df.index <= 20]
    
    # Create a dictionary of the Top 20 Companies in the format 'Name: Symbol'
    ns_dict = top20.set_index('Name')['Symbol'].to_dict()
    
    # Create a DataFrame consisting of Names and Symbols
    names = list(ns_dict.keys())
    symbols = list(ns_dict.values())
    ns = pd.DataFrame({'Name': names, 'Symbol': symbols})
    ns = ns.set_index(top20.index)
    
    # Initializing the DataFrame
    years = list(range(2003, 2023))
    company_dataframes = {}
    for year in years:
        variables = {
            'Year': year,
            'Company Name': list(ns_dict.keys()),
            'Symbol': list(ns_dict.values()),
            'Stock Price': [0.0] * len(ns_dict),
            'Market Cap': [0] * len(ns_dict),
            'Revenue': [0.0] * len(ns_dict),
            'Revenue Growth': [0.0] * len(ns_dict),
            'NetIncomeLoss': [0.0] * len(ns_dict),
            'Net Profit Margin': [0.0] * len(ns_dict),
            'EPS': [0.0] * len(ns_dict),
            'P/E Ratio': [0.0] * len(ns_dict)
        }
        df_name = f'df_{year}'
        company_dataframes[df_name] = pd.DataFrame(variables)
        
    # Get Market Cap Data and turn it into a DataFrame
    mc_data = {}
    for symbol in symbols:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="max")
        tz = hist.index.tz # Timezone
        shares_outstanding = ticker.info['sharesOutstanding']
        hist['Market Cap'] = hist['Close'] * shares_outstanding
        mc_dict = {}
        for year in years:
            x = pd.Timestamp(f'{year + 1}-01-01', tz=tz)
            while x not in hist.index:
                if x.year > year + 1:
                    break
                x += pd.Timedelta(days=1)
            if x in hist.index:
                marketcap = hist.loc[x, 'Market Cap']
            else:
                marketcap = None
            mc_dict[year] = marketcap
        mc_data[symbol] = mc_dict
    mc_df = pd.DataFrame(mc_data)
    
    # Updating the DataFrame with 'Market Cap' Values
    for year in years:
        df_year = company_dataframes[f'df_{year}']
        for index, row in df_year.iterrows():
            symbol = row['Symbol']
            market_cap = mc_df.loc[year, symbol] if symbol in mc_df.columns else None
            df_year.at[index, 'Market Cap'] = market_cap
        company_dataframes[f'df_{year}'] = df_year
        
    # Sort the DataFrame by 'Market Cap'
    for year in years:
        df_year = company_dataframes[f'df_{year}']
        df_year_reset = df_year.reset_index()
        df_year_sorted = df_year_reset.sort_values(by=['Market Cap', 'index'], ascending=[False, True], na_position='last')
        df_year_sorted.set_index('index', inplace=True)
        company_dataframes[f'df_{year}'] = df_year_sorted
        
    user = {'User-Agent': "bedirian@usc.edu"}        
    allCompanies = retrieve_company_tickers(user)
    filteredCompanies = allCompanies[allCompanies['ticker'].isin(ns['Symbol'])]
    topCompanies = filteredCompanies.copy()
    topCompanies.rename(columns={'cik_str': 'CIK', 'ticker': 'Ticker', 'title': 'Company Name'}, inplace=True)
    topCompanies.reset_index(drop=True, inplace=True)
    topCompanies['CIK'] = topCompanies['CIK'].astype(str).str.zfill(10)
    
    # Obtain financial data
    variable_list = ['EarningsPerShareBasic', 'Revenues', 'NetIncomeLoss']
    financial_data = {}
    user = {'User-Agent': "bedirian@usc.edu"}
    for variable in variable_list:
        variable_data = []
        for index, row in topCompanies.iterrows():
            cik = row['CIK']
            try:
                data = retrieve_financial_data(cik, user, variable)
                data['CIK'] = cik
                data['Ticker'] = row['Ticker']
                data['Company Name'] = row['Company Name']
                data['Variable'] = variable
                variable_data.append(data[['CIK', 'Ticker', 'Company Name', 'Start', 'End', 'Value', 'Variable']])
            except Exception as e:
                print(f"Error retrieving data for {row['Company Name']} ({variable}): {e}")
        financial_data[variable] = pd.concat(variable_data)
    print("\n")

    # Determine the year for each row of data
    for key, df in financial_data.items():
        df['Start'] = pd.to_datetime(df['Start'])
        df['End'] = pd.to_datetime(df['End'])

        df['Timeframe (Months)'] = df.apply(lambda row: month_diff(row['Start'], row['End']), axis=1)
        filtered_df = df[df['Timeframe (Months)'] >= 10].copy()

        filtered_df['Year'] = filtered_df.apply(lambda row: determine_year(row['Start'], row['End']), axis=1)
        financial_data[key] = filtered_df
        
    # Append variable data
    for year in years:
        df = company_dataframes[f'df_{year}']
        for symbol in df['Symbol'].unique():
            eps_value = financial_data['EarningsPerShareBasic'][
                (financial_data['EarningsPerShareBasic']['Ticker'] == symbol) & 
                (financial_data['EarningsPerShareBasic']['Year'] == year)
            ]['Value']
            if not eps_value.empty:
                df.loc[df['Symbol'] == symbol, 'EPS'] = eps_value.iloc[0]
            else:
                df.loc[df['Symbol'] == symbol, 'EPS'] = None
        company_dataframes[f'df_{year}'] = df

    for year in years:
        df = company_dataframes[f'df_{year}']
        for symbol in df['Symbol'].unique():
            revenue_value = financial_data['Revenues'][
                (financial_data['Revenues']['Ticker'] == symbol) & 
                (financial_data['Revenues']['Year'] == year)
            ]['Value']
            if not revenue_value.empty:
                df.loc[df['Symbol'] == symbol, 'Revenue'] = revenue_value.iloc[0]
            else:
                df.loc[df['Symbol'] == symbol, 'Revenue'] = None
        company_dataframes[f'df_{year}'] = df

    for year in years:
        df = company_dataframes[f'df_{year}']
        for symbol in df['Symbol'].unique():
            net_value = financial_data['NetIncomeLoss'][
                (financial_data['NetIncomeLoss']['Ticker'] == symbol) & 
                (financial_data['NetIncomeLoss']['Year'] == year)
            ]['Value']
            if not net_value.empty:
                df.loc[df['Symbol'] == symbol, 'NetIncomeLoss'] = net_value.iloc[0]
            else:
                df.loc[df['Symbol'] == symbol, 'NetIncomeLoss'] = None
        company_dataframes[f'df_{year}'] = df
        
    # Download stock data
    yearly_avg_prices = {}
    for symbol in symbols:
        try:
            stock_df = yf.download(symbol, period='max', interval='3mo')
            if not stock_df.empty:
                stock_df['Period Avg'] = stock_df[['Open', 'High', 'Low', 'Close']].mean(axis=1)
                stock_df['Year'] = stock_df.index.year
                yearly_avg = stock_df.groupby('Year')['Period Avg'].mean()
                yearly_avg_prices[symbol] = yearly_avg
        except Exception as e:
            print(f"Failed to download data for {symbol}: {e}")
    for year in years:
        if f'df_{year}' in company_dataframes:
            df = company_dataframes[f'df_{year}']
            df['Stock Price'] = float('nan')
            for index, row in df.iterrows():
                symbol = row['Symbol']
                if symbol in yearly_avg_prices and year in yearly_avg_prices[symbol].index:
                    df.at[index, 'Stock Price'] = yearly_avg_prices[symbol][year]
            company_dataframes[f'df_{year}'] = df
    print("\n")
            
    # Organize each DataFrame in the dictionary
    desired_order = ['Year', 'Company Name', 'Symbol', 'Stock Price', 'Market Cap', 'Revenue', 'Revenue Growth', 'NetIncomeLoss', 'Net Profit Margin', 'EPS', 'P/E Ratio']
    for year in years:
        df = company_dataframes.get(f'df_{year}')
        df = df[desired_order]
        df = df.reset_index(drop=True)
        company_dataframes[f'df_{year}'] = df
        
    # Calculate P/E Ratio, Revenue Growth, and Net Profit Margin
    for year in years:
        if f'df_{year}' in company_dataframes:
            df = company_dataframes[f'df_{year}']
            df['P/E Ratio'] = df.apply(lambda row: row['Stock Price'] / row['EPS'] if row['EPS'] and row['EPS'] != 0 else float('nan'), axis=1)
            company_dataframes[f'df_{year}'] = df
    for year in years:
        current_df = company_dataframes.get(f'df_{year}')
        previous_df = company_dataframes.get(f'df_{year - 1}')
        if previous_df is not None:
            for index, row in current_df.iterrows():
                symbol = row['Symbol']
                current_year_revenue = row['Revenue']
                previous_row = previous_df[previous_df['Symbol'] == symbol]
                if not previous_row.empty and not pd.isna(previous_row.iloc[0]['Revenue']):
                    previous_year_revenue = previous_row.iloc[0]['Revenue']
                    if not pd.isna(current_year_revenue) and not pd.isna(previous_year_revenue):
                        revenue_growth = ((current_year_revenue - previous_year_revenue) / previous_year_revenue) * 100
                    else:
                        revenue_growth = np.nan
                else:
                    revenue_growth = np.nan
                current_df.at[index, 'Revenue Growth'] = revenue_growth
        else:
            current_df['Revenue Growth'] = np.nan
        company_dataframes[f'df_{year}'] = current_df
    for year in years:
        df = company_dataframes.get(f'df_{year}')
        if df is not None:
            for index, row in df.iterrows():
                net_income_loss = row['NetIncomeLoss']
                total_revenue = row['Revenue']
                if pd.isna(net_income_loss) or pd.isna(total_revenue) or total_revenue == 0:
                    net_profit_margin = np.nan
                else:
                    net_profit_margin = (net_income_loss / total_revenue) * 100
                df.at[index, 'Net Profit Margin'] = net_profit_margin
            company_dataframes[f'df_{year}'] = df
            
    fred = Fred(api_key='3dba2f16e182a60610bea249ca0e9580') # Create an instance of Fred from 'fredapi' package
    gdp_growth_id = 'A191RL1Q225SBEA'  # Quarterly
    inflation_rate_id = 'FPCPITOTLZGUSA'  # Annual
    unemployment_rate_id = 'UNRATE'  # Monthly
    start_year = '2003-01-01'
    end_year = '2022-12-31'
    gdp_growth = get_fred_data(gdp_growth_id, start_year, end_year)
    inflation_rate = get_fred_data(inflation_rate_id, start_year, end_year)
    unemployment_rate = get_fred_data(unemployment_rate_id, start_year, end_year)
    gdp_growth.index = gdp_growth.index.year
    inflation_rate.index = inflation_rate.index.year
    unemployment_rate.index = unemployment_rate.index.year
    gdp_growth = gdp_growth.groupby(gdp_growth.index).mean()
    inflation_rate = inflation_rate.groupby(inflation_rate.index).mean()
    unemployment_rate = unemployment_rate.groupby(unemployment_rate.index).mean()
    data = pd.concat([gdp_growth, inflation_rate, unemployment_rate], axis=1)
    data.columns = ["GDP Growth (Annual Mean)", "Inflation Rate (Annual Mean)", "Unemployment Rate (Annual Mean)"]
    
    # Create a dictionary of dataframes for each year's economic data
    country_dataframes = {}
    for year in range(2003, 2023):
        year_data = data.loc[year:year]
        year_df = pd.DataFrame(year_data)
        country_dataframes[f'df_{year}'] = year_df
        
    # Combine both dataframes
    combined_dataframes = {}
    for key in country_dataframes.keys():
        combined_data = {
            'country_df': country_dataframes[key],
            'company_df': company_dataframes[key]
        }
        combined_dataframes[key] = combined_data

    for year_key, dataframes_dict in combined_dataframes.items():
        country_df = dataframes_dict['country_df']
        company_df = dataframes_dict['company_df']
        country_csv_path = os.path.join(output_directory, f'{year_key}_country.csv')
        company_csv_path = os.path.join(output_directory, f'{year_key}_company.csv')
        country_df.to_csv(country_csv_path, index=False)
        company_df.to_csv(company_csv_path, index=False)
    print(f"Successfully saved all DataFrames to CSV files in '{output_directory}' directory!")
    print("\n")

if __name__ == "__main__":
    main()