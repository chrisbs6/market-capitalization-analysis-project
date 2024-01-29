import os
import requests

def download_file(url, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as file:
            file.write(response.content)
        print(f"File downloaded successfully: {filename}")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")

def main():
    file_url = "https://companiesmarketcap.com/usa/largest-companies-in-the-usa-by-market-cap/?download=csv"
    output_file = "data/raw/marketcap.csv"
    download_file(file_url, output_file)

if __name__ == "__main__":
    main()