import yfinance as yf

def download_data(ticker):
    print(f"Downloading data for {ticker}...")
    data = yf.download(ticker, start='2015-01-01', end='2024-12-31')
    data.to_csv(f'{ticker}_data.csv')
    print("Done!")

if __name__ == "__main__":
    download_data('AAPL')