from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time

# Since yahoo finance is dynamic rendered in Javascript, it's almost impossible to scrape based on beautifulsoup after several trails.
# Thus I use selenium library to scrap the top 100 healthcare companies.

# set Chrome
chrome_options = Options()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--disable-gpu')
chrome_options.add_argument('--no-sandbox')

# initialize the driver
driver = webdriver.Chrome(options=chrome_options)
tickers= []
url = "https://finance.yahoo.com/screener/predefined/sec-ind_sec-largest-equities_healthcare/?count=100&offset=0" # here's the target URL

# scrapping
driver.get(url)
time.sleep(5) # wait for url loading

driver.execute_script("window.scrollTo(0, document.body.scrollHeight);") # roll the window
time.sleep(2)

# retrieve tickers
rows = driver.find_elements(By.XPATH, "//table[contains(@class, 'W(100%)')]/tbody/tr")
for row in rows:
    try:
        ticker = row.find_element(By.XPATH, './td[1]//a').text
        tickers.append(ticker)
    except Exception as e:
        print(f"Failed to retreive ticker information:{e}")

# remove any dupicates tickers
tickers = list(set(tickers)) 
print(f"Total tickers retrieved: {len(tickers)}")
print("The top 100 healthcare companies tickers:", tickers)

# close the chrome
driver.quit()

