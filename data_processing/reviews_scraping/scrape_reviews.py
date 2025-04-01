import pandas as pd
import requests
import time
from bs4 import BeautifulSoup
import logging
from tqdm import tqdm


# -------------------------------
# Setup Logging: Log to file and console
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    filename='wine_reviews_1.log',
    filemode='a',  # append mode
    format='%(asctime)s - %(levelname)s - %(message)s'
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger().addHandler(console)


# --------------------
# Load and Clean Data
# --------------------
def load_and_clean_data(csv_path):
    logging.info("Loading data from %s", csv_path)
    df = pd.read_csv(csv_path)
    logging.info("Initial DataFrame shape: %s", df.shape)

    nan_info = pd.DataFrame({
        'NaN Count': df.isna().sum(),
        'NaN Percentage': (df.isna().sum() / len(df)) * 100
    })
    logging.info("NaN Information:\n%s", nan_info)

    cols_to_check = ['country', 'designation', 'price', 'province', 'region_1', 'taster_name', 'variety']
    df = df.dropna(subset=cols_to_check)
    logging.info("DataFrame shape after dropna: %s", df.shape)
    return df

# --------------------------------
# Define the Helper Functions
# --------------------------------
def get_wine_id(wine_title):
    """Search for the wine on Vivino and extract its Wine ID."""
    search_url = f"https://www.vivino.com/search/wines?q={wine_title.replace(' ', '+')}"
    logging.info("---------------------------------------------------------------")
    logging.info("Searching: %s", search_url)

    headers = {
        "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/114.0.0.0 Safari/537.36")
    }

    response = requests.get(search_url, headers=headers)

    if response.status_code != 200:
        logging.error("Failed to load search page. Status Code: %s", response.status_code)
        return None, response.status_code

    soup = BeautifulSoup(response.text, "html.parser")
    wine_results = soup.find_all("div", class_="default-wine-card")

    if len(wine_results) == 1:
        wine_div = wine_results[0]
        if wine_div and wine_div.has_attr("data-wine"):
            wine_id = wine_div["data-wine"]
            logging.info("Found wine ID: %s", wine_id)
            return wine_id, response.status_code
    else:
        logging.warning("Multiple (%s) results found. Returning None.", len(wine_results))
        return None, response.status_code


def get_vivino_reviews(wine_id, per_page=10, language="en"):
    """Fetch reviews from Vivino's hidden API."""
    api_url = f"https://www.vivino.com/api/wines/{wine_id}/reviews?per_page={per_page}&language={language}"
    headers = {
        "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/114.0.0.0 Safari/537.36")
    }
    response = requests.get(api_url, headers=headers)

    if response.status_code != 200:
        logging.error("API request failed for wine_id %s. Status Code: %s", wine_id, response.status_code)
        return None, response.status_code

    reviews_json = response.json()
    if reviews_json:
        return reviews_json, response.status_code
    else:
        logging.info("No reviews found for wine_id %s.", wine_id)
        return None, response.status_code

def process_json(wine_id, wine_reviews, wine_title, reviews_df=None):
    if reviews_df is None:
        columns = ["title", "id"] + [f"review_{i+1}" for i in range(10)]
        reviews_df = pd.DataFrame(columns=columns)

    reviews_list = [(rev["note"], rev["rating"]) for rev in wine_reviews.get("reviews", [])]
    while len(reviews_list) < 10:
        reviews_list.append((None, None))
    new_row = pd.DataFrame([[wine_title, wine_id] + reviews_list], columns=reviews_df.columns)
    reviews_df = pd.concat([reviews_df, new_row], ignore_index=True)
    return reviews_df


def gather_reviews(title, reviews_df=None):
    wine_id, status_code_1 = get_wine_id(title)
    if wine_id:
        logging.info("Wine matched: %s", wine_id)
    elif status_code_1 == 200:
        logging.info("Wine not matched for title: %s", title)
        return reviews_df
    else:
        return reviews_df, 'error'

    wine_reviews, status_code_2 = get_vivino_reviews(wine_id)
    if wine_reviews:
        logging.info("Reviews gathered for wine_id: %s", wine_id)
    elif status_code_1 == 200:
        logging.info("No reviews found for wine_id: %s", wine_id)
        return reviews_df
    else:
        return reviews_df, 'error'

    reviews_df = process_json(wine_id, wine_reviews, title, reviews_df)
    return reviews_df

# ---------------------
# Main Loop Execution
# ---------------------
def main():
    csv_path = pd.read_csv("winemag-data-130k-v2.csv")
    df = load_and_clean_data(csv_path)
    names = df['title']
    names_to_process = names.iloc[35750:35750+2000]

    reviews_df = None
    file_count = 98

    check = 0

    for idx, name in zip(range(35751, 35751 + len(names_to_process)), tqdm(names_to_process, total=len(names_to_process))):
        print(f"\nProcessing wine #{idx}: {name}")
        backoff = 8
        while True:
            result = gather_reviews(name, reviews_df)
            if isinstance(result, tuple):
                logging.info("Error processing %s (possibly due to rate limiting). Retrying in %s seconds...", name, backoff)
                time.sleep(backoff)
                backoff *= 2
            else:
                reviews_df = result
                break
        check += 1

        # Every 1,000 gathered reviews, write the output to a CSV file and reset the DataFrame.
        if reviews_df is not None and check >= 500:
            check = 0
            filename = f"wine_reviews_{file_count}.csv"
            reviews_df.to_csv(filename, index=False)
            logging.info("Saved %s reviews to %s", len(reviews_df), filename)
            file_count += 1
            reviews_df = None

    if reviews_df is not None and not reviews_df.empty:
        filename = f"wine_reviews_{file_count}.csv"
        reviews_df.to_csv(filename, index=False)
        logging.info("Final batch: Saved %s reviews to %s", len(reviews_df), filename)

    logging.info("Review gathering completed.")

if __name__ == '__main__':
    main()