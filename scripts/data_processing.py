"""
Data Processing Script

This script processes raw Reddit data, cleaning text and preparing it for sentiment analysis.
"""

import os
import re
import pandas as pd
import datetime
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# download required NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# define data directories
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')

# create processed data directory if it doesn't exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def load_raw_data(filename):
    """
    Load raw data from a CSV file.

    Args:
        filename: Name of the CSV file to load

    Returns:
        Pandas DataFrame with loaded data
    """
    filepath = os.path.join(RAW_DATA_DIR, filename)
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} rows from {filepath}")
        return df
    except Exception as e:
        print(f"Error loading data from {filepath}: {e}")
        return pd.DataFrame()
    
def clean_text(text):
    """
    Clean and preprocess text data.

    Args:
        text: String text to clean

    Returns:
        Cleaned text string
    """
    if not isinstance(text, str):
        return ""
    
    # convert to lowercase
    text = text.lower()

    # remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # remove special characters and digits
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)

    # remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def tokenize_text(text):
    """
    Tokenize text and remove stopwords.

    Args:
        text: String text to tokenize

    Returns:
        List of tokens
    """
    tokens = word_tokenize(text)

    # remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    return tokens

def extract_hashtags(text):
    """
    Extract hashtags from text.

    Args:
        text: String text to extract hashtags from

    Returns:
        List of hashtags
    """
    if not isinstance(text, str):
        return []
    
    # find all hashtags
    hashtags = re.findall(r'#(\w+)', text)
    return hashtags

def process_data(df):
    """
    Process a DataFrame of Reddit posts.

    Args:
        df: Pandas DataFrame with Reddit posts

    Returns:
        Processed DataFrame
    """
    if df.empty:
        return df
    
    # create a copy
    processed_df = df.copy()

    # combine title and body for text analysis
    processed_df['combined_text'] = processed_df['title'] + ' ' + processed_df['body'].fillna('')

    # clean text
    processed_df['cleaned_text'] = processed_df['combined_text'].apply(clean_text)

    # tokenize text
    processed_df['tokens'] = processed_df['cleaned_text'].apply(tokenize_text)

    # extract hashtags
    processed_df['hashtags'] = processed_df['combined_text'].apply(extract_hashtags)

    # convert timestamp to datetime
    processed_df['created_datetime'] = pd.to_datetime(processed_df['created_utc'], unit='s')

    # drop columns that are no longer needed
    processed_df = processed_df.drop(['combined_text'], axis=1)

    print(f"Processed {len(processed_df)} posts")
    return processed_df

def save_processed_data(df, filename):
    """
    Save processed data to a CSV file.

    Args:
        df: Pandas DataFrame to save
        filename: Name for the processed CSV file
    """
    if df.empty:
        print("No data to save")
        return
    
    # create a processed filename
    processed_filename = f"processed_{filename}"
    filepath = os.path.join(PROCESSED_DATA_DIR, processed_filename)

    try:
        # convert lists to strings for CSV storage
        df_to_save = df.copy()
        df_to_save['tokens'] = df_to_save['tokens'].apply(lambda x: ' '.join(x))
        df_to_save['hashtags'] = df_to_save['hashtags'].apply(lambda x: ' '.join(x))

        # save to CSV
        df_to_save.to_csv(filepath, index=False)
        print(f"Saved processed data to {filepath}")

    except Exception as e:
        print(f"Error saving processed data: {e}")

def process_all_files():
    """
    Process all raw data files in the raw data directory.
    """
    # get all CSV files in raw data directory
    raw_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.csv')]

    if not raw_files:
        print("No raw data files found")
        return
    
    for file in raw_files:
        print(f"\nProcessing {file}...")
        df = load_raw_data(file)

        if not df.empty:
            processed_df = process_data(df)
            save_processed_data(processed_df, file)

def main():
    """
    Main function to process all raw data files.
    """
    print("Starting data processing...")
    process_all_files()
    print("\nData processing complete!")

if __name__ == "__main__":
    main()