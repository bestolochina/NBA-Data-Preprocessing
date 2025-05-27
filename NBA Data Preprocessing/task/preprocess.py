import pandas as pd
import os
import requests

# Checking ../Data directory presence
if not os.path.exists('../Data'):
    os.mkdir('../Data')

# Download data if it is unavailable.
if 'nba2k-full.csv' not in os.listdir('../Data'):
    print('Train dataset loading.')
    url = "https://www.dropbox.com/s/wmgqf23ugn9sr3b/nba2k-full.csv?dl=1"
    r = requests.get(url, allow_redirects=True)
    open('../Data/nba2k-full.csv', 'wb').write(r.content)
    print('Loaded.')

data_path = "../Data/nba2k-full.csv"

def clean_data(path: str) -> pd.DataFrame:
    """
    Load and clean NBA player data from a CSV file.

    This function performs the following steps:
    - Reads the dataset from the specified CSV path.
    - Converts birth date and draft year columns to datetime.
    - Fills missing team names with 'No Team'.
    - Extracts and converts metric height and weight to float.
    - Removes dollar sign from salary and converts to float.
    - Replaces non-USA countries with 'Not-USA'.
    - Replaces 'Undrafted' draft round values with '0'.

    Parameters:
        path (str): File path to the CSV file.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df = pd.read_csv(path)
    df['b_day'] = pd.to_datetime(df['b_day'], format='%m/%d/%y')
    df['draft_year'] = pd.to_datetime(df['draft_year'], format='%Y')
    df['team'] = df['team'].fillna('No Team')
    df['height'] = df['height'].apply(
        lambda s: s.split('/')[1] if isinstance(s, str) and '/' in s else None
    ).astype('float')
    df['weight'] = df['weight'].apply(
        lambda s: s.split('/')[1].removesuffix('kg.') if isinstance(s, str) and '/' in s else None
    ).astype('float')
    df['salary'] = df['salary'].apply(
        lambda s: s.replace('$', '') if isinstance(s, str) else None
    ).astype('float')
    df.loc[df['country'] != 'USA', 'country'] = 'Not-USA'
    df.loc[df['draft_round'] == 'Undrafted', 'draft_round'] = '0'

    return df


clean_data(data_path)
