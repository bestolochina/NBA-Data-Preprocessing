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


def feature_data(df: pd.DataFrame) -> pd.DataFrame:
    """
        Perform feature engineering and cleaning on the NBA player DataFrame.

        Steps performed:
        1. Convert the last two characters of the 'version' column to a datetime object
           representing the year (e.g., '20' → 2020).
        2. Create an 'age' feature calculated as the difference in years between 'version' and 'b_day'.
        3. Create an 'experience' feature calculated as the difference in years between 'version' and 'draft_year'.
        4. Compute the 'bmi' (body mass index) using the formula: bmi = weight / (height^2).
           Note: 'weight' is expected in kilograms and 'height' in meters.
        5. Drop original columns used for feature engineering: 'version', 'b_day', 'draft_year', 'weight', and 'height'.
        6. Identify and drop categorical columns with high cardinality (50 or more unique values).

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        pd.DataFrame
            The transformed DataFrame with new features and reduced dimensionality after dropping
            specified columns and high-cardinality categorical features.
        """
    df['version'] = pd.to_datetime(df['version'].str[-2:], format='%y')
    df['age'] = df['version'].dt.year - df['b_day'].dt.year
    df['experience'] = df['version'].dt.year - df['draft_year'].dt.year
    df['bmi'] = df['weight'] / (df['height']**2)
    df = df.drop(columns=['version', 'b_day', 'draft_year', 'weight', 'height'])
    high_card_features = df.select_dtypes(include='object').nunique() >= 50
    df = df.drop(columns=high_card_features[high_card_features].index)

    return df


if __name__ == '__main__':
    data_path = "../Data/nba2k-full.csv"
    df_cleaned = clean_data(data_path)
    df = feature_data(df_cleaned)
    print(df[['age', 'experience', 'bmi']].head())
