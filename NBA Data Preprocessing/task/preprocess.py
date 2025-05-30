import numpy as np
import pandas as pd
import os
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

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


def multicol_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove features that exhibit multicollinearity based on Pearson correlation.

    This function identifies pairs of numerical features with strong correlation (|r| >= 0.5)
    and drops the feature from each pair that has the weaker absolute correlation with the target
    variable 'salary'. The goal is to reduce redundancy and improve model robustness.

    Parameters:
    ----------
    df : pd.DataFrame
        A DataFrame with numerical features and the target variable 'salary'.

    Returns:
    -------
    pd.DataFrame
        A new DataFrame with multicollinear features removed.
    """
    df_numerical = df.select_dtypes(include='number')
    features_corr_matrix = df_numerical.drop(columns=['salary']).corr(method='pearson')
    target_corr_matrix = df_numerical.corr(method='pearson')['salary']

    # Extract feature pairs with strong correlation from the upper triangle to avoid redundant comparisons
    upper_mask = np.triu(np.ones(features_corr_matrix.shape), k=1).astype(bool)
    upper_triangle = features_corr_matrix.where(upper_mask)
    strong_corr = upper_triangle.stack()[lambda x: x.abs() >= 0.5]

    # Select features to drop based on correlation with salary
    features_to_drop = set()
    for f1, f2 in strong_corr.index:
        if abs(target_corr_matrix[f1]) < abs(target_corr_matrix[f2]):
            features_to_drop.add(f1)
        else:
            features_to_drop.add(f2)

    # Drop selected features
    return df.drop(columns=features_to_drop)


def transform_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocesses the input DataFrame by scaling numerical features and one-hot encoding categorical features.

    Steps performed:
    - Separates the target variable 'salary' from the features.
    - Scales all numerical features using StandardScaler.
    - One-hot encodes all nominal categorical features using OneHotEncoder.
    - Concatenates the transformed numerical and categorical features (numerical first, then categorical).

    Parameters:
    df (pd.DataFrame): The input DataFrame with features and the target variable 'salary'.

    Returns:
    tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
        - X (pd.DataFrame): The transformed feature set.
        - y (pd.DataFrame): The target variable 'salary'.
    """
    features, target = df.drop(columns=['salary']), df['salary']
    num_feat_df = features.select_dtypes('number')  # numerical features
    cat_feat_df = features.select_dtypes('object')  # categorical features

    scaler = StandardScaler()
    num_feat_scaled = pd.DataFrame(scaler.fit_transform(num_feat_df),
                                   columns=num_feat_df.columns, index=num_feat_df.index)

    encoder = OneHotEncoder(sparse_output=False)
    encoder.fit(cat_feat_df)
    col_names = [name for batch in encoder.categories_ for name in batch]
    cat_feat_encoded = pd.DataFrame(encoder.transform(cat_feat_df),
                                    columns=col_names, index=cat_feat_df.index)

    X = pd.concat([num_feat_scaled, cat_feat_encoded], axis=1)

    return X, target


if __name__ == '__main__':
    path = "../Data/nba2k-full.csv"
    df_cleaned = clean_data(path)
    df_featured = feature_data(df_cleaned)
    df = multicol_data(df_featured)
    X, y = transform_data(df)

    answer = {
        'shape': [X.shape, y.shape],
        'features': list(X.columns),
    }
    print(answer)