import pandas as pd




def transform_function(X):
    """
    Processes the input DataFrame X by performing several transformations:
    - Converts date columns to datetime objects and extracts relevant date parts.
    - Fills NaN values for certain columns with default values.
    - Maps Acquisition Campaign values to predefined groups.
    - Drops unnecessary columns.

    Parameters:
    - X: pd.DataFrame - The input DataFrame to be transformed.

    Returns:
    - pd.DataFrame - The transformed DataFrame.
    """

    # Ensure that X is a pandas DataFrame
    if not isinstance(X, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    # Date Transformation: Convert 'Created Date_x' to datetime and extract date components
    X['Created Date_x'] = pd.to_datetime(X['Created Date_x'])
    X['Weekday'] = X['Created Date_x'].dt.dayofweek
    X['Day_of_Month'] = X['Created Date_x'].dt.day
    X['Month'] = X['Created Date_x'].dt.month

    # NaN Value Handling: Define default values for NaN in 'Acquisition Campaign'
    X['Acquisition Campaign'].fillna('Unknown', inplace=True)

    # Campaign Group Mapping: Define a mapping from various campaigns to campaign groups

    campaign_group_mapping = {
        'VirtualMeetups': 'VirtualMeetups',
        'TradeShow 6': 'TradeShow',
        'Digital kit': 'Digital Marketing',
        'ArtFair': 'ArtFair',
        'TradeShow': 'TradeShow',
        'EducationExpo 2': 'Education',
        'Corporate Connect': 'Corporate',
        'Follow-up: digital guide 2': 'Follow-up',
        'Follow-up: digital guide': 'Follow-up',
        'TradeShow 5': 'TradeShow',
        'FestivalFever': 'Events',
        'TradeShow 7': 'TradeShow',
        'Digital kit 2': 'Digital Marketing',
        'Stu Campaign': 'Community & Specials',
        'TradeShow 8': 'TradeShow',
        'Branding book': 'Others',
        'Digital kit 4': 'Digital Marketing',
        'TradeShow 3': 'TradeShow',
        'EducationExpo': 'Education',
        'Event Management Guide': 'Events',
        'FB Campaign': 'Others',
        'CommunityOutreach': 'Community & Specials',
        'EducationExpo 3': 'Education',
        'TechEvent': 'Events',
        'Digital kit 3': 'Digital Marketing',
        'June Launch': 'Events',
        'Specials': 'Community & Specials',
        'recommendation': 'Follow-up & Recommendations',
        'Recommendation 4': 'Follow-up & Recommendations',
        'Corporate Event Guide A': 'Corporate',
        'Recommendation 2': 'Follow-up & Recommendations',
        'EducationExpo 6': 'Education',
        'Healthcare': 'Others',
        'Stars': 'Others',
        'FB Webinar': 'Others',
        'Cryptoshow': 'Events',
        'EducationExpo 5': 'Education',
        'Recommendation 6': 'Follow-up & Recommendations',
        'Recommendation 3': 'Follow-up & Recommendations'
    }

    # Apply Campaign Group Mapping and fill NaN values for 'Campaign_Group'
    X['Campaign_Group'] = X['Acquisition Campaign'].map(campaign_group_mapping)
    X['Campaign_Group'].fillna('Others', inplace=True)

    # City and Use Case Handling: Fill NaN values with 'Other'
    X['City'].fillna('Other', inplace=True)
    X['Use Case_x'].fillna('Other', inplace=True)

    # Column Dropping: Define a list of columns to be dropped
    columns_drop = ['Id', 'First Name', 'Status_x', 'Discarded/Nurturing Reason',
                    'Converted', 'Close Date', 'Price', 'Created Date_x',
                    'Created Date_y', 'Loss Reason', 'Discount code',
                    'Status_y', 'Use Case_y', 'Pain']

    # Drop the specified columns, ignore errors if columns do not exist
    X.drop(columns=columns_drop, axis=1, inplace=True, errors='ignore')

    return X
