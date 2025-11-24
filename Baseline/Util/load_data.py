import os
import pandas as pd


def read_excel(file_name):
    """
    Reads an Excel file from the 'Instance' folder located two levels above the current script directory.

    Parameters
    ----------
    file_name : str
        The name of the Excel file to be read.

    Returns
    -------
    list
        A list of rows from the Excel file, where each row is a list of values.
    """
    # Get the absolute path of the 'Instance' folder, two levels above the current script
    file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "Instance",
                             file_name)

    # Check if the file exists before reading
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Read the Excel file into a pandas DataFrame
    df = pd.read_excel(file_path)

    # Convert the DataFrame to a list of rows and return
    return df.values.tolist()