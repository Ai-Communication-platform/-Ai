import pandas as pd

# Path to your Excel file
excel_file_path = 'C:\\Users\\win\\Documents\\GitHub\\-Ai\\감성대화말뭉치(최종데이터)_Training.xlsx'

# Load the Excel file
df = pd.read_excel(excel_file_path)

# Path where you want to save the CSV file
csv_file_path = 'C:\\Users\\win\Documents\\GitHub\\-Ai\\감성대화말뭉치(최종데이터)_Training.csv'

# Save as CSV
df.to_csv(csv_file_path, index=False, encoding='utf-8-sig')

print("Excel file has been converted to CSV and saved at:", csv_file_path)