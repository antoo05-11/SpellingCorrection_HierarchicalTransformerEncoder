import pandas as pd

data = pd.read_csv('data/vi_processed.csv')

correct_text = []
error_text = []

for index, row in data.iterrows():
    correct_text.append(row.correct_text)
    error_text.append(row.error_text)
