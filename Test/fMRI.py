
import pandas as pd
df = pd.read_csv('PC.csv')
df.drop('Unnamed: 0',1,inplace=True)
fMRI = df.as_matrix()