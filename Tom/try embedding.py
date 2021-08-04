
import pandas as pd

df_com = pd.read_csv(r"C:\Users\Li\Desktop\icc3-comments.csv", lineterminator='\n')
df_com = df_com[~df_com['text'].isin(['[removed]', '[deleted]' ])].dropna(subset=['text']).reset_index(drop=True)
print(df_com.head)
