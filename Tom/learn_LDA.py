import pandas as pd


trp_sub = pd.read_csv(r"C:\Users\Li\Desktop\TheRedPill_submissions.csv", lineterminator='\n')
trp_com = pd.read_csv(r"C:\Users\Li\Desktop\TheRedPill_comments.csv", lineterminator='\n')

trp_sub = trp_sub[~trp_sub['selftext'].isin(['[removed]', '[deleted]' ])].dropna(subset=['selftext'])
trp_com = trp_com[~trp_com['body'].isin(['[removed]', '[deleted]' ])].dropna(subset=['body'])


trp_t = pd.merge(trp_sub, trp_com, how='inner', left_on='idstr', right_on='parent')


data_d = {}

for i, r in trp_t.iterrows():
    if r.idstr_x not in data_d.keys():
        data_d[r.idstr_x] = [r.selftext, r.body]
    else:
        data_d[r.idstr_x].append(r.body)
#print(data_d.values())
data = ['\n'.join(thread) for thread in data_d.values()]
print(data)