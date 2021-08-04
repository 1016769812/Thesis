import pandas as pd

sent = pd.read_csv(r"C:\Users\Li\Desktop\8latoutput.csv-bow-all-sent.csv")
vv = pd.read_csv(r"C:\Users\Li\Desktop\8latoutput.csv-bow-all-vv.csv")


value_p=sent[['care_p','fairness_p','loyalty_p','authority_p','sanctity_p']]
#print(value_p)
value_average=value_p.mean(axis=0)
print(value_average)
df1=pd.DataFrame(value_average)
print(df1)
df1.to_csv(r"C:\Users\Li\Desktop\value_average.csv")


vv_p=vv[['care.virtue','fairness.virtue','loyalty.virtue','authority.virtue','sanctity.virtue','care.vice','fairness.vice','loyalty.vice','authority.vice','sanctity.vice'
]]
vv_average=vv_p.mean(axis=0)
print(vv_average)
df2=pd.DataFrame(vv_average)
print(df2)
df2.to_csv(r"C:\Users\Li\Desktop\vv_average.csv")
#df2.to_excel(r"C:\Users\Li\Desktop\vv_average.xls")