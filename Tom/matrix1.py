import pandas as pd

df=pd.read_excel(r"C:\Users\Li\Desktop\LAD关键词.xlsx",index_col="序号")
theme=data['主题']

import pandas as pd
df_dict = {}
df_list = []

for i in df.index.values:
    # loc为按列名索引 iloc 为按位置索引，使用的是 [[行号], [列名]]
    df_line = df.loc[i, ['列1', '列2', '列3', '列4', ]].to_dict()
    # 将每一行转换成字典后添加到列表
    df_list.append(df_line)


    df_list = []
    for i in df.index.values:
        # loc为按列名索引 iloc 为按位置索引，使用的是 [[行号], [列名]]
        df_line = df.loc[i, ['列1', '列2', '列3', '列4',]].to_dict()
        # 将每一行转换成字典后添加到列表
        df_list.append(df_line)
    df_dict['data'] = df_list

    return df_dict
