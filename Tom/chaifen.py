open_diff = open(r"C:\Users\Li\Desktop\美国（txt）\all.txt", 'r',encoding='utf-8')  # 源文本文件
diff_line = open_diff.readlines()

line_list = []
for line in diff_line: line_list.append(line)

count = len(line_list)  # 文件行数
print('源文件数据行数：', count)
# 切分diff
diff_match_split = [line_list[i:i + 50] for i in range(0, len(line_list), 50)]  # 每个文件的数据行数

# 将切分的写入多个txt中
for i, j in zip(range(0, int(count / 50 + 1)), range(0, int(count / 50 + 1))):
    with open(r'C:\Users\Li\Desktop\美国（txt）\split\us%d.txt'% j,'w+',encoding='utf-8') as temp:
        for line in diff_match_split[i]: temp.write(line)
    print('拆分后文件的个数：', i + 1)

