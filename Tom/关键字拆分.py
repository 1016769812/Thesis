import re
import linecache


def fileParse():
    inputfile = r"C:\Users\Li\Desktop\美国（txt）\all.txt"

    fp = open(r"C:\Users\Li\Desktop\美国（txt）\all.txt",'r',encoding='utf-8')

    number = []
    lineNumber = 1
    keyword = 'Document'  ##输入你要切分的关键字
    outfilename = r"C:\Users\Li\Desktop\美国（txt）\split\PRout"  ##输出文件名，如out.txt则写out即可，后续输出的文件是out0.txt,out1.txt...

    for eachLine in fp:
        m = re.search(keyword, eachLine)  ##查询关键字
        if m is not None:
            number.append(lineNumber)  # 将关键字的行号记录在number中
        lineNumber = lineNumber + 1
    size = int(len(number))

    for i in range(0, size - 1):
        start = number[i]
        end = number[i + 1]
        destLines = linecache.getlines(inputfile)[start + 1:end - 1]  # 将行号为start+1到end-1的文件内容截取出来
        fp_w = open(outfilename + str(i) + '.txt', 'w',encoding='utf-8')  # 将截取出的内容保存在输出文件中
        for key in destLines:
            fp_w.write(key)
        fp_w.close()


if __name__ == "__main__":
    fileParse()