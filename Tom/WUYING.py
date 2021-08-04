import os


'''with open("C:\\Users\\Li\\Desktop\中国（txt）\\202001020239519.txt",encoding='utf-8') as f:
    data=f.read()
print(data)'''



path=os.listdir(r"C:\Users\Li\Desktop\中国（txt）")
datalist=[]


for i in path:
    domain= r"C:\Users\Li\Desktop\中国（txt）\\"+i
    #print(domain)
    with open(domain,"r",encoding="utf-8") as f:
        data=f.read()

        datalist.append(data)
print(datalist[0])
