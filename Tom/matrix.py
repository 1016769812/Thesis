# -*- coding: utf-8 -*-
# @Date    : 2017-04-05 09:30:04
# @Author  : Alan Lau (rlalan@outlook.com)
import numpy as np
import time
from pprint import pprint as p
import os


def log(func):
    def wrapper(*args, **kwargs):
        now_time = str(time.strftime('%Y-%m-%d %X', time.localtime()))
        print('------------------------------------------------')
        print('%s %s called' % (now_time, func.__name__))
        print('Document:%s' % func.__doc__)
        print('%s returns:' % func.__name__)
        re = func(*args, **kwargs)
        p(re)
        return re
    return wrapper


@log
def build_matirx(set_key_list):
    '''建立矩阵，矩阵的高度和宽度为关键词集合的长度+1'''
    edge = len(set_key_list)+1
    # matrix = np.zeros((edge, edge), dtype=str)
    matrix = [['' for j in range(edge)] for i in range(edge)]
    return matrix


@log
def init_matrix(set_key_list, matrix):
    '''初始化矩阵，将关键词集合赋值给第一列和第二列'''
    matrix[0][1:] = np.array(set_key_list)
    matrix = list(map(list, zip(*matrix)))
    matrix[0][1:] = np.array(set_key_list)
    return matrix


@log
def count_matrix_attr(matrix, formated_data):
    '''计算各个属性共现次数'''
    #zd=dbattr()[0]
    zd = {'贸易合作与投资': ['development',
  'cooperation',
  'project',
  'trade',
  'world',
  'people',
  'century maritime silk',
  'economy',
  'city',
  'area',
  'investment',
  'government',
  'industry',
  'company',
  'construction',
  ],
 '中国与东南亚合作共建': ['singapore',
  'city',
  'development',
  'investment',
  'government',
  'construction',
  'tailand',
  'infrastructure',
  'cooperation',
  'railway',
  'visit',
  'mr',
  'area',
  'zone',
  'laos'],
 '技术与产业': ['network',
  'industry',
  'research',
  'communication',
  'system',
  'service',
  'corporation',
  'report',
  'application',
  'design',
  'solution',
  'product',
  'association',
  'strategy',
  'association',
  'analysis',
  'logistic'],
 '市场及金融服务': ['market',
  'growth',
  'business',
  'payment',
  'group',
  'company',
  'loan',
  'service',
  'impact',
  'event',
  'commerce',
  'asia',
  'sector',
  'value',
  'financing',
  'canton fair'],
 '企业发展状况': ['bri',
  'employee',
  'partner',
  'client',
  'solution',
  'payment',
  'benefit',
  'platform',
  'ceo',
  'information',
  'employer',
  'copyright',
  'customer',
  'team',
  'startup',
  'bill'],
 '新冠疫情': ['vaccine','disease',
  'infection',
  'virus',
  'people',
  'case',
  'outbreak',
  'health',
  'infection',
  'globalisation',
  'test',
  'benaroya research',
  'immunology',
  'hubei'],
 '中国与南亚合作共建': ['cpec',
  'project',
  'energy',
  'power',
  'bri',
  'pakistani',
  'army',
  'investment',
  'country',
  'debt',
  'islamabad',
  'iran',
  'pakistan economic corridor'],
 '外交关系与国际竞争': [
  'security',
  'power',
  'competition',
  'sea',
  'governance',
  'tweet',
  'zhao',
  'cyber',
  'region',
  'community',
  'relation',
  'capability',
  'government',
  'state',
  'order'],
 '军备实力': ['beidou',
  'cooperation',
  'system',
  'hainan',
  'energy',
  'industry',
  'satellite',
  'bri',
  'coal',
  'beijing',
  'effort',
  'service',
  'expert',
  'capability',
  'precision'],
 '国家主权与安全': ['agent',
  'tibet',
  'espionage',
  'spy',
  'government',
  'security',
  'site',
  'case',
  'taiwan',
  'state',
  'prison',
  'mainland',
  'american',
  'department',
  'official'],
 '教育及人权保障': ['student',
  'refugee',
  'teacher',
  'silence',
  'crime',
  'learning',
  'phd',
  'netdragon',
  'consultancy',
  'master',
  'university',
  'teaching',
  'sentiment',
  'person',
  'education']}

    for row in range(1, len(matrix)):
        # 遍历矩阵第一行，跳过下标为0的元素
        for col in range(1, len(matrix)):
                # 遍历矩阵第一列，跳过下标为0的元素
                # 实际上就是为了跳过matrix中下标为[0][0]的元素，因为[0][0]为空，不为关键词
            if matrix[0][row] == matrix[col][0]:
                # 如果取出的行关键词和取出的列关键词相同，则其对应的共现次数为0，即矩阵对角线为0
                matrix[col][row] = str(0)
            else:
                counter = 0
                # 初始化计数器
                for ech in formated_data:
                        # 遍历格式化后的原始数据，让取出的行关键词和取出的列关键词进行组合，
                        # 再放到每条原始数据中查询
                    for w1 in zd[matrix[0][row]]:
                        for w2 in zd[matrix[col][0]]:
                            if w1 in ech and w2 in ech:
                                counter += 1
                            else:
                                continue
                matrix[col][row] = str(counter)
    return matrix



def main():

    output_path = r'C:\Users\Li\Desktop\三国网络关系图\中国matrix.csv'

    set_key_list = ['贸易合作与投资',
 '中国与东南亚合作共建',
 '技术与产业',
 '市场及金融服务','企业发展状况',
 '新冠疫情',
 '中国与南亚合作共建',
 '外交关系与国际竞争',
 '军备实力',
 '国家主权与安全',
 '教育及人权保障']


    path = os.listdir(r"C:\Users\Li\Desktop\中国（txt）")
    formated_data = []
    # len(path)

    for i in path:
        domain = r"C:\\Users\\Li\\Desktop\\中国（txt）\\"+ i
        with open(domain, "r", encoding="utf-8") as f:
            data = f.read().replace('Belt and Road','bri').lower()
            # print(data)
            formated_data.append(data)

    matrix = build_matirx(set_key_list)
    matrix = init_matrix(set_key_list, matrix)
    result_matrix = count_matrix_attr(matrix, formated_data)
    np.savetxt(output_path, result_matrix, fmt=('%s,'*len(matrix))[:-1])

if __name__ == '__main__':
    main()
