import pandas as pd
import numpy as np

d = {'x':1, 'y':2, 'z':3}
for i in range(5):
    print('\n')
    for key in d.keys():
        print(key + '  ')


rootPath = './data'
result_fileName = '/simResult.txt'
file = open(rootPath + result_fileName,'w')
similarity = {1:2,2:4,3:6}
file.write(str(similarity))

file.close()
