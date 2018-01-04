import math
import numpy as np

class tree_node(object):
    def __init__(self):
        self.data = '#'
        self.left_children = None
        self.right_children = None

class tree(tree_node):
    #create a tree
    def create_tree(self,t):
        data = t#going on, time to continue to code tree structure.

def load_POI_dataset():
    data = []
    file_obj = open('./POI.txt')
    for line in file_obj:
        if 'lng' in line or line == '\n':
            continue
        lng = float(line.replace('\n','').split('\t')[0])
        lat = float(line.replace('\n','').split('\t')[1])
        cate = line.replace('\n','').split('\t')[2]
        data.append([lng,lat,cate])
    POI_dataset = np.array(data)
    file_obj.close()

    return POI_dataset
        
def load_stay_points():
    stay_points = []
    try:
        file_obj = open('./stay_point_2017-11-27.txt')
    except Exception:
        return false
        
    for line in file_obj:
        if line.split(',')[0] == 'end_time' or line == '\n':
            continue
        y = float(line.replace('\n','').split(',')[1])
        x = float(line.replace('\n','').split(',')[2])
        end_time = line.replace('\n','').split(',')[0]
        start_time = line.replace('\n','').split(',')[4]
        stay_points.append([x,y,start_time,end_time])
    file_obj.close()
    
    return stay_points
    
def distance(p1,p2):
    try:
        len(p1) == 3 #debug
        len(p2) == 3
    except Exception:
        print(Exception + '\n your input arg is wrong, retry.');
        return -1;
        
    dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    return dist;

def interval(p1,p2):
    try:
        len(p1) == 3 
        len(p2) == 3
        p2[2] > p1[2]
    except Exception:
        print(Exception + '\n your input arg is wrong, retry.');
        return -1;
    
    timeInterval = p2[2] - p1[2]
    return timeInterval
    
def calc_stay_point(data, start, end):
    length_dataset = start - end + 1;
    sum_x  = 0.0;
    sum_y = 0.0;
    
    for point in data[start:end+1]:
        sum_x += point[0];
        sum_y += point[1];
    
    stay_point = [sum_x / length_dataset , sum_y/ length_dataset, data[start][2], data[end][2]]
    
    return stay_point
    
def calc_unique_POI(POI):
    categories = POI[:,2]
    num_category = len(set(categories))
    return num_category
    


    
    
    
    