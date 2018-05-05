# -*- coding: utf-8 -*-
import config
import numpy as np
train_X, train_Y = [],[]

def combine_fv(fv):
    # this function is to combine a week's feature vector [{day1},{day2}...] into {day1-7}
    res = fv[0]
    for key in res:
        res[key] = np.array(res[key])
        for i in range(1,len(fv)):
            res[key] += np.array(fv[i][key]) 
    return res
    
def normalize(data):
    #this function is to normalize a data(list, ultimate dimension)
    length = len(data[0])
    data = np.array(data,dtype = np.float32)
    for n in range(length):
        data[:,n][:] = data[:,n][:]/np.sum(data[:,n])
    return data
    
def judge_label(point, POI):
    #this function is to judge which POI this point belongs to. 
    #	1)Basic algorithm uses Python-Shapely to check if a polygon contains a point.
    #	2)Special circumstance is that all POI don't contain this point, reason is that we ignore some area in the map when we define POI region(polygon define is not full-map-coveraged). Under this, we use min-distance to define this point's type
    from shapely.geometry import Point
    from shapely.geometry.polygon import Polygon
    point_shapely = Point(point[0],point[1])
    distance = {}# a dict to save distance between point to a polygon.
    
    for label in POI:
        area = POI[label]
        
        if len(area) < 2:
            print('a POI region define error, go check POI.txt\n')
            continue
        if len(area) == 2:# if this polygon just define by two coordinate, we judge this polygon is a rectangle defined by two coordinates
        #First Turn Test Result��This polygon contains Alg didn't work, even I judge this point is definitely in a rec box, This is because Polygon make alg is influenced by points' order. So we should adjust points' order correctly.
            minX = min(area[0][0],area[1][0])
            minY = min(area[0][1],area[1][1])
            maxX = max(area[0][0],area[1][0])
            maxY = max(area[0][1],area[1][1])
            area = [(minX,minY),(minX,maxY),(maxX,maxY),(maxX,minY)]

        polygon = Polygon(area)
        mid = np.sum(area, axis = 0)/len(area)
        distance.setdefault(label, np.linalg.norm(mid - np.array([point[0],point[1]])))
        if polygon.contains(point_shapely):
            #print('region contains' + str(label)+ '\n')
            return label
            
    # 2) no region contains this point.
    #print('no region contains this point, distance calc mode.\n')
    return min(distance, key = distance.get)
    #attention, time to test label correctness. then think about feature vector then input it into nn.
    

def read_feature(stayPoint_path):
    # this function open a txt file(stay point), combining with POI file, return this sp's feature vector.
    # e.g. fv: {lib:{time: 3, minutes: 260.5}};
    stay_points = load_stay_point(stayPoint_path)
    POI = load_POI()
    # now we got a stay points' list, [[lng, lat, start_time, end_time],...,[]] and a POI dict, {1:[[x1,y1],[x2,y2]]}, then we use this two structure to calc this stay point file's feature_vector

    feature_vector = merge_sp_POI(stay_points, POI)
    
    return feature_vector


def merge_sp_POI(sp, POI):
    # this function is to merge stay_points(a list) and POI (a dict), then create a new dict that represents feature vector. 
    # e.g. feature_vector: {1:[seconds, time]}
    from datetime import datetime
    # here, we use python's time lib to calc time interval. (input two time string and output a float)
    
    feature_vector = {}
    for key in POI:
        feature_vector.setdefault(key,[0.0,0])
    
    for point in sp:
        # judge_label could return point label of this POI.
        label = judge_label(point, POI)
        time_format = '%H:%M:%S'
        diff = datetime.strptime(point[3].split(' ')[1], time_format)  - datetime.strptime(point[2].split(' ')[1], time_format)
        diff = diff.total_seconds()
        feature_vector[label][0] += diff
        feature_vector[label][1] += 1
    return feature_vector
    
def load_stay_point(filePath):
# this function is to read a stay point file's info. input a stay_point file' path. Return sp as a list [[1,1, 9:00:00, 10:00:00 ]...[lng, lat, start, end]]
    stay_points = []
    try:
        file_obj = open(filePath)
    except Exception:
        print('open this error' + filePath +'\n')
        return False
        
    for line in file_obj:
        if line.split(',')[0] == 'end_time' or line == '\n':
            continue
        lat = float(line.replace('\n','').split(',')[1])
        lng = float(line.replace('\n','').split(',')[2])
        end_time = line.replace('\n','').split(',')[0]
        start_time = line.replace('\n','').split(',')[4]
        stay_points.append([lng,lat,start_time,end_time])
    file_obj.close()
    
    return stay_points
    
def load_POI(filePath = config.POI_path):
#this function is to read POI file, input POI file, return all POI as a dict: {1:[[1,1],[2,2]]...,n:[[lng1,lat1],[lng2,lat2]]}
    POI = {}
    try:
        file_obj = open(filePath)
    except Exception:
        print('open this error' + filePath +'\n')
        return False
    
    for line in file_obj:
        if line.split(',')[0] == 'class' or line =='\n' or line[0] == '#':
            continue
        label = int(line.replace('\n','').split(',')[0])
        x1 = float(line.replace('\n','').split(',')[1])
        y1 = float(line.replace('\n','').split(',')[2])
        x2 = float(line.replace('\n','').split(',')[3])
        y2 = float(line.replace('\n','').split(',')[4])
        # improve: define POI as polygon, just modify code here. Define xn yn to save multiple coordinates, just use a for i in range(len(line.split(',')): operate_x_y)
        POI.setdefault(label,[[x1,y1],[x2,y2]])
    return POI
    
def data_to_feature(filePath = 'E:\College\������\����\����\���ݷ���\���1\\'):
    # this function is to read a classification standard(dir-setting-based) then read all data into train_X and train_Y, 1-weeks' data as one train_X, auto-recognize .txt file name. return train_X and train_Y as lists.
    train_X, train_Y = [],[]
    import os
    length_label = 0
    filePath = config.stayPoint_path
    label_name = os.listdir(filePath)
    label_length = len(label_name)
    dict_feature_with_label = {}
    for label in sorted(label_name):
        feature_vector = []
        for root, dirs, files in os.walk(filePath + label):
            if files:
                feature_vector.append([])
                trainData_names = sorted(list(set([file.split('gps')[0] for file in files])))
                current_name = trainData_names[0]
                
                for file in sorted(files):
                    if file.split('gps')[0] != current_name:
                        trainData_names.remove(current_name)
                        current_name = file.split('gps')[0]
                        feature_vector.append([])
                        feature_vector[-1].append(read_feature(root+'\\'+file))
                        
                    else:
                        feature_vector[-1].append(read_feature(root+'\\'+file))
        dict_feature_with_label.setdefault(label,feature_vector)	
    
    # here we got a feature_vector contains a week's feature day by day, now we need to combine them together to get a total-week one.
    for label in dict_feature_with_label:
        feature_vector = dict_feature_with_label[label]
        for week in feature_vector:
            res = combine_fv(week)
            #normalize input data(train_X)
            train_X.append(normalize(list(res.values())))
            train_Y.append(label)

                    
    return train_X,train_Y

                    
