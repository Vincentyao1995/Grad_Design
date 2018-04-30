import config

train_X, train_Y = [],[]

def judge_label(point, POI):
	#this function is to judge which POI this point belongs to. 
	#	1)Basic algorithm uses Python-Shapely to check if a polygon contains a point.
	#	2)Special circumstance is that all POI don't contain this point, reason is that we ignore some area in the map when we define POI region(polygon define is not full-map-coveraged). Under this, we use min-distance to define this point's type
	from shapely.geometry import Point
	from shapely.geometry.polygon import Polygon
	import numpy as np
	point_shapely = Point(point[0],point[1])
	distance = {}# a dict to save distance between point to a polygon.
	
	for label in POI:
		area = POI[label]
		
		if len(area) <= 2:# if this polygon just define by two coordinate, we judge this polygon is a rectangle defined by two coordinates
		#第一次测试结果：这个polygon contains的算法没有生效，即使我人眼判定是绝对在矩形框中的，现在推测这是由于创造polygon时的点输入顺序是有影响的。
			tmp1 = [min(area[0][0],area[1][0]),min(area[0][1],area[1][1])]
			tmp2 = [max(area[0][0],area[1][0]),max(area[0][1],area[1][1])]
			
			if tmp1 in area and tmp2 in area: # if define by (minx,miny),(maxX,maxY)
				tmp1 = [max(area[0][0],area[1][0]),min(area[0][1],area[1][1])]
				tmp2 = [min(area[0][0],area[1][0]),max(area[0][1],area[1][1])]
			area.append(tmp1)
			area.append(tmp2)

		polygon = Polygon([tuple(p) for p in area])
		mid = np.sum(area, axis = 0)/len(area)
		distance.setdefault(label, np.linalg.norm(mid - np.array([point[0],point[1]])))
		if polygon.contains(point_shapely):
			print('region contains' + str(label)+ '\n')
			return label
			
	# 2) no region contains this point.
	print('no region contains this point, distance calc mode.\n')
	return min(distance, key = distance.get)
	
	

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
	
def data_to_feature(filePath = 'E:\College\大四下\毕设\数据\数据分类\类别1\\'):
	# this function is to read a classification standard then read all data into train_X and train_Y, 1-weeks' data as one train_X, auto-recognize .txt file name. return train_X and train_Y as lists.
	import os
	length_label = 0
	filePath = config.stayPoint_path
	label_name = os.listdir(filePath)
	label_length = len(label_name)
	for label in sorted(label_name):
		for root, dirs, files in os.walk(filePath + label):
			if files:
				trainData_names = sorted(list(set([file.split('gps')[0] for file in files])))
				current_name = trainData_names[0]
				feature_vector = [[]]
				for file in sorted(files):
					if file.split('gps')[0] != current_name:
						trainData_names.remove(current_name)
						current_name = file.split('gps')[0]
						feature_vector.append([])
						feature_vector[-1].append(read_feature(root+'\\'+file))
						
					else:
						feature_vector[-1].append(read_feature(root+'\\'+file))
				for f in feature_vector:
					train_X.append(f) 

					
#train_X: [ [feature_vector1], [fv2]... ]
#feature_vector1: [[day1],[day2]...[day7]]
#应该把这个feature vector合并，因为每天去的地方可能有差异。形成这个人这周所有去过地方的fv，然后把所有人合并，形成所有人去过的地方的fv。
