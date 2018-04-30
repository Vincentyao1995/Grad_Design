#this file is to 

train_X, train_Y = [],[]

def read_feature(stayPoint_path):
	# this function open a txt file(stay point), combining with POI file, return this sp's feature vector.
	# e.g. fv: {lib:{time: 3, minutes: 260.5}};
    with open(stayPoint_path) as file:
        return [stayPoint_path]
    return False

def data_to_feature(filePath = 'E:\College\大四下\毕设\数据\数据分类\类别1\\'):
	import os
	length_label = 0
	filePath = 'E:\College\大四下\毕设\数据\数据分类\类别1\\'
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
应该把这个feature vector合并，因为每天去的地方可能有差异。形成这个人这周所有去过地方的fv，然后把所有人合并，形成所有人去过的地方的fv。
