import utils
import config
import pandas as pd


def SLH():
	
	'''
	'''

def stay_points_extraction(data):

	'''
	This function receive one arg: data representing GPS dataset for one person.
	Return stay points dataset extracted from this person's GPS trajectory
	In the paper, this function 
	'''

	stay_points = [];
	
	for i in range(len(data)):
		for j in range(i+1, len(data)):
			
			dist_i_j = utils.distance(data[i],data[j]);
			time_i_j = utils.interval(data[i],data[j]);
			dist_i_jp1 = utils.distance(data[i],data[j+1]);
			time_i_jm1 = utils.interval(data[i],data[j-1]);
			
			if (dist_i_j <= config.threshold_distance):
				continue
			else:
				if j == i+1:
					# not stay, just pass by
					break; # jump out for j, continue to do i+1;
				else:
					if (time_i_jm1 > config.threshold_time):
						# yes, stay here. output i - (j-1)
						stay_point = calc_stay_point(data,i,j-1);
						stay_points.append(stay_point)
						
					else:
						# not stay, just pass by
						break; # jump out for j, continue to do i+1;
						
	config.stay_points = stay_points
	return stay_points

def feature_vector_extraction(stay_points):
	'''
	This function receive all stay_points, then calc feature vectors basing on POI dataset. return all feature vectors.
	Each stay point is a stay region, and each stay region has a feature vector.
	'''
	#1) prepare room. dict_feature is to one stay point/region's feature vector, {food: weight, school: weight}
	# dict_regions_containing_i saves the number of regions that containing a certain feature; {feature1: 3, feature2: 0, ...}
	
	unique_POI = set(config.POI_dataset[:,3])
	num_unique_POI = len(unique_POI)
	dict_feature = {}	
	dict_regions_containing_i = {}
	# to calc dict_regions_containing_i, we need a temp pandas form column is feature, row is regions(stay points), value is 1/0;
	col_names = unique_POI
	row_names = [str(p) for p in stay_points]
	df_region_feature_temp = pd.DataFrame(0,columns = col_names, index = row_names)
	#set default value in dict_feature;
	for i in range(num_unique_POI):
		dict_feature.setdefault(unique_POI[i], 0)
		dict_regions_containing_i(unique_POI[i],0)
	
	#2) calc weight of a certain feature;
	Ni = 0 #number of POIs of category i located in region 'point'
	N = 0 #total number of POIs in region 'point'
	R = len(config.stay_points)
	
	# 2.1) this for loop is to calc regions containing i;
	for point in stay_points:
		for feature in dict_feature.keys:
			for POI in config.POI_dataset:
				if (if_POI_in_region(point,POI)):
					df_region_feature_temp[feature_POI][str(point)] = 1
	for feature in dict_feature.keys:
		dict_regions_containing_i[feature] = sum(df_region_feature_temp[feature])
	
	feature_vectors = {}
	for point in stay_points:
		for feature in dict_feature.keys:
			#a point is a region
			#2.2 calc Ni and N;
			fv = {}
			for POI in config.POI_dataset:
				if(if_POI_in_region(point, POI)):
					N += 1
					feature_POI = POI[2]
					
					if (feature_POI == feature):
						Ni += 1
		
			weight_feature = Ni/N * math.log(R/dict_regions_containing_i[feature])
			fv.setdefault(feature, weight_feature)
			Ni = 0
			N = 0 
			
		feature_vectors.setdefault(point, f_v)	
			
	return feature_vectors
	
if '__name__' == __main__:


	dataGPS = utils.loadData()
	stay_points = spExtraction()
	feature_vectors = feature_vector_extraction(stay_points)
	# going on, time to generate location history framework
	
	
	
	