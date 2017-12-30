import utils
import config



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
						
			
	return stay_points

def feature_vector_extraction(POI_dataset, stay_points):
	'''
	Each stay point is a stay region, and each stay region has a feature vector
	'''
	# attention: maybe POI_dataset should be set in config as a whole var
	unique_POI = set(POI_dataset)
	num_unique_POI = len(unique_POI)
	dict_features = {}	#{food: weight, school: weight}
	
	#set default value in dict_features;
	for i in range(num_unique_POI):
		dict_features.setdefault(unique_POI[i], 0)
	
	for point in stay_points:
		for feature in dict_features.keys:
			calc_weight_POI(point, feature)
			# going on, coding calc_weight_POI now.
	return feature_vector
	
if '__name__' == __main__:


	dataGPS = utils.loadData();
	stay_points = spExtraction();
	

	
	
	
	