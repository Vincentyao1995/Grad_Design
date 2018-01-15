import numpy as np
import utils
from sklearn.cluster import AgglomerativeClustering as AGCL
import math

threshold_time = 1.1
threshold_distance = 2.2
threshold_GPS_error_userSetting = 100 # unit: meter
threshold_sp_equal = 200 # this threshold controls maximal distance between two euqal stay point(meter)
threshold_c_suqual_ratio = 0.3 #this threshold controls how many stay points in semantic location to which the num of equal sp reach could say two c is equal
threshold_rou = 0.2

latitude = 30.52780029296875 # different city has different latitude leading to different threshold lng error
threshold_GPS_error_lng = 180 / 6371000 / math.pi / math.cos(latitude) * threshold_GPS_error_userSetting  #纬上跨经，与纬度有关。
threshold_GPS_error_lat = 180 / 6371000 / math.pi * threshold_GPS_error_userSetting  #经上跨纬，与纬度无关。

data_filePath = 'GPS_data.txt'

'''
	To generate location history framework, clustering algorithm should be introduced, and the different alg could be defined here.
	Agglomerative clustering alg is used here, u could change it in config file.
	Algorithm source, see details and para settings: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering
'''
clusteringModel = AGCL(n_clusters = 1, linkage = 'ward', compute_full_tree = True)

num_class_kmeans = 8
num_iteration_kmeans = 10

#maybe pandas is easier to operate
#GPS_data_format = [[x,y,t],[x2,y2,t2]];
#POI_data_format = np.array[[x,y,category],[x2,y2,category2]]

stay_points = []
# POI_dataset = utils.get_POI_dataset()
POI_dataset = utils.load_POI_dataset() # np.array([1,1,'school'],[2.2,3.3,'canteen'])
feature_vectors = {} #{sp1:{f1:w1,f2:w2....}}


