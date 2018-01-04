import numpy as np
import utils
import sklearn.cluster.agglometrativeClustering as AGCL

threshold_time = 1.1
threshold_distance = 2.2
threshold_GPS_error = 2.2
data_filePath = 'GPS_data.txt'

'''
	To generate location history framework, clustering algorithm should be introduced, and the different alg could be defined here.
	Agglomerative clustering alg is used here, u could change it in config file.
	Algorithm source, see details and para settings: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering
'''
clusteringModel = AGCL(n_clusters = 1, linkage = 'ward', compute_full_tree = true).fit


num_class_kmeans = 8
num_iteration_kmeans = 10


#maybe pandas is easier to operate
GPS_data_format = [[x,y,t],[x2,y2,t2]];
POI_data_format = np.array[[x,y,category],[x2,y2,category2]]

stay_points = []
# POI_dataset = utils.get_POI_dataset()
POI_dataset = utils.load_POI_dataset() # np.array([1,1,'school'],[2.2,3.3,'canteen'])
feature_vectors = {} #{sp1:{f1:w1,f2:w2....}}


