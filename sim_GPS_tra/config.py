import numpy as np
import utils

threshold_time = 1.1
threshold_distance = 2.2.
threshold_GPS_error = 2.2

#maybe pandas is easier to operate
GPS_data_format = [[x,y,t],[x2,y2,t2]];
POI_data_format = np.array[[x,y,category],[x2,y2,category2]]

stay_points = []
# POI_dataset = utils.get_POI_dataset()
POI_dataset = np.array([1,1,'school'],[2.2,3.3,'canteen'])

