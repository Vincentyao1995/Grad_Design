import utils
import config
import pandas as pd
import numpy as np
import math
from anytree import Node, RenderTree, LevelOrderGroupIter, PreOrderIter

def SLH(filepath = './stay_point_2017-11-27.txt'):
    '''
    history is a list saves ci semantic location history,[ [[index of this level], [c1,c2,c3..]], .....] c1 = [sp1,sp2](Node)
    '''
    stay_points = utils.load_stay_points()
    if stay_points == False:
        stay_points = spExtraction()
    feature_vectors = feature_vector_extraction(stay_points)
    # going on, time to generate location history framework
    history = generate_location_history_framework(feature_vectors) #attention, debug.
    return history

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

            if dist_i_j <= config.threshold_distance:
                continue
            else:
                if j == i+1:
                    # not stay, just pass by
                    break; # jump out for j, continue to do i+1;
                else:
                    if time_i_jm1 > config.threshold_time:
                        # yes, stay here. output i - (j-1)
                        stay_point = utils.calc_stay_point(data,i,j-1);
                        stay_points.append(stay_point)

                    else:
                        # not stay, just pass by
                        break; # jump out for j, continue to do i+1;

    return stay_points

def feature_vector_extraction(stay_points):
    '''
    This function receive all stay_points, then calc feature vectors basing on POI dataset. return all feature vectors.
    Each stay point is a stay region, and each stay region has a feature vector.
    '''
    #1) prepare room. dict_feature is to one stay point/region's feature vector, {food: weight, school: weight}
    # dict_regions_containing_i saves the number of regions that containing a certain feature; {feature1: 3, feature2: 0, ...}

    unique_POI_temp = set(config.POI_dataset[:,2])
    unique_POI = []
    for POI in unique_POI_temp:
        unique_POI.append(POI)
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
        dict_regions_containing_i.setdefault(unique_POI[i],0)

    #2) calc weight of a certain feature;
    Ni = 0 #number of POIs of category i located in region 'point'
    N = 0 #total number of POIs in region 'point'
    R = len(stay_points)

    # 2.1) this for loop is to calc regions containing i;
    for point in stay_points:
        for POI in config.POI_dataset:
            if (if_POI_in_region(point,POI)):
                df_region_feature_temp[POI[2]][str(point)] = 1
    for feature in dict_feature.keys():
        dict_regions_containing_i[feature] = sum(df_region_feature_temp[feature])

    feature_vectors = {}
    for point in stay_points:
        fv = {}
        for feature in dict_feature.keys():
            #a point is a region
            #2.2 calc Ni and N;
            for POI in config.POI_dataset:
                if(if_POI_in_region(point, POI)):
                    N += 1
                    feature_POI = POI[2]

                    if (feature_POI == feature):
                        Ni += 1
            # going on, df(region containing i all are 0), debug
            if dict_regions_containing_i[feature] == 0:
                weight_feature = 0
            else:
                weight_feature = Ni/N * math.log(R/dict_regions_containing_i[feature])
            fv.setdefault(feature, weight_feature)
            Ni = 0
            N = 0
        feature_vectors.setdefault(str(point), fv)

    return feature_vectors

def generate_location_history_framework(feature_vectors):

    '''
        This function is to do clustering alg, then generate tree-structured framework.
        Different clustering alg could be defined in config.py
    '''

    fv_ndArray = convert_featureVectors_ndArray(feature_vectors)
    tree = config.clusteringModel.fit(fv_ndArray[0]).children_
    root_node = build_tree(feature_vectors, tree, fv_ndArray)
    history = build_individual_location_history(root_node)
    return history

def convert_featureVectors_ndArray(feature_vectors):
    '''
    This function is to convert feature_vectors to the format matched with kmeans alg in scipy lib.
    this function receive one para: feature_vectors, return an ndArray, stay_point indexes and feature indexes.
    '''
    stay_point_index = []


    feature_index = []
    data_temp = []
    data_list_temp = []

    #attention, cuz dict feature vectors are converted to ndarray, if keys in feature_vector[sp] are changing, data would be poluted, we should make sure keys' oder is unchangable
    for stay_point in feature_vectors.keys():
        stay_point_index.append(stay_point)
        if data_temp != []:
            data_list_temp.append(data_temp)
        data_temp = []
        for feature in feature_vectors[stay_point].keys():
            data_temp.append(feature_vectors[stay_point][feature])
            if len(feature_index) == len(feature_vectors[stay_point].keys()):
                continue
            feature_index.append(feature)
    data_list_temp.append(data_temp) # append the last stay_point info

    ndArray = np.array(data_list_temp)
    return ndArray, stay_point_index, feature_index

def build_tree(feature_vectors, tree, fv_ndArray):
    '''
        This build tree function build tree using anytree lib, according to sklearn.cluster.AgglomerativeClustering.fit(X) return value children_ (see its(agg.children_) meaning at:)
         http://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering.fit_predict
         tree data structure see there:
         https://pypi.python.org/pypi/anytree
    '''
    stay_points_index = fv_ndArray[1]
    feature_index = fv_ndArray[2]
    N = len(feature_vectors)
    nodes = {}
    for i in range(len(tree)):
        for j in range(len(tree[0])):
            if tree[i][j] < N:
                key_sp = stay_points_index[tree[i][j]]
                nodes.setdefault(tree[i][j], Node(key_sp, parent = None))
            elif tree[i][j] >= N:
                key_sp = str(tree[i][j])
                nodes.setdefault(tree[i][j], Node(key_sp,parent = None))
                children_index = tree[i][j] - N
                for child in tree[children_index]:
                    if child >= 8:
                        key_sp = str(child)
                        nodes.setdefault(child,Node(key_sp,parent = None))
                    else:
                        key_sp = stay_points_index[child]
                        nodes.setdefault(child, Node(key_sp, parent = None))
                    nodes[child].parent = nodes[tree[i][j]]
    root = Node('root')
    for node_index in nodes.keys():
        if nodes[node_index].parent == None:
            nodes[node_index].parent = root
    return root

def if_POI_in_region(point,POI):
    '''
    point defines the region, rect [point-r,point+r] in (x,y) space, POI is the POI waited to be judge whether it's in this region or not.
    Return true/false: this POI in/not this region(point)
    '''

    length_r_lat = config.threshold_GPS_error_lat
    length_r_lng = config.threshold_GPS_error_lng
    if float(POI[0]) >= point[0] - length_r_lng and float(POI[0]) <= point[0] + length_r_lng:
        if float(POI[1]) >= point[1] - length_r_lat and float(POI[1]) <= point[1] + length_r_lat:
            return True

    return False

def build_individual_location_history(root):
    #1) Visit tree structure as the order of layer.
    #2) alg of this function could be found in Definition 6.
    nodes_level = [[node for node in children] for children in LevelOrderGroupIter(root)]
    history = []
    for i in range(len(nodes_level)):
        history.append([])
        history[i].append([])
        history[i].append([])
        for j in range(len(nodes_level[i])):
            history[i][1].append([node for node in PreOrderIter(nodes_level[i][j])])
        
        index_of_this_level = get_index_in_tree_level(history[i][1])
        history[i][0] = index_of_this_level
      
    return history

def get_index_in_tree_level(list):
    ori_list = list
    index = []
    earliest_time = -1# this should be earlist node. for loop to find it.
    earliest_point = -1
    earliest_index = -1
    while found_done == 0:

        for c_index in range(len(list)):
            for stay_point in list[c_index]: # sp info saved in a Node
                #resolve string and get the earliest visiting node.
                if len((stay_point.name).split(',')) == 4: # this Node is a leave which saves info of sp, other else this would be a inter-node.
                    if (stay_point.name).split(',')[2] < early_time:  # start time.
                       earliest_time = stay_point.name.split(',')[2]
                       earliest_point = stay_point
                       earliest_index = c_index
                else:
                    sp_index += 1
                    continue
        list[earliest_index].remove(earliest_point)
        index.append(earliest_index)
        #judge whether there is still any sp not get indexed.
        condition_found_done = 1
        for c in list:
            for sp in c:
                if len((stay_point.name).split(',')) == 4:
                    condition_found_done = 0
        if condition_found_done:
            found_done = 1
    return index

def build_graph(history1, history2):
    for c1 in history1:
        for c2 in history2:
            if judge_whether_equal(c1[1],c1[2]):


def judge_whether_equal(c1,c2):
    return True

if __name__ == '__main__':

    history1 = SLH('./stay_point_2017-11-27.txt')
    history2 = SLH('./stay_point_2017-11-27.txt')

    graph = build_graph(history1,history2)
    maximal_match = get_maximal_match(graph)
    similarity = cal_simUser(h1,h2)
