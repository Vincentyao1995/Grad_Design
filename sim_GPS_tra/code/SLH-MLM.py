import utils
import config
import pandas as pd
import numpy as np
import math
from anytree import Node, RenderTree, LevelOrderGroupIter, PreOrderIter

opener_top10 = 0

'''
    Question and influence factor:
    1) judge_whether_equal(c1,c2). this is to 
    2) the level of two tree built through two stay points may differ, 
'''
stay_points_users = []
feature_vectors_users = []

def SLH(filepath = './stay_point_2017-11-27.txt'):
    '''
    history is a list saves ci semantic location history,[ [[index of this level], [c1,c2,c3..]], .....] c1 = [sp1,sp2](Node)
    '''
    global stay_points_users, feature_vectors_users
    stay_points = utils.load_stay_points(filepath)
    stay_points_users.append(stay_points)
    if stay_points == False:
        stay_points = spExtraction()
        stay_points_users.append(stay_points)
    feature_vectors = feature_vector_extraction(stay_points)
    feature_vectors_users.append(feature_vectors)
    history = generate_location_history_framework(feature_vectors)
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
            if dict_regions_containing_i[feature] == 0 or N == 0:
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
                    if child >= N:
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
    '''
    1) Visit tree structure as the order of layer.
    2) alg of this function could be found in Definition 6.
    3) history is a list  [[level1], [level2] ...[level_n]], level_n is also a list:[[index_of_this_level],[nodes_of_this_level]]
    '''
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
    earliest_time = " '2030-11-27 10:05:56'" # this should be earlist node. for loop to find it.
    earliest_point = -1
    earliest_index = -1
    found_done = 0
    num_sp = 0
    list_indexed = []
    #this is to find the number of leave this tree == len(stay_points)
    for c_index in range(len(list)):
        for stay_point in list[c_index]:
            if len((stay_point.name).split(',')) == 4:
                num_sp += 1

    while found_done == 0:
        for c_index in range(len(list)):
            for stay_point in list[c_index]:
                #resolve string and get the earliest visiting node.
                if len((stay_point.name).split(',')) == 4: # this Node is a leave which saves info of sp, other else this would be a inter-node.
                    if stay_point in list_indexed:
                        continue
                    if utils.judge_time_former((stay_point.name).split(',')[2], earliest_time) == 1:  # start time.
                        earliest_time = stay_point.name.split(',')[2]
                        earliest_point = stay_point
                        earliest_index = c_index
                else:
                    continue
        list_indexed.append(earliest_point)
        index.append(earliest_index)
        #judge whether there is still any sp not get indexed.
        condition_found_done = 1

        if len(list_indexed) != num_sp:
            earliest_time = " '2030-11-27 10:05:56'"
            condition_found_done = 0
        if condition_found_done:
            found_done = 1
    return index

def LHM(history1, history2):
    '''
    attention: assume that two history tree has the same num of level
    '''
    graph = []
    graph_nodes = []
    simUser = 0
    #if len(history1) != len(history2):
    #    print("two semantic history tree don't have the same levels")
    #    exit(0)

    for num_level in range(min(len(history1),len(history2))): # attention, here I calc sim from the top level, which should be bottom, and ignore top levels when two tree doesn't have same levels.
        for index1 in range(len(history1[num_level][1])):
            semantic_location1 = history1[num_level][1][index1]
            index_list1 = history1[num_level][0]
            for index2 in range(len(history2[num_level][1])):
                semantic_location2 = history2[num_level][1][index2]
                index_list2 = history2[num_level][0]
                if judge_semanticLocation_equal(semantic_location1,semantic_location2):
                    graph_nodes.append([[index_list1,index1,semantic_location1], [index_list2,index2,semantic_location2]])
        if graph_nodes != []:
            simSq = maximal_travel_match(graph_nodes)
            simUser = simUser + simSq* math.pow(2,num_level-1)
            graph_nodes = []
        else:
            return 0.0
    return simUser


def maximal_travel_match(graph_nodes):
    '''
    this function realize for l from 2 to k algorithm in paper algrithm3. return similarity between two semantic location sequences(this a a set of semantic location at a layer, calc by all maximal travel match).
    warning: this function wasn't fully compelted.
    '''
    similarity_Sq = 0
    k = len(graph_nodes)
    for l in range(2,k):
        for t in range(1,l):
            if judge_precedence(graph_nodes[l],graph_nodes[l - t]) or judge_precedence(graph_nodes[l - t],graph_nodes[l]):
                #attention, should build graph here, but didn't build it, think all (v_l - v_t) is a maximal travel match.
                #so here, graph_nodes1 and graph_nodes[l-t] is a maximal match couple. calc their similarity and sum. max match length is 2
                sim_temp1 = judge_semanticLocation_equal(graph_nodes[l][0][2],graph_nodes[l-t][0][2],return_ratio = True)
                sim_temp2 = judge_semanticLocation_equal(graph_nodes[l][1][2],graph_nodes[l-t][1][2],return_ratio = True)
                sg = (sim_temp1+sim_temp2)/2 * math.pow(2,math.pow(2,2-1))
                similarity_Sq += sg
    length_s1 = len(graph_nodes[0][0][0])
    length_s2 = len(graph_nodes[0][1][0])
    similarity_Sq = similarity_Sq/(length_s1*length_s2)
    return similarity_Sq

def judge_precedence(v_l, v_t):
    '''
    this function is to judge whether v_l is the precedence of v_t(see details in paper - Algorithm3.)
    '''
    l_former_t_s1 = -1
    l_former_t_s2 = -1
    l_former_t = -1
    for i in range(len(v_l[0][0])):
        index_temp = v_l[0][0][i]
        if l_former_t_s1 == 1:
            break
        if index_temp == v_l[0][1]:
            for j in range(i, len(v_l[0][0])):
                if v_l[0][0][j] == v_t[0][1]:
                    l_former_t_s1 = 1
                    break
    for i in range(len(v_l[1][0])):
        index_temp = v_l[1][0][i]
        if l_former_t_s2 == 1:
            break
        if index_temp == v_l[1][1]:
            for j in range(i, len(v_l[1][0])):
                if v_l[1][0][j] == v_t[1][1]:
                    l_former_t_s2 = 1
                    break
    if l_former_t_s1 == 1 and l_former_t_s2 == 1:
        l_former_t = 1
        return True
    else:
        return False
    #attention, here, ignore condition2 in definition8 cuz don't know how to compare two semantic locations' time interval

    
def judge_semanticLocation_equal(semantic_location1, semantic_location2, return_ratio = False):
    '''
    this function is to judge whether two semantic locations are equal, see details and why in algorithm3
    '''
    if semantic_location1 == semantic_location2:
        if return_ratio:
            return 0.0
        else:
            return False
    length_c = min(len(semantic_location1), len(semantic_location2))
    num_equal_node = 0
    for node1 in semantic_location1:
        if len(node1.name.split(',')) != 4:
                continue
        for node2 in semantic_location2:
            if len(node2.name.split(',')) != 4:
                continue
            lat1 = float(node1.name.split(',')[1])
            lng1 = float(node1.name.split(',')[0].replace('[',''))
            lat2 = float(node2.name.split(',')[1])
            lng2 = float(node2.name.split(',')[0].replace('[',''))
            d_lat = lat1 - lat2
            d_lng = lng1 - lng2
            d_x = d_lng * 6371000 * math.pi/180
            d_y = d_lat * 6371000 * math.cos(lat1)* math.pi/180
            dis_node1_node2 = math.sqrt(d_x**2 + d_y**2)
            if dis_node1_node2 <= config.threshold_sp_equal:
                num_equal_node += 1
                break
    if num_equal_node >= float(length_c*config.threshold_c_equal_ratio):
        if return_ratio:
            if float(num_equal_node/length_c) <= 1.0: 
                return float(num_equal_node/length_c)
            else:
                return 1.0
        else:
            return True
    else:
        if return_ratio:
            return float(num_equal_node/length_c)
        else:
            return False
def simRes_person_others(person = './data/stay_point_2017-11-27.txt', others_path = './data', write_file = False, opener_sorted = False):
    '''
        This function is to get most similar person
        Input:
        1. person: the person on attention, input this person's stay point file path.
        2. others_path: input a directory path, other persons' stay points files are in this dir.
        3.(choose) write_file: write result into txt(simResult_person) or not.
        4.(choose) opener_sorted: True/False, return sorted basing on similarity result or oringinal one.
        Output:
        1. a dictionary which saves this person and all others' similarity.
        2. a list. if u input arg 'sorted = True', return a sorted list(from top to bottom).
    '''
    import os
    similarity = {}
    history_path1 = person
    history1 = SLH(history_path1)
    for root, dirs, files in os.walk(others_path):
        for i in range(len(files)):
            history_path2 = others_path + '/' + files[i]
            print(person + '\t' + history_path2 + 'is processing.\n')
            key = person + ',' + files[i]
            history2 = SLH(history_path2)
            sim_user = LHM(history1,history2)
            similarity.setdefault(key,sim_user)
    if write_file:
        result_fileName = './simResult_person.txt'
        file = open(result_fileName,'w')
        simRes = sorted(similarity.items(), key = lambda x:x[1], reverse = True)
        for item in simRes:
            file.write(str(item) + '\n')
        file.close()
    if sorted:
        return sorted(similarity.items(), key = lambda x:x[1], reverse = True)
    return similarity
def normal_result():
    import os
    similarity = {}
    similarity_normal = {}
    rootPath = './data'
    result_fileName = './simResult.txt'
    
    for root, dirs, files in os.walk(rootPath):
        for i in range(len(files)):
            for j in range(i,len(files)):
                history_path1 = rootPath + '/' + files[i]
                history_path2 = rootPath + '/' + files[j]
                print(history_path1 + '\t'+ history_path2 + ' is processing.\n')
                key = files[i] + ',' + files[j]
                history1 = SLH(history_path1)
                history2 = SLH(history_path2)
                sim_user = LHM(history1,history2)
                similarity.setdefault(key,sim_user)
    for key in similarity.keys():
        normal_value = float(similarity[key]/sum(similarity.values()))
        similarity_normal[key] = normal_value
    file = open(result_fileName,'w')
    if opener_top10:
        top10_fileName = './simResult_top10.txt'
        file_top10 = open(top10_fileName,'w')
        sorting_result = sorted(similarity.items(), key = lambda x:x[1],reverse = True)
        for item in sorting_result:
            file_top10.write(str(item))
            file_top10.write('\n')
        file_top10.close()
    file.write(str(similarity))
    file.write('\n similarity after normalization:\n')
    for key in similarity_normal.keys():
        file.write(key + ': ' + str(similarity_normal[key]))
        file.write('\n')
    file.close()
if __name__ == '__main__':
    simRes_person_others(write_file = True,opener_sorted = True)