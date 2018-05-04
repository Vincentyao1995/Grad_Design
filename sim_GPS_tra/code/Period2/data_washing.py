# this script is for dealing with raw_data, clear them into good-classes
import os
import K_cluster as kc
out_path = 'E:\College\大四下\毕设\数据\数据分类\类别2/'

label_dict = {'李浩男':2016,'李均':2016,'刘艺婷':2016, '孙上哲':2016,'陈江媛':2015, '李雁':2015, '吕访贤':2015, '武亚明':2015, '向枫':2015, '张小格':2015, '张歆越':2014, '姚博睿':2014, '刘瑾':2014,'李楠楠':2014, '胡舒':2014}

rawData_path = 'E:\College\大四下\毕设\数据/raw_data/raw_data/'
week_name = []

week_name = os.listdir(rawData_path)

for week in week_name:
	path1 = rawData_path + week + '/' + '14_16' + '/'
	path2 = rawData_path + week + '/' + '17' + '/'
	paths = [path1, path2]
	for i, path in enumerate(paths):
		if i == 0:#14_16
			persons = os.listdir(path)
			for person in persons:
				if person not in label_dict:
					print(person + 'not in label dict, root is:' + path+person + '\n')
					continue
				if label_dict[person] == 2014:
					week_dir = out_path + '大四/' + week + '/'
                    if not os.path.isdir(week_dir):
                        os.mkdir(week_dir)
					print(week_dir + person)
                    kc.K_cluster_dir(path + person + '/', week_dir)
                if label_dict[person] == 2015:
					week_dir = out_path + '大三/' + week + '/'
                    if not os.path.isdir(week_dir):
                        os.mkdir(week_dir)
					print(week_dir + person)
                    kc.K_cluster_dir(path + person + '/', week_dir)
                if label_dict[person] == 2016:
					week_dir = out_path + '大二/' + week + '/'
                    if not os.path.isdir(week_dir):
                        os.mkdir(week_dir)
					print(week_dir + person)
                    kc.K_cluster_dir(path + person + '/', week_dir)
        elif i == 1:#17
			week_dir = out_path + '大一/' + week + '/'
            if not os.path.isdir(week_dir):
                os.mkdir(week_dir)
            print(week_dir)
            kc.K_cluster_dir(path, week_dir)
print('data washing done!\n')