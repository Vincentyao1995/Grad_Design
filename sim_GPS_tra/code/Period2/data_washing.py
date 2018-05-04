# this script is for dealing with raw_data, clear them into good-classes
import os

label_dict = {'李浩男':2016,'李均':2016,'刘艺婷':2016, '孙上哲':2016,'陈江媛':2015, '李雁':2015, '吕访贤':2015, '武亚明':2015, '向枫':2015, '张小格':2015, '张歆越':2014, '姚博睿':2014, '刘瑾':2014,'李楠楠':2014, '胡舒':2014}

rawData_path = 'E:\College\大四下\毕设\数据\raw_data\raw_data'
week_name = []

week_name = os.listdir(rawData_path)
for week in week_name:
	14to16_path = rawData_path + '\\' + week + '\\' + '14_16' + '\\'
	17_path = rawData_path + '\\' + week + '\\' + '17' + '\\'
	paths = [14to16_path, 17_path]
	for i, path in enumerate(paths):
		if i == 0:#14_16
			
		elif i == 1:#17
			files = os.listdir(path)
			track = [x for x in files if 'gps.txt' in x]