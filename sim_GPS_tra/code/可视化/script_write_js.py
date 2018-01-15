import os
data_path = '../data/'
for root,dirs,files in os.walk(data_path):
	for file in files:
		file_obj_data = open(data_path+file, 'r')
		filePath_js = './'+ file.replace('.txt','').split('_')[-1] + '.js'
		file_obj_js = open(filePath_js, 'w')
		file_obj_js.write('var provinces = [')
		index_line = 0
		for line in file_obj_data:
			if line.split(',')[0] == 'end_time':
				continue
			index_line += 1 
			lat = float(line.split(',')[1])
			lng = float(line.split(',')[2])
			file_obj_js.write(' {\n\t\"name\": \"'+ str(index_line)+'\",\n\t\"center\": \"' + str(lng) + ',' + str(lat) + '\",\n\t\"type\": 0\n\t},')
		file_obj_js.write('];')
		file_obj_js.close()
print('done!\n')
