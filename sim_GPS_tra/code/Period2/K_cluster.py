# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 16:35:03 2017

@author: Alina
"""
import numpy as np
import math
import time as TIME
import pandas as pd
import os
EARTH_RADIUS=6371		   # 地球平均半径，6371km




def hav(theta):  
	s = math.sin(theta / 2)  
	return s * s

def get_distance_hav(lat0, lng0, lat1, lng1):  
#	"用haversine公式计算球面两点间的距离。返回单位为米
	# 经纬度转换成弧度  
	lat0 = math.radians(lat0)  
	lat1 = math.radians(lat1)  
	lng0 = math.radians(lng0)  
	lng1 = math.radians(lng1)  

	dlng = math.fabs(lng0 - lng1)  
	dlat = math.fabs(lat0 - lat1)  
	h = hav(dlat) + math.cos(lat0) * math.cos(lat1) * hav(dlng)  
	distance = 2 * EARTH_RADIUS * math.asin(math.sqrt(h))  
	return distance*1000

def cluster(df,time_limit,stay_time,stay_range):
	lngs=df['lng']
	lats=df['lat']
	time=df['utc'].values
	time_last=np.array([time[0]]+list(time[:-1]))
	deta_times=time-time_last
	assert np.sum(np.less(np.array(deta_times),0))==0
	pt_collection=[[]]
	span_collection=[0]  
	time_collection=[]
	cluster_num=0
	for i,(lng,lat,deta_time) in enumerate(zip(lngs,lats,deta_times)):
		cur_pts=pt_collection[cluster_num]
		if len(cur_pts)==0:
			start_time=df.loc[i,'time']
		cur_max_dis=0
		for pt in cur_pts[::-1]:
			dis=get_distance_hav(pt[1],pt[0],lat,lng)
			cur_max_dis=cur_max_dis if dis<cur_max_dis else dis
			if cur_max_dis>stay_range or deta_time>time_limit: 
				if i==len(lngs)-1:
					pass
				else:				
					span_collection.append(0)
					pt_collection.append([])
				cluster_num+=1
				time_collection.append((start_time,df.loc[i,'time']))
				break
		if cur_max_dis<=stay_range and deta_time<=time_limit:
			if i==len(lngs)-1:
				time_collection.append((start_time,df.loc[i,'time']))						 
			pt_collection[cluster_num].append((lng,lat))
			span_collection[cluster_num]+=deta_time
		
	assert len(pt_collection)==len(span_collection)==len(time_collection)
	result=[]
	mean_pos_loop=[]
	pt_num_loop=[]
	for pts in pt_collection:
		lng=np.array([pt[0] for pt in pts])
		lat=np.array([pt[1] for pt in pts])
		mean_pos_loop.append((np.mean(lng),np.mean(lat)))
		pt_num_loop.append(len(pts))
	span_loop=span_collection.copy()
	time_loop=time_collection.copy()
	not_merge=False
	cnt=0
	while not not_merge:
		mean_pos=[]
		pt_num=[]
		span_list=[]
		time_list=[]

		last_time=time_loop[0]
		last_pt=mean_pos_loop[0]
		last_span=span_loop[0]
		last_pt_num=pt_num_loop[0]	
			
		not_merge=True
		for i,(pt,num,span,time) in enumerate(zip(mean_pos_loop[1:],pt_num_loop[1:],span_loop[1:],time_loop[1:])):
			cur_pt=(pt[0],pt[1])
			cur_pt_num=num	 
			cur_span=span
			cur_time=time
			deta=TIME.mktime(TIME.strptime(cur_time[0], "%Y-%m-%d %H:%M:%S"))-TIME.mktime(TIME.strptime(last_time[1], "%Y-%m-%d %H:%M:%S"))
			#print(deta)
			if deta<time_limit/2:
				#print(get_distance_hav(cur_pt[1],cur_pt[0],last_pt[1],last_pt[0]))
				if not_merge and get_distance_hav(cur_pt[1],cur_pt[0],last_pt[1],last_pt[0])<10: 
					#print(get_distance_hav(cur_pt[1],cur_pt[0],last_pt[1],last_pt[0]))
					lg=(cur_pt_num*cur_pt[0]+last_pt_num*last_pt[0])/(cur_pt_num+last_pt_num)
					lt=(cur_pt_num*cur_pt[1]+last_pt_num*last_pt[1])/(cur_pt_num+last_pt_num)	  
					#print(lg,lt)
					mean_pos.append((lg,lt))
					pt_num.append(cur_pt_num+last_pt_num)
					span_list.append(cur_span+last_span)
					time_list.append((last_time[0],cur_time[1]))						
					#print(last_time,cur_time,cnt)
					not_merge=False
					#print(cnt,len(mean_pos))
					mean_pos.extend(mean_pos_loop[i+2:])
					span_list.extend(span_loop[i+2:])
					pt_num.extend(pt_num_loop[i+2:])
					time_list.extend(time_loop[i+2:])
					break					   
				else:					  
					mean_pos.append(last_pt)
					span_list.append(last_span)
					pt_num.append(last_pt_num)
					time_list.append(last_time)
			else:
				mean_pos.append(last_pt)
				span_list.append(last_span)
				pt_num.append(last_pt_num)
				time_list.append(last_time)
			last_pt=cur_pt
			last_pt_num=cur_pt_num
			last_span=cur_span
			last_time=cur_time
			
			if i==len(span_loop)-2:
				mean_pos.append(cur_pt)
				span_list.append(cur_span)
				pt_num.append(cur_pt_num)
				time_list.append(cur_time) 
				cnt+=1
		mean_pos_loop=mean_pos.copy()
		pt_num_loop=pt_num.copy()
		span_loop=span_list.copy()
		time_loop=time_list.copy()
		#print(len(mean_pos))
		#assert 1==0
	
	for pos,span,time in zip(mean_pos_loop,span_loop,time_loop):					   
		if span>stay_time:				  
			item={'lng':pos[0],'lat':pos[1],'span':span,'start_time':str(time[0]),'end_time':str(time[1])}
			#print(item)
			result+=[item]
	result=pd.DataFrame.from_dict(result)			
	 
	return result

def drop_item(df,cur_day):
	st=TIME.mktime(TIME.strptime(cur_day+" 08:30:00", "%Y-%m-%d %H:%M:%S"))
	et=TIME.mktime(TIME.strptime(cur_day+" 22:30:00", "%Y-%m-%d %H:%M:%S"))
	di=[]
	#	  assert 1==0
	for index,utc in zip(df.index,df['utc']):
		if utc<st or utc>et:
			  di.append(index)
	df.drop(di,inplace=True)

	df=df.reset_index(drop=True)
	return df

def K_cluster_dir(path = '', out_path = ''):
	# this function input a dir path, processing all .txt files in this dir.
	if not path:
		ROOT='E:/17/'
		out_path = ROOT
		print('no path input here! \n')
	else:
		ROOT = path
	days=['2017-11-%.2d'%i for i in range(20,31)]+['2017-12-%.2d'%i for i in range(1,25)]
	
	# here, we consider gps.txt file in the child-directory, to open corresponding file well, we save each xxxgps.txt file's path too.(into file_path_dict.)
	track = []
	file_path_dict = {}
	for root, dirs, files in os.walk(ROOT):
		for file in files:
			if 'gps.txt' in file and file not in track:
				track.append(file)
				file_path_dict.setdefault(file, root+'/'+file)
	cnt=0
	for name in track:
		df_all=pd.read_csv(file_path_dict[name],sep='\t',header=None,index_col=False,names=['time','utc','ec','lng','lat'])
		df=df_all[df_all['ec']==0]
		df=df.reset_index(drop=True) 
		df['utc']=df['utc'].values/1000
		for day in days: 
			df_day=df[df['time'].str.contains(day)].reset_index(drop=True)
			if len(df_day)==0:
				#print('lack data of ',name,day)
				continue
			df_day=drop_item(df_day,day)		  
			#print(df_day)
			if len(df_day)==0:
				#print('lack data of ',name,day)
				continue
			result=cluster(df_day,1200,1200,50) 
			if len(result)>1: 
				if not out_path:
					if not os.path.isdir(out_path + 'stay_point'):
						os.mkdir(out_path + 'stay_point')
					result.to_csv(out_path + 'stay_point/'+name[:-4]+'_'+day+'.txt',index=False)
				else:
					result.to_csv(out_path + name[:-4]+'_'+day+'.txt',index=False)
		cnt+=1
			
			
