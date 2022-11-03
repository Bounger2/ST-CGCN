#Get weather data
#Reference address:https://blog.csdn.net/weixin_42578412/article/details/113880442

import requests
import re
import json
from bs4 import BeautifulSoup
import pandas as pd

#PEMS03
# months = [9,10,11]
# years = [2018,]

#PEMS04
# months = [1,2,]
# years = [2018,]

#PEMS07
# months = [5,6,7,8,]
# years = [2017,]

#PEMS08
# months = [7,8,]
# years = [2016,]

#SZ-taxi
months = [1,]
years = [2015,]

index_ = ['date','MaxTemp','MinTemp', 'Weather','Wind']  # 选取的气象要素
data = pd.DataFrame(columns=index_)  # 建立一个空dataframe


for y in years:
	for m in months:
		url = 'http://tianqi.2345.com/Pc/GetHistory?areaInfo[areaId]=59493&areaInfo[areaType]=2&date[year]='+str(y)+'&date[month]='+str(m)
		response = requests.get(url=url)
		if response.status_code == 200:  # 防止url请求无响应
			#print(json.loads(response.text)['data'])
			html_str = json.loads(response.text)['data']
			soup = BeautifulSoup(html_str,'lxml')
			tr = soup.table.select("tr")
			for i in tr[1:]:

				td = i.select('td')
				tmp = []
				for j in td:
					#print(re.sub('<.*?>',"",str(j)))
					tmp.append(re.sub('<.*?>',"",str(j)))
				#print(tmp)
				data_spider = pd.DataFrame(tmp).T
				data_spider.columns = index_  # 修改列名
				#data_spider.index = date  # 修改索引
				data = pd.concat((data,data_spider), axis=0)  # 数据拼接
	#print(data)
data.to_excel('weather_data/weatherdata_SZ_SPEED.xlsx')