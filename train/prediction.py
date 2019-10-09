#coding=utf-8
from train.fine_get_model import get_model
from train.process_data import _gen_test_data
import numpy as np
get_data=_gen_test_data()
model,(test_x,test_y)=get_model()
a=model.predict(get_data)
print(a)
print(a.shape)
b=np.argmax(a,axis=-1)[0]
# a=np.reshape(a,[-1])
# b=np.round(a)
print(b)
label_tag=[0,1]
# c=[label_tag[i] for i in b]
# print(c)
# b=a.reshape([-1])
# print(max(b))
str = "潍坊市食品药品监督管理局通过其官方网站公布了“2016年3月份流通环节食品市级监督抽检结果的通告”。此次抽检品种为肉制品、蔬菜制品、调味品、食用农产品等13个种类，经检测发现不合格样品12批次，抽检合格率为98.83%。其中，山东临朐百货大楼有限公司龙泉路广场所售茌平县恒源食品有限公司生产的金针菇，因二氧化硫含量超标，遭到通报。近日，潍坊市食品药品监督管理局通过其官方网站公布了“2016年3月份流通环节食品市级监督抽检结果的通告”。据悉，潍坊市食品药品监督管理局3月份抽检品种为肉制品、蔬菜制品、调味品、食用农产品等13个种类，以非食用物质、药残留、重金属、食品添加剂等易出现问题的项目为检测重点，共计700批次。经检测发现不合格样品12批次，抽检合格率为98.83%。其中，山东临朐百货大楼有限公司龙泉路广场所售茌平县恒源食品有限公司生产的金针菇，因二氧化硫含量超标，遭到通报。通报如下：被抽检产品为金针菇（香辣），被抽检单位为山东临朐百货大楼有限公司龙泉路广场，标称生产企业为茌平县恒源食品有限公司，规格为175g/瓶，生产日期为2015/11/8，不合格项目为二氧化硫。对上述抽检中发现的不合格产品，抽样场所所在地县级食品药品监管部门（市场监督管理部门）按照《中华人民共和国》的规定，依法进行处理。标称市外生产的不合格产品信息已通报相关食品生产企业所在地食品药品监督管理局。"
print(b[:len(str)])
e=[]
for i in range(len(str)):
    e.append([str[i],b[i]])
print(e)