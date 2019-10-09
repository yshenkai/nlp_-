#con
from bs4 import BeautifulSoup
import requests
import re
import pymongo
import uuid
import hashlib
import time
import numpy
from nltk.corpus import wordnet as wn

YOUDAO_URL = 'http://openapi.youdao.com/api'
APP_KEY = '469ec439f8531788'
APP_SECRET = 'EgZW4NyCOAd4RbP950bKCcUcXnfSaHzK'

selectordict = {"https://www.fsai.ie": "#content",
                "http://news.foodmate.net": "body > div.m2 > div.m_l.f_l > div"}

header = {  # 请求头部
    # "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
    # "Accept-Encoding": "gzip, deflate",
    # "Accept-Language": "zh-CN,zh;q=0.9",
    # "Connection": "keep-alive",
    # "Cookie": "__gads=ID=4433e5d2360ef5b5:T=1547000790:S=ALNI_MbnkAFV6ky3siM2Wp-Icxy3ZdAfUg; Hm_lvt_2aeaa32e7cee3cfa6e2848083235da9f=1553136133,1553482283; __51cke__=; yunsuo_session_verify=6deccc98239c33fee8a57fbd2a0bfa3d; Hm_lpvt_2aeaa32e7cee3cfa6e2848083235da9f=1553494494; __tins__1636283=%7B%22sid%22%3A%201553493298546%2C%20%22vd%22%3A%203%2C%20%22expires%22%3A%201553496293927%7D; __51laig__=20",
    # "Host": "news.foodmate.net",
    # "If-Modified-Since": "Mon, 25 Mar 2019 06:14:53 GMT",
    # "If-None-Match": "W/"5c9871dd-b1e9""
    # "Referer": "http://news.foodmate.net/guonei/",
    # "Upgrade-Insecure-Requests": "1",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36"
}


def get_charset(response):  # 根据请求返回的响应获取数据()
    """
        get charset from response
    :param response:
    :return:
    """
    _charset = requests.utils.get_encoding_from_headers(response.headers)
    if _charset == 'ISO-8859-1':
        __charset = requests.utils.get_encodings_from_content(response.text)
        if __charset:
            _charset = __charset[0]
        else:
            _charset = response.apparent_encoding

    return _charset


def readtagged():
    #     从MongoDB中读入已标注过的新闻，返回一个包括新闻url和标注内容的字典
    #     newslist = {"https://www.fsai.ie/details.aspx?id=16909":{'国家地区':'爱尔兰', '发布机构':'FSAI', '发布日期':'2019-04-06',
    #                                                              '企业':'USN', '品牌':'USN', '原产地':'/', '产品名称及描述':'类固醇食品补充剂，规格45/90/180片/瓶，召回涉及保质期2021-11-01前的所有批次产品',
    #                                                              '不合格原因':'致敏原', '采取措施':'预警召回'}}
    newslist = {}
    myclient = pymongo.MongoClient(host="180.167.46.217",port=27017)
    mydb = myclient["fsi"]
    mydb.authenticate(name="dataint",password="58281698")
    mycol = mydb["warning_recall"]
    myquery = {"original_link": {"$regex": ".*.*"}}
    #     myquery = {"table_1":{"$elemMatch":{"$elemMatch":{"$in":["丹麦"]}}}}
    mydoc = mycol.find(myquery)
    for x in mydoc:
        if "table_1" in x.keys():
            dic = {}
            for i in range(len(x["table_1"][0])):
                try:
                    dic[x["table_1"][0][i]] = x["table_1"][1][i]
                except Exception:
                    continue
            newslist[x["original_link"]] = dic


    return newslist
def read_htmlbody(url):
    myclient=pymongo.MongoClient(host="180.167.46.217",port=27017)
    mydb=myclient["fsi"]
    mydb.authenticate(name="dataint",password="58281698")
    mycol=mydb["origin_content"]
    mydoc=mycol.find({"docUrl":url},{"docContent":1})

    for x in mydoc:
        doc=x['docContent'].replace("\n","").encode().decode("utf-8")
    myclient.close()
    return doc



def findnewstext(url):
    response = requests.get(url, headers=header)
    response = response.content.decode(get_charset(response))
    selector = "html"
    for website in selectordict.keys():
        if url.find(website):
            selector = selectordict[website]
    try:
        newsbody = BeautifulSoup(response, 'lxml').select(selector)[0]
    except Exception:
        return ;
    for script in newsbody.find_all('script'):
        script.clear()

    return newsbody.get_text("", strip=True)


def truncate(q):
    if q is None:
        return None
    size = len(q)
    return q if size <= 20 else q[0:10] + str(size) + q[size - 10:size]


def youdaotranslate(chinesestr):
    q = chinesestr
    data = {}
    data['from'] = 'zh-CHS'
    data['to'] = 'EN'
    data['signType'] = 'v3'
    curtime = str(int(time.time()))
    data['curtime'] = curtime
    salt = str(uuid.uuid1())
    signStr = APP_KEY + truncate(q) + salt + curtime + APP_SECRET
    hash_algorithm = hashlib.sha256()
    hash_algorithm.update(signStr.encode('utf-8'))
    sign = hash_algorithm.hexdigest()
    data['appKey'] = APP_KEY
    data['q'] = q
    data['salt'] = salt
    data['sign'] = sign
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    response = requests.post(YOUDAO_URL, data=data, headers=headers)
    translated = str(response.content)
    beginIndex = translated.find("translation\":[\"") + 15
    endIndex = translated.find("\"],\"errorCode")
    translated = translated[beginIndex:endIndex]
    return translated


def find_lcsubstr(s1, s2):
    m = [[0 for i in range(len(s2) + 1)] for j in range(len(s1) + 1)]  # 生成0矩阵，为方便后续计算，比字符串长度多了一列
    mmax = 0  # 最长匹配的长度
    p = 0  # 最长匹配对应在s1中的最后一位
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i + 1][j + 1] = m[i][j] + 1
                if m[i + 1][j + 1] > mmax:
                    mmax = m[i + 1][j + 1]
                    p = i + 1
    return s1[p - mmax:p], mmax  # 返回最长子串及其长度
def save_file(id,htmlbody,label,mode="data"):
    if id=="国家地区":
        with open("../data/mongo/guojia/train_{}.data".format(mode),'a+',encoding="utf-8") as f:
            for i in range(len(label)):
                f.write(htmlbody[i]+"\t"+str(label[i])+"\n")
            f.write("\n")
    elif id=="发布机构":
        with open("../data/mongo/jigou/train_{}.data".format(mode),'a+',encoding="utf-8") as f:
            for i in range(len(label)):
                f.write(htmlbody[i]+"\t"+str(label[i])+"\n")
            f.write("\n")
    elif id=="企业":
        with open("../data/mongo/guojia/train_{}.data".format(mode),'a+',encoding="utf-8") as f:
            for i in range(len(label)):
                f.write(htmlbody[i]+"\t"+str(label[i])+"\n")
            f.write("\n")
    elif id=="品牌":
        with open("../data/mongo/pinpai/train_{}.data".format(mode),'a+',encoding="utf-8") as f:
            for i in range(len(label)):
                f.write(htmlbody[i]+"\t"+str(label[i])+"\n")
            f.write("\n")
    elif id=="原产地":
        with open("../data/mongo/yuanchandi/train_{}.data".format(mode),'a+',encoding="utf-8") as f:
            for i in range(len(label)):
                f.write(htmlbody[i]+"\t"+str(label[i])+"\n")
            f.write("\n")
    elif id=="产品名称及描述":
        with open("../data/mongo/miaoshu/train_{}.data".format(mode),'a+',encoding="utf-8") as f:
            for i in range(len(label)):
                f.write(htmlbody[i]+"\t"+str(label[i])+"\n")
            f.write("\n")
    elif id=="不合格原因":
        with open("../data/mongo/yuanyin/train_{}.data".format(mode),'a+',encoding="utf-8") as f:
            for i in range(len(label)):
                f.write(htmlbody[i]+"\t"+str(label[i])+"\n")
            f.write("\n")
    elif id=="采取措施":
        with open("../data/mongo/cuoshi/train_{}.data".format(mode),'a+',encoding="utf-8") as f:
            for i in range(len(label)):
                f.write(htmlbody[i]+"\t"+str(label[i])+"\n")
            f.write("\n")

def check_contain_chinese(check_str):
    for ch in check_str.decode('utf-8'):
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False


label=['国家地区', '发布机构', '企业', '品牌', '原产地', '产品名称及描述', '不合格原因', '采取措施']
def tagchinews(newsbody, contentdic):  # 国家 企业 品牌 产地
    #     输入一条新闻的正文文本，将人工识别的字段及其内容在文中标记出来
    taggednews = {}
    taggednews["origin"] = newsbody
    index=0
    for k, c in contentdic.items():

        if k in label and check_contain_chinese(newsbody.encode("utf-8")):


            for body in newsbody.split("。"):
                tagged = [0 for x in range(len(body))]
                hasMatched = True  # 每次取出标注文本和待标注文本的最长公共子串，直到剩下的标注文本子串都为单字
                v = c
                remainLen = len(v)
                while hasMatched and len(v) > 1:
                    subStr, chaNum = find_lcsubstr(v, body)
                    if chaNum > 1:
                        v = v.replace(subStr, "")
                        idx = body.find(subStr)
                        for i in range(chaNum):
                            tagged[idx + i] = 1
                    else:
                        hasMatched = False
                taggednews[k] = tagged
                # print(k, c)  ##
                # for j in range(len(newsbody)):
                #     if tagged[j] == 1:
                #         print(newsbody[j], end="")  ##
                # print(tagged)  ##
                if(tagged.__contains__(1)):
                    print("保存成功")
                    save_file(k,body,tagged)
    return taggednews


def tagengnews(newsbody, contentdic):  # 国家 企业 品牌 产地
    #     输入一条新闻的正文文本，将人工识别的字段及其内容在文中标记出来
    taggednews = {}
    newsbody = re.split(r'\s+', newsbody.lower())
    taggednews["origin"] = newsbody
    for k, c in contentdic.items():

        tagged = [0 for x in range(len(newsbody))]
        hasMatched = True  # 每次取出标注文本和待标注文本的最长公共子串，直到剩下的标注文本子串都为单字
        v = youdaotranslate(c).split(' ')
        print(v)  ##
        remainLen = len(v)
        while hasMatched and len(v) > 1:
            subStr, chaNum = find_lcsubstr(v, newsbody)
            if chaNum > 1:
                v = v.replace(subStr, "")
                idx = newsbody.find(subStr)
                for i in range(chaNum):
                    tagged[idx + i] = 1
            else:
                hasMatched = False
        taggednews[k] = tagged
        print(k, c)  ##
        for j in range(len(newsbody)):
            if tagged[j] == 1:
                print(newsbody[j], end="")  ##
        print(tagged)  ##
    return taggednews


newslist = readtagged()
print("======================链接加载完毕")
print(newslist)
for url, data in newslist.items():
        try:
            print(url)
            print(url,
              "\n" + read_htmlbody(url))  # ,"\n国家地区:",youdaotranslate(data[1][0]),"\n发布机构: ",youdaotranslate(data[1][1]))
            tagchinews(read_htmlbody(url), data)
        except Exception:
            continue

#     tagengnews(findnewstext(url),data)