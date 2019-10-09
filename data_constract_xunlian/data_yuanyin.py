from xml.dom.minidom import parse
import numpy as np
import os
labels=["国家和地区","发布机构","企业","品牌","产地","产品描述","不合格原因","采取措施"]
def save_file(htmlbody,label,mode="data"):
    with open("data_xunlian/yuanyin/train_{}.data".format(mode), 'a+', encoding="utf-8") as f:
        for i in range(len(label)):
            f.write(htmlbody[i] + "\t" + str(label[i]) + "\n")
        f.write("\n")
num_count=0
def data_qiye(xmlpath):
    doc=parse(xmlpath)
    root=doc.documentElement
    test=root.getElementsByTagName("labeled")[0]
    print(test.hasChildNodes())
    labeled=root.getElementsByTagName("labeled")[0].firstChild.data
    content_text=root.getElementsByTagName("content")[0].firstChild.data
    label_tag = np.zeros(shape=(len(content_text)), dtype=np.int)
    if(labeled):
        outputs=root.getElementsByTagName("outputs")
        for output in outputs:
            anaotations=output.getElementsByTagName("annotation")
            for anaotation in  anaotations:
                Ts=anaotation.getElementsByTagName("T")
                for T in Ts:
                    items=T.getElementsByTagName("item")
                    for item in items:
                        names=item.getElementsByTagName("name")
                        for name in names:
                            str_name=name.childNodes[0].data
                            if str_name=="不合格原因":
                                start_index = int(item.getElementsByTagName("start")[0].childNodes[0].data)
                                end_index = int(item.getElementsByTagName("end")[0].childNodes[0].data)
                                label_tag[start_index:end_index]=1
                                save_file(content_text,label_tag)
                                global num_count
                                num_count += 1


def constact_qiye_data(dirpath):
    paths=os.listdir(dirpath)
    print(paths)

    for path in paths:
        try:
            data_qiye(os.path.join(dirpath,path))

        except Exception as e:
            continue
    print("合格数据为",num_count,"条")


if __name__=="__main__":
    constact_qiye_data("C:\\Users\\Administrator.USER-20180601PP\\Desktop\\整合\\源文件重合\\outputs(322)")