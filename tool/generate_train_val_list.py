import os


def list_save(filename, data):  # filename为写入CSV文件的路径，data为要写入数据列表.
    file = open(filename, 'a')
    for i in range(len(data)):
        s = data[i] + '\n'
        file.write(s)
    file.close()
    print("保存成功")


datapath = "/media/Deepin/003/chen_data/semi-supervision data/LIDC/new_data_2d/data"
outputdir = "/media/Deepin/003/chen_data/semi-supervision data/LIDC/new_data_2d/val.list"
datalist = os.listdir(datapath)
for idx, data in enumerate(datalist):
    if '.h5' not in data:
        datalist.remove(data)
        continue
    datalist[idx] = data[:-3]
list_save(outputdir, datalist)