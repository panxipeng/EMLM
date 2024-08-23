import os

path = "/home/Deepin/PycharmProjects/SSL4MIS-master/data/GD_liang2/data/slices"
list1 = os.listdir(path)
list2 = [item[:-3] for item in list1]
file = open("train_slies.list", 'a')
for i in range(len(list2)):
    s = list2[i] + '\n'
    file.write(s)
file.close()
print("保存成功")