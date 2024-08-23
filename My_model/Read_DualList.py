import os
import statistics
import numpy as np
from scipy import stats
#1 3 5
data_list = []
data_path = "/data1/code/My-Semi-Supervision/code/My_model/performence2/LIDC/10"
file_list = os.listdir(data_path)
for data_file in file_list:
    with open(os.path.join(data_path, data_file), 'r') as file:
        lines = file.readlines()
        source_list = [line.strip().strip('[').strip(']').split(',') for line in lines]
        data_list.append(source_list)
# with open('./performence2/LIDC/EMLM_LIDC_50_preformence.txt', 'r') as file:
#     lines=file.readlines()
#     source_list = [line.strip().strip('[').strip(']').split(',') for line in lines]
# with open('./performence2/LIDC/CPS_LIDC_50_preformence.txt', 'r') as file:
#     lines=file.readlines()
#     target_list = [line.strip().strip('[').strip(']').split(',') for line in lines]


dice_list = [[float(item[0]) for item in item0] for item0 in data_list]
quantiles = statistics.quantiles(dice_list[0])
print(quantiles)
IoU_list = [[float(item[1]) for item in item0] for item0 in data_list]
HD95_list = [[float(item[2]) for item in item0] for item0 in data_list]
ASD_list = [[float(item[3]) for item in item0] for item0 in data_list]
stat, p = stats.kruskal(dice_list[0],dice_list[1],dice_list[2],dice_list[3],dice_list[4],dice_list[5],dice_list[6],dice_list[7])
print("DSC")
print("stat: {}".format(stat))
print("p: {}".format(p))

stat, p = stats.kruskal(IoU_list[0],IoU_list[1],IoU_list[2],IoU_list[3],IoU_list[4],IoU_list[5],IoU_list[6],IoU_list[7])
print("IoU")
print("stat: {}".format(stat))
print("p: {}".format(p))

stat, p = stats.kruskal(HD95_list[0],HD95_list[1],HD95_list[2],HD95_list[3],HD95_list[4],HD95_list[5],HD95_list[6],HD95_list[7])
print("HD95")
print("stat: {}".format(stat))
print("p: {}".format(p))

stat, p = stats.kruskal(ASD_list[0],ASD_list[1],ASD_list[2],ASD_list[3],ASD_list[4],ASD_list[5],ASD_list[6],ASD_list[7])
print("ASD")
print("stat: {}".format(stat))
print("p: {}".format(p))
# new_soure_list = [float(item[0]) for item in source_list]
# new_target_list = [float(item[0]) for item in target_list]
# u_statistic, p_value = stats.mannwhitneyu(new_soure_list, new_target_list)
# print("u_statistic: {}, p_value: {}".format(u_statistic, p_value))
# new_soure_list = [float(item[1]) for item in source_list]
# new_target_list = [float(item[1]) for item in target_list]
# u_statistic, p_value = stats.mannwhitneyu(new_soure_list, new_target_list)
# print("u_statistic: {}, p_value: {}".format(u_statistic, p_value))
#
# new_soure_list = [float(item[2]) for item in source_list]
# new_target_list = [float(item[2]) for item in target_list]
# u_statistic, p_value = stats.mannwhitneyu(new_soure_list, new_target_list)
# print("u_statistic: {}, p_value: {}".format(u_statistic, p_value))
#
# new_soure_list = [float(item[3]) for item in source_list]
# new_target_list = [float(item[3]) for item in target_list]
# u_statistic, p_value = stats.mannwhitneyu(new_soure_list, new_target_list)
# print("u_statistic: {}, p_value: {}".format(u_statistic, p_value))