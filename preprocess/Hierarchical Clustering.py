# -*- coding: utf-8 -*-
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

def Plot(data, class_arr=None):
    from scipy.io import loadmat  # 导入 loadmat, 用于对 mat 格式文件进行操作
    import matplotlib.pyplot as plt  # 导入绘图操作用到的库

    # 读取数据得到一个字典类型数据，需要根据键名 ’data‘ 取出对应的值。
    fig = plt.figure()  # 创建画布
    ax = fig.add_subplot(111)
    # 绘制散点图
    p1 = ax.scatter(data[:, 0], data[:, 1], marker='.', color='black', s=8)
    plt.show()  # 显示散点图


if __name__ == '__main__':
    data_path = "/media/Deepin/003/chen_data/semi-supervision data/shengyi-linhuan/My_data_enhancement/Labeled Radiomics.csv"
    data_df = pd.read_csv(data_path, header=0, index_col=0)
    data_id = data_df.index.tolist()
    X = np.array(data_df)
    scaler = MinMaxScaler()
    scaler.fit(X)
    X_norm = scaler.transform(X)

    # pca = PCA(n_components=2)
    # pca.fit(X)
    #
    # new_X = pca.transform(X_norm)
    # Plot(new_X)

    y_pred = KMeans(n_clusters=1, random_state=618).fit_predict(X_norm)
    # [1 0 1 1 0 0 1 2 1 2 0 2 2 1 2 1 1 1 0 1 1 2 1 1 2 1 1 0 0 1 0 1 1 0 1 1 0
    #  1 0 1 1 0 1 1 0 0 0 0 0 2 2]

    print(y_pred)
    print(data_id)
    for idx, item in enumerate(data_id):
        imageid, lesion_id = item.split('_')
        data_id[idx] = "img" + imageid + "_lesion" + lesion_id
    print(data_id)
