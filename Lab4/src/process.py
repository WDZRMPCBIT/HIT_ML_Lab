import numpy as np
from data import Dataset
from copy import deepcopy


class Process(object):
    def __init__(self, data: Dataset, name: str):
        """
        对给定点集进行PCA降维
        """
        self.__name = name
        self.__data = deepcopy(data)

    def PCA(self, dim: int):
        """
        利用PCA进行降维

        :param dim: 降低到的维数
        """
        x = [[]] * self.__data.cnt()
        for i in range(self.__data.cnt()):
            x[self.__data.y()[i]].append(self.__data.x()[i])

        res = []
        for i in range(self.__data.kind()):
            res = res + self.__pca_process(x[i], dim)

        self.__data = Dataset(np.array(res), self.__data.y())

    def __pca_process(self, x, dim: int):
        x = np.array(x)
        cnt, feature = x.shape

        mean = np.sum(x, axis=0) / cnt
        x = np.array(x - [np.array.tolist(mean)] * cnt)
        print(x)

        # return np.array.tolist(x)

    def show2D(self) -> None:
        """
        图形化显示点集中的各个点及其所属聚类
        只显示前两维
        """
        flag = self.__classifier.predicate(self.__data)
        color = ['b', 'c', 'g', 'k', 'm', 'r', 'w', 'y']
        if self.__data.dim() < 2:
            print("illegal dim")
            return

        import matplotlib.pyplot as plt

        plt.title(self.__name)
        for i in range(self.__data.cnt()):
            plt.scatter(self.__data.x()[i][0],
                        self.__data.x()[i][1],
                        color=color[flag[i]])

        plt.show()

    def show3D(self) -> None:
        """
        图形化显示点集中的各个点及其所属聚类
        只显示前三维
        """
        color = ['b', 'c', 'g', 'k', 'm', 'r', 'w', 'y']
        if self.__data.dim() < 3:
            print("illegal dim")
            return

        import matplotlib.pyplot as plt

        ax = plt.subplot(111, projection='3d')
        for i in range(self.__data.cnt()):
            ax.scatter(self.__data.x()[i][0], self.__data.x()[
                i][1], self.__data.x()[i][2], color=color[self.__data.y()[i]])

        plt.show()
