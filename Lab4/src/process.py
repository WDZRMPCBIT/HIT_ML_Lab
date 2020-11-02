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
        self.__PCA_flag = False

    def PCA(self, dim: int):
        """
        利用PCA进行降维

        :param dim: 降低到的维数
        """
        self.__mean = np.mean(self.__data.x(), axis=0)
        norm = self.__data.x() - self.__mean

        self.__scope = np.max(norm, axis=0) - np.min(norm, axis=0)
        X = norm / self.__scope

        Sigma = (1.0 / self.__data.cnt()) * np.dot(X.T, X)

        U, S, V = np.linalg.svd(Sigma)
        self.__U_reduce = U[:, 0:dim].reshape(self.__data.dim(), dim)

        self.__data = Dataset(np.dot(X, self.__U_reduce), self.__data.y())
        self.__PCA_flag = True

    def rePCA(self):
        """
        对PCA做还原
        需要已调用过PCA，否则不做任何处理
        """
        if self.__PCA_flag is False:
            return

        x = self.__data.x() @ self.__U_reduce.T
        self.__data = Dataset(x * self.__scope + self.__mean, self.__data.y())
        self.__PCA_flag = False

    def show2D(self) -> None:
        """
        图形化显示点集中的各个点及其所属聚类
        只显示前两维
        """
        color = ['b', 'c', 'g', 'k', 'm', 'r', 'w', 'y']
        if self.__data.dim() < 2:
            print("illegal dim")
            return

        import matplotlib.pyplot as plt

        plt.title(self.__name)
        for i in range(self.__data.cnt()):
            plt.scatter(self.__data.x()[i][0],
                        self.__data.x()[i][1],
                        color=color[self.__data.y()[i]])

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
            ax.scatter(self.__data.x()[i][0],
                       self.__data.x()[i][1],
                       self.__data.x()[i][2],
                       color=color[self.__data.y()[i]])

        plt.show()
