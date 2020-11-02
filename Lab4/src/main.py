import load
import random
import numpy as np
from data import Dataset
from process import Process
from config import args

if __name__ == "__main__":
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    if args.file_type == "csv":
        train_x, train_y = load.load_csv(args.path + "/train.csv", 3)
        test_x, test_y = load.load_csv(args.path + "/test.csv", 3)
    data = Dataset(train_x, train_y)

    module = Process(data, "PCA")
    module.PCA(2)
    module.show2D()
