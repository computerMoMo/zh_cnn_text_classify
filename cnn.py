# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import json
import os
from collections import OrderedDict

Data_dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")


class MyCNN(object):
    def __init__(self, configs):
        self.sentence_len = configs["sentence_len"]
        self.num_classes = configs["num_classes"]
        self.embedding_size = configs["embedding_size"]
        self.filter_sizes = configs["filter_sizes"].split(" ")
        self.num_filters = configs["num_filters"]
        self.l2_reg_lambda = configs["l2_reg_lambda"]



# for test
if __name__ == "__main__":
    cnn_configs = json.load(open(os.path.join(Data_dir_path, "cnn_configs.json")))
    cnn_example = MyCNN(cnn_configs)
