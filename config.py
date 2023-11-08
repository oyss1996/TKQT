import configparser
import numpy as np
import json


class Config(object):
    def __init__(self, dataset):
        config_file = './data/' + dataset + '.ini'
        conf = configparser.ConfigParser()
        try:
            conf.read(config_file)
        except:
            print("loading config: %s failed" % config_file)

        # Parameter
        self.dataset = dataset
        self.seed = conf.getint("Model_Setup", "seed")
        self.features = np.array(json.loads(conf.get("Model_Setup", "features")))
        self.tr_ranks = np.array(json.loads(conf.get("Model_Setup", "tr_ranks")))
        self.window = conf.getint("Model_Setup", "window")

        # Dataset
        self.num_dim = np.array(json.loads(conf.get("Data_Setting", "ndim")))
        # self.trans = tuple(json.loads(conf.get("Data_Setting", "tran")))
        # self.num_row = conf.getint("Data_Setting", "nrow")
        # self.num_col = conf.getint("Data_Setting", "ncol")
        # self.num_time = conf.getint("Data_Setting", "ntime")

        self.data_path = conf.get("Data_Setting", "data_path")
        self.location_path = conf.get("Data_Setting", "location_path")
        # self.location_path = None
        self.data_size = conf.getint("Data_Setting", "data_size")
