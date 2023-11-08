import numpy as np


class Dataset(object):
    def __init__(self, config, train_ratio):
        super(Dataset, self).__init__()
        self.num_dim = config.num_dim
        self.window = config.window

        self.train_data, self.train_mask, self.test_data, self.test_mask, self.num_train, self.num_test, self.max_value\
            = self.get_dataset(config, train_ratio)
        print('-Train entries number:', self.num_train)
        print('-Test entries number:', self.num_test)
        print('-Max value:', self.max_value)

    def get_batch_data(self, records):
        for i in range(len(records)):
            x = records[i, 0, :]
            y = records[i, 1, :]
            z = records[i, 2, :]
            w = records[i, 3, :]
            yield x, y, z, w

    def get_all_batch_data(self):
        ds = np.stack([self.train_data, self.train_mask, self.test_data, self.test_mask], 0)
        records = self.process2records(ds)

        all_batch_data = []
        batch_data = self.get_batch_data(records)
        for ds in batch_data:
            all_batch_data.append(ds)

        return all_batch_data

    def get_dataset(self, config, train_ratio):
        data = np.load(config.data_path).astype(np.float32)
        max_value = np.max(data)
        data = data / max_value
        location = np.load(config.location_path).astype(int)
        data = data[0: config.num_dim[0]]
        location = location[0: config.num_dim[0]]

        if config.dataset == 'MBD' or config.dataset == 'PMU':
            train_mask = []
            test_mask = []
            for i in range(0, config.num_dim[0] // config.window):
                loc = location[i * config.window: (i + 1) * config.window]
                num_sample = int(np.ceil(np.max(loc) * train_ratio))
                train_mask += [((loc > 0) & (loc <= num_sample)).astype(np.float32)]
                test_mask += [(loc > num_sample).astype(np.float32)]
            train_mask = np.concatenate(train_mask)
            test_mask = np.concatenate(test_mask)

        else:
            num_sample = int(np.ceil(np.max(location) * train_ratio))
            train_mask = ((location > 0) & (location <= num_sample)).astype(np.float32)
            test_mask = (location > num_sample).astype(np.float32)
        train_data = train_mask * data
        test_data = test_mask * data

        num_train = np.sum(train_mask)
        num_test = np.sum(test_mask)

        return train_data, train_mask, test_data, test_mask, num_train, num_test, max_value

    def process2records(self, data_tensor):
        data_list = []
        for i in range(0, self.num_dim[0]//self.window):
            data_batch = data_tensor[:, i*self.window: (i+1)*self.window]
            data_list.append(data_batch)
        return np.asarray(data_list)
