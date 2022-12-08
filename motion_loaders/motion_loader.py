from torch.utils.data import DataLoader, Dataset
from utils.get_opt import get_opt
import numpy as np
from torch.utils.data._utils.collate import default_collate
import glob
import re
import os

def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)

class A2MGeneratedDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        file_list = []
        for path, currentDirectory, files in os.walk(opt.result_path):
            for file in files:
                if file.endswith(".npy"):
                    file_list.append(os.path.join(path, file))
        self.dataset = []
        self.labels = set()
        for file in file_list:
            pose_raw = np.load(file)
            motion_mat = pose_raw.reshape((-1, opt.joints_num * 3))
            path = os.path.normpath(file)
            folder = path.split(os.sep)[-4]
            file_name = os.path.basename(file)
            action = '_'.join(file_name.split('_')[:-1])[:-1]
            # idx = int(re.findall(r'\d+', prefix)[0])
            # idx = int(folder[-2:])
            label = opt.enumerator_rev[action]
            self.labels.add(label)
            # if self.dataset.__contains__(label):
            #     self.dataset[label].append(motion_mat)
            # else:
            #     self.dataset[label] = list(motion_mat)
            self.dataset.append({"label": label, "motion": motion_mat})
        self.labels = list(self.labels)
        self.labels.sort()
        self.label_enc = dict(zip(self.labels, np.arange(len(self.labels))))
        self.label_enc_rev = dict(zip(np.arange(len(self.labels)), self.labels))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        label = self.dataset[item]['label']
        motion = self.dataset[item]['motion']
        # label = self.labels[item]
        # data = self.dataset[label]
        return motion, self.label_enc[label]



def get_motion_loader(opt):
    dataset = A2MGeneratedDataset(opt)

    # motion_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, drop_last=True, num_workers=4)
    motion_loader = DataLoader(dataset, batch_size=1, num_workers=1)

    print('Generated Dataset Loading Completed!!!')

    return motion_loader