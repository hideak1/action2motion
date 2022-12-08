import numpy as np
from os.path import join as pjoin
from torch.utils.data import DataLoader
from utils.get_opt import get_opt
from utils import paramUtil
from data import dataset

def get_dataset_motion_loader(opt, batch_size = 1):
    if opt.dataset_type == "humanact12":
        opt.data_root = "./dataset/humanact12"
        input_size = 72
        joints_num = 24
        raw_offsets = paramUtil.humanact12_raw_offsets
        kinematic_chain = paramUtil.humanact12_kinematic_chain
        data = dataset.MotionFolderDatasetHumanAct12V2(opt.data_root, opt, lie_enforce=opt.lie_enforce, do_offset=True)
        enumerator = paramUtil.humanact12_coarse_action_enumerator
        label_dec = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    elif opt.dataset_type == "mocap":
        opt.data_root = "./dataset/mocap/mocap_3djoints/"
        clip_path = './dataset/mocap/pose_clip.csv'
        input_size = 60
        joints_num = 20
        raw_offsets = paramUtil.mocap_raw_offsets
        kinematic_chain = paramUtil.mocap_kinematic_chain
        data = dataset.MotionFolderDatasetMocap(clip_path, opt.data_root, opt)
        enumerator = paramUtil.mocap_action_enumerator
        label_dec = [0, 1, 2, 3, 4, 5, 6, 7]

    elif opt.dataset_type == "ntu_rgbd_vibe":
        file_prefix = "./dataset"
        motion_desc_file = "ntu_vibe_list.txt"
        joints_num = 18
        input_size = 54
        labels = paramUtil.ntu_action_labels
        raw_offsets = paramUtil.vibe_raw_offsets
        kinematic_chain = paramUtil.vibe_kinematic_chain
        data = dataset.MotionFolderDatasetNtuVIBE(file_prefix, motion_desc_file, labels, opt, joints_num=joints_num,
                                              offset=True, extract_joints=paramUtil.kinect_vibe_extract_joints)
        enumerator = paramUtil.ntu_action_enumerator
        label_dec = [6, 7, 8, 9, 22, 23, 24, 38, 80, 93, 99, 100, 102]
    else:
        raise NotImplementedError('This dataset is unregonized!!!')
    
    motion_loader = DataLoader(data, batch_size=batch_size, num_workers=1)
    print('Ground Truth Dataset Loading Completed!!!')
    return motion_loader