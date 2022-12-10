import os
from argparse import Namespace
import re
from os.path import join as pjoin
from utils.word_vectorizer import POS_enumerator
from utils import paramUtil

def is_float(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')    # 去除正数(+)、负数(-)符号
    try:
        reg = re.compile(r'^[-+]?[0-9]+\.[0-9]+$')
        res = reg.match(str(numStr))
        if res:
            flag = True
    except Exception as ex:
        print("is_float() - error: " + str(ex))
    return flag


def is_number(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')    # 去除正数(+)、负数(-)符号
    if str(numStr).isdigit():
        flag = True
    return flag


def get_opt(opt_path, device):
    opt = Namespace()
    opt_dict = vars(opt)

    skip = ('-------------- End ----------------',
            '------------ Options -------------',
            '\n')
    print('Reading', opt_path)
    with open(opt_path) as f:
        for line in f:
            if line.strip() not in skip:
                # print(line.strip())
                key, value = line.strip().split(': ')
                if value in ('True', 'False'):
                    opt_dict[key] = (value == 'True')
                #     print(key, value)
                elif is_float(value):
                    opt_dict[key] = float(value)
                elif is_number(value):
                    opt_dict[key] = int(value)
                else:
                    opt_dict[key] = str(value)

    # print(opt)
    opt_dict['which_epoch'] = 'finest'
    opt.save_root = pjoin(opt.checkpoints_dir, 'a2m', opt.dataset_type, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')
    opt.eval_dir = pjoin(opt.save_root, 'animation')
    opt.log_dir = pjoin('./log', opt.dataset_type, opt.name)

    if opt.dataset_type == "humanact12":
        opt.data_root = "./dataset/humanact12"
        opt.input_size = 72
        opt.joints_num = 24
        opt.raw_offsets = paramUtil.humanact12_raw_offsets
        opt.kinematic_chain = paramUtil.humanact12_kinematic_chain
        opt.enumerator = paramUtil.humanact12_coarse_action_enumerator
        opt.enumerator_rev = paramUtil.humanact12_coarse_action_enumerator_rev
        opt.label_dec = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    elif opt.dataset_type == "mocap":
        opt.data_root = "./dataset/mocap/mocap_3djoints/"
        opt.clip_path = './dataset/mocap/pose_clip.csv'
        opt.input_size = 60
        opt.joints_num = 20
        opt.raw_offsets = paramUtil.mocap_raw_offsets
        opt.kinematic_chain = paramUtil.mocap_kinematic_chain
        opt.enumerator = paramUtil.mocap_action_enumerator
        opt.enumerator_rev = paramUtil.mocap_action_enumerator_rev
        opt.label_dec = [0, 1, 2, 3, 4, 5, 6, 7]

    elif opt.dataset_type == "ntu_rgbd_vibe":
        opt.file_prefix = "./dataset"
        opt.motion_desc_file = "ntu_vibe_list.txt"
        opt.joints_num = 18
        opt.input_size = 54
        opt.labels = paramUtil.ntu_action_labels
        opt.raw_offsets = paramUtil.vibe_raw_offsets
        opt.kinematic_chain = paramUtil.vibe_kinematic_chain
        opt.enumerator = paramUtil.ntu_action_enumerator
        opt.label_dec = [6, 7, 8, 9, 22, 23, 24, 38, 80, 93, 99, 100, 102]
    else:
        raise NotImplementedError('This dataset is unregonized!!!')

    opt.dim_word = 300
    opt.num_classes = 200
    opt.dim_pos_ohot = len(POS_enumerator)
    opt.is_train = False
    opt.is_continue = False
    opt.device = device
    opt.result_path = os.path.join('./eval_results/vae/', opt.dataset_type, opt.name)

    return opt