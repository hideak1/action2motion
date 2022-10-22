import imp
import os

from os.path import join as pjoin

import utils.paramUtil as paramUtil
from options.evaluate_options import TestOptions
from utils.plot_script import *

from networks.transformer import TransformerV1, TransformerV2
from networks.quantizer import *
from networks.modules import *
# from scripts.motion_process import *
from torch.utils.data import DataLoader
from utils.word_vectorizer import WordVectorizerV2
from utils.utils import *
from tqdm import tqdm
from data import dataset

def plot(data, label):
    for i in range(data.shape[0]):
        class_type = label
        motion_orig = data[i]
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        keypoint_path = os.path.join(result_path, 'keypoint')
        if not os.path.exists(keypoint_path):
            os.makedirs(keypoint_path)
        file_name = os.path.join(result_path, class_type + str(i) + ".gif")
        offset = np.matlib.repmat(np.array([motion_orig[0, 0], motion_orig[0, 1], motion_orig[0, 2]]),
                                        motion_orig.shape[0], joints_num)

        motion_mat = motion_orig - offset

        motion_mat = motion_mat.reshape(-1, joints_num, 3)
        np.save(os.path.join(keypoint_path, class_type + str(i) + '_3d.npy'), motion_mat)

        if opt.dataset_type == "humanact12":
            plot_3d_motion_v2(motion_mat, kinematic_chain, save_path=file_name, interval=80)

        elif opt.dataset_type == "ntu_rgbd_vibe":
            plot_3d_motion_v2(motion_mat, kinematic_chain, save_path=file_name, interval=80)

        elif opt.dataset_type == "mocap":
            plot_3d_motion_v2(motion_mat, kinematic_chain, save_path=file_name, interval=80, dataset="mocap")


def build_models(opt):
    vq_decoder = VQDecoderV3(opt.dim_vq_latent, dec_channels, opt.n_resblk, opt.n_down)
    quantizer = Quantizer(opt.codebook_size, opt.dim_vq_latent, opt.lambda_beta)

    checkpoint = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_type, opt.tokenizer_name, 'model', 'finest.tar'),
                            map_location=opt.device)
    for k, v in checkpoint.items():
        print(k)
    vq_decoder.load_state_dict(checkpoint['vq_decoder'])
    quantizer.load_state_dict(checkpoint['quantizer'])
    return vq_decoder, quantizer



if __name__ == '__main__':
    parser = TestOptions()
    opt = parser.parse()

    opt.device = torch.device("cpu" if opt.gpu_id==-1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)
    if opt.gpu_id != -1:
        torch.cuda.set_device(opt.gpu_id)

    opt.save_root = os.path.join(opt.checkpoints_dir, opt.dataset_type, opt.name)
    result_path = os.path.join(opt.result_path, opt.dataset_type, opt.name + opt.name_ext)
    opt.result_dir = pjoin(opt.result_path, opt.dataset_type, opt.name + opt.name_ext)
    opt.joint_dir = pjoin(opt.result_dir, 'joints')
    opt.animation_dir = pjoin(opt.result_dir, 'animations')

    os.makedirs(opt.joint_dir, exist_ok=True)
    os.makedirs(opt.animation_dir, exist_ok=True)

    if opt.dataset_type == "humanact12":
        opt.data_root = "./dataset/humanact12"
        input_size = 72
        joints_num = 24
        raw_offsets = paramUtil.humanact12_raw_offsets
        kinematic_chain = paramUtil.humanact12_kinematic_chain
        data = dataset.MotionFolderDatasetHumanAct12(opt.data_root, opt, lie_enforce=opt.lie_enforce)

    elif opt.dataset_type == "mocap":
        opt.data_root = "./dataset/mocap/mocap_3djoints/"
        clip_path = './dataset/mocap/pose_clip.csv'
        input_size = 60
        joints_num = 20
        raw_offsets = paramUtil.mocap_raw_offsets
        kinematic_chain = paramUtil.mocap_kinematic_chain
        data = dataset.MotionFolderDatasetMocap(clip_path, opt.data_root, opt)

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
    else:
        raise NotImplementedError('This dataset is unregonized!!!')

    enc_channels = [1024, opt.dim_vq_latent]
    dec_channels = [opt.dim_vq_latent, 1024, input_size]


    vq_decoder, quantizer = build_models(opt)

    vq_decoder.to(opt.device)
    quantizer.to(opt.device)

    vq_decoder.eval()
    quantizer.eval()

    with torch.no_grad():
        for i in tqdm(range(1024)):
            m_token = torch.LongTensor(1, 1).fill_(i).to(opt.device)
            vq_latent = quantizer.get_codebook_entry(m_token)
            gen_motion = vq_decoder(vq_latent)

            plot(gen_motion.cpu().numpy(), str(i))

