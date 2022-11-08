import os

from os.path import join as pjoin

import utils.paramUtil as paramUtil
from options.train_options import TrainOptions
from utils.plot_script import *

from networks.modules import *
from networks.quantizer import *
from data import dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
import codecs as cs


# def plot_t2m(data, save_dir):
#     data = train_dataset.inv_transform(data)
#     for i in range(len(data)):
#         joint_data = data[i]
#         joint = recover_from_ric(torch.from_numpy(joint_data).float(), opt.joints_num).numpy()
#         save_path = pjoin(save_dir, '%02d.mp4' % (i))
#         plot_3d_motion(save_path, kinematic_chain, joint, title="None", fps=fps, radius=radius)
def plot(data, label, result_path):
    for i in range(data.shape[0]):
        class_type = enumerator[label_dec[label]]
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

def loadVQModel(opt):
    vq_encoder = VQEncoderV3(input_size - 4, enc_channels, opt.n_down)
    vq_decoder = VQDecoderV3(opt.dim_vq_latent, dec_channels, opt.n_resblk, opt.n_down)
    quantizer = Quantizer(opt.codebook_size, opt.dim_vq_latent, opt.lambda_beta)
    checkpoint = torch.load(pjoin(opt.checkpoints_dir, 'a2m', opt.dataset_type, opt.name, opt.tokenizer_name, 'model', 'finest.tar'),
                            map_location=opt.device)
    vq_encoder.load_state_dict(checkpoint['vq_encoder'])
    quantizer.load_state_dict(checkpoint['quantizer'])
    vq_decoder.load_state_dict(checkpoint['vq_decoder'])
    return vq_encoder, quantizer, vq_decoder


if __name__ == '__main__':
    parser = TrainOptions()
    opt = parser.parse()
    enumerator = None
    opt.is_train = False

    opt.device = torch.device("cpu" if opt.gpu_id==-1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)
    if opt.gpu_id != -1:
        torch.cuda.set_device(opt.gpu_id)

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_type, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')

    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.meta_dir, exist_ok=True)

    if opt.dataset_type == "humanact12":
        opt.data_root = "./dataset/humanact12"
        input_size = 72
        joints_num = 24
        label_dec = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        raw_offsets = paramUtil.humanact12_raw_offsets
        kinematic_chain = paramUtil.humanact12_kinematic_chain
        enumerator = paramUtil.humanact12_coarse_action_enumerator
        data = dataset.MotionFolderDatasetHumanAct12(opt.data_root, opt, lie_enforce=opt.lie_enforce)

    elif opt.dataset_type == "mocap":
        opt.data_root = "./dataset/mocap/mocap_3djoints/"
        clip_path = './dataset/mocap/pose_clip.csv'
        input_size = 60
        joints_num = 20
        raw_offsets = paramUtil.mocap_raw_offsets
        kinematic_chain = paramUtil.mocap_kinematic_chain
        label_dec = [0, 1, 2, 3, 4, 5, 6, 7]
        data = dataset.MotionFolderDatasetMocap(clip_path, opt.data_root, opt)
        enumerator = paramUtil.mocap_action_enumerator

    elif opt.dataset_type == "ntu_rgbd_vibe":
        file_prefix = "./dataset"
        motion_desc_file = "ntu_vibe_list.txt"
        joints_num = 18
        input_size = 54
        labels = paramUtil.ntu_action_labels
        raw_offsets = paramUtil.vibe_raw_offsets
        kinematic_chain = paramUtil.vibe_kinematic_chain
        label_dec = [6, 7, 8, 9, 22, 23, 24, 38, 80, 93, 99, 100, 102]
        data = dataset.MotionFolderDatasetNtuVIBE(file_prefix, motion_desc_file, labels, opt, joints_num=joints_num,
                                              offset=True, extract_joints=paramUtil.kinect_vibe_extract_joints)
        enumerator = paramUtil.ntu_action_enumerator
    else:
        raise NotImplementedError('This dataset is unregonized!!!')

    # mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
    # std = np.load(pjoin(opt.meta_dir, 'std.npy'))

    # all_split_file = pjoin(opt.data_root, 'all.txt')

    enc_channels = [1024]
    dec_channels = [1024, input_size]

    vq_encoder, quantizer, vq_decoder = loadVQModel(opt)

    all_params = 0
    pc_vq_enc = sum(param.numel() for param in vq_encoder.parameters())
    print(vq_encoder)
    print("Total parameters of encoder net: {}".format(pc_vq_enc))
    all_params += pc_vq_enc

    pc_quan = sum(param.numel() for param in quantizer.parameters())
    print(quantizer)
    print("Total parameters of codebook: {}".format(pc_quan))
    all_params += pc_quan

    print('Total parameters of all models: {}'.format(all_params))

    all_dataset = dataset.MotionDataset(data, opt)

    all_loader = DataLoader(all_dataset, batch_size=1, num_workers=1, pin_memory=True)

    token_data_dir = pjoin(opt.data_root, opt.tokenizer_name)
    os.makedirs(token_data_dir, exist_ok=True)

    start_token = opt.codebook_size
    end_token = opt.codebook_size + 1
    pad_token = opt.codebook_size + 2

    max_length = 55
    num_replics = 5
    opt.unit_length = 4

    vq_encoder.to(opt.device)
    quantizer.to(opt.device)
    vq_decoder.to(opt.device)
    vq_encoder.eval()
    quantizer.eval()
    vq_decoder.eval()
    with torch.no_grad():
        # Since our dataset loader introduces some randomness (not much), we could generate multiple token sequences
        # to increase the robustness.
        for e in range(num_replics):
            for i, data in enumerate(tqdm(all_loader)):
                motion, name = data
                class_type = enumerator[label_dec[name[0] - 1]]
                motion = motion.detach().to(opt.device).float()
                pre_latents = vq_encoder(motion[..., :-4])
                indices = quantizer.map2index(pre_latents)
                indices = list(indices.cpu().numpy())
                # indices = [start_token] + indices + [end_token] + [pad_token] * (max_length - len(indices) - 2)
                indices = [str(token) for token in indices]
                # with cs.open(pjoin(token_data_dir, '%s.txt'%int(name[0])), 'a+') as f:
                #     f.write(' '.join(indices))
                #     f.write('\n')
                _, vq_latents, _, _ = quantizer(pre_latents)
                # print(self.vq_latents.shape)
                recon_motions = vq_decoder(vq_latents)
                plot(recon_motions.cpu().numpy(), name, pjoin("remote_train/test24/", 'gen_motion_%02d_L%03d' % (i, motion.shape[1])))