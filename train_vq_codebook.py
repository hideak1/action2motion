import os

from os.path import join as pjoin

import utils.paramUtil as paramUtil
from options.train_options import TrainOptions
from utils.plot_script import *

from networks.modules import *
from networks.quantizer import *
from networks.trainers import VQTokenizerTrainerV3, VQTokenizerTrainer
from data import dataset
from torch.utils.data import DataLoader
from utils.plot_script import plot_loss
from utils.utils import save_logfile

# def plot_t2m(data, save_dir):
#     data = train_dataset.inv_transform(data)
#     for i in range(len(data)):
#         joint_data = data[i]
#         joint = recover_from_ric(torch.from_numpy(joint_data).float(), opt.joints_num).numpy()
#         save_path = pjoin(save_dir, '%02d.mp4' % (i))
#         plot_3d_motion(save_path, kinematic_chain, joint, title="None", fps=fps, radius=radius)

def load_models(opt):
    vq_encoder = VQEncoderV3(opt.input_size - 4, enc_channels, opt.n_down)
    vq_decoder = VQDecoderV3(opt.dim_vq_latent, dec_channels, opt.n_resblk, opt.n_down)

    quantizer = Quantizer(opt.codebook_size, opt.dim_vq_latent, opt.lambda_beta)

    if opt.is_continue:
        checkpoint = torch.load(pjoin(opt.checkpoints_dir, 'a2m', opt.dataset_type,
                                      opt.name, opt.tokenizer_name, 'model', 'finest.tar'),
                                map_location=opt.device)
        vq_encoder.load_state_dict(checkpoint['vq_encoder'])
        vq_decoder.load_state_dict(checkpoint['vq_decoder'])
        quantizer.load_state_dict(checkpoint['quantizer'])
    return vq_encoder, vq_decoder, quantizer


if __name__ == '__main__':
    parser = TrainOptions()
    opt = parser.parse()

    opt.device = torch.device("cuda:" + str(opt.gpu_id) if torch.cuda.is_available() else "cpu")
    if opt.use_wandb:
        import wandb
        wandb.init(project='ece740-a2m', config=opt)
    opt.save_root = os.path.join(opt.checkpoints_dir, 'a2m', opt.dataset_type, opt.name, opt.tokenizer_name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.joints_path = os.path.join(opt.save_root, 'joints')
    opt.log_path = os.path.join(opt.save_root, "log.txt")
    opt.eval_dir = pjoin(opt.save_root, 'animation')

    if not os.path.exists(opt.model_dir):
        os.makedirs(opt.model_dir)
    if not os.path.exists(opt.joints_path):
        os.makedirs(opt.joints_path)

    dataset_path = ""
    joints_num = 0
    input_size = 72
    data = None

    if opt.dataset_type == "humanact12":
        dataset_path = "./dataset/humanact12"
        input_size = 72
        joints_num = 24
        raw_offsets = paramUtil.humanact12_raw_offsets
        kinematic_chain = paramUtil.humanact12_kinematic_chain
        data = dataset.MotionFolderDatasetHumanAct12V2(dataset_path, opt, lie_enforce=opt.lie_enforce, do_offset=False)

    elif opt.dataset_type == "mocap":
        dataset_path = "./dataset/mocap/mocap_3djoints/"
        clip_path = './dataset/mocap/pose_clip.csv'
        input_size = 60
        joints_num = 20
        raw_offsets = paramUtil.mocap_raw_offsets
        kinematic_chain = paramUtil.mocap_kinematic_chain
        data = dataset.MotionFolderDatasetMocap(clip_path, dataset_path, opt)
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
        label_dec = [6, 7, 8, 9, 22, 23, 24, 38, 80, 93, 99, 100, 102]
    else:
        raise NotImplementedError('This dataset is unregonized!!!')

    opt.mean = np.load(pjoin(dataset_path, 'zscore', 'Mean.npy'))
    opt.std = np.load(pjoin(dataset_path, 'zscore', 'Std.npy'))

    opt.dim_category = len(data.labels)
    # arbitrary_len won't limit motion length, but the batch size has to be 1
    if opt.arbitrary_len:
        opt.batch_size = 1
        motion_loader = DataLoader(data, batch_size=opt.batch_size, drop_last=True, num_workers=1, shuffle=True)
    else:
        motion_dataset = dataset.VQVaeMotionDataset(data, opt)
        motion_loader = DataLoader(motion_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=2, shuffle=True)
    opt.pose_dim = input_size

    # if opt.time_counter:
    #     opt.input_size = input_size + opt.dim_category + 1
    # else:
    #     opt.input_size = input_size + opt.dim_category

    opt.input_size = input_size
    opt.output_size = input_size

    # mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
    # std = np.load(pjoin(opt.data_root, 'Std.npy'))

    # train_split_file = pjoin(opt.data_root, 'train.txt')
    # val_split_file = pjoin(opt.data_root, 'val.txt')

    enc_channels = [1024, opt.dim_vq_latent]
    dec_channels = [opt.dim_vq_latent, 1024, input_size]

    # vq_encoder = VQEncoderV2(dim_pose-4, opt.dim_vq_enc_hidden, opt.dim_vq_latent

    vq_encoder, vq_decoder, quantizer = load_models(opt)

    # for name, parameters in vq_decoder.named_parameters():
    #     print(name, ':', parameters.size())
        # parm[name] = parameters.detach().numpy()

    discriminator = VQDiscriminator(input_size, opt.dim_vq_dis_hidden, opt.n_layers_dis)

    all_params = 0
    pc_vq_enc = sum(param.numel() for param in vq_encoder.parameters())
    print(vq_encoder)
    print("Total parameters of encoder net: {}".format(pc_vq_enc))
    all_params += pc_vq_enc

    pc_quan = sum(param.numel() for param in quantizer.parameters())
    print(quantizer)
    print("Total parameters of codebook: {}".format(pc_quan))
    all_params += pc_quan

    pc_vq_dec = sum(param.numel() for param in vq_decoder.parameters())
    print(vq_decoder)
    print("Total parameters of decoder net: {}".format(pc_vq_dec))
    all_params += pc_vq_dec

    pc_vq_dis = sum(param.numel() for param in discriminator.parameters())
    print(discriminator)
    print("Total parameters of discriminator net: {}".format(pc_vq_dis))
    all_params += pc_vq_dis

    print('Total parameters of all models: {}'.format(all_params))

    trainer = VQTokenizerTrainerV3(opt, vq_encoder, quantizer, vq_decoder, discriminator)


    logs = trainer.train(motion_loader, motion_loader)
    plot_loss(logs, os.path.join(opt.save_root, "loss_curve.png"), opt.plot_every)
    save_logfile(logs, opt.log_path)