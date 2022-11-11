import os

from os.path import join as pjoin

import utils.paramUtil as paramUtil
from options.train_options import Action2MotionOptions
from utils.plot_script import *

from networks.transformer import TransformerV5, TransformerV6
from networks.quantizer import *
from networks.modules import *
from networks.trainers import TransformerA2MTrainer
from data import dataset
from torch.utils.data import DataLoader
from utils.word_vectorizer import WordVectorizerV2
from data.dataset import ActionTokenDataset

if __name__ == '__main__':
    parser = Action2MotionOptions()
    opt = parser.parse()

    opt.device = torch.device("cpu" if opt.gpu_id==-1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)
    if opt.gpu_id != -1:
        torch.cuda.set_device(opt.gpu_id)

    opt.save_root = pjoin(opt.checkpoints_dir, 'a2m', opt.dataset_type, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')
    opt.eval_dir = pjoin(opt.save_root, 'animation')
    opt.log_dir = pjoin('./log', opt.dataset_type, opt.name)

    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.meta_dir, exist_ok=True)
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    if opt.dataset_type == "humanact12":
        opt.data_root = "./dataset/humanact12"
        input_size = 72
        joints_num = 24
        raw_offsets = paramUtil.humanact12_raw_offsets
        kinematic_chain = paramUtil.humanact12_kinematic_chain
        data = dataset.MotionFolderDatasetHumanAct12V2(opt.data_root, opt, lie_enforce=opt.lie_enforce, do_offset = False)
        label_dec = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

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
                                               extract_joints=paramUtil.kinect_vibe_extract_joints)
    else:
        raise NotImplementedError('This dataset is unregonized!!!')


    w_vectorizer = WordVectorizerV2('./glove', 'our_vab')

    n_mot_vocab = opt.codebook_size + 3
    opt.mot_start_idx = opt.codebook_size
    opt.mot_end_idx = opt.codebook_size + 1
    opt.mot_pad_idx = opt.codebook_size + 2

    n_txt_vocab = len(w_vectorizer) + 1
    _, _, opt.txt_start_idx = w_vectorizer['sos/OTHER']
    _, _, opt.txt_end_idx = w_vectorizer['eos/OTHER']
    opt.txt_pad_idx = 0

    opt.dim_category = len(data.labels)

    enc_channels = [opt.dim_vq_latent]
    dec_channels = [opt.dim_vq_latent, input_size]

    # if opt.t2m_v2:
    # a2m_transformer = TransformerV5(12, opt.txt_pad_idx, n_mot_vocab, opt.mot_pad_idx, d_src_word_vec=12,
    #                                 d_trg_word_vec=12,
    #                                 d_model=12, d_inner=opt.d_inner_hid, n_enc_layers=opt.n_enc_layers,
    #                                 n_dec_layers=opt.n_dec_layers, n_head=opt.n_head, d_k=opt.d_k, d_v=opt.d_v,
    #                                 dropout=0.1,
    #                                 n_src_position=50, n_trg_position=100,
    #                                 trg_emb_prj_weight_sharing=opt.proj_share_weight
    #                                     )

    # 我改了transformer的数
    a2m_transformer = TransformerV6(13, opt.txt_pad_idx, n_mot_vocab, opt.mot_pad_idx, d_src_word_vec=12,
                                    d_trg_word_vec=12,
                                    d_model=12, d_inner=opt.d_inner_hid, n_enc_layers=opt.n_enc_layers,
                                    n_dec_layers=opt.n_dec_layers, n_head=opt.n_head, d_k=opt.d_k, d_v=opt.d_v,
                                    dropout=0.1,
                                    n_src_position=50, n_trg_position=100,
                                    trg_emb_prj_weight_sharing=opt.proj_share_weight
                                        )

    # else:
    #     t2m_transformer = TransformerV1(n_mot_vocab, opt.mot_pad_idx, d_src_word_vec=300, d_trg_word_vec=512,
    #                                     d_model=opt.d_model, d_inner=opt.d_inner_hid, n_enc_layers=opt.n_enc_layers,
    #                                     n_dec_layers=opt.n_dec_layers, n_head=opt.n_head, d_k=opt.d_k, d_v=opt.d_v, dropout=0.1,
    #                                     n_src_position=50, n_trg_position=100, trg_emb_prj_weight_sharing=opt.proj_share_weight)


    all_params = 0
    pc_transformer = sum(param.numel() for param in a2m_transformer.parameters())
    print(a2m_transformer)
    print("Total parameters of t2m_transformer net: {}".format(pc_transformer))
    all_params += pc_transformer

    print('Total parameters of all models: {}'.format(all_params))

    trainer = TransformerA2MTrainer(opt, a2m_transformer)

    train_dataset = ActionTokenDataset(data, opt, w_vectorizer)
    val_dataset = ActionTokenDataset(data, opt, w_vectorizer)

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=0,
                              shuffle=True, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=0,
                            shuffle=True, pin_memory=False)

    trainer.train(train_loader, val_loader, None)