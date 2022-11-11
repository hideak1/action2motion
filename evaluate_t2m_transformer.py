import os

from os.path import join as pjoin

import utils.paramUtil as paramUtil
from options.evaluate_options import TestOptions
from utils.plot_script import *

from networks.transformer import TransformerV5
from networks.quantizer import *
from networks.modules import *
from networks.trainers import TransformerA2MTrainer
from data import dataset
from torch.utils.data import DataLoader
from utils.word_vectorizer import WordVectorizerV2
from data.dataset import TestActionTokenDataset
from tqdm import tqdm

def plot(data, label, result_path, do_offset = True):
    for i in range(data.shape[0]):
        class_type = enumerator[label_dec[label]]
        motion_orig = data[i]
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        keypoint_path = os.path.join(result_path, 'keypoint')
        if not os.path.exists(keypoint_path):
            os.makedirs(keypoint_path)
        file_name = os.path.join(result_path, class_type + str(i) + ".gif")


        # offset = np.matlib.repmat(np.array([motion_orig[0, 0], motion_orig[0, 1], motion_orig[0, 2]]),
        #                                 motion_orig.shape[0], joints_num)

        # motion_mat = motion_orig - offset

        motion_mat = motion_orig
        if do_offset:
            for j in range(1, motion_orig.shape[0], 1):
                offset = np.matlib.repmat(np.array([motion_orig[j - 1, 0], motion_orig[j - 1, 1], motion_orig[j - 1, 2]]),
                                            1, joints_num)

                motion_mat[j] = motion_orig[j] + offset

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

    checkpoint = torch.load(pjoin(opt.checkpoints_dir, 'a2m', opt.dataset_type, opt.name, opt.tokenizer_name, 'model', 'finest.tar'),
                            map_location=opt.device)
    vq_decoder.load_state_dict(checkpoint['vq_decoder'])
    quantizer.load_state_dict(checkpoint['quantizer'])

    a2m_transformer = TransformerV5(13, opt.txt_pad_idx, n_mot_vocab, opt.mot_pad_idx, d_src_word_vec=12,
                                    d_trg_word_vec=12,
                                    d_model=12, d_inner=opt.d_inner_hid, n_enc_layers=opt.n_enc_layers,
                                    n_dec_layers=opt.n_dec_layers, n_head=opt.n_head, d_k=opt.d_k, d_v=opt.d_v,
                                    dropout=0.1,
                                    n_src_position=50, n_trg_position=100,
                                    trg_emb_prj_weight_sharing=opt.proj_share_weight)


    checkpoint = torch.load(pjoin(opt.checkpoints_dir, 'a2m', opt.dataset_type, opt.name, 'model', '%s.tar'%(opt.which_epoch)),
                            map_location=opt.device)
    a2m_transformer.load_state_dict(checkpoint['transformer'])
    print('Loading t2m_transformer model: Epoch %03d Total_Iter %03d' % (checkpoint['ep'], checkpoint['total_it']))

    return vq_decoder, quantizer, a2m_transformer



if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    # torch.multiprocessing.set_sharing_strategy('file_system')
    parser = TestOptions()
    opt = parser.parse()

    opt.device = torch.device("cpu" if opt.gpu_id==-1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)
    if opt.gpu_id != -1:
        torch.cuda.set_device(opt.gpu_id)

    opt.save_root = os.path.join(opt.checkpoints_dir, 'a2m', opt.dataset_type, opt.name, opt.tokenizer_name)
    opt.result_dir = pjoin(opt.result_path, opt.dataset_type, opt.name, opt.ext)
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
        data = dataset.MotionFolderDatasetHumanAct12V2(opt.data_root, opt, lie_enforce=opt.lie_enforce, do_offset=False)
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
                                               extract_joints=paramUtil.kinect_vibe_extract_joints)
        enumerator = paramUtil.ntu_action_enumerator
        label_dec = [6, 7, 8, 9, 22, 23, 24, 38, 80, 93, 99, 100, 102]
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

    enc_channels = [1024, opt.dim_vq_latent]
    dec_channels = [opt.dim_vq_latent, 1024, input_size]

    vq_decoder, quantizer, a2m_transformer = build_models(opt)

    dataset = TestActionTokenDataset(data, opt, w_vectorizer)
    data_loader = DataLoader(dataset, batch_size=opt.batch_size,num_workers=0, shuffle=True, pin_memory=False)

    vq_decoder.to(opt.device)
    quantizer.to(opt.device)
    a2m_transformer.to(opt.device)

    vq_decoder.eval()
    quantizer.eval()
    a2m_transformer.eval()

    opt.repeat_times = opt.repeat_times if opt.sample else 1

    '''Generating Results'''
    print('Generating Results')
    result_dict = {}
    with torch.no_grad():
        for i, batch_data in enumerate(tqdm(data_loader)):
            print('%02d_%03d'%(i, opt.num_results))
            motions, word_emb, label, cap_lens, m_tokens, m_tokens_len = batch_data

            # word_emb, word_ids, caption, cap_lens, m_tokens, len_tokens = batch_data
            word_emb = word_emb.detach().to(opt.device).float()
            m_tokens = m_tokens.detach().to(opt.device).long()
            # word_ids = word_ids.detach().to(opt.device).long()
            # gt_tokens = motions[:, :m_lens[0]]

            # print(captions[0])
            # print('Ground Truth Tokens')
            # print(gt_tokens[0])

            # rec_vq_latent = quantizer.get_codebook_entry(gt_tokens)
            # rec_motion = vq_decoder(rec_vq_latent)
            l = len(motions[0])
            name = 'L%03dC%03d' % (l, i)
            item_dict = {
                'length': l,
                'label': label,
                'gt_motion': motions[:, :l].cpu().numpy()
            }
            for t in range(opt.repeat_times):
                # if opt.t2m_v2:
                #     pred_tokens = t2m_transformer.sample(word_ids, trg_sos=opt.mot_start_idx, trg_eos=opt.mot_end_idx,
                #                                          max_steps=80, sample=opt.sample, top_k=opt.top_k)
                # else:
                pred_tokens = a2m_transformer.sample(word_emb, m_tokens_len, trg_sos=opt.mot_start_idx,
                                                     trg_eos=opt.mot_end_idx, max_steps=80, sample=opt.sample,
                                                     top_k=opt.top_k, input_onehot=True)
                pred_tokens = pred_tokens[:, 1:]
                print('Sampled Tokens %02d'%t)
                print(pred_tokens[0])
                if len(pred_tokens[0]) == 0:
                    continue
                vq_latent = quantizer.get_codebook_entry(pred_tokens)
                gen_motion = vq_decoder(vq_latent)

                sub_dict = {}
                sub_dict['motion'] = gen_motion.cpu().numpy()
                sub_dict['length'] = len(gen_motion[0])
                item_dict['result_%02d'%t] = sub_dict

            result_dict[name] = item_dict
            if i > opt.num_results:
                break

    print('Animating Results')
    '''Animating Results'''
    for i, (key, item) in enumerate(result_dict.items()):
        print('%02d_%03d' % (i, opt.num_results))
        label = item['label']
        gt_motions = item['gt_motion']
        joint_save_path = pjoin(opt.joint_dir, key)
        animation_save_path = pjoin(opt.animation_dir, key)

        os.makedirs(joint_save_path, exist_ok=True)
        os.makedirs(animation_save_path, exist_ok=True)

        # np.save(pjoin(joint_save_path, 'gt_motions.npy'), gt_motions)
        plot(gt_motions, label, pjoin(animation_save_path, 'gt_motion'))
        for t in range(opt.repeat_times):
            sub_dict = item['result_%02d' % t]
            motion = sub_dict['motion']
            # np.save(pjoin(joint_save_path, 'gen_motion_%02d_L%03d.npy' % (t, motion.shape[1])), motion)
            plot(motion, label, pjoin(animation_save_path, 'gen_motion_%02d_L%03d' % (t, motion.shape[1])))