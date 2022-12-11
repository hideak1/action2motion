# action2motion
The code is based on https://github.com/EricGuo5513/TM2T


use following command to setup the environment.
```
conda create -f environment.yaml
conda activate a2m
```
or install by pip
```
conda install pytorch torchvision torchaudio cpuonly -c pytorch
pip install 
scipy
numpy
tensorflow       # For use of tensorboard only
spacy
tqdm
ffmpeg
matplotlib
numpy
opencv-python
pandas
joblib
```

### Data

We use three datasets and they are: `HumanAct12`, `NTU-RGBD` and `CMU Mocap`. All datasets have been properly pre-transformed to better fit our purpose. Details are provided in our project [webpage](https://ericguo5513.github.io/action-to-motion/) or dataset documents. 

Create a folder for dataset

```sh
mkdir ./dataset/
```

#### Download HumanAct12 Dataset
If you'd like to use HumanAct12 dataset, download the data folder [here](https://drive.google.com/drive/folders/1hGNkxI9jPbdFueUHfj8H8zC2f7DTb8NG?usp=sharing), and place it in `dataset/`

#### Download NTU-RGBD Dataset
The dataset is not public now.

#### Download CMU Mocap Dataset
If you'd like to use CMU-Mocap dataset, download the data folder [here](https://drive.google.com/drive/folders/1_2jbZK48Li6sm1duNJnR_eyQjVdJQDoU?usp=sharing), and place it in `dataset/`


### Pre-trained model
We have trained models for different dataset, if you want to have a try, you can use our pre-traiend models.

#### HumanAct12 Dataset
If you'd like to use model trained with HumanAct12, download the zip file [here](https://drive.google.com/file/d/1DZFInlqKFA7Q2vAcPfK4cmUbriKR3ZZx/view?usp=share_link), and unzip it in `checkpoint/a2m/`

#### Download NTU-RGBD Dataset
Not available

#### Download CMU Mocap Dataset
If you'd like to use model trained with CMU-Mocap, download the zip file [here](https://drive.google.com/file/d/1sFkBqqhxBVbvbLqcp-QQVhUDtzsbdyWM/view?usp=share_link), and unzip it in `checkpoint/a2m/`

### Train VQ-VAE
#### HumanAct12 Dataset
```
python train_vq_codebook.py --name xxxx(place your experiment name here) --dataset_type humanact12 --batch_size 32 --motion_length 60 --coarse_grained --lambda_kld 0.001 --eval_every 2000 --plot_every 50 --print_every 20 --save_every 2000 --save_latest 50 --time_counter --gpu_id 0 --iters 50000 --n_resblk 3 --dim_vq_latent 1024 --codebook_size 1024 --lr_scheduler_every_e 100 --n_down 2 
```

#### NTU-RGBD Dataset
```
python train_vq_codebook.py --name xxxx(place your experiment name here) --dataset_type ntu_rgbd_vibe --batch_size 32 --motion_length 60 --coarse_grained --lambda_kld 0.001 --eval_every 2000 --plot_every 50 --print_every 20 --save_every 2000 --save_latest 50 --time_counter --gpu_id 0 --iters 50000 --n_resblk 3 --dim_vq_latent 1024 --codebook_size 1024 --lr_scheduler_every_e 100 --n_down 2 
```

#### CMU Mocap Dataset
```
python train_vq_codebook.py --name xxxx(place your experiment name here) --dataset_type mocap --batch_size 32 --motion_length 60 --coarse_grained --lambda_kld 0.001 --eval_every 2000 --plot_every 50 --print_every 20 --save_every 2000 --save_latest 50 --time_counter --gpu_id 0 --iters 50000 --n_resblk 3 --dim_vq_latent 1024 --codebook_size 1024 --lr_scheduler_every_e 100 --n_down 2 
```

### Generate Tokens
#### HumanAct12 Dataset
```
python tokenize_script.py --name xxxx(place your experiment name here) --dataset_type humanact12 --batch_size 1 --motion_length 60 --coarse_grained --lambda_kld 0.001 --eval_every 2000 --plot_every 50 --print_every 20 --save_every 2000 --save_latest 50 --time_counter --gpu_id 0 --iters 50000 --n_resblk 3 --dim_vq_latent 1024 --codebook_size 1024 --lr_scheduler_every_e 100 --n_down 2 
```

#### NTU-RGBD Dataset
```
python tokenize_script.py --name xxxx(place your experiment name here) --dataset_type ntu_rgbd_vibe --batch_size 1 --motion_length 60 --coarse_grained --lambda_kld 0.001 --eval_every 2000 --plot_every 50 --print_every 20 --save_every 2000 --save_latest 50 --time_counter --gpu_id 0 --iters 50000 --n_resblk 3 --dim_vq_latent 1024 --codebook_size 1024 --lr_scheduler_every_e 100 --n_down 2  
```

#### CMU Mocap Dataset
```
python tokenize_script.py --name xxxx(place your experiment name here) --dataset_type mocap --batch_size 1 --motion_length 60 --coarse_grained --lambda_kld 0.001 --eval_every 2000 --plot_every 50 --print_every 20 --save_every 2000 --save_latest 50 --time_counter --gpu_id 0 --iters 50000 --n_resblk 3 --dim_vq_latent 1024 --codebook_size 1024 --lr_scheduler_every_e 100 --n_down 2  
```

### Train Transformer
#### HumanAct12 Dataset
```
python train_action2motion.py --name xxxx(place your experiment name here) --dataset_type humanact12 --batch_size 32 --motion_length 60 --coarse_grained --lambda_kld 0.001 --eval_every 2000 --plot_every 50 --print_every 20 --save_every 2000 --save_latest 50 --time_counter --gpu_id 0 --iters 50000 --lr_scheduler_every_e 100 
```

#### Download NTU-RGBD Dataset
```
python train_action2motion.py --name xxxx(place your experiment name here) --dataset_type ntu_rgbd_vibe --batch_size 32 --motion_length 60 --coarse_grained --lambda_kld 0.001 --eval_every 2000 --plot_every 50 --print_every 20 --save_every 2000 --save_latest 50 --time_counter --gpu_id 0 --iters 50000 --lr_scheduler_every_e 100 
```

#### Download CMU Mocap Dataset
```
python train_action2motion.py --name xxxx(place your experiment name here) --dataset_type mocap --batch_size 32 --motion_length 60 --coarse_grained --lambda_kld 0.001 --eval_every 2000 --plot_every 50 --print_every 20 --save_every 2000 --save_latest 50 --time_counter --gpu_id 0 --iters 50000 --lr_scheduler_every_e 100  
```


### Evaluation
You can directly use our pre-trained model for evaluation. Just replace the corresponding experiment name.

#### HumanAct12 Dataset
```
python evaluate_a2m_transformer.py --name xxxx(place your experiment name here) --dataset_type humanact12 --batch_size 1 --motion_length 60 --coarse_grained --n_resblk 3 --dim_vq_latent 1024 --codebook_size 1024 --n_down 2 --n_dec_layers 4 --n_enc_layers 4 --n_head 8 --repeat_times 10 --sample 
```

#### Download NTU-RGBD Dataset
```
python evaluate_a2m_transformer.py --name xxxx(place your experiment name here) --dataset_type ntu_rgbd_vibe --batch_size 1 --motion_length 60 --coarse_grained --n_resblk 3 --dim_vq_latent 1024 --codebook_size 1024 --n_down 2 --n_dec_layers 4 --n_enc_layers 4 --n_head 8 --repeat_times 10 --sample
```

#### Download CMU Mocap Dataset
```
python evaluate_a2m_transformer.py --name xxxx(place your experiment name here) --dataset_type mocap --batch_size 1 --motion_length 60 --coarse_grained --n_resblk 3 --dim_vq_latent 1024 --codebook_size 1024 --n_down 2 --n_dec_layers 4 --n_enc_layers 4 --n_head 8 --repeat_times 10 --sample
```