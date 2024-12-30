# Improved Residual Vision Transformer for CT to MRI Translation

This repository contains the official implementation of the paper **Improved Residual Vision Transformer for CT to MRI Translation**, presented at the *IEEE International Conference on Transdisciplinary AI 2023*.

---

## Citation

If you use this code or find it helpful in your work, please cite our paper:

```
@inproceedings{kundu2023improved,
  title={Improved Residual Vision Transformer for CT to MRI Translation},
  author={Kundu, Souraja and Iwahori, Yuji and Bhuyan, Manas Kamal and Bhatt, Manish and Ouchi, Akira and Shimizu, Yasuhiro},
  booktitle={2023 Fifth International Conference on Transdisciplinary AI (TransAI)},
  pages={58--65},
  year={2023},
  organization={IEEE}
}
```

This repository includes updates to the codebase of the work **ResViT**, published in *IEEE Transactions on Medical Imaging (TMI)*. Please cite the ResViT paper if you are using this code repository

```
@article{dalmaz2022resvit,
  title={ResViT: residual vision transformers for multimodal medical image synthesis},
  author={Dalmaz, Onat and Yurt, Mahmut and {\c{C}}ukur, Tolga},
  journal={IEEE Transactions on Medical Imaging},
  volume={41},
  number={10},
  pages={2598--2614},
  year={2022},
  publisher={IEEE}
}
```

---

## Code Updates

This repository builds upon the ResViT repository. The following files have been updated or modified from the original ResViT implementation:

- `base_model.py`
- `__init__.py`
- `networks.py`
- `residual_transformers.py`
- `resvit_one.py`
- `unaligned_dataset.py`

The code structure follows the same organization as in the ResViT repository to ensure consistency and ease of understanding.

In the above files, if two file names are `X.py` and `X (copy).py`, remember that `X.py` contains the modified code, while `X (copy).py` contains the original ResViT code.

---


## Installation

1. Clone this repository:

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Data Setup
1. Prepare your dataset and organize it under the `Datasets` folder with the following subfolders:
   - `trainA`, `trainB`
   - `valA`, `valB`
   - `testA`, `testB`

2. Create additional folders with the same names and structure as in the ResViT repository:
   - `checkpoints`
   - `data`
   - `model`
   - `models`
   - `options`
   - `results`
   - `util`

3. Replace specific files in the `models` and `data` folders with the updated files listed above.

### Training Steps

#### Step 1: Pretraining
Run the following command to pretrain the model:
```bash
python3 train.py --dataroot Datasets/MRI_CT \
  --name CT_MRI_pre_trained --gpu_ids 0 --model resvit_one \
  --which_model_netG res_cnn --which_direction AtoB --lambda_A 100 \
  --dataset_mode unaligned --norm batch --pool_size 0 --output_nc 1 \
  --input_nc 1 --loadSize 256 --fineSize 256 --niter 50 \
  --niter_decay 50 --save_epoch_freq 5 --checkpoints_dir checkpoints/ \
  --display_id 0 --lr 0.0002
```

#### Step 2: Training the Improved ResViT Model
Run the following command to train the improved model:
```bash
python3 train.py --dataroot Datasets/MRI_CT \
  --name CT_MRI_resvit --gpu_ids 0 --model resvit_one \
  --which_model_netG resvit --which_direction AtoB --lambda_A 100 \
  --dataset_mode unaligned --norm batch --pool_size 0 --output_nc 1 \
  --input_nc 1 --loadSize 256 --fineSize 256 --niter 25 \
  --niter_decay 25 --save_epoch_freq 5 --checkpoints_dir checkpoints/ \
  --display_id 0 --pre_trained_transformer 1 --pre_trained_resnet 1 \
  --pre_trained_path checkpoints/CT_MRI_pre_trained/latest_net_G.pth \
  --lr 0.001
```

### Testing Steps

#### Testing the Pretrained Model
Run the following command to test only the pretrained model:
```bash
python3 test.py --dataroot Datasets/MRI_CT \
  --name CT_MRI_pre_trained --gpu_ids 0 --model resvit_one \
  --which_model_netG res_cnn --dataset_mode unaligned --norm batch \
  --phase test --output_nc 1 --input_nc 1 --how_many 125 \
  --serial_batches --fineSize 256 --loadSize 256 --results_dir results/ \
  --checkpoints_dir checkpoints/ --which_epoch latest \
  --pre_trained_path checkpoints/CT_MRI_pre_trained/latest_net_G.pth
```

#### Testing the Trained Improved ResViT Model
Run the following command to test the improved model:
```bash
python3 test.py --dataroot Datasets/MRI_CT \
  --name CT_MRI_resvit --gpu_ids 0 --model resvit_one \
  --which_model_netG resvit --dataset_mode unaligned --norm batch \
  --phase test --output_nc 1 --input_nc 1 --how_many 125 \
  --serial_batches --fineSize 256 --loadSize 256 --results_dir results/ \
  --checkpoints_dir checkpoints/ --which_epoch latest \
  --pre_trained_path checkpoints/CT_MRI_pre_trained/latest_net_G.pth
```

---

## License

This repository is licensed under the [MIT License](LICENSE).

