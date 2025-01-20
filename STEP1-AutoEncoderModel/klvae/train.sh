
accelerate launch train_klvae.py \
  --train_data_dir="/mnt/realccvl15/zzhou82/data/AbdomenAtlasPro" \
  --validation_images ../../../../Dataset_raw/reconFELIX/ct/BDMAP_V0000001.nii.gz ../../../../Dataset_raw/reconFELIX/ct/BDMAP_V0000002.nii.gz \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --report_to="wandb" \
  --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
  --max_train_steps=100000 \
  --vae_loss="l1" \
  --learning_rate=1e-6 \
  --use_ema