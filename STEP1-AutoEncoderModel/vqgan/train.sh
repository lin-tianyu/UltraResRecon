accelerate launch train_vqgan.py \
  --train_data_dir="/mnt/realccvl15/zzhou82/data/AbdomenAtlasPro" \
  --validation_images ../../../../Dataset_raw/reconFELIX/ct/BDMAP_V0000001.nii.gz ../../../../Dataset_raw/reconFELIX/ct/BDMAP_V0000002.nii.gz \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --report_to="wandb" \
  --model_config_name_or_path="models/vq-f8/config.json" \
  --max_train_steps=100000 \
  --discriminator_iter_start=10000 \
  --vae_loss="l1"