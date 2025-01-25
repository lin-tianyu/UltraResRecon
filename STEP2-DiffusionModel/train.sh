export SD_MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export FT_VAE_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export TRAIN_DIR="/mnt/data/tlin67/Dataset_raw/FELIXtemp/FELIXh5" # Temporary FELIX path!!!!

accelerate launch --mixed_precision="no" train_unconditional.py \
  --pretrained_model_name_or_path=$SD_MODEL_NAME \
  --finetuned_vae_name_or_path=$FT_VAE_NAME \
  --train_data_dir=$TRAIN_DIR \
  --resolution=512 \
  --train_batch_size=8 \
  --dataloader_num_workers=2 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=100000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --report_to=wandb \
  --validation_steps=1000 \
  --validation_images ../../../Dataset_raw/FELIXtemp/FELIXh5/BDMAP_A0000001/ct.h5 ../../../Dataset_raw/FELIXtemp/FELIXh5/BDMAP_A0000001/ct.h5 \
  --output_dir="uncond-sd-training-slurm"