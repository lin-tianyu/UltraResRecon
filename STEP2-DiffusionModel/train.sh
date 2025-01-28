export SD_MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export FT_VAE_NAME="../STEP1-AutoEncoderModel/klvae/klvae-output-KL-weight-1e-6/checkpoint-80000"
export TRAIN_DATA_DIR="/mnt/T9/AbdomenAtlas/image_mask_h5" # Temporary FELIX path!!!!


# accelerate launch --mixed_precision="no" train_unconditional.py \
#   --pretrained_model_name_or_path=$SD_MODEL_NAME \
#   --finetuned_vae_name_or_path=$FT_VAE_NAME \
#   --train_data_dir=$TRAIN_DATA_DIR \
#   --resolution=512 \
#   --train_batch_size=16 \
#   --dataloader_num_workers=16 \
#   --gradient_accumulation_steps=1 \
#   --max_train_steps=1_000_000 \
#   --learning_rate=1e-04 \
#   --max_grad_norm=1 \
#   --lr_scheduler="constant" \
#   --report_to=wandb \
#   --validation_steps=1000 \
#   --validation_images ../../../Dataset_raw/FELIXtemp/FELIXh5/BDMAP_A0000001/ct.h5 ../../../Dataset_raw/FELIXtemp/FELIXh5/BDMAP_V0000001/ct.h5 \
#   --output_dir="uncond-sd-training-1k-attn1st"


accelerate launch --mixed_precision="no"  train_text_to_image.py \
  --pretrained_model_name_or_path=$SD_MODEL_NAME \
  --finetuned_vae_name_or_path=$FT_VAE_NAME \
  --train_data_dir=$TRAIN_DATA_DIR \
  --resolution=512 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=4 \
  --dataloader_num_workers=2 \
  --gradient_checkpointing \
  --max_train_steps=1_000_000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --report_to=wandb \
  --validation_steps=1000 \
  --checkpointing_steps=1000 \
  --validation_prompt 'An Arterial CT slice.' 'A Portal-venous CT slice.' \
  --output_dir="sd-training"