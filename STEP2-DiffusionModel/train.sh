export SD_MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export FT_VAE_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export TRAIN_DIR="/mnt/realccvl15/zzhou82/data/AbdomenAtlasPro"

accelerate launch --mixed_precision="no" train_unconditional.py \
  --pretrained_model_name_or_path=$SD_MODEL_NAME \
  --finetuned_vae_name_or_path=$FT_VAE_NAME \
  --train_data_dir=$TRAIN_DIR \
  --resolution=512 \
  --center_crop \
  --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="sd-naruto-model"