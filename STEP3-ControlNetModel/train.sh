export VAE_MODEL_DIR="../STEP1-AutoEncoderModel/klvae/klvae-output-KL-weight-1e-6/checkpoint-80000"
export UNET_MODEL_DIR="../STEP2-DiffusionModel/uncond-sd-training-1k/checkpoint-20000"
export OUTPUT_DIR="path to save model"

accelerate launch train_controlnet.py \
    --finetuned_vae_name_or_path=$UNET_MODEL_DIR \
    --pretrained_model_name_or_path=$MODEL_DIR \
    --output_dir=$OUTPUT_DIR \
    --dataset_name=fusing/fill50k \
    --resolution=512 \
    --learning_rate=1e-5 \
    --validation_image "./conditioning_image_1.png" "./conditioning_image_2.png" \
    --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4