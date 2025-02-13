import diffusers
from diffusers import DiffusionPipeline
import torch

from PIL import Image
import numpy as np

from diffusers import AutoencoderKL
from diffusers import VQModel
from diffusers import UNet2DModel

from diffusers import LDMPipeline, StableDiffusionPipeline
import nibabel as nib
import os

TRAINED = True
# # Initialize AutoencoderKL for LDM-VQ-8 (config file aligned with stable diffusion paper)
# # >>>>>>>>>>>>>>>>>>>>>> Famous Stable Diffusion Huggingface Repo >>>>>>>>>>
# # 1. stabilityai/stable-diffusion-2-1               (trained on 768x768)
# # 2. stabilityai/stable-diffusion-2                 (trained on 768x768)
# # 3. stabilityai/stable-diffusion-2-base            (trained on 512x512)
# # 4. CompVis/stable-diffusion-v1-4                  (trained on 512x512)
# # 5. stable-diffusion-v1-5/stable-diffusion-v1-5    (trained on 512x512)
# # <<<<<<<<<<<<<<<<<<<<<< They are all using the same AutoencoderKL >>>>>>>>>
if not TRAINED:                     # A. initialize the model from scratch
    print("create model from scratch")
    config = AutoencoderKL.load_config("STEP1-AutoEncoderModel/klvae/models/kl-f8/config.json")
    kl_model_8 = AutoencoderKL.from_config(config)
else:                               # B. load pre-trained model
    print("load pre-trained model")
    kl_model_8 = AutoencoderKL.from_pretrained("STEP1-AutoEncoderModel/klvae/vae_kl6_lr4_std/checkpoint-119000", subfolder="vae")

print(kl_model_8.config.scaling_factor)                             # scaling_factor doesn't matter for VQmodel
data_dir = "/mnt/realccvl15/zzhou82/data/AbdomenAtlasPro/"
bdmap_id = "BDMAP_A0000001"
slice_id = 150
nii = nib.load(os.path.join(data_dir, bdmap_id, "ct.nii.gz"))
nii_slice = nii.dataobj[:, :, slice_id:slice_id+3].copy()
nii_slice[nii_slice > 1000] = 1000
nii_slice[nii_slice < -1000] = -1000
nii_slice = (nii_slice + 1000) / 2000
input_image = nii_slice * 255
image = np.asarray(input_image)/255 * 2 - 1
# image = np.asarray(Image.open("example.png").convert("RGB"))/255 * 2 - 1
image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
print(image.shape, image.min(), image.max())
encoding = kl_model_8.encode(image)
posterior = encoding.latent_dist
latents = posterior.mode()
kl_loss = posterior.kl().mean()
reconstructed = kl_model_8.decode(latents).sample
print("AutoencoderKL latents:", latents.shape, latents.dtype, latents.min(), latents.max())
print("AutoencoderKL output:", reconstructed.shape, reconstructed.min(), reconstructed.max())        # two outputs: (recon_output, commit_loss)
print("AutoencoderKL downsampling rate:", f"{reconstructed.shape[2] // latents.shape[2]}", "\n")
print("AutoencoderKL KL-loss:", kl_loss)
print(latents.std(), 1/latents.std())

# tensor = torch.tensor(reconstructed)
# tensor = torch.clamp(tensor, 0.0, 1.0) * 255
# out_image = Image.fromarray(tensor[0].detach().numpy().transpose(1, 2, 0).astype(np.uint8))
# out_image.save("example.png")




# pipeline = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", vae=kl_model_8)
# image = pipeline(prompt="dog", num_inference_steps=20, output_type="latent").images[0]
# print(image.detach().numpy().shape, image.detach().numpy().min(), image.detach().numpy().max())
# image_w_factor = pipeline.vae.decode(image[None] / pipeline.vae.config.scaling_factor).sample[0]
# image_wo_factor = pipeline.vae.decode(image[None] / 1).sample[0]
# print(image_w_factor.detach().numpy().shape, image_w_factor.detach().numpy().min(), image_w_factor.detach().numpy().max())
# print(image_wo_factor.detach().numpy().shape, image_wo_factor.detach().numpy().min(), image_wo_factor.detach().numpy().max())
# image_w_factor = ((image_w_factor.detach().numpy() / 2 + 0.5).clip(0, 1) * 255).astype(np.uint8).transpose(1, 2, 0)
# image_wo_factor = ((image_wo_factor.detach().numpy() / 2 + 0.5).clip(0, 1) * 255).astype(np.uint8).transpose(1, 2, 0)

# image_w_factor = Image.fromarray(image_w_factor)
# image_w_factor.save("image_w_factor_rescale.png")
# image_wo_factor = Image.fromarray(image_wo_factor)
# image_wo_factor.save("image_wo_factor_rescale.png")
# # image.save("example.png")


# # Initialize VQModel for LDM-VQ-8 (config file aligned with stable diffusion paper)
# if not TRAINED:                     # A. initialize the model from scratch
#     config = VQModel.load_config("STEP1-AutoEncoderModel/models/vq-f8/config.json")
#     vq_model_8 = VQModel.from_config(config)
# else:                               # B. load pre-trained model
#     vq_model_8 = VQModel.from_pretrained("STEP1-AutoEncoderModel/vqgan-output/checkpoint-9000", subfolder="vqmodel")

# print(vq_model_8.config.scaling_factor)                             # scaling_factor doesn't matter for VQmodel
# image = np.asarray(Image.open("/ccvl/net/ccvl15/tlin67/3DReconstruction/Tianyu/STEP1-AutoEncoderModel/images/dog.jpg").convert("RGB"))
# image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
# print(image.shape)
# latents = vq_model_8.encode(image).latents   # (b, c, h, w) -> (b, 4, h//f, w//f), f==8
# print("VQModel latents:", latents.shape, latents.dtype)
# out = vq_model_8.decode(latents)                                    # (b, 4, h//f, w//f) -> (b, c, h, w), f==8
# print("VQModel output:", out[0].shape, out[0].dtype, out[1])        # two outputs: (recon_output, commit_loss)
# print("VQModel downsampling rate:", f"{out[0].shape[2] // latents.shape[2]}", "\n")
# print(out[0].min(), out[0].max())

# tensor = torch.tensor(out[0])
# tensor = torch.clamp(tensor, 0.0, 1.0) * 255
# out_image = Image.fromarray(tensor[0].detach().numpy().transpose(1, 2, 0).astype(np.uint8))
# out_image.save("example.png")







# # Initialize UNet2DModel for LDMs
# unet_ldm = UNet2DModel(
#     sample_size=32,  # Latent space resolution (e.g., 256/8 for LDM-KL-8)
#     in_channels=4,  # Latent channels from the autoencoder
#     out_channels=4,
#     layers_per_block=2,
#     block_out_channels=(128, 256, 512),  # Example feature map sizes
#     down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
#     up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D")
# )
# print(unet_ldm)







# pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)
# pipeline.to("cuda")
# image = pipeline("An image of a squirrel in Picasso style").images[0]

# image.save("example.png")