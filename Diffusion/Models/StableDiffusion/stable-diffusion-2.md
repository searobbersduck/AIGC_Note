## [Stable Diffusion v2-base Model Card](https://huggingface.co/stabilityai/stable-diffusion-2-base)

* **Training**
  * **Training Data**
* **Training Procedure**
  * Images are encoded through an encoder, which turns images into latent representations. The autoencoder uses a relative downsampling factor of 8 and maps images of shape H x W x 3 to latents of shape H/f x W/f x 4
  * Text prompts are encoded through the OpenCLIP-ViT/H text-encoder.
  * The output of the text encoder is fed into the UNet backbone of the latent diffusion model via cross-attention.
  * The loss is a reconstruction objective between the noise that was added to the latent and the prediction made by the UNet. We also use the so-called v-objective, see https://arxiv.org/abs/2202.00512.
* **checkpoints**
  * <font color='red'>512-base-ema.ckpt</font>: <font color='gree'>550k steps at resolution 256x256</font> on a subset of LAION-5B filtered for explicit pornographic material, using the LAION-NSFW classifier with punsafe=0.1 and an aesthetic score >= 4.5. <font color='gree'>850k steps at resolution 512x512 on the same dataset</font> with resolution >= 512x512.
  *  <font color='red'>768-v-ema.ckpt</font>: <font color='gree'>Resumed from 512-base-ema.ckpt</font> and <font color='gree'>trained for 150k steps using a v-objective on the same dataset</font>. Resumed for <font color='gree'>another 140k steps on a 768x768 subset of our dataset</font>.
  *  <font color='red'>512-depth-ema.ckpt</font>: Resumed from 512-base-ema.ckpt and <font color='gree'>finetuned for 200k steps</font>. Added an <font color='gree'>extra input channel to process the (relative) depth prediction produced by MiDaS (dpt_hybrid)</font> which is used as an additional conditioning. <font color='gree'>The additional input channels of the U-Net which process this extra information were zero-initialized</font>.
  *  **<font color='red'>512-inpainting-ema.ckpt</font>**: Resumed from 512-base-ema.ckpt and trained for <font color='gree'>another 200k steps</font>. Follows the <font color='gree'>mask-generation strategy presented in LAMA</font> which, in combination with the latent VAE representations of the masked image, are used as an additional conditioning. <font color='gree'>The additional input channels of the U-Net which process this extra information were zero-initialized</font>. The same strategy was used to train the 1.5-inpainting checkpoint.
  *  <font color='red'>x4-upscaling-ema.ckpt</font>: Trained for 1.25M steps on a 10M subset of LAION containing images >2048x2048. The model was trained on crops of size 512x512 and is a text-guided latent upscaling diffusion model. In addition to the textual input, it receives a noise_level as an input parameter, which can be used to add noise to the low-resolution input according to a predefined diffusion schedule.

## [Stable Diffusion v2 Model Card](https://huggingface.co/stabilityai/stable-diffusion-2)

## TODO
- [ ] [MiDas](https://github.com/isl-org/MiDaS)
- [ ] [LaMa: Resolution-robust Large Mask Inpainting with Fourier Convolutions](https://github.com/advimman/lama)
