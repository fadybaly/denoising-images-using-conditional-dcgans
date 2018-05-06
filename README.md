# denoising-images-using-conditional-dcgans

In this project we explored the capabilities of the pix2pix model to filter different types of noises (gaussian, pixelated images, salt and pepper, and speckle). the model performed above expectations when trained to denoise the image per noise.
Future work includes tweaking the model to learn to filter all the earlier mentioned noises using the same trained model (not to train a specific model per noise as before).

The model is trained on the Labeled Faces in the Wild dataset using NVIDIA GTX 1050TI with 4GB RAM.

The model is forked from https://github.com/Eyyub/tensorflow-pix2pix

all future modifications are from the current authors of this repository
