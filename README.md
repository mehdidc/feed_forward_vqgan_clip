Feed forward VQGAN-CLIP model, where the goal is to eliminate the need for optimizing the latent
space of VQGAN for each input prompt. This  is done by training a model that takes as input
a dataset of text prompts, and returns as an output the VQGAN latent space, which is then
transformed into an RGB image. The loss function is minimizing the distance between
the CLIP generated image features and the CLIP input text features.

# How to install?

- `pip install -r requirements.txt`

# How to use?

- Example usage: `python main.train configs/example.yaml`

# Acknowledgment

- The training code is heavily based on the VQGAN-CLIP notebook <https://colab.research.google.com/drive/1ZAus_gn2RhTZWzOWUpPERNC0Q8OhZRTZ>, thanks
to all the authors who contributed to the notebook (@crowsonkb, @advadnoun, @Eleiber, @Crimeacs, @Abulafia)
- Thanks to @lucidrains, the MLP mixer model (`mlp_mixer_pytorch.py`)  is from <https://github.com/lucidrains/mlp-mixer-pytorch>.
- Thanks to Taming Transformers authors <https://github.com/CompVis/taming-transformers>, the code uses VQGAN pre-trained model and
VGG16 feature space perceptual loss <https://github.com/CompVis/taming-transformers/blob/master/taming/modules/losses/lpips.py>
- Thanks to @afiaka87, who provided the blog captions dataset for experimentation.
