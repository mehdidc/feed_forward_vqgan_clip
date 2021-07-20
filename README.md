Feed forward VQGAN-CLIP model, where the goal is to eliminate the need for optimizing the latent
space of VQGAN for each input prompt. This  is done by training a model that takes as input
a dataset of text prompts, and returns as an output the VQGAN latent space, which is then
transformed into an RGB image. The loss function is minimizing the distance between
the CLIP generated image features and the CLIP input text features. Additionally,
a diversity loss can be used to make increase the diversity of the generated
images given the same prompt.

# How to install?

### Download the 16384 Dimension Imagenet VQGAN (f=16)

Links:
- http://mirror.io.community/blob/vqgan/vqgan_imagenet_f16_16384.ckpt
- http://mirror.io.community/blob/vqgan/vqgan_imagenet_f16_16384.yaml

### Install dependencies. 

#### conda
```bash
conda create -n ff_vqgan_clip_env python=3.8
conda activate ff_vqgan_clip_env
# Install pytorch/torchvision - See https://pytorch.org/get-started/locally/ for more info.
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
pip install -r requirements.txt
```
#### pip/venv
```bash
conda deactivate # Make sure to use your global python3
python3 -m pip install venv
python3 -m venv ./ff_vqgan_clip_venv
source ./ff_vqgan_clip_venv/bin/activate
$ (ff_vqgan_clip_venv) python -m pip install -r requirementst.txt
```

# How to use?

- Example usage: `python main.py train configs/example.yaml`

# Acknowledgment

- The training code is heavily based on the VQGAN-CLIP notebook <https://colab.research.google.com/drive/1ZAus_gn2RhTZWzOWUpPERNC0Q8OhZRTZ>, thanks
to all the authors who contributed to the notebook ([@crowsonkb](https://github.com/crowsonkb), [@advadnoun](https://twitter.com/advadnoun), [@Eleiber](https://github.com/Eleiber), [@Crimeacs](https://twitter.com/earthml1), @Abulafia)
- Thanks to [@lucidrains](https://github.com/lucidrains), the MLP mixer model (`mlp_mixer_pytorch.py`)  is from <https://github.com/lucidrains/mlp-mixer-pytorch>.
- Thanks to Taming Transformers authors <https://github.com/CompVis/taming-transformers>, the code uses VQGAN pre-trained model and
VGG16 feature space perceptual loss <https://github.com/CompVis/taming-transformers/blob/master/taming/modules/losses/lpips.py>
- Thanks to [@afiaka87](https://github.com/afiaka87), who provided the blog captions dataset for experimentation.
