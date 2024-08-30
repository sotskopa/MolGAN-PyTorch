# MolGAN PyTorch

PyTorch Implemenentation of [MolGAN: An implicit generative model for small molecular graphs](https://arxiv.org/abs/1805.11973). The code is based on:
- https://github.com/nicola-decao/MolGAN
- https://github.com/yongqyu/MolGAN-pytorch
- https://github.com/MiloszGrabski/MolGAN-TF2-

## Installation
### Dependencies
- `numpy`
- `rdkit`
- `torch`
- `tqdm`

Install the dependencies using:
```sh
pip install -r requirements.txt
```
Then, the QM9 dataset can be downloaded using:
```sh
./data/download_dataset.sh
```
Finally, preprocess the dataset into the correct format using:
```sh
./data/preprocess_dataset.sh
```

## Training
Training is parameterized by a number of command line arguments. To display these options, use:
```sh
python train.py -h # Show training options
```
Alternatively, run with default options using:
```sh
python train.py # Use default arguments
```

## Citation
If you find this code useful in your research, please consider citing the original paper by Nicolas De Cao and Thomas Kipf:
```
@article{de2018molgan,
  title={{MolGAN: An implicit generative model for small
  molecular graphs}},
  author={De Cao, Nicola and Kipf, Thomas},
  journal={ICML 2018 workshop on Theoretical Foundations 
  and Applications of Deep Generative Models},
  year={2018}
}
```
