## RMAFF-PSN: A Residual Multi-Scale Attention Feature FusionPhotometric Stereo Network

**Author:** 
Kai Luo, Yakun Ju, Lin Qi, Kaixuan Wang and Junyu Dong

## Getting Started

#### Train RMAFF-PSN
```shell
# Train RMAFF-PSN on both synthetic datasets using 32 images-light pairs
CUDA_VISIBLE_DEVICES=0 python main.py --concat_data --in_img_num 32 --normalize --item normalize
```

#### Test RMAFF-PSN on the DiLiGenT main dataset
```shell
CUDA_VISIBLE_DEVICES=0 python eval/run_model.py --retrain data/Training/normalize/train/check_26.pth.tar --in_img_num 96 --normalize --train_img_num 32
```

## Results on the DiLiGenT benchmark dataset:

We have provided the estimated surface normals and error maps on the DiLiGenT benchmark dataset, in document ``Results''

## Acknowledgement:
Our code is partially based on: https://github.com/guanyingc/PS-FCN, https://github.com/Kelvin-Ju/MF-PSN.<br>
We are grateful for the help of Guanying Chen(https://guanyingc.github.io/),<br> Satoshi Ikehata(https://satoshi-ikehata.github.io/),<br> and Yakun Ju(https://kelvin-ju.github.io/yakunju/) in academic research.

## Citation

If you find our work useful, please consider citing our paper:

```bibtex
@inproceedings{luo2023rmaff,
  title={RMAFF-PSN: A Residual Multi-Scale Attention Feature Fusion Photometric Stereo Network},
  author={Luo, Kai and Ju, Yakun and Qi, Lin and Wang, Kaixuan and Dong, Junyu},
  booktitle={Photonics},
  volume={10},
  number={5},
  pages={548},
  year={2023},
  organization={MDPI}
}
```
