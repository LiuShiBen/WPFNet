## Wavelet-pixel domain progressive fusion network for underwater image enhancement (WPFNet)

The *official* repository for  [Wavelet-pixel domain progressive fusion network for underwater image enhancement](https://www.sciencedirect.com/science/article/abs/pii/S095070512400683X).

```
Our work proposes an “Wavelet-pixel domain progressive fusion network for underwater image enhancement” (WPFNet). The proposed WPFNet progressively merge frequency features capturing fine-grained details and spatial features with rich color and illumination information.
```

![](.\doc\WPFNet.png)

## Getting Started

### Requirements

- Python 3.8
- Pytorch 1.7.0

### Prepare Datasets

```
data
├── UIEB
│   └──trainA
│   └──trainB
├── Underwater-Dark
│   └──trainA
│   └──trainB
```

### Training

```
python Underwater_train.py --data_dir=./your dataset path
```

## Contact

If you have any questions, please contact Shiben Liu at [liushiben@sia.cn].