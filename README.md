- ## WPFNet

  ```
  Our work proposes an “Wavelet-pixel domain progressive fusion network for underwater image enhancement” (WPFNet). The proposed WPFNet progressively merge frequency features capturing fine-grained details and spatial features with rich color and illumination information.
  ```

  ![](.\doc\WPFNet.png)

  ## Getting Started

  ### Requirements

  - Python 3.8
  - Pytorch 1.7.0

  ### Evaluation

  ```
  python Underwater_test.py --data_dir=./your dataset path
  --sample_dir=./your results output path
  --model_path=./checkpoint/generator_600.pth
  ```

  ## Contact
  
  If you have any questions, please contact Shiben Liu at liushiben@sia.cn.

  ## Contact
@article{liu2024wavelet,
  title={Wavelet-pixel domain progressive fusion network for underwater image enhancement},
  author={Liu, Shiben and Fan, Huijie and Wang, Qiang and Han, Zhi and Guan, Yu and Tang, Yandong},
  journal={Knowledge-Based Systems},
  pages={112049},
  year={2024},
  publisher={Elsevier}
}
