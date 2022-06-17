# 基于PSFRGAN的超声重建图像深度学习算法优化

> **PSFRGAN 原文及作者**
>
> [Progressive Semantic-Aware Style Transformation for Blind Face Restoration](https://arxiv.org/abs/2009.08709)  
> [Chaofeng Chen](https://chaofengc.github.io), [Xiaoming Li](https://csxmli2016.github.io/), [Lingbo Yang](https://lotayou.github.io), [Xianhui Lin](https://dblp.org/pid/147/7708.html), [Lei Zhang](https://www4.comp.polyu.edu.hk/~cslzhang/), [Kwan-Yee K. Wong](https://i.cs.hku.hk/~kykwong/)
>
> 查看原始项目说明，参见 `Original-README.md` 。

## 环境要求

- CUDA 10.1

- Python 3.7， 可以使用以下命令来安装依赖的库：

  `pip install -r requirements.txt`

- 下载预训练模型：

  - [BaiduNetDisk](todo)，提取码：`xxxx`
  - 将所有预训练模型放置在路径 `./pretrain_models` 下。

## 测试运行

所有输入图片应为 `512 * 512` 像素的rgb图片。

使用以下命令来测试运行：

```
python test_enhance_single_unalign.py --test_img_path ./test_dir/147.png --results_dir test_result --gpus 1
```

- `--test_img_path` 为输入图片的位置。
- `--result_dir` 为输出结果的路径。
  - `test_results/LQ_faces` 为需要处理的低通图像。
  - `test_results/ParseMaps` 为低通图像的语义图。
  - `test_results/HQ` 为处理结果。
- `-gpus` 为需要使用的GPU，`<=0` 表示使用CPU来运行。

## 训练模型

### 准备训练数据

- 将数据集中的数据放在`../datasets/reconstructed_ultrasound_images/imgs1024` 路径下，所有训练数据的长宽比应为**1：1**。

- 运行以下命令，生成对应的语义图：

  ```
  python generate_masks.py --test_img_path  ../datasets/reconstructed_ultrasound_images/imgs1024
  --results_dir
  ../datasets/reconstructed_ultrasound_images/masks512
  ```

### 训练PSFRGAN

使用以下命令来训练PSFRGAN：

```
python train.py --gpus 1 --model enhance --name PSFRGAN_v001 --g_lr 0.0001 --d_lr 0.0004 --beta1 0.5 --gan_mode hinge --lambda_pix 10 --lambda_fm 10 --lambda_ss 1000 --Dinput_nc 22 --D_num 3 --n_layers_D 4 --batch_size 2 --dataroot ../datasets/reconstructed_ultrasound_images --visual_freq 100 --print_freq 10 --dataset ffhq
```

- 对于不同的实验，需要更改`--name` 选项。相同名字的实验结果会被覆盖。
- `--gpus` 表示训练需要使用的GPU数量。
