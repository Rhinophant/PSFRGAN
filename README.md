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



