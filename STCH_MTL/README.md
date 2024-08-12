# STCH for MTL

The code of STCH_MTL is developed using the ``LibMTL`` [library](https://github.com/median-research-group/LibMTL). The implementation of STCH can be found in ``LibMTL/weighting/STCH.py``, and the trained model for each problem can be found in the corresponding folder under ``examples``.

If you find this code useful for your research, please also cite the MTL library:

```
@article{lin2023libmtl,
  title={{LibMTL}: A {P}ython Library for Multi-Task Learning},
  author={Baijiong Lin and Yu Zhang},
  journal={Journal of Machine Learning Research},
  volume={24},
  number={209},
  pages={1--7},
  year={2023}
}
```


# LibMTL

[![Documentation Status](https://readthedocs.org/projects/libmtl/badge/?version=latest)](https://libmtl.readthedocs.io/en/latest/?badge=latest) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/median-research-group/LibMTL/blob/main/LICENSE) [![PyPI version](https://badge.fury.io/py/LibMTL.svg)](https://badge.fury.io/py/LibMTL) [![Supported Python versions](https://img.shields.io/pypi/pyversions/LibMTL.svg?logo=python&logoColor=FFE873)](https://github.com/median-research-group/LibMTL) [![CodeFactor](https://www.codefactor.io/repository/github/median-research-group/libmtl/badge/main)](https://www.codefactor.io/repository/github/median-research-group/libmtl/overview/main) [![paper](https://img.shields.io/badge/Accepted%20by-JMLR-b31b1b.svg)](https://www.jmlr.org/papers/v24/22-0347.html) [![coverage](./tests/coverage.svg)](https://github.com/median-research-group/LibMTL) [![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fmedian-research-group%2FLibMTL&count_bg=%23763DC8&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=visitors&edge_flat=false)](https://hits.seeyoufarm.com) [![Made With Love](https://img.shields.io/badge/Made%20With-Love-orange.svg)](https://github.com/median-research-group/LibMTL) 

``LibMTL`` is an open-source library built on [PyTorch](https://pytorch.org/) for Multi-Task Learning (MTL). See the [latest documentation](https://libmtl.readthedocs.io/en/latest/) for detailed introductions and API instructions.

:star: Star us on GitHub — it motivates us a lot!

## News

- **[Feb 08 2024]** Added support for [DB-MTL](https://arxiv.org/abs/2308.12029).
- **[Aug 16 2023]**: Added support for [MoCo](https://openreview.net/forum?id=dLAYGdKTi2) (ICLR 2023). Many thanks to the author's help [@heshandevaka](https://github.com/heshandevaka).
- **[Jul 11 2023]** Paper got accepted to [JMLR](https://jmlr.org/papers/v24/22-0347.html).
- **[Jun 19 2023]** Added support for [Aligned-MTL](https://openaccess.thecvf.com/content/CVPR2023/html/Senushkin_Independent_Component_Alignment_for_Multi-Task_Learning_CVPR_2023_paper.html) (CVPR 2023).
- **[Mar 10 2023]**: Added [QM9](https://github.com/median-research-group/LibMTL/tree/main/examples/qm9) and [PAWS-X](https://github.com/median-research-group/LibMTL/tree/main/examples/xtreme) examples.
- **[Jul 22 2022]**: Added support for [Nash-MTL](https://proceedings.mlr.press/v162/navon22a/navon22a.pdf) (ICML 2022).
- **[Jul 21 2022]**: Added support for [Learning to Branch](http://proceedings.mlr.press/v119/guo20e/guo20e.pdf) (ICML 2020). Many thanks to [@yuezhixiong](https://github.com/yuezhixiong) ([#14](https://github.com/median-research-group/LibMTL/pull/14)).
- **[Mar 29 2022]**: Paper is now available on the [arXiv](https://arxiv.org/abs/2203.14338).

## Table of Content

- [Features](#features)
- [Overall Framework](#overall-framework)
- [Supported Algorithms](#supported-algorithms)
- [Supported Benchmark Datasets](#supported-benchmark-datasets)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Download Dataset](#download-dataset)
  - [Run a Model](#run-a-model)
- [Citation](#citation)
- [Contributor](#contributor)
- [Contact Us](#contact-us)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Features

- **Unified**:  ``LibMTL`` provides a unified code base to implement and a consistent evaluation procedure including data processing, metric objectives, and hyper-parameters on several representative MTL benchmark datasets, which allows quantitative, fair, and consistent comparisons between different MTL algorithms.
- **Comprehensive**: ``LibMTL`` supports many state-of-the-art MTL methods including 8 architectures and 16 optimization strategies. Meanwhile, ``LibMTL`` provides a fair comparison of several benchmark datasets covering different fields.
- **Extensible**:  ``LibMTL`` follows the modular design principles, which allows users to flexibly and conveniently add customized components or make personalized modifications. Therefore, users can easily and fast develop novel optimization strategies and architectures or apply the existing MTL algorithms to new application scenarios with the support of ``LibMTL``.

## Overall Framework

 ![framework](./docs/docs/images/framework.png)

Each module is introduced in [Docs](https://libmtl.readthedocs.io/en/latest/docs/user_guide/framework.html).

## Supported Algorithms

``LibMTL`` currently supports the following algorithms:

| Optimization Strategies                                                                                                                                                                                           | Venues             | Arguments                                                                                                         |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ | ---------------------------------------------------------------------------------------------------------------- |
| Equal Weighting (EW)                                                                                                                                                                                              | -                  | ``--weighting EW``                                                                                                |
| Gradient Normalization ([GradNorm](http://proceedings.mlr.press/v80/chen18a/chen18a.pdf))                                                                                                                         | ICML 2018          | ``--weighting GradNorm``                                                                                                |
| Uncertainty Weights ([UW](https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf))                                                                          | CVPR 2018          | ``--weighting UW``                                                                                               |
| [MGDA](https://papers.nips.cc/paper/2018/hash/432aca3a1e345e339f35a30c8f65edce-Abstract.html) ([official code](https://github.com/isl-org/MultiObjectiveOptimization))                                            | NeurIPS 2018       | ``--weighting MGDA``         |
| Dynamic Weight Average ([DWA](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_End-To-End_Multi-Task_Learning_With_Attention_CVPR_2019_paper.pdf)) ([official code](https://github.com/lorenmt/mtan))   | CVPR 2019          | ``--weighting DWA``                               |
| Geometric Loss Strategy ([GLS](https://openaccess.thecvf.com/content_CVPRW_2019/papers/WAD/Chennupati_MultiNet_Multi-Stream_Feature_Aggregation_and_Geometric_Loss_Strategy_for_Multi-Task_CVPRW_2019_paper.pdf)) | CVPR 2019 Workshop | ``--weighting GLS``                                                                                                |
| Projecting Conflicting Gradient ([PCGrad](https://papers.nips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html))                                                                                 | NeurIPS 2020       | ``--weighting PCGrad``                                                                                                |
| Gradient sign Dropout ([GradDrop](https://papers.nips.cc/paper/2020/hash/16002f7a455a94aa4e91cc34ebdb9f2d-Abstract.html))                                                                                         | NeurIPS 2020       | ``--weighting GradDrop``                                                                                                |
| Impartial Multi-Task Learning ([IMTL](https://openreview.net/forum?id=IMPnRXEWpvr))                                                                                                                               | ICLR 2021          | ``--weighting IMTL``                                                                                                |
| Gradient Vaccine ([GradVac](https://openreview.net/forum?id=F1vEjWK-lH_))                                                                                                                                         | ICLR 2021          | ``--weighting GradVac``                                                                                                |
| Conflict-Averse Gradient descent ([CAGrad](https://openreview.net/forum?id=_61Qh8tULj_)) ([official code](https://github.com/Cranial-XIX/CAGrad))                                                                 | NeurIPS 2021       | ``--weighting CAGrad``                          |
| [Nash-MTL](https://proceedings.mlr.press/v162/navon22a/navon22a.pdf) ([official code](https://github.com/AvivNavon/nash-mtl))                                                                                     | ICML 2022          | ``--weighting Nash_MTL``                          |
| Random Loss Weighting ([RLW](https://openreview.net/forum?id=jjtFD8A1Wx))                                                                                                                                         | TMLR 2022          | ``--weighting RLW``                                                                                                |
| [MoCo](https://openreview.net/forum?id=dLAYGdKTi2)                                                                                                                                                                | ICLR 2023          | ``--weighting MoCo`` |
| [Aligned-MTL](https://openaccess.thecvf.com/content/CVPR2023/html/Senushkin_Independent_Component_Alignment_for_Multi-Task_Learning_CVPR_2023_paper.html) ([official code](https://github.com/SamsungLabs/MTL))   | CVPR 2023          | ``--weighting Aligned_MTL``                             |
| [DB-MTL](https://arxiv.org/abs/2308.12029)                                                                                                                                                                        | arXiv              | ``--weighting DB_MTL``                                                                                                |

| Architectures                                                                                                                                                           | Venues          | Arguments                                                                                                                           |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| Hard Parameter Sharing ([HPS](https://dl.acm.org/doi/10.5555/3091529.3091535))                                                                                          | ICML 1993       | ``--arch HPS``                                                                                                                  |
| Cross-stitch Networks ([Cross_stitch](https://openaccess.thecvf.com/content_cvpr_2016/papers/Misra_Cross-Stitch_Networks_for_CVPR_2016_paper.pdf))                      | CVPR 2016       | ``--arch Cross_stitch``                                                                                                                  |
| Multi-gate Mixture-of-Experts ([MMoE](https://dl.acm.org/doi/10.1145/3219819.3220007))                                                                                  | KDD 2018        | ``--arch MMoE``                                                                                                                  |
| Multi-Task Attention Network ([MTAN](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_End-To-End_Multi-Task_Learning_With_Attention_CVPR_2019_paper.pdf)) ([official code](https://github.com/lorenmt/mtan))     | CVPR 2019       | ``--arch MTAN``                                                  |
| Customized Gate Control ([CGC](https://dl.acm.org/doi/10.1145/3383313.3412236)), Progressive Layered Extraction ([PLE](https://dl.acm.org/doi/10.1145/3383313.3412236)) | ACM RecSys 2020 |``--arch CGC``, ``--arch PLE``                                                                                                                  |
| Learning to Branch ([LTB](http://proceedings.mlr.press/v119/guo20e/guo20e.pdf))                                                                                         | ICML 2020       | ``--arch LTB``                                                                                                                  |
| [DSelect-k](https://proceedings.neurips.cc/paper/2021/hash/f5ac21cd0ef1b88e9848571aeb53551a-Abstract.html) ([official code](https://github.com/google-research/google-research/tree/master/dselect_k_moe))   | NeurIPS 2021    |``--arch DSelect_k``  |

## Supported Benchmark Datasets

| Datasets                                                                                 | Problems                      | Task Number  | Tasks                                                                      | multi-input | Supported Backbone             |
|:---------------------------------------------------------------------------------------- |:-----------------------------:|:------------:|:--------------------------------------------------------------------------:|:-----------:|:--------------------:|
| [NYUv2](https://github.com/median-research-group/LibMTL/tree/main/examples/nyu)          | Scene Understanding           | 3            | Semantic Segmentation+<br/>Depth Estimation+<br/>Surface Normal Prediction | ✘           | ResNet50/<br/>SegNet |
| [Office-31](https://github.com/median-research-group/LibMTL/tree/main/examples/office)   | Image Recognition             | 3            | Classification                                                             | ✓           | ResNet18             |
| [Office-Home](https://github.com/median-research-group/LibMTL/tree/main/examples/office) | Image Recognition             | 4            | Classification                                                             | ✓           | ResNet18             |
| [QM9](https://github.com/median-research-group/LibMTL/tree/main/examples/qm9)            | Molecular Property Prediction | 11 (default) | Regression                                                                 | ✘           | GNN                  |
| [PAWS-X](https://github.com/median-research-group/LibMTL/tree/main/examples/xtreme)      | Paraphrase Identification     | 4 (default)  | Classification                                                             | ✓           | Bert                 |

## Installation

1. Create a virtual environment
   
   ```shell
   conda create -n libmtl python=3.8
   conda activate libmtl
   pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
   ```

2. Clone the repository
   
   ```shell
   git clone https://github.com/median-research-group/LibMTL.git
   ```

3. Install `LibMTL`
   
   ```shell
   cd LibMTL
   pip install -r requirements.txt
   pip install -e .
   ```

## Quick Start

We use the NYUv2 dataset as an example to show how to use ``LibMTL``.

### Download Dataset

The NYUv2 dataset we used is pre-processed by [mtan](https://github.com/lorenmt/mtan). You can download this dataset [here](https://www.dropbox.com/sh/86nssgwm6hm3vkb/AACrnUQ4GxpdrBbLjb6n-mWNa?dl=0).

### Run a Model

The complete training code for the NYUv2 dataset is provided in [examples/nyu](./examples/nyu). The file [main.py](./examples/nyu/main.py) is the main file for training on the NYUv2 dataset.

You can find the command-line arguments by running the following command.

```shell
python main.py -h
```

For instance, running the following command will train an MTL model with EW and HPS on NYUv2 dataset.

```shell
python main.py --weighting EW --arch HPS --dataset_path /path/to/nyuv2 --gpu_id 0 --scheduler step --mode train --save_path PATH
```

More details is represented in [Docs](https://libmtl.readthedocs.io/en/latest/docs/getting_started/quick_start.html).

## Citation

If you find ``LibMTL`` useful for your research or development, please cite the following:

```latex
@article{lin2023libmtl,
  title={{LibMTL}: A {P}ython Library for Multi-Task Learning},
  author={Baijiong Lin and Yu Zhang},
  journal={Journal of Machine Learning Research},
  volume={24},
  number={209},
  pages={1--7},
  year={2023}
}
```

## Contributor

``LibMTL`` is developed and maintained by [Baijiong Lin](https://baijiong-lin.github.io).

## Contact Us

If you have any question or suggestion, please feel free to contact us by [raising an issue](https://github.com/median-research-group/LibMTL/issues) or sending an email to ``bj.lin.email@gmail.com``.

## Acknowledgements

We would like to thank the authors that release the public repositories (listed alphabetically):  [CAGrad](https://github.com/Cranial-XIX/CAGrad), [dselect_k_moe](https://github.com/google-research/google-research/tree/master/dselect_k_moe), [MultiObjectiveOptimization](https://github.com/isl-org/MultiObjectiveOptimization), [mtan](https://github.com/lorenmt/mtan), [MTL](https://github.com/SamsungLabs/MTL), [nash-mtl](https://github.com/AvivNavon/nash-mtl), [pytorch_geometric](https://github.com/pyg-team/pytorch_geometric), and [xtreme](https://github.com/google-research/xtreme).

## License

``LibMTL`` is released under the [MIT](./LICENSE) license.
