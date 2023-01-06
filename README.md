# KoopmanLab
[![PyPI Version](https://img.shields.io/pypi/v/koopmanlab?color=54B435&label=PyPI)](
https://pypi.org/project/koopmanlab/)
[![Code Sizez](https://img.shields.io/github/languages/code-size/Koopman-Laboratory/KoopmanLab?label=Code%20Size)](
https://github.com/Koopman-Laboratory/KoopmanLab)
[![License](https://img.shields.io/pypi/l/koopmanlab?color=FF7000&label=License)](
https://github.com/Koopman-Laboratory/KoopmanLab/blob/main/LICENSE)

Thisi is a package for Koopman Neural Operator with Pytorch.

For more information, please refer to the following paper, where we provid detailed mathematical derivations, computational designs, and code explanations. 

  - Wei Xiong, Muyuan Ma, Pei Sun and Yang Tian. "[KoopmanLab: A PyTorch module of Koopman neural operator family for solving partial differential equations](https://arxiv.org/abs/2301.01104)." arXiv preprint arXiv:2301.01104 (2023).

# Installation
KoopmanLab requires the following dependencies to be installed:
- PyTorch >= 1.10
- Numpy >= 1.23.2
- Matplotlib >= 3.3.2

You can install KoopmanLab package via the following approaches:

- Install the stable version with `pip`:

```
$ pip install koopmanlab
```

- Install the current version by source code with `pip`:
```
$ git clone https://github.com/Koopman-Laboratory/KoopmanLab.git
$ cd KoopmanLab
$ pip install -e .
```

# Usage
You can read `demo_ns.py` to learn about the basic API and workflow of KoopmanLab. If you want to run `demo_ns.py`, the following data need to be prepared in your computing resource. 
- [Dataset](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-)

If you want to generation Navier-Stokes Equation data by yourself, the data generation configuration file can be found in the following link.

- [File](https://github.com/zongyi-li/fourier_neural_operator/tree/master/data_generation/navier_stokes)

Our package provides an easy way to create a Koopman neural operator model.
``` python
import koopmanlab as kp
MLP_KNO_2D = kp.model.koopman(backbone = "KNO2d", autoencoder = "MLP", device = device)
MLP_KNO_2D = kp.model.koopman(backbone = "KNO2d", autoencoder = "MLP", o = o, m = m, r = r, t_in = 10, device = device)
MLP_KNO_2D.compile()
## Parameter definitions:
# o: the dimension of the learned Koopman operator
# f: the number of frequency modes below frequency truncation threshold
# r: the power of the Koopman operator
# T_in: the duration length of input data
# device : if CPU or GPU is used for calculating

ViT_KNO = kp.model.koopman_vit(decoder = "MLP", resolution=(64, 64), patch_size=(2, 2),
            in_chans=1, out_chans=1, head_num=16, embed_dim=768, depth = 16, parallel = True, high_freq = True, device=device)
ViT_KNO.compile()
## Parameter definitions:
# depth: the depth of each head 
# head_num: the number of heads
# resolution: the spatial resolution of input data
# patch_size: the size of each patch (i.e., token)
# in_chans: the number of target variables in the data set
# out_chans: the number of predicted variables by ViT-KNO , which is usually same as in_chans
# embed_dim: the embeding dimension
# parallel: if data parallel is applied
# high_freq: if high-frequency information complement is applied
```
Once the model is compiled, an optimizer setting is required to run your own experiments. If you want a more customized setting of optimizer and scheduler, you could use any PyTorch method to create them and assign them to Koopman neural operator object, eg. `MLP_KNO_2D.optimizer` and `MLP_KNO_2D.scheduler`.
``` python
MLP_KNO_2D.opt_init("Adam", lr = 0.005, step_size=100, gamma=0.5)
```
If you use burgers equation and navier-stokes equation data or the shallow water data provided by PDEBench, there are three specifc data interfaces that you can consider.
``` python
train_loader, test_loader = kp.data.burgers(path, batch_size = 64, sub = 32)
train_loader, test_loader = kp.data.shallow_water(path, batch_size = 5, T_in = 10, T_out = 40, sub = 1)
train_loader, test_loader = kp.data.navier_stokes(path, batch_size = 10, T_in = 10, T_out = 40, type = "1e-3", sub = 1)
## Parameter definitions:
# path: the file path of the downloaded data set
# T_in: the duration length of input data
# T_out: the duration length required to predict
# Type: the viscosity coefficient of navier-stokes equation data set.
# sub: the down-sampling scaling factor. For instance, a scaling factor sub=2 acting on a 2-dimensional data with the spatial resoluion 64*64 will create a down-sampled space of 32*32. The same factor action on a 1 dimensional data with the spatial resoluion 1*64 implies a down-sampled space of 1*32.
```
We recommend that you process your data by pytorch method `torch.utils.data.DataLoader`. In KNO model, the shape of 2D input data is `[batchsize, x, y, t_len]` and the shape of output data and label is `[batchsize, x, y, T]`, where t_len is defined in `kp.model.koopman` and T is defined in train module. In Koopman-ViT model, the shape of 2D input data is `[batchsize, in_chans, x, y]` and the shape of output data and label is `[batchsize, out_chans, x, y]`.

The KoopmanLab provides two training and two testing methods of the compact KNO sub-family. If your scenario is single step prediction, you can consider to use `train_single` method or use `train` with `T_out = 1`. Our package provides a method to save and visualize your prediction results in `test`.
``` python
MLP_KNO_2D.train_single(epochs=ep, trainloader = train_loader, evalloader = eval_loader)
MLP_KNO_2D.train(epochs=ep, trainloader = train_loader, evalloader = eval_loader, T_out = T)
MLP_KNO_2D.test_single(test_loader)
MLP_KNO_2D.test(test_loader, T_out = T, path = "./fig/ns_time_error_1e-4/", is_save = True, is_plot = True)
```
As for the ViT-KNO sub-family, `train` and `test` method is set with a single step predicition scenario. Specifically, `train_multi` and `test_multi` method provide multi-step iteration prediction, where the model iterates `T_out` times in training and testing. 
``` python
ViT_KNO.train_single(epochs=ep, trainloader = train_loader, evalloader = eval_loader)
ViT_KNO.test_single(test_loader)
ViT_KNO.train_multi(epochs=ep, trainloader = train_loader, evalloader = eval_loader, T_out = T_out)
ViT_KNO.test_multi(test_loader)
## Parameter definitions:
# epoch: epoch number of training
# trainloader: dataloader of training, which is returning variable from torch.utils.data.DataLoader
# evalloader: dataloader of evaluating, which is returning variable from torch.utils.data.DataLoader
# test_loader: dataloader of testing, which is returning variable from torch.utils.data.DataLoader
# T_out: the duration length required to predict
```
Once your model has been trained, you can use the saving module provided in KoopmanLab to save your model. Saved variable has three attribute. where `koopman` is the model class variable (i.e., the saved `kno_model` variable), `model` is the trained model variable (i.e., the saved `kno_model.kernel` variable), and `model_params` is the parameters dictionary of trained model variable (i.e., the saved `kno_model.kernel.state_dict()` variable).
``` python
MLP_KNO_2D.save(save_path)
## Parameter definitions:
# save_path: the file path of the result saving
```
# Citation
If you use KoopmanLab package for academic research, you are encouraged to cite the following paper:
```
@article{xiong2023koopmanlab,
  title={KoopmanLab: A PyTorch module of Koopman neural operator family for solving partial differential equations},
  author={Xiong, Wei and Ma, Muyuan and Sun, Pei and Tian, Yang},
  journal={arXiv preprint arXiv:2301.01104},
  year={2023}
}
```

# License
[GPL-3.0 License](https://github.com/Koopman-Laboratory/KoopmanLab/blob/main/LICENSE)
