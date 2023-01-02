# KoopmanLab
The fundamental package for Koopman Neural Operator with Pytorch.

# Installation
KoopmanLab requires the following dependencies to be installed:
- PyTorch >= 1.10
- Numpy >= 1.23.2
- Matplotlib >= 3.3.2

Then you can install KoopmanLab package:

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
You can read `demo_ns.py` to familiar with the basic API and workflow of our pacakge. If you want to run `demo_ns.py`, the data need to be prepared in your computing resource.
- [Dataset](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-)

If you want to generation Navier-Stokes Equation data by yourself, the data generation configuration file can be found in the link.

- [File](https://github.com/zongyi-li/fourier_neural_operator/tree/master/data_generation/navier_stokes)

Our package gives you an easy way to create a koopman model.
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

ViT_KNO = kp.model.koopman_vit(decoder = "MLP", resolution=(64, 64), patch_size=(2, 2),
            in_chans=1, out_chans=1, head_num=16, embed_dim=768, depth = 16, parallel = True, high_freq = True, device=device)
ViT_KNO.compile()
## Parameter definitions:
# depth: the depth of each head 
# head_num: the number of heads
# resolution: the spatial resolution of input data
# patch_size: the size of each patch (i.e., token)
# in_chans:
# out_chans:
# num_blocks:
# embed_dim: 
# parallel: if data parallel is applied
# high_freq: if high-frequency information complement is applied
```
If you use burgers equation and navier-stokes equation data by the link or shallow water data by PDEBench, there are three specifc data interface are provided.
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
We recommend you process your data by pytorch method `torch.utils.data.DataLoader`. In KNO model, the shape of 2D input data is `[batchsize, x, y, t_len]`, the shape of output data and label is `[batchsize, x, y, T]`, where t_len is defined in `kp.model.koopman` and T is defined in train module. In Koopman-ViT model, the shape of 2D input data is `[batchsize, in_chans, x, y]`, the shape of output data and label is `[batchsize, out_chans, x, y]`.

In KNO model, The package provides two train methods and two test methods. If your scenario is single step prediction, you'd better use `train_single` method or use `train` setting `T_out = 1`. The package provides prediction result saving method and result ploting method in `test`.
``` python
MLP_KNO_2D.train_single(epochs=ep, trainloader = train_loader, evalloader = eval_loader)
MLP_KNO_2D.train(epochs=ep, trainloader = train_loader, evalloader = eval_loader, T_out = T)
MLP_KNO_2D.test_single(test_loader)
MLP_KNO_2D.test(test_loader, T_out = T, path = "./fig/ns_time_error_1e-4/", is_save = True, is_plot = True)
```
In Koopman-Vit model, `train` and `test` method for training and testing the model in single step predicition scenario. Because of Koopman-ViT structure, `train_multi` and `test_multi` method provide multi-step iteration prediction, which meanse the model is iterated by `T_out` times in training and testing method. 
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
Having trained your own model, save module is provided in our package. Saved variable has three attribute. `koopman` is the model class variable, which means save `kno_model` variable. `model` is the trained model variable, which means save `kno_model.kernel` variable. `model_params` is the parameters dictionary of trained model variable, which means `kno_model.kernel.state_dict()` variable.
``` python
MLP_KNO_2D.save(save_path)
## Parameter definitions:
# save_path: the file path of the result saving
```
# Cite KoopmanLab
If you use KoopmanLab package for academic research, you are encouraged to cite the following paper:
```

```


