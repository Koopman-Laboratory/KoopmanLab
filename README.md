# KoopmanLab
The fundamental package for Koopman Neural Operator with Pytorch.

# Installation
KoopmanLab requires the following dependencies to be installed:
- PyTorch >= 1.10
- Numpy >= 1.23.2

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

Our package gives you a easy way to create a koopman model.
``` python
import koopman as kp
kno_model = kp.model.koopman(backbone = "KNO2d", autoencoder = "MLP", device = device)
kno_model = kp.model.koopman(backbone = "KNO2d", autoencoder = "MLP", o = o, m = m, r = r, t_len = 10, device = device)
# koopmanVit_model = model.koopman_vit(decoder = "MLP", L_layers = 16, resolution=(64, 64), patch_size=(2, 2),
            in_chans=1, out_chans=1, parallel = True, high_freq = True, device=device)
kno_model.compile()
```
If you use burgers equation and navier-stokes equation data by the link or shallow water data by PDEBench, there are three specifc data interface are provided.
``` python
train_loader, test_loader = kp.data.burgers(path, batch_size = 64, sub = 32)
train_loader, test_loader = kp.data.shallow_water(path, batch_size = 5, T_in = 10, T = T, sub = 1)
train_loader, test_loader = kp.data.navier_stokes(path, batch_size = 10, T_in = 10, T = 40, type = "1e-3", sub = 1)
```
We recommend you process your data by pytorch method `torch.utils.data.DataLoader`. In KNO model, the shape of 2D input data is `[batchsize, x, y, t_len]`, the shape of output data and label is `[batchsize, x, y, T]`, where t_len is defined in `kp.model.koopman` and T is defined in train module. In Koopman-ViT model, the shape of 2D input data is `[batchsize, in_chans, x, y]`, the shape of output data and label is `[batchsize, out_chans, x, y]`.

In KNO model, The package provides two train methods and two test methods. If your scenario is single step prediction, you'd better use `train_single` method or use `train` setting `T_out = 1`. The package provides prediction result saving method and result ploting method in `test`.
```
kno_model.train_single(epochs=ep, trainloader = train_loader, evalloader = eval_loader)
kno_model.train(epochs=ep, trainloader = train_loader, evalloader = eval_loader, T_out = T)
kno_model.test_single(test_loader)
kno_model.test(test_loader, T_out = T, path = "./fig/ns_time_error_1e-4/", is_save = True, is_plot = True)
```
In Koopman-Vit model, `train` and `test` method for training and testing the model in single step predicition scenario. Because of Koopman-ViT structure, `train_multi` and `test_multi` method provide multi-step iteration prediction, which meanse the model is iterated by `T_out` times in training and testing method. 
```
koopmanVit_model.train(epochs=ep, trainloader = train_loader, evalloader = eval_loader)
koopmanVit_model.test(test_loader)
koopmanVit_model.train_multi(epochs=ep, trainloader = train_loader, evalloader = test_loader, T_out = T_out)
koopmanVit_model.test_multi(test_loader)
```
# Cite KoopmanLab
If you use KoopmanLab package for academic research, you are encouraged to cite the following paper:
```

```


