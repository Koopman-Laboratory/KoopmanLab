from . import kno
from . import utils
from . import koopmanViT

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer

class koopman:
    def __init__(self, backbone = "KNO1d", autoencoder = "MLP", o = 16, m = 16, r = 8, t_in = 1, device = False):
        self.backbone = backbone
        self.autoencoder = autoencoder
        self.operator_size = o
        self.modes = m
        self.decompose = r
        self.device = device
        self.t_in = t_in
        # Core Model
        self.params = 0
        self.kernel = False
        # Opt Setting
        self.optimizer = False
        self.scheduler = False
        self.loss = torch.nn.MSELoss()
    def compile(self):
        if self.autoencoder == "MLP":
            encoder = kno.encoder_mlp(self.t_in, self.operator_size)
            decoder = kno.decoder_mlp(self.t_in, self.operator_size)
            print("The autoencoder type is MLP.")
        elif self.autoencoder == "Conv1d":
            encoder = kno.encoder_conv1d(self.t_in, self.operator_size)
            decoder = kno.decoder_conv1d(self.t_in, self.operator_size)
            print("The autoencoder type is Conv1d.")
        elif self.autoencoder == "Conv2d":
            encoder = kno.encoder_conv2d(self.t_in, self.operator_size)
            decoder = kno.decoder_conv2d(self.t_in, self.operator_size)
            print("The autoencoder type is Conv2d.")
        else:
#            encoder = kno.encoder_mlp(self.t_in, self.operator_size)
#            decoder = kno.decoder_mlp(self.t_in, self.operator_size)
#            print("The autoencoder type is MLP.")
            print("Wrong!")
        if self.backbone == "KNO1d":
            self.kernel = kno.KNO1d(encoder, decoder, self.operator_size, modes_x = self.modes, decompose = self.decompose).to(self.device)
            print("KNO1d model is completed.")
        
        elif self.backbone == "KNO2d":
            self.kernel = kno.KNO2d(encoder, decoder, self.operator_size, modes_x = self.modes, modes_y = self.modes,decompose = self.decompose).to(self.device)
            print("KNO2d model is completed.")

        if not self.kernel == False:
            self.params = utils.count_params(self.kernel)
            print("Koopman Model has been compiled!")
            print("The Model Parameters Number is ",self.params)
    def opt_init(self, opt, lr, step_size, gamma):
        if opt == "Adam":
            self.optimizer = utils.Adam(self.kernel.parameters(), lr= lr, weight_decay=1e-4)
        if not step_size == False:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)

    def train_single(self, epochs, trainloader, evalloader = False):
        for ep in range(epochs):
            # Train
            self.kernel.train()
            t1 = default_timer()
            train_recons_full = 0
            train_pred_full = 0
            for xx, yy in trainloader:
                l_recons = 0
                bs = xx.shape[0]
                xx = xx.to(self.device)
                yy = yy.to(self.device)
                pred,im_re = self.kernel(xx)
                
                l_recons = self.loss(im_re.reshape(bs, -1), xx.reshape(bs, -1))
                l_pred = self.loss(pred.reshape(bs, -1), yy.reshape(bs, -1))

                train_pred_full += l_pred.item()
                train_recons_full += l_recons.item()

                loss = 5*l_pred + 0.5*l_recons
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            train_pred_full = train_pred_full / len(trainloader)
            train_recons_full = train_recons_full / len(trainloader)
            t2 = default_timer()
            test_pred_full = 0
            test_recons_full = 0
            mse_test = 0
            # Test
            if evalloader:
                with torch.no_grad():
                    for xx, yy in evalloader:
                        bs = xx.shape[0]
                        loss = 0
                        xx = xx.to(self.device)
                        yy = yy.to(self.device)

                        pred,im_re = self.kernel(xx)


                        l_recons = self.loss(im_re.reshape(bs, -1), xx.reshape(bs, -1))
                        l_pred = self.loss(pred.reshape(bs, -1), yy.reshape(bs, -1))


                        test_pred_full += l_pred.item()
                        test_recons_full += l_recons.item()
                        
                test_pred_full = test_pred_full/len(evalloader)
                test_recons_full = test_recons_full/len(evalloader)
                
            self.scheduler.step()

            if evalloader:
                if ep == 0:
                    print("Epoch","Time","[Train Recons MSE]","[Train Pred MSE]","[Eval Recons MSE]","[Eval Pred MSE]")
                print(ep, t2 - t1, train_recons_full, train_pred_full, test_recons_full, test_pred_full)
            else:
                if ep == 0:
                    print("Epoch","Time","Train Recons MSE","Train Pred MSE")
                print(ep, t2 - t1, train_recons_full, train_pred_full)

    def test_single(self, testloader):
        test_pred_full = 0
        test_recons_full = 0
        with torch.no_grad():
            for xx, yy in testloader:
                bs = xx.shape[0]
                loss = 0
                xx = xx.to(self.device)
                yy = yy.to(self.device)

                pred,im_re = self.kernel(xx)

                l_recons = self.loss(im_re.reshape(bs, -1), xx.reshape(bs, -1))
                l_pred = self.loss(pred.reshape(bs, -1), yy.reshape(bs, -1))

                test_pred_full += l_pred.item()
                test_recons_full += l_recons.item()
        test_pred_full = test_pred_full/len(testloader)
        test_recons_full = test_recons_full/len(testloader)
        print("Total prediction test mse error is ",test_pred_full)
        print("Total reconstruction test mse error is ",test_recons_full)
        return test_pred_full


    def train(self, epochs, trainloader, step = 1, T_out = 40, evalloader = False):
        T_eval = T_out
        for ep in range(epochs):
            self.kernel.train()
            t1 = default_timer()
            train_recons_full = 0
            train_pred_full = 0
            for xx, yy in trainloader:
                l_recons = 0
                xx = xx.to(self.device)
                yy = yy.to(self.device)
                bs = xx.shape[0]
                for t in range(0, T_out):
                    y = yy[..., t:t + 1]

                    im,im_re = self.kernel(xx)
                    l_recons += self.loss(im_re.reshape(bs, -1), xx.reshape(bs, -1))
                    if t == 0:
                        pred = im[...,-1:]
                    else:
                        pred = torch.cat((pred, im[...,-1:]), -1)
                    
                    xx = torch.cat((xx[..., step:], im[...,-1:]), dim=-1)
                
                l_pred = self.loss(pred.reshape(bs, -1), yy.reshape(bs, -1))
                loss = 5 * l_pred + 0.5 * l_recons
                
                train_pred_full += l_pred.item()
                train_recons_full += l_recons.item()/T_out

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            train_pred_full = train_pred_full / len(trainloader)
            train_recons_full = train_recons_full / len(trainloader)
            t2 = default_timer()
            test_pred_full = 0
            test_recons_full = 0
            loc = 0
            mse_error = 0
            if evalloader:
                with torch.no_grad():
                    for xx, yy in evalloader:
                        loss = 0
                        xx = xx.to(self.device)
                        yy = yy.to(self.device)

                        for t in range(0, T_eval):
                            y = yy[..., t:t + 1]
                            im, im_re = self.kernel(xx)
                            l_recons += self.loss(im_re.reshape(bs, -1), xx.reshape(bs, -1))
                            if t == 0:
                                pred = im[...,-1:]
                            else:
                                pred = torch.cat((pred, im[...,-1:]), -1)
                            xx = torch.cat((xx[..., 1:], im[...,-1:]), dim=-1)
                        l_pred = self.loss(pred.reshape(bs, -1), yy.reshape(bs, -1))

                        test_recons_full += l_recons.item() / T_eval
                        test_pred_full += l_pred.item()
                        
                        loc = loc + 1
                    mse_error = mse_error / loc
                test_recons_full = test_recons_full / len(evalloader)
                test_pred_full = test_pred_full / len(evalloader)
            self.scheduler.step()

            if evalloader:
                if ep == 0:
                    print("Epoch","Time","[Train Recons MSE]","[Train Pred MSE]","[Eval Recons MSE]","[Eval Pred MSE]")
                print(ep, t2 - t1, train_recons_full, train_pred_full, test_recons_full, test_pred_full)
            else:
                if ep == 0:
                    print("Epoch","Time","Train Recons MSE","Train Pred MSE")
                print(ep, t2 - t1, train_recons_full, train_pred_full)
    def test(self, testloader, step = 1, T_out = 40, path = False, is_save = False, is_plot = False):
        time_error = torch.zeros([T_out,1])
        test_pred_full = 0
        test_recons_full = 0
        loc = 0
        with torch.no_grad():
            for xx, yy in testloader:
                loss = 0
                bs = xx.shape[0]
                xx = xx.to(self.device)
                yy = yy.to(self.device)
                l_recons = 0
                for t in range(0, T_out):
                    y = yy[..., t:t + 1]
                    im, im_re = self.kernel(xx)
                    l_recons += self.loss(im_re.reshape(bs, -1), xx.reshape(bs, -1))
                    t_error = self.loss(im[...,-1:],y)
                    if t == 0:
                        pred = im[...,-1:]
                    else:
                        pred = torch.cat((pred, im[...,-1:]), -1)
                    time_error[t] = time_error[t] + t_error.item()
                    xx = torch.cat((xx[..., 1:], im[...,-1:]), dim=-1)

                test_recons_full += l_recons.item() / T_out
                l_pred = self.loss(pred.reshape(bs, -1), yy.reshape(bs, -1))
                test_pred_full += l_pred.item()
                if(loc == 0 & is_save):
                    torch.save({"pred":pred, "yy":yy}, path+ "pred_yy.pt")
                
                if(loc == 0 & is_plot):
                    for i in range(T_out):
                        plt.subplot(1,3,1)
                        plt.title("Predict")
                        plt.imshow(pred[0,...,i].cpu().detach().numpy())
                        plt.subplot(1,3,2)
                        plt.imshow(yy[0,...,i].cpu().detach().numpy())
                        plt.title("Label")
                        plt.subplot(1,3,3)
                        plt.imshow(pred[0,...,i].cpu().detach().numpy()-yy[0,...,i].cpu().detach().numpy())
                        plt.title("Error")
                        plt.show()
                        plt.savefig(path + "time_"+str(i)+".png")
                        plt.close()
                loc = loc + 1
        test_pred_full = test_pred_full / loc
        test_recons_full = test_recons_full / loc
        time_error = time_error / len(testloader)
        print("Total prediction test mse error is ",test_pred_full)
        print("Total reconstruction test mse error is ",test_recons_full)
        return time_error
        
    def save(self, path):
        (fpath,_) = os.path.split(path)
        if not os.path.isfile(fpath):
            os.makedirs(fpath)
        torch.save({"koopman":self,"model":self.kernel,"model_params":self.kernel.state_dict()}, path)

class koopman_vit:
    def __init__(self, decoder = "MLP", depth = 16, resolution=(256, 256), patch_size=(4, 4),
            in_chans=1, out_chans=1, head_num = 16, embed_dim=768, high_freq = True, parallel = False, device = False):
        # Model Hyper-parameters
        self.decoder = decoder
        self.resolution = resolution
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.embed_dim = embed_dim
        self.num_blocks = head_num
        self.depth = depth
        # Core Model
        self.params = 0
        self.kernel = False
        # Opt Setting
        self.optimizer = False
        self.scheduler = False
        self.device = device
        self.parallel = parallel
        self.high_freq = high_freq
        self.loss = torch.nn.MSELoss()
    def compile(self):
        self.kernel = koopmanViT.ViT(img_size=self.resolution, patch_size=self.patch_size, in_chans=self.in_chans, out_chans=self.out_chans, num_blocks=self.num_blocks, embed_dim = self.embed_dim, depth=self.depth, settings = self.decoder).to(self.device)
        if self.parallel:
            self.kernel = torch.nn.DataParallel(self.kernel)
        self.params = utils.count_params(self.kernel)
        
        print("Koopman Fourier Vision Transformer has been compiled!")
        print("The Model Parameters Number is ",self.params)
        
    def opt_init(self, opt, lr, step_size, gamma):
        if opt == "Adam":
            self.optimizer = utils.Adam(self.kernel.parameters(), lr= lr, weight_decay=1e-4)
        if not step_size == False:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        
    def train_multi(self, epochs, trainloader, T_out = 10, evalloader = False):
        T_eval = T_out
        for ep in range(epochs):
            self.kernel.train()
            t1 = default_timer()
            train_recons_full = 0
            train_pred_full = 0
            for xx, yy in trainloader:
                l_recons = 0
                xx = xx.to(self.device) # [batchsize,1,x,y]
                yy = yy.to(self.device) # [batchsize,T,x,y]
                bs = xx.shape[0]
                for t in range(0, T_out):
                    y = yy[:, t:t + 1]
                    im,im_re = self.kernel(xx)
                    l_recons += self.loss(im_re.reshape(bs, -1), xx.reshape(bs, -1))
                    
                    if t == 0:
                        pred = im[:, -1:]
                    else:
                        pred = torch.cat((pred, im[:, -1:]), -1)
                    
                    xx = im
                
                l_pred = self.loss(pred.reshape(bs, -1), yy.reshape(bs, -1))
                loss = 5 * l_pred + 0.5 * l_recons
                
                train_pred_full += l_pred.item()
                train_recons_full += l_recons.item()/T_out

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            train_pred_full = train_pred_full / len(trainloader)
            train_recons_full = train_recons_full / len(trainloader)
            t2 = default_timer()
            test_pred_full = 0
            test_recons_full = 0
            loc = 0
            mse_error = 0
            if evalloader:
                with torch.no_grad():
                    for xx, yy in evalloader:
                        loss = 0
                        xx = xx.to(self.device)
                        yy = yy.to(self.device)

                        for t in range(0, T_eval):
                            y = yy[:, t:t + 1]
                            im, im_re = self.kernel(xx)
                            
                            l_recons += self.loss(im_re.reshape(bs, -1), xx.reshape(bs, -1))
                            
                            if t == 0:
                                pred = im
                            else:
                                pred = torch.cat((pred, im), 1)
                                
                            xx = im
                            
                        l_pred = self.loss(pred.reshape(bs, -1), yy.reshape(bs, -1))

                        test_recons_full += l_recons.item() / T_eval
                        test_pred_full += l_pred.item()
                        
                test_recons_full = test_recons_full / len(evalloader)
                test_pred_full = test_pred_full / len(evalloader)
            self.scheduler.step()

            if evalloader:
                if ep == 0:
                    print("Epoch","Time","[Train Recons MSE]","[Train Pred MSE]","[Eval Recons MSE]","[Eval Pred MSE]")
                print(ep, t2 - t1, train_recons_full, train_pred_full, test_recons_full, test_pred_full)
            else:
                if ep == 0:
                    print("Epoch","Time","Train Recons MSE","Train Pred MSE")
                print(ep, t2 - t1, train_recons_full, train_pred_full)
    
    def test_multi(self, testloader, step = 1, T_out = 5, path = False, is_save = False, is_plot = False):
        time_error = torch.zeros([T_out,1])
        test_pred_full = 0
        test_recons_full = 0
        loc = 0
        with torch.no_grad():
            for xx, yy in testloader:
                loss = 0
                bs = xx.shape[0]
                xx = xx.to(self.device)
                yy = yy.to(self.device)
                l_recons = 0
                for t in range(0, T_out):
                    y = yy[:, t:t + 1]
                    im, im_re = self.kernel(xx)
                    
                    
                    l_recons += self.loss(im_re.reshape(bs, -1), xx.reshape(bs, -1))
                    t_error = self.loss(im, y)
                    
                    xx = im
                    
                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), 1)
                    time_error[t] = time_error[t] + t_error.item()
    
                test_recons_full += l_recons.item() / T_out
                l_pred = self.loss(pred.reshape(bs, -1), yy.reshape(bs, -1))
                test_pred_full += l_pred.item()

                if(loc == 0 & is_save):
                    torch.save({"pred":pred, "yy":yy}, path+ "pred_yy.pt")
                
                if(loc == 0 & is_plot):
                    for i in range(T_out):
                        plt.subplot(1,3,1)
                        plt.title("Predict")
                        plt.imshow(pred[0,i].cpu().detach().numpy())
                        plt.subplot(1,3,2)
                        plt.imshow(yy[0,i].cpu().detach().numpy())
                        plt.title("Label")
                        plt.subplot(1,3,3)
                        plt.imshow(pred[0,i].cpu().detach().numpy()-yy[0,i].cpu().detach().numpy())
                        plt.title("Error")
                        plt.show()
                        plt.savefig(path + "time_"+str(i)+".png")
                        plt.close()

                loc = loc + 1
        test_pred_full = test_pred_full / loc
        test_recons_full = test_recons_full / loc
        time_error = time_error / len(testloader)
        print("Total prediction test mse error is ",test_pred_full)
        print("Total reconstruction test mse error is ",test_recons_full)
        return time_error
        
        
    def train_single(self, epochs, trainloader, evalloader = False):
        for ep in range(epochs):
            self.kernel.train()
            t1 = default_timer()
            train_recons_full = 0
            train_pred_full = 0
            for x, y in trainloader:
                l_recons = 0
                x = x.to(self.device) # [batchsize,1,64,64]
                y = y.to(self.device) # [batchsize,1,64,64]
                bs = x.shape[0]
                
                im,im_re = self.kernel(x)
                
                l_recons = self.loss(im_re.reshape(bs, -1), x.reshape(bs, -1))
                l_pred = self.loss(im.reshape(bs, -1), y.reshape(bs, -1))
                
                loss = 5 * l_pred + 0.5 * l_recons
                
                train_pred_full += l_pred.item()
                train_recons_full += l_recons.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            train_pred_full = train_pred_full / len(trainloader)
            train_recons_full = train_recons_full / len(trainloader)
            t2 = default_timer()
            test_pred_full = 0
            test_recons_full = 0
            loc = 0
            mse_error = 0
            if evalloader:
                with torch.no_grad():
                    for x, y in evalloader:
                        loss = 0
                        x = x.to(self.device)
                        y = y.to(self.device)
                        
                        im, im_re = self.kernel(x)

                        l_recons = self.loss(im_re.reshape(bs, -1), x.reshape(bs, -1))
                        l_pred = self.loss(im.reshape(bs, -1), y.reshape(bs, -1))

                        test_recons_full += l_recons.item()
                        test_pred_full += l_pred.item()
                        
                test_recons_full = test_recons_full / len(evalloader)
                test_pred_full = test_pred_full / len(evalloader)
            self.scheduler.step()

            if evalloader:
                if ep == 0:
                    print("Epoch","Time","[Train Recons MSE]","[Train Pred MSE]","[Eval Recons MSE]","[Eval Pred MSE]")
                print(ep, t2 - t1, train_recons_full, train_pred_full, test_recons_full, test_pred_full)
            else:
                if ep == 0:
                    print("Epoch","Time","Train Recons MSE","Train Pred MSE")
                print(ep, t2 - t1, train_recons_full, train_pred_full)
                
    def test_single(self, testloader, T_out = 1, path = False, is_save = False, is_plot = False):
        time_error = torch.zeros([T_out,1])
        test_pred_full = 0
        test_recons_full = 0
        loc = 0
        idx = np.random.randint(0,len(testloader))
        with torch.no_grad():
            for xx, yy in testloader:
                loss = 0
                bs = xx.shape[0]
                xx = xx.to(self.device)
                yy = yy.to(self.device)
                l_recons = 0
                for t in range(0, T_out):
                    y = yy[:, t:t + 1]
                    im, im_re = self.kernel(xx)
                    
                    l_recons += self.loss(im_re.reshape(bs, -1), xx.reshape(bs, -1))
                    t_error = self.loss(im, y)
                    
                    xx = im
                    
                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), 1)
                    time_error[t] = time_error[t] + t_error.item()
    
                test_recons_full += l_recons.item() / T_out
                l_pred = self.loss(pred.reshape(bs, -1), yy.reshape(bs, -1))
                test_pred_full += l_pred.item()

                if(loc == 0 & is_save):
                    torch.save({"pred":pred, "yy":yy}, path+ "pred_yy.pt")
                
                if(loc == 0 & is_plot):
                    for i in range(T_out):
                        plt.subplot(1,3,1)
                        plt.title("Predict")
                        plt.imshow(pred[0,i].cpu().detach().numpy())
                        plt.subplot(1,3,2)
                        plt.imshow(yy[0,i].cpu().detach().numpy())
                        plt.title("Label")
                        plt.subplot(1,3,3)
                        plt.imshow(pred[0,i].cpu().detach().numpy()-yy[0,i].cpu().detach().numpy())
                        plt.title("Error")
                        plt.show()
                        plt.savefig(path + "time_"+str(i)+".png")
                        plt.close()
                loc = loc + 1

        test_pred_full = test_pred_full / len(testloader)
        test_recons_full = test_recons_full / len(testloader)
        time_error = time_error / len(testloader)
        print("Total prediction test mse error is ",test_pred_full)
        print("Total reconstruction test mse error is ",test_recons_full)
        
        return time_error
        
    def save(self, path):
#        (fpath,_) = os.path.split(path)
#        print(fpath, os.path.isfile(fpath))
#        if not os.path.isfile(fpath):
#            os.makedirs(fpath)
        torch.save({"koopman":self,"model":self.kernel,"model_params":self.kernel.state_dict()}, path)
