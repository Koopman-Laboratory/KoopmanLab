import torch
import koopmanlab as kp

torch.cuda.set_device(0)

device = torch.device("cuda")
# Loading Data
path = "./data/ns_V1e-3_N5000_T50.mat"
train_loader, test_loader = kp.data.navier_stokes(path, batch_size = 10, T_in = 10, T_out = 40, type = "1e-3", sub = 1)
# Hyper parameters
ep = 1
o = 32
m = 16
r = 8
# Model
koopman_model = kp.model.koopman(backbone = "KNO2d", autoencoder = "MLP", o = o, m = m, r = r, t_in = 10, device = device)
koopman_model.compile()
koopman_model.opt_init("Adam", lr = 0.005, step_size=100, gamma=0.5)
koopman_model.train(epochs=ep, trainloader = train_loader, evalloader = test_loader)
time_error = koopman_model.test(test_loader, path = "./fig/ns_time_error_1e-3_conv/", is_save = True, is_plot = True)
print(time_error)
filename = "ns_time_error_op" + str(op) + "m" + str(m) + "d" +str(d) + ".pt"
torch.save({"time_error":time_error,"params":a.params},"./result/ns_time_error_1e-3/"+filename)

