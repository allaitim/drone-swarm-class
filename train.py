
import numpy as np 
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score
from torchsummary import summary
import torch.nn as nn
from model import STFTANet
from loaders import get_train_valid_loader



def trainer (model, criterion, optimizer, train_batches, test_batches,
            epochs, patience, verbose = True):
  
    min_test_loss = np.inf

    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    consec_increases = 0        
    for epoch in range(num_epochs):  
        train_batch_loss = 0
        train_batch_acc = 0
        test_batch_loss = 0
        test_batch_acc = 0
        for i, (X_tr, y_tr) in enumerate(train_batches): 
            
            optimizer.zero_grad()
            if torch.cuda.is_available():
                X_tr, y_tr = X_tr.cuda(), y_tr.cuda()
            
            y_hat = model(X_tr)
            y_hat1 = y_hat.argmax(dim = 1).to(device).numpy()
            loss = loss_function(y_hat, y_tr)
            loss.backward() 
            optimizer.step() 
            train_batch_loss += loss.item() 
            train_batch_acc += accuracy_score(y_tr , y_hat1)
            
        train_loss.append(train_batch_loss / len(train_batches))
        train_accuracy.append(train_batch_acc / len(train_batches))  
        model.eval() 
        
        
        with torch.no_grad():
            # this stops pytorch doing computation and saves memory and time
            for X_tes, y_tes in (test_batches):
                if torch.cuda.is_available():
                    X_tes, y_tes  = X_tes.cuda(), y_tes.cuda()
                y_hat = model(X_tes)
                y_hat1 = y_hat.argmax(dim = 1).to(device).numpy()
                loss = loss_function(y_hat, y_tes)
                test_batch_loss += loss.item()
                test_batch_acc += accuracy_score(y_tes , y_hat1)
        test_loss.append(test_batch_loss / len(test_batches))       
        test_accuracy.append(test_batch_acc/len(test_batches))  
        
        model.train() 
        test_norm_loss = test_batch_loss/ len(test_batches)   
        if min_test_loss > test_norm_loss:                
            min_test_loss = test_norm_loss
            name_mod = (conf1['SNR'],conf1["f_s"],conf1["signal_duration"])
            torch.save(model.state_dict(), (model_path +'m_.'+str(name_mod)+'.pth'))

        if verbose:
            if  epoch % ep_print==0:

                print(f"\n Epoch {epoch } \n",
                  f"Train Loss: {train_loss[-1]:.3f}.",
                  f"Test Loss: {test_loss[-1]:.3f}\n",
                  f"Train Accuracy: {train_accuracy[-1]:.4f}.",
                  f"Test Accuracy: {test_accuracy[-1]:.4f}.\n",
                  )
        
        # Early stopping
        if epoch > 0 and test_loss[-1] > test_loss[-2]:
            consec_increases += 1
        else:
            consec_increases = 0
            
        if consec_increases == patience:
            print (f"stopped early at epoch {epoch + 1} - val loss increased for {consec_increases} consecutive epochs!")
            break

    results = {"train_loss": train_loss,
               "test_loss": test_loss,
               "train_accuracy": train_accuracy,
               "test_accuracy": test_accuracy,
               }
    return results




conf1 = {
        
        "epochs": 150,  #100,
        "SNR":10,  # 5,
        "f_s": 2_000,
        "f_c": 9.4e10,
        "signal_duration":0.3,   # 0.16,
        "totalset_size": 15_000
    }




totalset_size = 2500
batch_size = 128
drone_classes = [0,1,2,3,4,5]
totalsize = totalset_size*len(drone_classes)
windowsize = 512
N = 4
random_seed = 42
valid_size = 0.8

num_epochs = 150  
numepplot = 1  

ep_print = num_epochs / 15

X_asinacos = np.zeros((totalsize, 2, windowsize, N))
y_dr = np.zeros((totalsize))

phc = np.load(total_path + "Amcosproc_swarms3--SNR10--fs1_2000--14999.npy")
phs = np.load(total_path + "Amsinproc_swarms3--SNR10--fs1_2000--14999.npy")
dr_cl = np.load(total_path + "dron_class_swarms3--SNR10--fs1_2000--14999.npy")


phs1 = phs.reshape(totalsize,1,windowsize, N)
phc1 = phc.reshape(totalsize,1,windowsize,N)
y_dr = np.array(dr_cl)


X_asinacos[:,0:1,:,:] = np.array (phc1) 
X_asinacos[:,1:2,:,:] = np.array (phs1)


train_batches,test_batches = get_train_valid_loader(batch_size,totalsize,
            windowsize,N,X_asinacos,y_dr,random_seed,valid_size,shuffle=True)
                                                     

model = STFTANet(num_classes = 6)

summary (model, (2, windowsize, N))

if torch.cuda.is_available():
    device = torch.device('cuda')
    not_device = torch.device('cpu')
else:
    device = torch.device('cpu')
    not_device = torch.device('cuda')
    
model.to(device)

criterion = nn.CrossEntropyLoss()
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

results = trainer (model, loss_function, optimizer, train_batches, test_batches, 
                   epochs = num_epochs, patience = 7)


trloss = np.zeros((num_epochs))
testloss = np.zeros((num_epochs))
traccur = np.zeros((num_epochs))
testaccur = np.zeros((num_epochs))

trloss = np.array(results["train_loss"]).reshape(num_epochs)
testloss = np.array(results["test_loss"]).reshape(num_epochs)
traccur = np.array(results["train_accuracy"]).reshape(num_epochs)
testaccur = np.array(results["test_accuracy"]).reshape(num_epochs)

trloss1 = trloss[numepplot:]
testloss1 = testloss[numepplot:] 
traccur1 = traccur[numepplot:]
testaccur1 = testaccur[numepplot:]


np.save (totalres_path + 'trainloss_'+ str(num_epochs) + '-'+str('epochs'),trloss)
np.save (totalres_path + 'testloss_'+ str(num_epochs) + '-'+str('epochs') ,testloss)
np.save (totalres_path + 'accurtrain_'+ str(num_epochs) + '-'+str('epochs') ,traccur)
np.save (totalres_path + 'accurtest_'+ str(num_epochs) + '-'+str('epochs'),testaccur)


fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 10), sharex=True)

ax1.plot(testloss1)
ax1.plot(trloss1)
ax1.set_ylabel("test_train_losses")

ax2.plot(testaccur1)
ax2.plot(traccur1)
ax2.set_ylabel("test_train_accuracy")

ax2.set_xlabel("epochs") 

print (f"\n optimizer {optimizer} \n")


