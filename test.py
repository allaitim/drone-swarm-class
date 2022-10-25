
import torch
import numpy as np
import seaborn as sn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from model import STFTANet
from loaders import GetTestLoader




def testclassifier(model_path, conf, dataset_size, vis=False):
    print(f"\nstarting testing of {str(conf)}\n")

    with torch.no_grad():
        confusm1 = np.zeros((num_classes,num_classes))
        cmrow = []
        cmnorm = np.zeros((num_classes,num_classes))
        
        test_loss = []
        test_accuracy = []
        test_batch_loss = 0
        test_batch_acc = 0

        loss_function = torch.nn.CrossEntropyLoss()

        model.load_state_dict(torch.load(model_path1+ "m_.(10, 2000, 0.3).pth"))
        model.eval()

        
        for i,  (X_tes,y_tes) in enumerate (test_batches):
    
            if torch.cuda.is_available():
                 X_tes, y_tes  = X_tes.cuda(), y_tes.cuda()
                 
            y_hat = model(X_tes)    
            _, predicted = torch.max(y_hat.data,1)
            
            y_hat1 = y_hat.argmax(dim = 1).to(device).numpy()
                         
            loss = loss_function(y_hat, y_tes)   # Calculate loss based on output
            test_batch_loss += loss.item()
            test_batch_acc += accuracy_score(y_tes , y_hat1)
            
            y_hat = y_hat.to(device)  

            y_tr, y_pr = torch.Tensor.cpu(y_tes).numpy(),torch.Tensor.cpu(predicted).numpy()
            cmatr = confusion_matrix(y_tr, y_pr)

            confusm1 = np.add(confusm1, cmatr)
                    
        for i in range(0,num_classes):
            cmrow.append(np.sum(confusm1[i,:]))
        
        for j in range (0,num_classes):
            cf = []
            for i in range (0,num_classes):
                cadd = confusm1[j,i]/cmrow[j]
                cf.append(cadd) 
            cmnorm[j,:] = cf
        print(f"cmnorm: \n {cmnorm} \n")
                
        
        test_loss.append(test_batch_loss / len(test_batches))       
        test_accuracy.append(test_batch_acc / len(test_batches))  # accuracy
        
        print(
                  f"\n Test Loss: {test_loss[-1]:.3f}\n",
                  f"Test Accuracy: {test_accuracy[-1]:.4f}.",
                  )
        
        if vis:
            
            ax = sn.heatmap(cmnorm, annot=True,cmap= 'RdYlGn_r',fmt=".3g")
            ax.set_xlabel('\n PREDICTED')
            ax.set_ylabel('actual')
            ax.xaxis.set_ticklabels([0,1,2,3,4,5])
            ax.yaxis.set_ticklabels([0,1,2,3,4,5])
            plt.show()
                       
           
        results = {
               "test_loss": test_loss,
               "test_accuracy": test_accuracy,
               "loss":loss,
               }
        return results
            



if __name__ == "__main__":
    conf = {
        "epochs": 1,
        "learning_rate": 0.001,
        "batch_size": 128,
        "f_s": 2_000,
        "f_c": 9.4e10,
        "signal_duration": 0.3,
    }
    
    num_classes = 6
    batch_size = 128
    f_s = 2_000   #10_000
    totalset_size = 500
    drone_classes = [0,1,2,3,4,5]
    totsize = totalset_size*len(drone_classes)
    windowsize = 512
    random_seed = 42
    N = 4
    
    X_asinacos = np.zeros((totsize,2,windowsize,4))
    y_dr = np.zeros((totsize))
    
    
    phct = np.load(total_path + "Amcosproc_swarms3--SNR10--fs1_2000--2999.npy")
    phst = np.load(total_path + "Amsinproc_swarms3--SNR10--fs1_2000--2999.npy")
    dr_clt = np.load(total_path + "dron_class_swarms3--SNR10--fs1_2000--2999.npy")


    phs1 = phst.reshape(totsize,1,windowsize,4)
    phc1 = phct.reshape(totsize,1,windowsize,4)
    y_dr = np.array(dr_clt)

    X_asinacos[:,0:1,:,:] = np.array (phc1) 
    X_asinacos[:,1:2,:,:] = np.array (phs1)
    
    
    test_batches = GetTestLoader(batch_size, totsize, windowsize, N,
                  X_asinacos, y_dr, random_seed, shuffle=True)


    model = STFTANet(num_classes = 6)


    if torch.cuda.is_available():
        device = torch.device('cuda')
        not_device = torch.device('cpu')
    else:
        device = torch.device('cpu')
        not_device = torch.device('cuda')

    model.to(device)


    model_path2 = model_path1 + "m_.(10, 2000, 0.3).pth" 
    results = testclassifier(model_path2, conf=conf, dataset_size=totsize, vis=True)


