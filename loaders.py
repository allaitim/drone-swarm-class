
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from fastload import FastTensorDataLoader



def get_train_valid_loader(batch_size, totalsize, windowsize, N, 
                X_asinacos, y_dr, random_seed, valid_size, shuffle=True):
    
    
    def shufflearrays(tens1,tens2):
        
        tens11 = np.zeros((totalsize,2,windowsize,N))
        tens12 = np.zeros((totalsize,1))
        
        dataset_len = len(tens1[:])
        indices = list(range(dataset_len))
        

        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(indices)
           
        for i, indx in enumerate(indices):
            tens11[i,:,:,:] =tens1[indx,:,:,:]
            tens12[i,:] = tens2[indx,:]
        
        return  tens11, tens12   

    X_tot, y_tot = shufflearrays(X_asinacos, y_dr)
    y_tot = y_tot.reshape(totalsize)
    
    
    X_train = torch.tensor(X_tot, dtype = torch.float32)
    y_train = torch.tensor(y_tot, dtype=torch.int64)
    
    X_tr ,X_tes, y_tr, y_tes = train_test_split(X_train, y_train, 
                                            test_size=0.2, random_state=42) 



    train_batches = FastTensorDataLoader(X_tr, y_tr,
                                       batch_size=batch_size, shuffle=False)

    test_batches = FastTensorDataLoader(X_tes, y_tes,
                                        batch_size=batch_size, shuffle=False)
    
    return (train_batches, test_batches )



def GetTestLoader(batch_size, totsize, windowsize, N,
                  X_asinacos, y_dr, random_seed, shuffle=True):
    
        
    def shufflearrays(tens1, tens2):
        
        tens11 = np.zeros((totsize, 2, windowsize ,N))
        tens12 = np.zeros((totsize, 1))
        
        dataset_len = len(tens1[:])
        indices = list(range(dataset_len))
        

        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(indices)
           
        for i, indx in enumerate(indices):
            tens11[i,:,:,:] =tens1[indx,:,:,:]
            tens12[i,:] = tens2[indx,:]
        
        return  tens11, tens12   

    X_tot,y_tot = shufflearrays(X_asinacos, y_dr)
    y_tot = y_tot.reshape(totsize)
    
    
    X_tes = torch.tensor(X_tot, dtype = torch.float32)
    y_tes = torch.tensor(y_tot, dtype=torch.int64)
    

    test_batches = FastTensorDataLoader(X_tes, y_tes,
                                        batch_size=batch_size, shuffle=False)
    
    return  test_batches 





