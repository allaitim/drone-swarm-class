
import numpy as np
from calculator import generateData1, calculatorSTFTAcAs, psigenerator3 
from data import drones1
from graphics import plotlongSTFTamp, colsplotlongSTFT, plotcolrow
   



def generate_item(index, f_s1, sample_length, SNR, drone_class, windowsize):
    
    c = 2.998e8  # speed of light in m/s
    scenario = {
            "SNR": SNR, 
            "R": np.random.uniform(low=1000, high=5000),
            "A_r": np.random.chisquare(4),
            "thetran" : np.random.uniform(low=np.pi/16, high=np.pi/2),
            "fip" : np.random.uniform(low=0, high=np.pi/4),
            "lamb": 0.02998,
            "f_c": c/0.02998,
    }

    
    p = psigenerator3(**dict(scenario, **drones1[drone_class]))
    x1, yr1, yi1, ys1 = generateData1(p, f_s, sample_length)
    Lampl, Acos, Asin, f, t = calculatorSTFTAcAs(p, f_s1, sample_length, 
                                             windowsize, offset=False)

    return Lampl, Acos, Asin, f, t, drone_class
 
   


def GenerateImage(f_s1, sample_length, totalset_size, SNR,windowsize,
                                             drone_classes):  
    
    lampl1, Acos, Asin, f1, t1, drone_number = generate_item(1, f_s1, sample_length,
                                                 SNR, 0, windowsize)
    Famp11 = np.zeros((totalset_size*len(drone_classes),windowsize,len(t1)))
    Amcos = np.zeros((totalset_size*len(drone_classes),windowsize,len(t1)))
    Amsin = np.zeros((totalset_size*len(drone_classes),windowsize,len(t1)))
    drones_tot = np.zeros((totalset_size*len(drone_classes),1))
    
    for drone_number in (drone_classes):
       
            for x in range(totalset_size):
                x1 = totalset_size*drone_number + x
                lampl, Acs, Asn, f, t, drone_number = generate_item(x1,f_s1,
                             sample_length, SNR, drone_number, windowsize)             

                drones_tot[x1,:] = np.array(drone_number) 
                Famp11[x1,:,:] = np.array(lampl)
                Amcos[x1,:,:] = np.array(Acs)
                Amsin[x1,:,:] = np.array(Asn)
            
    np.save (totalset_path + 'Amcosproc_'+ 'swarms' + str(swarms)+'--'+'SNR'+ str(SNR)+ '--' + 'fs1_'+str(f_s1)+'--'+ str(x1), Amcos)
    np.save (totalset_path + 'Amsinproc_' + 'swarms' + str(swarms)+ '--'+ 'SNR'+str(SNR)+'--'+ 'fs1_'+str(f_s1)+'--' +str(x1), Amsin)
    np.save (totalset_path + 'dron_class_'+ 'swarms' + str(swarms) + '--'+'SNR'+str(SNR)+'--' + 'fs1_'+str(f_s1)+'--' +str(x1), drones_tot)
    np.save (totalset_path + 't_' + 'swarms' + str(swarms) +'--'+ 'SNR'+str(SNR)+'--' + 'fs1_'+str(f_s1)+'--' +str(x1), t)        
    np.save (totalset_path + 'f_' + 'swarms' + str(swarms) + '--' + 'SNR'+str(SNR)+'--' + 'fs1_'+str(f_s1)+'--' +str(x1), f)        
    
    return Amcos, Amsin, drones_tot, t, f 







f_s = 2000   
totalsetsize = 2_500 
testtotalsize = 500    #400   #1000  #2000  #400  #2000  #400  #2000  #400  #2000       #400         #2_000  # per denomination
drone_classes = [0,1,2,3,4,5]
windowsize = 512
sample_length = 0.3   
SNR = 10 
swarms = 3

        
Amcos,Amsin,drones_tot,t,f = GenerateImage(f_s, sample_length, totalsetsize,
                                                SNR, windowsize, drone_classes) 


ph11 = np.zeros((windowsize,len(t)))
ph12 = np.zeros((windowsize,len(t)))
ph21 = np.zeros((windowsize,len(t)))
ph22 = np.zeros((windowsize,len(t)))


phs11 = np.zeros((windowsize,len(t)))
phs12 = np.zeros((windowsize,len(t)))
phs21 = np.zeros((windowsize,len(t)))
phs22 = np.zeros((windowsize,len(t)))


ph11[:,:] = Amcos[500:501,:,:] 
ph12[:,:] = Amcos[6000:6001,:,:] 

ph21[:,:] = Amcos[1000:1001,:,:] 
ph22[:,:] = Amcos[14000:14001,:,:] 

phs11[:,:] = Amsin[500:501,:,:] 
phs12[:,:] = Amsin[6000:6001,:,:] 

phs21[:,:] = Amsin[1000:1001,:,:] 
phs22[:,:] = Amsin[14000:14001,:,:] 


sample_length1 = 1
N = int(windowsize*sample_length1)
td = np.arange(N)/windowsize


colsplotlongSTFT(14, 10, t, f, ph11, phs11, ph12, phs12)
    
plotcolrow(10, 10, td, ph11, ph12, ph21, ph22)

plotlongSTFTamp(14,10, td, phs11, phs12)



Amcos,Amsin,drones_tot,t,f = GenerateImage(f_s, sample_length, testtotalsize,
                                                SNR, windowsize, drone_classes) 

