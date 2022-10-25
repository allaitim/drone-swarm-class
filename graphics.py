
import numpy as np 
import matplotlib.pyplot as plt


    
def plotlongSTFTamp(hsize,vsize,td,pwr11,pwr12):
    
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(hsize,vsize))
    plt.xlim([min(td),max(td)])
    
    plt.subplot(2,1,1) #col, row_total, row
    plt.plot(td,pwr11)
    plt.ylabel('magnitude')
    
    plt.subplot(2,1,2)
    plt.plot(td,pwr12)
    plt.ylabel('magnitude')
   
    plt.tight_layout()
    plt.xlabel('time')
    plt.show()



def colsplotlongSTFT(hsize,vsize,t,f,amplog11,amplog12,amplog21,amplog22):
    
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(hsize,vsize))
    
    axs[0,0].set_title("long-window STFT")
    axs[0,0].pcolormesh(t, np.fft.fftshift(f), amplog11)
    axs[0,0].set_ylabel("frequency [Hz]")
    axs[0,0].set_xlabel("time [s]")
    
    axs[0,1].set_title("long-window STFT")
    axs[0,1].pcolormesh(t, np.fft.fftshift(f), amplog12)
    axs[0,0].set_ylabel("frequency [Hz]")
    axs[0,0].set_xlabel("time [s]")
    
    axs[1,0].set_title("long-window STFT")
    axs[1,0].pcolormesh(t, np.fft.fftshift(f), amplog21)
    axs[0,0].set_ylabel("frequency [Hz]")
    axs[0,0].set_xlabel("time [s]")
    
    axs[1,1].set_title("long-window STFT")
    axs[1,1].pcolormesh(t, np.fft.fftshift(f), amplog22)
    axs[0,0].set_ylabel("frequency [Hz]")
    axs[0,0].set_xlabel("time [s]")


    fig.tight_layout()
    plt.show()
    
    

def plotcolrow(hsize,vsize,td,pwr1,pwr2,pwr3,pwr4):
    fig,axs = plt.subplots(nrows=2, ncols=2, figsize=(hsize,vsize)) 
    
    
    axs[0,0].plot(td,pwr1,color='C1')
    axs[0,0].set_ylabel("magnitude")
    axs[0,0].set_xlabel("time")
    
    
    axs[0,1].plot(td,pwr2,color='C1')
    axs[0,1].set_ylabel("magnitude")
    axs[0,1].set_xlabel("time")
    
    axs[1,0].plot(td,pwr3,color='C2')
    axs[1,0].set_ylabel("magnitude")
    axs[1,0].set_xlabel("time")
    
    axs[1,1].plot(td,pwr4,color='C2')
    axs[1,1].set_ylabel("magnitude")
    axs[1,1].set_xlabel("time")
    

    fig.tight_layout()
    plt.show()


