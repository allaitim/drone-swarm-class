import math
import cmath
import numpy as np
from scipy import signal

j = complex(0, 1)  # sqrt negative 1


def sinc(x):
    if(x == 0):
        return 1  # this makes it continuous
    return math.sin(x) / x



def psigenerator3(f_c, lamb,RD,Thet,swarm_amount, N, L_1, L_2, f_rot, SNR=None,
                                           deterministic=False, **kwargs):
   
    
    sign = lambda x:math.copysign(1,x)
    
    if deterministic:
        A_r = 1
        R = 0
        #theta = 0
        V_rad = 0

    else:
        A_r = np.random.chisquare(4)  
        R = np.random.uniform(low=1000, high=5000)
        thetran = np.random.uniform(low=np.pi/16, high=np.pi/2)
        fip = np.random.uniform(low=0, high=np.pi/4)
        V_rad = 0
        

    def ex(t,rdist):
        rsum = rdist + 1000
        ex =  cmath.exp(j*(2*math.pi*f_c*t - (4*math.pi/lamb)*(R+rsum+V_rad*t)))
        prephase = A_r * ex
        return prephase

    def ampterm(t,th):
        accum = complex(0, 0)
        theta = thetran + th
        for n in range(N):
            arg = 2*math.pi*(f_rot*t+(n/N))
            exponential = cmath.exp(-j*(4*math.pi/lamb)*(((L_1 + L_2)/2) *
                                                math.cos(theta)*math.sin(arg)))
            sincterm = sinc(((4*math.pi)/lamb)*((L_2 - L_1)/2) *
                            math.cos(theta)*math.sin(arg))
            signterm = ((math.sin(abs(theta)+fip) + math.sin(abs(theta)-fip)) + 
                    sign(theta)*(math.sin(abs(theta)+fip) - 
                                 math.sin(abs(theta)-fip)))*math.cos(arg)
            
            accum += exponential*sincterm*signterm
            return accum
        
    def fi(t):
        accum = complex(0,0)
        alt = complex(0,0)
        prephase = complex(0,0)
        if swarm_amount == 1:
            prephase = ex(t,RD)
            accum = ampterm(t,Thet)
            alt += accum*prephase
        else:    
            for i,rd in enumerate(RD): 
                prephase = ex(t,rd)
                for k,th in enumerate(Thet): 
                    if k==i:
                        accum = ampterm(t,th)
                        alt += accum*prephase
        return alt                        
    
                 

    if not SNR is None:
        # this is a re-arrangement of dB = 10\log_{10}{A_r^2/\sigma^2}
        variance = 10**(2*math.log10(A_r) - (SNR/10))

        def fuzzypsi(t):
            real_noise = np.random.normal(0, math.sqrt(variance))
            imag_noise = np.random.normal(0, math.sqrt(variance))
            return fi(t) + real_noise + imag_noise *j

        return fuzzypsi

    else: 
        return fi
    
    

def noisegenerator(f_c, lamb, N, L_1, L_2, SNR=None,
                                           deterministic=False, **kwargs):
    """
    This function returns a psi function, which represents the RADAR signal off
    of a drone.
    :param SNR: Signal-to-noise ratio, given in dB. If None, don't add noise
    """
    if deterministic:
        A_r = 1
        R = 0
        theta = 0

    else:
        A_r = np.random.chisquare(4)  # A_r is a random value from X^2 with 4 dof
        R = np.random.uniform(low=1000, high=5000)
        theta = np.random.uniform(low=0, high=np.pi/2)

    def noise(t):
        prefactor = A_r * \
            cmath.exp(j*(2*math.pi*f_c*t - (4*math.pi/lamb)*R))
        accum = complex(0, 0)
        for n in range(N):
            exponential = cmath.exp(-j*(4*math.pi/lamb)*(((L_1 + L_2)/2) *
                                         math.cos(theta)*math.sin(2*math.pi*n/N)))
            sincterm = sinc(((4*math.pi)/lamb)*((L_2 - L_1)/2) *
                            math.cos(theta)*math.sin(2*math.pi*n/N))
            accum += exponential*sincterm

        return prefactor * accum

    if not SNR is None:
        variance = 10**(2*math.log10(A_r) - (SNR/10))

        def fuzzynoise(t):
            real_noise = np.random.normal(0, math.sqrt(variance))
            imag_noise = np.random.normal(0, math.sqrt(variance))
            return noise(t) + real_noise + imag_noise *j
        
        def fuzzynoise1(t):
            real_noise = np.random.normal(0, math.sqrt(variance))
            imag_noise = np.random.normal(0, math.sqrt(variance))
            return real_noise + imag_noise *j

        return fuzzynoise, fuzzynoise1 

    else:  
        return noise
  


def generateData1(fi, f_s, sample_length, offset=False):
   
    xs = []
    ys = []
    real_ys = []
    imaginary_ys = []
    offset_val = 0
    if offset:
        # offset a sample anywhere from 0 to 1 seconds
        offset_val = np.random.uniform(0, 1)
    for i in range(int(f_s*sample_length)):
        x = i/f_s
        xs.append(x)
        val = fi(offset_val + x)
        vri = val.real + val.imag*j
        real_ys.append(val.real)
        imaginary_ys.append(val.imag)
        ys.append(vri)
    return xs, real_ys, imaginary_ys, ys



def calculatorSTFTAcAs(fi, f_s1, sample_length,long_window_size, offset=False):
    
    xs, r_ys, i_ys, ys = generateData1(fi, f_s1, sample_length, offset=True)
    
    Amp = [abs(numb)for numb in ys]

    f,t,Zreal = signal.stft(r_ys, f_s1, window='hamming', nperseg=long_window_size,
                     noverlap=long_window_size//2, return_onesided=False)    
    Acos = 20*np.log10(np.abs(np.fft.fftshift(Zreal, axes=0)))
       
    f,t,Zimag = signal.stft(i_ys, f_s1, window='hamming', nperseg=long_window_size,
                     noverlap=long_window_size//2, return_onesided=False)
    Asin = 20*np.log10(np.abs(np.fft.fftshift(Zimag, axes=0)))
    
    
    f, t, Lamp = signal.stft(Amp, f_s1, window='hamming', nperseg=long_window_size,
                            noverlap=long_window_size//2, return_onesided=False)
    Lampl = (20*np.log10(np.abs(np.fft.fftshift(Lamp, axes=0))))
   
    return Lampl,Acos,Asin,f,t


