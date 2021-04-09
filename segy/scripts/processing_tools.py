import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def phase_vel_spectrum(data,dx,dt,fmax,vmin,vmax,side=0,f_abs=0):

    f    = np.fft.fftshift(np.fft.rfft2(data.T),axes=0)
    if f_abs == 0:
        f  = 20.0*np.log10(abs(f))
        f -= f.max()
    elif f_abs == 1:
        f = np.angle(f)

    freq = np.fft.rfftfreq(data.shape[0],d=dt)
    k    = np.fft.rfftfreq(data.shape[1],d=dx)
    kmax = k[-1]

    # Find max frequency index to plot
    ifmax = [i for i,x in enumerate(freq>fmax) if x][0]
    fmax_plot  = freq[ifmax]

    dk = (k[-1] - k[0])/len(k)
    k_ax = np.arange(dk,k[-1],dk)
    mid_k = len(k_ax)

    v_fmin = freq[1]/k_ax
    v_fmax = freq[-1]/k_ax

    vnew   = np.arange(vmin,vmax,5)

    fv = np.zeros((len(vnew),len(freq[:ifmax])))
    for i,w in enumerate(freq[:ifmax]):
        if side==0:
            tmp   = f[-len(k_ax):,i]
        elif side==1:
            tmp   = f[mid_k:mid_k-len(k_ax):-1,i]
        v     = w/k_ax
        vlow  = max(vmin,v.min())
        vhigh = min(vmax,v.max())
        ind   = (v<=vhigh) & (v>=vlow)
        if sum(ind)>2:
            int_func = interp1d(v[ind],tmp[ind],fill_value='extrapolate')
            a = int_func(vnew)
            fv[:,i] = int_func(vnew)

    mask = np.ones_like(fv)
    for i in range(len(vnew)):
        for j in range(len(freq[:ifmax])):
            if (freq[j] != 0):
                tmp = 0.5*vnew[i]/freq[j]
                if (tmp < dx):
                    mask[i,j] = 0
    fv = mask*fv

    return fv,vnew,freq[:ifmax]


def fk_spectrum(dataset,dx,dt):

    f    = np.fft.fftshift(np.abs(np.fft.rfft2(dataset.T)),axes=0)
    #f    = 20.0*np.log10(f)
    #f   -= f.max()
    freq = np.fft.rfftfreq(dataset.shape[0],d=dt)
    k    = np.fft.rfftfreq(dataset.shape[1],d=dx)

    return f,k,freq

def fk_view(dataset,dx,dt,clip=-30):
    f,k,freq = fk_spectrum(dataset,dx,dt)

    f    = 20.0*np.log10(f)
    f   -= f.max()
    plt.figure()
    plt.imshow(f.T, aspect='auto', vmin=clip,vmax=0,
               extent=[-1*k[-1], k[-1], freq[-1], freq[0]])
    plt.colorbar()

def fk_design(dataset,dx,dt,v1,smolen,v2=0):

    f,kpos,freq = fk_spectrum(dataset,dx,dt)
    kmax = kpos[-1]

    # Negative k-axis: exclude k=0, reverse order and flip sign
    mid = len(kpos)-1
    if (np.mod(f.shape[0],2) == 0) :
        # Even number of samples along the x-direction
        kneg = kpos[1:-1][::-1]
    else:
        # Odd number of samples along the x-direction
        kneg   = kpos[1:][::-1]

    k_axis = np.hstack([kneg,kpos])[:, None]
    one    = np.ones_like(f)
    row    = freq*one
    column = k_axis*one

    # Ignore divide by zero warnings
    with np.errstate(invalid='ignore',divide='ignore'):
        m  = np.divide(row,column)
        if v2 == 0:
            m = 1.*(m > v1)
        else:
            m1 = 1.*(m > v1)
            m2 = 1.*(m < v2)
            m  = m1+m2
            m  = np.where(m<2,0,1)

    vec= np.ones(smolen)/(smolen*1.0)
    smoothed_m = np.apply_along_axis(lambda m: np.convolve(m, vec, mode='same'), axis=0, arr=m)
    valid = smoothed_m.shape[0]
    m[:valid, :] = smoothed_m
    #plt.figure()
    #plt.imshow(m.T, aspect='auto', extent=[-1*kmax, kmax, freq[-1], freq[0]])
    #plt.colorbar()
    z = m.copy()

    return z

def fk_filter(dataset, fk_filter):
    nt,nx  = dataset.shape
    #delta  = abs(nt - fk_filter.shape[1])
    # FK spectrum of data
    f = np.fft.fftshift(np.fft.rfft2(dataset.T),axes=0)
    # Element-wise multiply with data
    fkfilt = f*fk_filter
    # FFT back to TX
    result = np.fft.irfft2(np.fft.ifftshift(fkfilt,axes=0)).T
    # Return fk filtered spectrum
    fkfilt = np.abs(f*fk_filter)

    return result,fkfilt

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def bandpass(dataset, **kwargs):

        # Sample rate and desired cutoff frequencies (in Hz).
        fs = 1./kwargs['dt']
        lowcut = kwargs['lowcut']
        highcut = kwargs['highcut']

        dataset['trace'] = butter_bandpass_filter(np.fliplr(dataset['trace']), lowcut, highcut, fs, order=3)
        dataset['trace'] = butter_bandpass_filter(np.fliplr(dataset['trace']), lowcut, highcut, fs, order=3)
        return dataset

