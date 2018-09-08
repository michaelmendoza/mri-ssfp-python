import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob
from lib.coil import calculate_csm_inati_iter
from experiments.spectra_generation.line_simulation import *
from scipy.optimize import minimize

# Add mr_utils
import sys
sys.path.append('/home/nicholas/Documents/mr_utils')
from load_data import load_raw

def loader(use='bart'):
    '''Specific to current dataset.'''

    if use == 'bart' or use == 'rdi':
        RO = 1024
    else:
        RO = 512

    # files = [ 'data/meas_MID362_TRUFI_STW_TE3_FID29379.dat','data/meas_MID363_TRUFI_STW_TE3_dphi_45_FID29380.dat' ]
    files = sorted(glob.glob('data/*.dat'))

    ims = np.zeros((len(files),RO,256,4,16),dtype='complex')
    for ii,file in enumerate(files):
        ims[ii,:,:,:,:] = load_raw(file,use=use,s2i_ROS=True)*100000

    return(ims)

def squash(im):
    # Average it out
    im = np.mean(im,axis=3)

    # Get the coil maps, start with coil images
    coil_images = np.fft.fftshift(np.fft.ifft2(im,axes=(0,1))).transpose((2,1,0)) # expects axes in reverse order

    # Inati Iterative CSM, port from Gadgetron, default 5 iters seems to be enough
    coil_map,coil_combined = calculate_csm_inati_iter(coil_images,smoothing=5,niter=5,thresh=1e-3,verbose=False)
    coil_combined = coil_combined.transpose((1,0)) # get our axis convention back

    # # Walsh iterative method - not as good as Inati coil estimation
    # csm_est,rho = calculate_csm_walsh(coil_images,smoothing=5,niter=5)
    # coil_combined = np.sum(csm_est*coil_images,axis=0)
    # coil_combined = coil_combined.transpose((1,0))

    return(coil_combined)

def get_edges(im):
    center_col = int(im.shape[1]/2)
    changes = np.abs(im[:,center_col]) - np.roll(np.abs(im[:,center_col]),1)
    edges = np.argsort(np.abs(changes))[-2::]
    return(edges[1],edges[0])
    # return(im[edges[1]-pad:edges[0]+pad,:])

def preprocess():
    # Load in the dataset in question
    ims_unprocessed = loader(use='s2i')

    # Remove averages, coils, empty space
    ims = None
    for ii in range(ims_unprocessed.shape[0]):
        # Squash the averages,coils
        tmp = squash(ims_unprocessed[ii,:,:,:,:])

        if ims is None:
            # Trim down to just the region of interest
            first,last = get_edges(tmp)
            tmp = tmp[first:last,:]
            ims = np.zeros((ims_unprocessed.shape[0],tmp.shape[0],tmp.shape[1]),dtype='complex')
            ims[ii,:,:] = tmp
        else:
            ims[ii,:,:] = tmp[first:last,:]

    # Save for later use
    np.save('data/measured_ims.npy',ims)

def extract_center_lines(ims,pad=5,lpf=True,show=False):

    center_col = int(ims.shape[2]/2)
    center_lines = np.fft.fftshift(np.fft.fft(ims[:,:,center_col].T,axis=0))

    # Low-pass filter
    if lpf:
        center_row = int(center_lines.shape[0]/2)
        pad = 50
        center_lines[center_row-pad:center_row+pad+1,:] = 0

    center_lines = np.fft.ifft(center_lines,axis=0)

    if show:
        plt.plot(np.abs(center_lines))
        plt.show()

    return(center_lines)

def demo_fft_question(ims):
    '''Show that phase data changes when fft,ifft applied in succession.'''

    center_col = int(ims.shape[2]/2)
    center_lines_without_fft = ims[:,:,center_col].T
    center_lines_with_fft = np.fft.ifft(np.fft.fftshift(np.fft.fft(ims[:,:,center_col].T,axis=0)),axis=0)

    plt.subplot(1,3,1)
    plt.plot(np.abs(center_lines_without_fft[:,0]))
    plt.plot(np.angle(center_lines_without_fft[:,0]))
    plt.title('TE=3ms,dphi=0, No FFT')

    plt.subplot(1,3,2)
    plt.plot(np.abs(center_lines_with_fft[:,0]))
    plt.plot(np.angle(center_lines_with_fft[:,0]))
    plt.title('TE=3ms,dphi=0, IFFT(FFTSHIFT(FFT))')

    plt.subplot(1,3,3)
    plt.plot(np.angle(center_lines_without_fft[:,0]),label='No FFT')
    plt.plot(np.angle(center_lines_with_fft[:,0]),label='FFT')
    plt.legend()
    plt.title('Comparision of Phase')
    plt.show()

def apply_coeffs(ims,coeffs):
    # Correct phase in images - same issue as before...
    ims = np.fft.ifft2(np.fft.fftshift(np.fft.fft2(ims,axes=(1,2))),axes=(1,2))

    # Apply the coefficients
    filtered_im = ims.transpose(1,2,0).dot(coeffs)
    return(filtered_im)

def add_const_im(ims):
    const_im = np.ones((1,ims.shape[1],ims.shape[2]),dtype='complex')*.1
    ims = np.append(ims,const_im,axis=0)
    return(ims)

def get_D(x,A):
    # Penalties
    if np.any(x < 0):
        return(1e8)
    if x[0] < x[1]:
        return(1e8)

    D = np.zeros(A.shape,dtype='complex')
    idx = 0
    for TE in [ 3e-3,6e-3,12e-3 ]:
        TR = TE/2
        fs = np.linspace(0,500,A.shape[0])
        for dphi in [ (ii*np.pi/4) for ii in reversed(range(8)) ]:
            if (TE == 12e-3) and (dphi == 2*np.pi/4):
                # print('Skipping')
                pass
            else:
                f0 = f(x,fs,TE,dphi,phi=0)
                D[:,idx] = f0/np.linalg.norm(f0) - A[:,idx]/np.linalg.norm(A[:,idx])
                idx += 1
    return(D)

def f(x,fs,TE,dphi,phi=0,M0=1):
    T1,T2,alpha = x[:]
    TR = 2*TE
    beta = 2*np.pi*fs*TR
    theta = beta - (dphi + phi)
    Mbottom = (1 - np.exp(-TR/T1)*np.cos(alpha))*(1 - np.exp(-TR/T2)*np.cos(theta)) - np.exp(-TR/T2)*(np.exp(-TR/T1) - np.cos(alpha))*(np.exp(-TR/T2) - np.cos(theta))
    Mx = M0 * (1 - np.exp(-TR/T1))*np.sin(alpha)*(1 - np.exp(-TR/T2)*np.cos(theta))/Mbottom
    My = M0 * (1 - np.exp(-TR/T1))*np.exp(-TR/T2)*np.sin(alpha)*np.sin(theta)/Mbottom
    Mc = Mx + 1j*My
    Mc = Mc*np.exp(1j*beta*(TE/TR))*np.exp(-TE/T2)
    return(Mc)

def get_sim_basis(ims,niter=100):

    # A = extract_center_lines(ims,pad=0,lpf=False,show=False)
    A = np.zeros((ims.shape[1],ims.shape[0]),dtype='complex')
    cntr = int(ims.shape[1]/2)
    for ii in range(ims.shape[0]):
        A[:,ii] = ims[ii,:,cntr]

    # x = [ T1,T2,alpha ]
    t1 = []
    t2 = []
    alpha = []
    for ii in range(20):
        x0 = np.random.random(3)
        sol = minimize(lambda x: np.linalg.norm(get_D(x,A))**2,x0,method='Nelder-Mead')
        t1.append(sol['x'][0])
        t2.append(sol['x'][1])
        alpha.append(sol['x'][2])
        print(sol)

        fs = np.linspace(0,500,A.shape[0])
        f0 = f(sol['x'],fs,3e-3,0,phi=0)
        plt.plot(np.abs(f0/np.linalg.norm(f0)))
        plt.plot(np.abs(A[:,0]/np.linalg.norm(A[:,0])))
        plt.show()

    plt.subplot(3,1,1)
    plt.hist(t1,bins=10)
    plt.subplot(3,1,2)
    plt.hist(t2,bins=10)
    plt.subplot(3,1,3)
    plt.hist(alpha,bins=10)
    plt.show()

def write_movie_freq_sweep(ims,center_lines,freqs,w=10):

    def updatefig(frame,*fargs):
        f = square(num_samples,freqs[frame],w=w)
        est,MSE,coeffs = solve(center_lines,f,show=False)
        filtered_im = apply_coeffs(ims,coeffs)
        im.set_array(np.abs(filtered_im))
        plt.imsave('freq_sweep/test%03d.png' % frame,np.abs(filtered_im),cmap='gray')
        return im,

    fig = plt.figure()
    f = square(num_samples,freqs[0],w=30)
    est,MSE,coeffs = solve(center_lines,f,show=False)
    filtered_im = apply_coeffs(ims,coeffs)
    im = plt.imshow(np.abs(filtered_im),cmap='gray',vmin=0,vmax=1,animated=True)

    ani = animation.FuncAnimation(fig,updatefig,interval=0,frames=len(freqs),repeat=False,blit=True)
    plt.show()

if __name__ == '__main__':

    # Preprocessing takes a while... Only run once if you can help it
    # preprocess()

    # Load in preprocessed data, shape=(num_ims,rows,cols)
    ims = np.load('data/measured_ims_scaled.npy')

    # Add a constant image to end
    ims = add_const_im(ims)

    # Generate simulated basis vectors
    # center_lines = get_sim_basis(ims)


    # Extract center lines to form basis vecors, I don't think the LPF does much
    center_lines = extract_center_lines(ims,pad=0,lpf=False,show=False)
    num_samples = center_lines.shape[0]

    # Some strange behavior to ask about...
    # Might be from coil sensistivy maps and subsequent coil combination?
    # demo_fft_question(ims)

    # Do a sweep of frequencies and write the result as a movie file
    # write_movie_freq_sweep(ims,center_lines,freqs=range(20,num_samples-20),w=20)

    # Solve for the coefficients and filter the image
    f = square(num_samples,5*num_samples/8,w=30)
    est,MSE,coeffs = solve(center_lines,f,show=False)
    filtered_im = apply_coeffs(ims,coeffs)

    plt.stem(np.abs(coeffs))
    plt.title('Magnitude of Coefficients')
    plt.show()

    # Show me what you got
    plt.subplot(2,1,1)
    plt.imshow(np.rot90(np.abs(filtered_im)),cmap='gray')
    plt.subplot(2,1,2)
    center = int(filtered_im.shape[1]/2)
    plt.plot(np.abs(filtered_im[:,center]))
    plt.plot(np.abs(center_lines.dot(coeffs)))
    plt.plot(np.abs(f))
    plt.show()
