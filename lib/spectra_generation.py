import ssfp
from rawdatarinator.readMeasDataVB15 import readMeasDataVB15 as rmd
import numpy as np
import matplotlib.pyplot as plt
import glob

def get_func(Ns,width,offset=0,amplitude=1):
    sq = np.zeros([ Ns,1 ])
    sq[int(offset-width/2):int(offset+width/2)] = 1
    return(sq)

def rmdTokSpace(files,rawdata=False):
    kspace = None
    for idx,f in enumerate(files):
        if rawdata:
            data = rmd(f,writeToFile=True)
            print('Processed file %d' % idx)
        else:
            data = np.load(f)
            print('Read in file %d' % idx)
        if kspace is None:
            kspace = np.zeros([ data['kSpace'].shape[0],data['kSpace'].shape[1],data['kSpace'].shape[2],data['kSpace'].shape[3],len(files) ],dtype='complex')

        kspace[:,:,:,:,idx] = data['kSpace']
    return(kspace)

def loader(directory='/home/nicholas/Documents/mri-ssfp-matlab/lib/spectral_profiles/data/1212017_SSFP_Spectral_Profile_Phantom_Set2'):
    print('Loading data...')

    # If we have already processed the data, no need to redo the raw data processing
    files = sorted(glob.glob('%s/*.npz' % directory))
    if len(files) > 0:
        print('NPZ files found!')
        # load in those files
        kspace = rmdTokSpace(files)
    else:
        # Find all the .dat files in the directory
        files = glob.glob('%s/*.dat' % directory)
        kspace = rmdTokSpace(files,True)

    return(kspace)

def run(kSpace,
        width=150,
        offset=None,
        TEs=[ 3./1000,6./1000,12./1000,24./1000 ],
        dphis=[ 0,np.pi ],
        alpha = np.pi/2,
        Ns = 500):

    if offset is None:
        offset = int(Ns/2)

    # Arguments for ssfp.SSFP_SpectrumF()
    args = {
        'M0': 1.0,
        'alpha': alpha,
        'phi': 0.0,
        'Nr': 200,
        'T1': 800.0/1000.0,
        'T2': 200.0/1000.0,
        'Ns': Ns,
        'f0': 0,
        'f1': 166
    }

    # Generate the basis
    sim_basis = np.zeros([ args['Ns'],len(TEs)*len(dphis)+1 ],dtype='complex')
    idx = 1
    for TE in TEs:
        for dphi in dphis:
            f,Mc = ssfp.SSFP_SpectrumF(**args,TR=2*TE,TE=TE,dphi=dphi)
            # Tuck Mc away
            sim_basis[:,idx] = Mc
            idx += 1

    sim_basis[:,0] = sim_basis[:,0] + np.mean(np.mean(np.absolute(sim_basis[:,1:]),axis=1))
    plt.subplot(2,1,1)
    plt.plot(np.absolute(sim_basis))
    plt.title('Simulated Basis')

    # Get a filter
    f = get_func(Ns=Ns,width=width,offset=offset)
    plt.subplot(2,1,2)
    plt.plot(f)

    # solve for coefficients and approximate the function
    c = np.linalg.lstsq(sim_basis,f,rcond=None)[0]
    approx = sim_basis.dot(c)

    # check out how well we did
    plt.plot(np.absolute(approx))
    plt.title('Function Approximation')
    plt.show()

    # Now apply the coefficients to the images
    res = np.zeros([ kSpace.shape[0],kSpace.shape[1],kSpace.shape[3] ],dtype='complex')
    for coil in range(kSpace.shape[3]):
        for ii in range(kSpace.shape[4]):
            data = np.squeeze(kSpace[:,:,:,:,ii])

            num_avgs = data.shape[2]
            avg = (np.squeeze(np.sum(data,axis=2))/num_avgs)[:,:,coil]
            imData = np.fft.ifftshift(np.fft.ifft2(avg))

            # Put together the image
            res[:,:,coil] += imData*c[ii]

    final_res = np.sum(res**2,axis=2)
    plt.imshow(np.abs(final_res),cmap='gray')
    plt.show()

    # You really need to to a t1, t2 map to use the simulated data, otherwise use
    # measured data or use a dictionary approach ala MRF

if __name__ == '__main__':
    # Load in all our data
    kSpace = loader()

    # Run with data
    run(kSpace)
