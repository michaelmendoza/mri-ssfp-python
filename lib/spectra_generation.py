import ssfp
from rawdatarinator.readMeasDataVB15 import readMeasDataVB15 as rmd
import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.optimize import minimize

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

def SSFP_data_formatter(alpha,Ns,T1,T2):
    return({
        'M0': 1.0,
        'alpha': alpha,
        'phi': -1.2,
        'Nr': 200,
        'T1': T1,
        'T2': T2,
        'Ns': Ns,
        'f0': 0,
        'f1': 250
    })

def gen_sim_basis(args,TEs,dphis,orthog=False):
    sim_basis = np.zeros([ args['Ns'],len(TEs)*len(dphis)+1 ],dtype='complex')
    idx = 1
    for dphi in dphis:
        for TE in TEs:
            f,Mc = ssfp.SSFP_SpectrumF(**args,TR=2*TE,TE=TE,dphi=dphi)
            # Tuck Mc away
            sim_basis[:,idx] = Mc
            idx += 1

    # Add constant basis vector with value to be the mean of all the others
    sim_basis[:,0] = sim_basis[:,0] + np.mean(np.mean(np.absolute(sim_basis[:,1:]),axis=1))

    if orthog:
        print('Orthog not implemented yet, sorry about that...')

    return(sim_basis)

def solve_coeffs(sim_basis,f):
    c = np.linalg.lstsq(sim_basis,f,rcond=None)[0]
    approx = sim_basis.dot(c)
    return(c,approx)

def apply_coeffs(kSpace,c):
    res = np.zeros([ kSpace.shape[0],kSpace.shape[1],kSpace.shape[3] ],dtype='complex')
    for coil in range(kSpace.shape[3]):
        for ii in range(kSpace.shape[4]):
            data = np.squeeze(kSpace[:,:,:,:,ii])

            num_avgs = data.shape[2]
            avg = (np.squeeze(np.sum(data,axis=2))/num_avgs)[:,:,coil]
            imData = np.fft.ifftshift(np.fft.ifft2(avg))

            # Put together the image
            res[:,:,coil] += imData*c[ii]
    res = np.sum(np.abs(res)**2,axis=2)
    return(res)

def est_obj(x,imData,args,TEs,dphis,show=False):
    args['T1'] = x[0]
    args['T2'] = x[1]
    args['alpha'] = x[2]
    # args['f1'] = 250
    # phi0 = -1.2 # constant phi offset to match up the start

    cost = 0
    center = int(imData.shape[1]/2)
    pad = 10
    # for ii in range(imData.shape[-1]):
    for ii in [ 4 ]:
        prof = np.mean(np.abs(imData[:,center-pad:center+pad,ii]),axis=1)
        prof /= np.max(prof)
        prof[np.abs(prof) < .5*np.mean(prof)] = 0
        prof = prof[prof > 0] # we don't want empty space
        prof = prof[2:-2] # we don't want the edges
        prof -= np.min(prof)
        prof /= np.max(prof)

        args['Ns'] = len(prof)
        sim_basis = gen_sim_basis(args,TEs,np.add(dphis,args['phi']))
        sim_prof = np.abs(sim_basis[:,ii+1]) #+ np.min(np.abs(prof))
        sim_prof /= np.max(sim_prof)

        if show:
            plt.plot(sim_prof,label='Simulated Profile')
            plt.plot(prof,label='Center line')
            plt.title('T1,T2,alpha Estimation')
            plt.xlabel('T1: %g, T2: %g, alpha: %g' % (x[0],x[1],x[2]))
            plt.legend()
            plt.show()

        cost += np.sum(np.sqrt((sim_prof - prof)**2))
    return(cost)

def est_params(kSpace,args,TEs,dphis):
    res = np.zeros([ kSpace.shape[0],kSpace.shape[1],kSpace.shape[3],kSpace.shape[4] ],dtype='complex')
    for coil in range(kSpace.shape[3]):
        for ii in range(kSpace.shape[4]):
            data = np.squeeze(kSpace[:,:,:,coil,ii])
            num_avgs = data.shape[2]
            avg = np.squeeze(np.sum(data,axis=2))/num_avgs
            imData = np.fft.ifftshift(np.fft.ifft2(avg))
            res[:,:,coil,ii] = np.abs(imData**2)

    res = np.sqrt(np.sum(res,axis=2))

    # Look at them to make sure we did alright
    # for ii in range(res.shape[-1]):
    #     plt.imshow(np.abs(res[:,:,ii]),cmap='gray')
    #     plt.show()

    # Set up an optimization problem to find T1,T2
    x0 = np.array([ 0.8,0.3,np.pi/3 ],dtype=np.float32)
    def con(x):
        return(x[0] - x[1])
    x = minimize(lambda x: est_obj(x,res,args,TEs,dphis),x0,bounds=[ (0,2.),(0,1.),(0,np.pi/3) ],constraints={ 'type':'ineq','fun':con })
    print('T1: %g \t T2: %g \t alpha: %g' % (x['x'][0],x['x'][1],x['x'][2]))
    est_obj(x['x'],res,args,TEs,dphis,show=True)

    # x = np.array([ .25,.24,250,-1.2 ],dtype=np.float32)
    # est_obj(x,res,args,TEs,dphis,show=True)

    return(x['x'][0],x['x'][1],x['x'][2])


def plotter(sim_basis,f,approx,res):
    # Show the simulated basis
    plt.subplot(2,1,1)
    plt.plot(np.absolute(sim_basis))
    plt.title('Simulated Basis')
    plt.subplot(2,1,2)
    plt.plot(f,label='Function')
    plt.legend()
    plt.show()

    # check out how well we did
    plt.subplot(3,1,1)
    plt.imshow(np.transpose(np.abs(res)),cmap='gray')
    plt.title('Filtered SOS Image')
    plt.subplot(3,1,2)
    plt.plot(np.absolute(approx),label='Function Approx.')
    plt.legend()
    plt.subplot(3,1,3)
    plt.plot(np.abs(res[:,int(res.shape[1]/2)]),label='center line')
    plt.legend()
    plt.show()

def run(kSpace,
        width=150,
        offset=None,
        TEs=[ 3./1000,6./1000,12./1000,24./1000 ],
        dphis=[ 0,np.pi ],
        alpha=np.pi/3,
        Ns=500):

    if offset is None:
        offset = int(Ns/2)


    # Arguments for ssfp.SSFP_SpectrumF()
    args = SSFP_data_formatter(alpha,Ns,800./1000,200./1000)

    # Estimate T1,T2
    T1,T2,alpha = est_params(kSpace,args.copy(),TEs,dphis)
    args['T1'] = T1
    args['T2'] = T2
    args['alpha'] = alpha

    # Generate the basis
    sim_basis = gen_sim_basis(args,TEs,dphis)

    # Get a filter
    f = get_func(Ns=Ns,width=width,offset=offset)

    # solve for coefficients and approximate the function
    c,approx = solve_coeffs(sim_basis,f)


    ## TODO: Make sure coefficients and images line up!
    # Now apply the coefficients to the images
    res = apply_coeffs(kSpace,c)

    # You really need to to a t1, t2 map to use the simulated data, otherwise use
    # measured data or use a dictionary approach ala MRF

    # Show me the money
    plotter(sim_basis,f,approx,res)

if __name__ == '__main__':
    # Load in all our data
    kSpace = loader()

    # Run with data
    run(kSpace)
    # get_imData(kSpace)
