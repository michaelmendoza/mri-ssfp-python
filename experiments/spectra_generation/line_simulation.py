import numpy as np
import matplotlib.pyplot as plt
from lib.ssfp import SSFP_SpectrumF as ssfp_sim

def square(l,idx,w=1,val=1):
    '''Create a square function.

    l -- total length of 1D function
    w -- width of function
    idx -- index location of the center of the square
    val -- value of the non-zero samples
    '''

    sig = np.zeros(l,dtype='complex')
    sig[int(idx - w/2):int(idx + w/2)] = val
    return(sig)

def get_const_args(num_samples,T1,T2,phi):
    return({
        'M0': 1.0,
        'Nr': 200,
        'T1': T1,
        'T2': T2,
        'Ns': num_samples,
        'f0': 0.,
        'f1': 250,
        'phi': phi
    })

def get_var_args():
    '''Specific to this dataset.'''

    # Set variable parameters, those that we adjust image to image
    TEs =    [3e-3]*8 + [6e-3]*8 + [12e-3]*8
    dphis =  [ (ii*np.pi/4) for ii in range(8) ]*int((len(TEs)/8))
    alphas = [ np.pi/2 ]*len(TEs)

    # WE DIDN'T GET THE 90 dphi ON THE TE=12ms IMAGES
    # Remove that entry from the previous 3 lists
    TEs = TEs[:-1]
    alphas = alphas[:-1]
    dphis = np.delete(dphis,20) # 20 is the index that lines up with TE=12,dphi=90

    # Check consistency of params
    assert(len(TEs) == len(dphis))

    return({
        'TEs': TEs,
        'dphis': dphis,
        'alphas': alphas
    })

def get_images(const_args,var_args,const_im=True,orthog=False,show=False):

    # Unpack the values we need in the loop
    phi = const_args['phi']
    num_samples = const_args['Ns']
    TEs = var_args['TEs']
    dphis = var_args['dphis']
    alphas = var_args['alphas']

    if const_im:
        num_ims = len(TEs) + 1
    else:
        num_ims = len(TEs)

    # Do the thing
    ims = np.ones((num_samples,num_ims),dtype='complex')*0.3
    for ii in range(len(TEs)):
        # output of ssfp_sim: [0] is frequency axis, [1] is the simulated vectors
        ims[:,ii] = ssfp_sim(**const_args,alpha=alphas[ii],TE=TEs[ii],TR=2*TEs[ii],dphi=dphis[ii]+phi)[1]

    # Try orthogonalizing the basis
    if orthog:
        q,_ = np.linalg.qr(ims)
        ims = q.copy()
        # Didn't seem to do anything?

    if show:
        plt.plot(np.abs(ims))
        plt.title('%d Basis Vectors' % ims.shape[-1])
        plt.show()

    return(ims)

def solve(ims,f,show=False):
    sol = np.linalg.lstsq(ims,f,rcond=None)

    # Grab the coefficients and estimate as well as the rank
    coeffs = sol[0]
    est = ims.dot(coeffs)
    rank = sol[2]
    print('Rank: %g' % rank)

    # Show the result and get MSE
    MSE = (np.abs(est - f)**2).mean()
    if show:
        plt.plot(np.abs(est))
        plt.plot(np.abs(f))
        plt.xlabel('MSE: %g' % MSE)
        plt.show()

    return(est,MSE,coeffs)

if __name__ == '__main__':

    # Set constant parameters, those that don't change image to image
    num_samples = 200
    const_args = get_const_args(num_samples=num_samples,T1=180e-3,T2=180e-3,phi=0)

    # Get each image
    ims = get_images(const_args,get_var_args(),const_im=True,orthog=False,show=True)

    # Define the function to match and solve the least squares problem
    f = square(num_samples,3*num_samples/4,w=30)
    est,MSE,coeffs = solve(ims,f,show=True)

    plt.stem(np.abs(coeffs))
    plt.title('Magnitude of Coefficients')
    plt.show()
