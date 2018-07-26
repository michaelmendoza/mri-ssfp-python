import numpy as np
import ssfp
import scipy
import matplotlib.pyplot as plt

def make_comparable(line,Mc):
    Mc_norm = (Mc - np.min(Mc))
    Mc_norm /= np.max(Mc_norm)

    line_norm = (line - np.min(line))
    line_norm /= np.max(line_norm)

    return(line_norm,Mc_norm)

def obj(x,lines,emit_basis=False):
    T1 = x[0]
    T2 = x[1]
    alpha = x[2]
    phis = x[3:]

    TEs = [ 3./1000,6./1000,12./1000,24./1000 ]
    dphis = [ 0.,np.pi/2 ]

    args = {
        'M0': 1.0,
        'alpha': alpha,
        'Nr': 200,
        'T1': T1,
        'T2': T2,
        'Ns': lines.shape[0],
        'f0': 0.,
        'f1': 250
    }

    Mc = np.zeros(lines.shape,dtype='complex')
    for kk,dphi in enumerate(dphis):
        for ii,TE in enumerate(TEs):
            f,tmp = ssfp.SSFP_SpectrumF(**args,TE=TEs[ii],TR=TEs[ii]*2,dphi=dphis[kk]+phis[ii+len(TEs)*kk],phi=phis[ii+len(TEs)*kk])
            for jj in range(lines.shape[1]):
                Mc[:,jj,ii+len(TEs)*kk] = tmp

    cost = 0
    for ii in range(lines.shape[-1]):
        line0,Mc0 = make_comparable(lines[:,:,ii],Mc[:,:,ii])

        sos = np.sqrt( np.sum( np.abs(line0**2) ,axis=1) )
        fit = np.polyfit(range(len(sos)),sos,1)
        fix = fit[1] + fit[0]*range(len(sos))
        sos_fixed = sos/fix
        sos_fixed /= np.max(sos_fixed)

        mc_sos = np.sqrt( np.sum( np.abs(Mc0**2) ,axis=1) )
        mc_sos /= np.max(mc_sos)

        cost += np.sum(np.abs(sos - mc_sos))
        cost += np.sum(np.abs(np.angle(line0) - np.angle(Mc0)),axis=(0,1))


    if emit_basis:
        return(Mc,cost)
    else:
        return(cost)

def con(x):
    return(x[0] - x[1])


def gen_sim_basis(lines):
    # Find params T1,T2,alpha
    x0 = np.array([ .8,.5,np.pi/3 ] + [1.5]*lines.shape[-1])
    bnds = [ (0,2),(0,1),(0,np.pi/2) ] + [(0,6)]*lines.shape[-1]

    f = lambda x: obj(x,lines)
    x = scipy.optimize.minimize(f,x0,bounds=bnds,constraints={ 'type':'ineq','fun':con })['x']
    print(x)

    Mc,cost = obj(x,lines,emit_basis=True)

    for ii in range(lines.shape[-1]):
        line0,Mc0 = make_comparable(lines[:,:,ii],Mc[:,:,ii])

        sos = np.sqrt( np.sum( np.abs(line0**2) ,axis=1) )
        fit = np.polyfit(range(len(sos)),sos,1)
        fix = fit[1] + fit[0]*range(len(sos))
        sos_fixed = sos/fix
        sos_fixed /= np.max(sos_fixed)

        mc_sos = np.sqrt( np.sum( np.abs(Mc0**2) ,axis=1) )
        mc_sos /= np.max(mc_sos)

        plt.plot(sos_fixed)
        plt.plot(mc_sos)
        plt.show()

    return(Mc)

if __name__ == '__main__':
    pass
