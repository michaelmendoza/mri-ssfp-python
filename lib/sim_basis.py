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
            idx = ii + len(TEs)*kk
            phi = phis[idx]
            f,tmp = ssfp.SSFP_SpectrumF(**args,TE=TE,TR=2*TE,dphi=dphi+phi,phi=phi)
            for jj in range(lines.shape[1]):
                Mc[:,jj,idx] = tmp

    cost = 0
    for ii in range(lines.shape[-1]):
        # line0,Mc0 = make_comparable(lines[:,:,ii],Mc[:,:,ii])

        sos_line = np.sqrt( np.sum( np.abs(lines[:,:,ii]**2) ,axis=1) )
        fit = np.polyfit(range(len(sos_line)),sos_line,1)
        fix = fit[1] + fit[0]*range(len(sos_line))
        sos_fixed = sos_line/fix
        sos_fixed -= np.min(sos_fixed)
        sos_fixed /= np.max(sos_fixed)

        mc_sos = np.sqrt( np.sum( np.abs(Mc[:,:,ii]**2) ,axis=1) )
        mc_sos /= np.max(mc_sos)

        # Could also consider phase?
        cost += np.sum(np.abs(np.abs(sos_fixed) - np.abs(mc_sos)))

    if emit_basis:
        return(Mc,cost)
    else:
        return(cost)

def con(x):
    return(x[0] - x[1]) # enforce T1 > T2


def gen_sim_basis(lines):
    # Find params T1,T2,alpha
    x0 = np.array([ .8,.5,np.pi/3 ] + [2]*lines.shape[-1])
    bnds = [ (0,2),(0,1),(0,np.pi/2) ] + [(0,6)]*lines.shape[-1]

    f = lambda x: obj(x,lines)
    cons = { 'type':'ineq','fun':con }
    opts = { 'disp':True }
    x = scipy.optimize.minimize(f,x0,bounds=bnds,constraints=cons,options=opts)['x']
    print('T1: %g \n T2: %g \n alpha: %g \n phis: ' % (x[0],x[1],x[2]),end='')
    print(x[3:])

    Mc,cost = obj(x,lines,emit_basis=True)
    return(Mc)

if __name__ == '__main__':
    pass
