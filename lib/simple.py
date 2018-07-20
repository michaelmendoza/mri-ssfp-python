import numpy as np
from spectra_generation import loader
import matplotlib.pyplot as plt

def simple_csm(imdata,rows=None):
    if rows is None:
        rows = range(imdata.shape[0])

    csm = np.zeros([ len(rows),imdata.shape[1],imdata.shape[2],imdata.shape[3] ],dtype='complex')
    for ii in range(imdata.shape[-1]):
        # Get SOS recon for each image index
        sos = np.zeros([ len(rows),imdata.shape[1] ])
        for c in range(imdata[:,:,:,ii].shape[2]):
            sos += np.abs(imdata[rows,:,c,ii]**2)
        sos = np.sqrt(sos)

        # Divide SOS recon by each coil to get sensitivity map
        for c in range(imdata[:,:,:,ii].shape[2]):
            csm[:,:,c,ii] = imdata[rows,:,c,ii]/sos

    return(csm)

def get_imspace(kspace):
    '''Give back image space data.

    Inputs:
        kspace -- Array from rawdatarinator with size
                  (rows,cols,avgs,coils,idx).

    Outputs:
        imdata -- Image space data with size (rows,cols,coils,idx).
    '''
    # rows,cols,avgs,coils,idx = kspace.shape
    imdata = np.fft.ifftshift(np.fft.ifft2(np.mean(kspace,axis=2),axes=[ 0,1 ]))
    return(imdata)

def trim_to_center_line(imdata):
    '''Gives back only center line of image space data.

    Inputs:
        imdata -- Image space data with size (rows,cols,coils,idx).

    Outputs:
                 (rows,1,coils,idx).
                 lines -- Image space slices through the center with size
    '''
    center_idx = int(imdata.shape[1]/2)
    lines = imdata[:,center_idx,:,:]
    return(lines)

def trim_to_basis(lines,idx=None):
    '''Gives back only the line through the phantom, not empty space.'''

    if idx is None:
        idx = range(lines.shape[-1])

    edges = np.zeros([ 2,len(idx) ],dtype=int)
    for ii in idx:
        changes = np.abs(lines[:,1,ii]) - np.roll(np.abs(lines[:,1,ii]),1)
        edges[:,ii] = np.argsort(np.abs(changes))[-2::]

    # Find the most frequent starts and stops
    first = np.bincount(edges[0,:]).argmax()
    last = np.bincount(edges[1,:]).argmax()

    # Make the lines go from start to stop
    trimmed = lines[first:last,:,idx]
    return(trimmed,first,last)

def combine_line_coils(lines,idx=None):
    '''Gives back the sum of squares of each line, collapsing the coil dim.'''

    if idx is None:
        idx = range(lines.shape[-1])
    combined = np.sqrt(np.sum(np.abs(lines[:,:,idx]),axis=1))
    return(combined)

def show_center_lines(lines,idx=None):
    '''Plots the absolute value of the lines.'''

    if idx is None:
        idx = range(lines.shape[-1])

    # Combine coils
    combined = combine_line_coils(lines)

    plt.figure()
    for ii in idx:
        plt.subplot(len(idx),1,ii+1)
        plt.plot(combined[:,ii])
    plt.show()

def show_csm(csm):
    for ii in range(csm.shape[0]):
        plt.imshow(np.abs(csm[ii,:,:]),cmap='gray')
        plt.show()

def get_func(Ns,width,offset=0,amplitude=1):
    sq = np.zeros(Ns)
    sq[int(Ns/2-width/2):int(Ns/2+width/2)] = 1
    sq = np.roll(sq,int(offset*Ns))
    return(1-sq)

def get_coeffs(lines,f):
    coeffs = np.squeeze(np.linalg.lstsq(lines,f,rcond=None)[0])
    approx = lines.dot(coeffs)
    return(coeffs,approx)

def add_constant_im(imdata):
    # add constant image
    imdata_tmp = np.ones([ imdata.shape[0],imdata.shape[1],imdata.shape[2],imdata.shape[3]+1 ],dtype='complex')
    imdata_tmp[:,:,:,range(imdata.shape[3])] = imdata

    #  Get the mask of a SOS image over all coils and all images
    const_im = np.sqrt(np.sum(np.abs((np.sqrt(np.sum(np.abs(imdata_tmp**2),axis=2)))**2),axis=2))
    const_im[np.abs(const_im) > np.mean(const_im)] = 1
    const_im[const_im != 1] = 0

    # Place the constant image in each of the coil slots
    for c in range(imdata_tmp.shape[2]):
        imdata_tmp[:,:,c,imdata.shape[3]] = const_im

    return(imdata_tmp)

def use_meas_basis(kspace,use_sim_coeff=False):
    # Get image space data
    imdata = get_imspace(kspace)

    # Add constant term
    # imdata = add_constant_im(imdata)

    # reduce to only center lines
    lines,first,last = trim_to_basis(trim_to_center_line(imdata))
    im_squares = imdata[first:last,:,:,:]

    # Try using complex values sensitivity map?
    csm = simple_csm(imdata,rows=range(first,last))
    csm_lines = trim_to_center_line(csm)

    # Find sensitivity corrected coil lines
    combined = combine_line_coils(lines)
    corrected_coil_lines = np.zeros(lines.shape,dtype='complex')
    corrected_coil_ims = np.zeros(im_squares.shape,dtype='complex')
    for ii in range(lines.shape[-1]):
        for c in range(lines[:,:,ii].shape[1]):
            # Correct the line for this coil using the sensitivity profile
            corrected_coil_lines[:,c,ii] = lines[:,c,ii]/csm_lines[:,c,ii]

            # Correct the coil image using the sensitivity profile
            corrected_coil_ims[:,:,c,ii] = im_squares[:,:,c,ii]/csm[:,:,c,ii]

    if use_sim_coeff:
        # Find T1,T2,alpha
        # Generate simulated basis
        pass

    # Generate forcing function
    f = get_func(Ns=corrected_coil_lines.shape[0],width=50,offset=0)

    # Solve for coefficients
    coeffs = np.zeros([ corrected_coil_lines.shape[2],corrected_coil_lines.shape[1] ],dtype='complex')
    line_res = np.zeros(f.shape)
    im_res = np.zeros([ im_squares.shape[0],im_squares.shape[1] ])
    for c in range(coeffs.shape[-1]):
        # get coeffs for the ii'th image's c'th coil
        coeffs[:,c],approx = get_coeffs(corrected_coil_lines[:,c,:],f)

        # We need to apply coefficients to each coil line
        line_res += np.abs((corrected_coil_lines[:,c,:].dot(coeffs[:,c]))**2)

        # Now try applying the coefficients to each coil image proper
        im_res += np.abs((corrected_coil_ims[:,:,c,:].dot(coeffs[:,c]))**2)

    line_res = np.sqrt(line_res)
    im_res = np.sqrt(im_res)

    # Grab a mid slice from the image to compare against the line
    mid,pad = int(im_res.shape[1]/2),10
    im_mid_slice = np.mean(im_res[:,mid-pad:mid+pad],axis=1)

    plt.subplot(3,1,1)
    plt.imshow(np.transpose(im_res),cmap='gray')
    plt.title('Filtered SOS Image')
    plt.subplot(3,1,2)
    plt.plot(im_mid_slice,label='Center Slices of Image')
    plt.plot(line_res,label='Single Line Approx.')
    plt.plot(f,label='Forcing Function')
    plt.legend()
    plt.subplot(3,1,3)
    plt.plot(np.abs(line_res - im_mid_slice),label='Error Between Single line approx and center slices of image')
    plt.legend()
    plt.show()
