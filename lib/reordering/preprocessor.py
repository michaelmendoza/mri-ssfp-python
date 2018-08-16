## TODO:
# Use correct BART python interface to avoid creating files
# OR...
# Delete all files after every run...

import numpy as np
import matplotlib.pyplot as plt
import subprocess,os

class Data(object):

    def __init__(self,filename,accel=2,caldim=32,avg_weights=True,axis=0,force_create=False):
        self.filename = filename
        self.force_create = force_create
        self.accel = accel
        self.caldim = caldim
        self.avg_weights = avg_weights
        self.axis = axis

        # No masking
        self.load_raw()
        self.compute_avg()
        self.compute_csm()
        self.compute_pics()
        self.get_true_reordering()
        self.compute_kspace()

        # With mask
        self.compute_mask()
        self.compute_masked_kspace()
        self.compute_masked_imspace()
        self.compute_masked_avg()
        self.compute_masked_csm()
        self.compute_pics_on_masked_avg()

        # Comparison images and reorderings
        self.compute_lowres()
        self.compute_reordered_lowres()
        self.compute_reordered_masked_avg()
        self.compute_gradients()

    # FROM BART
    def readcfl(self,name):
        # get dims from .hdr
        h = open(name + ".hdr", "r")
        h.readline() # skip
        l = h.readline()
        h.close()
        dims = [int(i) for i in l.split( )]

        # remove singleton dimensions from the end
        n = np.prod(dims)
        dims_prod = np.cumprod(dims)
        dims = dims[:np.searchsorted(dims_prod, n)+1]

        # load data and reshape into dims
        d = open(name + ".cfl", "r")
        a = np.fromfile(d, dtype=np.complex64, count=n);
        d.close()
        return a.reshape(dims, order='F') # column-major

    def run_bash(self,cmd):
        process = subprocess.Popen(cmd.split(),stdout=subprocess.PIPE)
        output,error = process.communicate()
        if output is not None:
            print(output.decode('utf-8'))
        if error is not None:
            print(error)

    def _exists(self,file):
        # Ask if the file exists, or insist that it doesn't
        if  self.force_create:
            return(False)
        else:
            return(os.path.exists('%s.cfl' % file))

    def load_raw(self):
        # Use twixread to convert to BART compatible file
        self.bart_filename = '%s_bart' % self.filename
        if not self._exists(self.bart_filename):
            print('Starting twixread...')
            self.run_bash('bart twixread -A %s %s' % (self.filename,self.bart_filename))
            print('%s.clf created successfully!' % self.bart_filename)

        # Find out rows and cols
        kspace = self.readcfl(self.bart_filename)
        self.rows = kspace.shape[0]
        self.cols = kspace.shape[1]

    def compute_avg(self):
        # Calculate (weighted) average along dimensions specified by bitmask
        if self.avg_weights is True:
            weight_f = '-w'
        else:
            weight_f = ''
        bitmask = 1 << 14
        self.avg_filename = '%s_avg' % self.bart_filename
        if not self._exists(self.avg_filename):
            self.run_bash('bart avg %s %d %s %s' % (weight_f,bitmask,self.bart_filename,self.avg_filename))
            self.do_reshape()
        self.load_avg()

    def load_avg(self):
        self.avg = self.readcfl(self.avg_filename).squeeze()

    def do_reshape(self):
        # Put the data in the order that BART expects it to be in
        self.run_bash('bart reshape 7 1 %d %d %s %s' % (self.rows,self.cols,self.avg_filename,self.avg_filename))

    def compute_csm(self):
        # Estimate coil sensitivities using ESPIRiT calibration.
        self.csm_filename = '%s_csm' % self.bart_filename
        if not self._exists(self.csm_filename):
            self.run_bash('bart ecalib %s %s' % (self.avg_filename,self.csm_filename))
        self.load_csm()

    def load_csm(self):
        self.csm = self.readcfl(self.csm_filename).squeeze()

    def compute_masked_avg(self):
        self.masked_avg_filename = '%s_masked_avg' % self.bart_filename
        if not self._exists(self.masked_avg_filename):
            self.run_bash('bart fmac %s %s %s' % (self.avg_filename,self.mask_filename,self.masked_avg_filename))
        self.load_masked_avg()

    def load_masked_avg(self):
        self.masked_avg = self.readcfl(self.masked_avg_filename).squeeze()

    def compute_masked_csm(self):
        # Estimate coil sensitivities using ESPIRiT calibration.
        self.masked_csm_filename = '%s_masked_csm' % self.bart_filename
        if not self._exists(self.masked_csm_filename):
            self.run_bash('bart ecalib %s %s' % (self.masked_avg_filename,self.masked_csm_filename))
        self.load_masked_csm()

    def load_masked_csm(self):
        self.masked_csm = self.readcfl(self.masked_csm_filename).squeeze()

    def compute_mask(self):
        # We also need an undersampling mask
        self.mask_filename = 'mask_accel_%s_caldim_%d_%d_%d' % (str(self.accel),self.caldim,self.rows,self.cols)
        if not self._exists(self.mask_filename):
            print('Starting poisson...')
            self.run_bash('bart poisson -Y %d -Z %d -y %d -z %d -C %d -v %s' % (self.rows,self.cols,self.accel,self.accel,self.caldim,self.mask_filename))
            print('%s.clf created successfully!' % self.mask_filename)
        self.load_mask()

    def load_mask(self):
        self.mask = self.readcfl(self.mask_filename).squeeze()

    def compute_pics(self):
        # Get the Parallel-imaging compressed-sensing reconstruction
        self.pics_filename = '%s_pics' % self.bart_filename
        if not self._exists(self.pics_filename):
            self.run_bash('bart pics %s %s %s' % (self.avg_filename,self.csm_filename,self.pics_filename))
            # Remove the extra map
            self.run_bash('bart slice 4 0 %s %s' % (self.pics_filename,self.pics_filename))
        self.load_pics()

    def load_pics(self):
        self.pics = self.readcfl(self.pics_filename).squeeze()

    def compute_kspace(self):
        self.kspace_filename = '%s_pics_kspace' % self.bart_filename
        if not self._exists(self.kspace_filename):
            bitmask = 7
            self.run_bash('bart fft -u %d %s %s' % (bitmask,self.pics_filename,self.kspace_filename))
        self.load_kspace()

    def load_kspace(self):
        self.kspace = self.readcfl(self.kspace_filename).squeeze()

    def compute_masked_kspace(self):
        self.masked_kspace_filename = '%s_pics_masked_kspace' % self.bart_filename
        if not self._exists(self.masked_kspace_filename):
            self.run_bash('bart fmac %s %s %s' % (self.kspace_filename,self.mask_filename,self.masked_kspace_filename))
        self.load_masked_kspace()

    def load_masked_kspace(self):
        self.masked_kspace = self.readcfl(self.masked_kspace_filename).squeeze()

    def compute_pics_on_masked_avg(self):
        self.pics_masked_avg_filename = '%s_pics_masked_avg' % self.bart_filename
        if not self._exists(self.pics_masked_avg_filename):
            self.run_bash('bart pics %s %s %s' % (self.masked_avg_filename,self.masked_csm_filename,self.pics_masked_avg_filename))
            # Remove the extra map
            self.run_bash('bart slice 4 0 %s %s' % (self.pics_masked_avg_filename,self.pics_masked_avg_filename))
        self.load_pics_masked_avg()

    def load_pics_masked_avg(self):
        self.pics_masked_avg = self.readcfl(self.pics_masked_avg_filename).squeeze()

    def compute_lowres(self):
        # Get the center of kspace
        self.lowres_kspace = np.zeros(self.kspace.shape,dtype='complex')
        cr,cc = int(self.rows/2),int(self.cols/2)
        pad = int(self.caldim/2)
        self.lowres_kspace[cr-pad:cr+pad,cc-pad:cc+pad] = self.kspace[cr-pad:cr+pad,cc-pad:cc+pad]

        # Now get the low-res imspace, simple fft
        self.lowres_impace = np.fft.fftshift(np.fft.fft2(self.lowres_kspace))

    def pixel_reorder(self,complex_im,axis=0):
        # For the real
        reordered_imspace_real = np.sort(np.real(complex_im),axis=axis)
        order_real = np.argsort(np.real(complex_im),axis=axis)

        # For the imag
        reordered_imspace_imag = np.sort(np.imag(complex_im),axis=axis)
        order_imag = np.argsort(np.imag(complex_im),axis=axis)

        # Combine real,imag to get reordered complex image
        reordered_imspace = reordered_imspace_real + 1j*reordered_imspace_imag

        return(order_real,order_imag,reordered_imspace)

    def get_true_reordering(self):
        self.true_order_real,self.true_order_imag,self.reordered_pics = self.pixel_reorder(self.pics)
        return(self.true_order_real,self.true_order_imag)

    def sort_by_order(self,arr2d):
        ii = np.argsort(arr2d,axis=0)
        jj = np.arange(arr2d.shape[1])
        return(ii,jj)

    def sort_complex_by_order(self,complex_im,sorter):
        ii,jj = self.sort_by_order(np.real(sorter))
        real = np.real(complex_im)[ii,jj]
        ii,jj = self.sort_by_order(np.imag(sorter))
        imag = np.imag(complex_im)[ii,jj]
        return(real + 1j*imag)

    def compute_reordered_lowres(self):
        # Reorder the fully sampled pics according to the ordering found using
        # the lowres image
        self.lowres_order_real,self.lowres_order_imag,_ = self.pixel_reorder(self.lowres_impace)
        self.reordered_lowres = self.sort_complex_by_order(self.pics,self.lowres_impace)

    def compute_reordered_masked_avg(self):
        # Reorder the fully sampled pics according to the ordering found using
        # the masked avg image
        self.masked_avg_order_real,self.masked_avg_order_imag,_ = self.pixel_reorder(self.pics_masked_avg)
        self.reordered_masked_avg = self.sort_complex_by_order(self.pics,self.pics_masked_avg)

    def compute_masked_imspace(self):
        self.masked_imspace = np.fft.fftshift(np.fft.ifft2(self.masked_kspace))

    def show_pics(self):
        plt.imshow(np.abs(self.pics),cmap='gray')
        plt.show()

    def show_masked_pics(self):
        plt.imshow(np.abs(self.pics_masked_avg),cmap='gray')
        plt.show()

    def show_csm(self):
        csm = self.csm[:,:,:,0]
        s = csm.shape
        plt.imshow(np.abs(csm.reshape(s[0],s[1]*s[2])),cmap='gray')
        plt.show()

    def show_masked_csm(self):
        csm = self.masked_csm[:,:,:,0]
        s = csm.shape
        plt.imshow(np.abs(csm.reshape(s[0],s[1]*s[2])),cmap='gray')
        plt.show()

    def show_kspace(self):
        plt.imshow(np.log(np.abs(self.kspace)),cmap='gray')
        plt.show()

    def show_masked_kspace(self):
        plt.imshow(np.log(np.abs(self.masked_kspace)+.000001),cmap='gray')
        plt.show()

    def show_masked_imspace(self):
        plt.imshow(np.abs(self.masked_imspace),cmap='gray')
        plt.show()

    def show_mask(self):
        plt.imshow(np.abs(self.mask),cmap='gray')
        plt.show()

    def show_coils(self):
        num_coils = self.avg.shape[-1]
        coils = np.fft.fftshift(np.fft.ifft2(self.avg,axes=(0,1)))
        for ii in range(num_coils):
            plt.subplot(1,num_coils,ii+1)
            plt.imshow(np.abs(coils[:,:,ii]),cmap='gray')
        plt.show()

    def show_masked_coils(self):
        num_coils = self.masked_avg.shape[-1]
        coils = np.fft.fftshift(np.fft.ifft2(self.masked_avg,axes=(0,1)))
        for ii in range(num_coils):
            plt.subplot(1,num_coils,ii+1)
            plt.imshow(np.abs(coils[:,:,ii]),cmap='gray')
        plt.show()

    def compare_pics_masked_imspace(self):
        plt.subplot(1,2,1)
        plt.imshow(np.abs(self.pics),cmap='gray')
        plt.title('PICS')
        plt.subplot(1,2,2)
        plt.imshow(np.abs(self.masked_imspace),cmap='gray')
        plt.title('Undersampled FFT Recon')
        plt.show()

    def compare_pics(self):
        plt.subplot(1,2,1)
        plt.imshow(np.abs(self.pics),cmap='gray')
        plt.title('PICS, Fully Sampled')
        plt.subplot(1,2,2)
        plt.imshow(np.abs(self.pics_masked_avg),cmap='gray')
        plt.title('PICS, Undersampled')
        plt.show()

    def show_lowres_kspace(self):
        plt.imshow(np.log(np.abs(self.lowres_kspace)+.1),cmap='gray')
        plt.show()

    def show_lowres_imspace(self):
        plt.imshow(np.abs(self.lowres_impace),cmap='gray')
        plt.show()

    def show_reordered_pics(self):
        plt.imshow(np.abs(self.reordered_pics),cmap='gray')
        plt.show()

    def show_reordered_lowres(self):
        plt.imshow(np.abs(self.reordered_lowres),cmap='gray')
        plt.show()

    def show_reordered_masked_avg(self):
        plt.imshow(np.abs(self.reordered_masked_avg),cmap='gray')
        plt.show()

    def compare_reorderings(self):
        true_lowres = self.reordered_pics - self.reordered_lowres
        plt.subplot(2,2,1)
        plt.imshow(np.real(true_lowres),cmap='gray')
        plt.title('real(True - Low-res)')
        plt.subplot(2,2,3)
        plt.imshow(np.imag(true_lowres),cmap='gray')
        plt.title('imag(True - Low-res)')

        true_cs = self.reordered_pics - self.reordered_masked_avg
        plt.subplot(2,2,2)
        plt.imshow(np.real(true_cs),cmap='gray')
        plt.title('real(True - CS)')
        plt.subplot(2,2,4)
        plt.imshow(np.imag(true_cs),cmap='gray')
        plt.title('imag(True - CS)')
        plt.show()

    def grad(self,complex_im,axis=0):
        g_r = np.gradient(np.real(complex_im),edge_order=2,axis=axis)
        g_i = np.gradient(np.imag(complex_im),edge_order=2,axis=axis)
        return(g_r,g_i)

    def compute_gradients(self):
        self.g_orig_r,self.g_orig_i = self.grad(self.pics)
        self.g_true_r,self.g_true_i = self.grad(self.reordered_pics)
        self.g_lowres_r,self.g_lowres_i = self.grad(self.reordered_lowres)
        self.g_masked_avg_r,self.g_masked_avg_i = self.grad(self.reordered_masked_avg)

    def cost_by_gradient(self,g_r,g_i):
        cost_r = np.sqrt(np.sum(g_r**2,axis=(0,1)))
        cost_i = np.sqrt(np.sum(g_i**2,axis=(0,1)))
        return(cost_r + cost_i)

    def compare_costs(self):
        print('true:         %g' % self.cost_by_gradient(d.g_true_r,d.g_true_i))
        print('cs prior:     %g' % self.cost_by_gradient(d.g_masked_avg_r,d.g_masked_avg_i))
        print('lowres prior: %g' % self.cost_by_gradient(d.g_lowres_r,d.g_lowres_i))
        print('orig:         %g' % self.cost_by_gradient(d.g_orig_r,d.g_orig_i))

    def show_gradients(self):
        plt.subplot(2,4,1)
        plt.imshow(self.g_true_r,cmap='gray')
        plt.title('grad real(true)')
        plt.subplot(2,4,5)
        plt.imshow(self.g_true_i,cmap='gray')
        plt.title('grad imag(true)')

        plt.subplot(2,4,2)
        plt.imshow(self.g_lowres_r,cmap='gray')
        plt.title('grad real(lowres)')
        plt.subplot(2,4,6)
        plt.imshow(self.g_lowres_i,cmap='gray')
        plt.title('grad imag(lowres)')

        plt.subplot(2,4,3)
        plt.imshow(self.g_masked_avg_r,cmap='gray')
        plt.title('grad real(cs)')
        plt.subplot(2,4,7)
        plt.imshow(self.g_masked_avg_i,cmap='gray')
        plt.title('grad imag(cs)')

        plt.subplot(2,4,4)
        plt.imshow(self.g_orig_r,cmap='gray')
        plt.title('grad real(orig)')
        plt.subplot(2,4,8)
        plt.imshow(self.g_orig_i,cmap='gray')
        plt.title('grad imag(orig)')

        plt.show()

    def cost(self,ind2d_real,ind2d_imag):
        real = np.real(self.pics)
        imag = np.imag(self.pics)
        for jj in range(real.shape[1]):
            real[:,jj] = real[ind2d_real[:,jj],jj]
            imag[:,jj] = imag[ind2d_imag[:,jj],jj]
        g_r,g_i = self.grad(real + 1j*imag)
        return(self.cost_by_gradient(g_r,g_i))

    def remove_files(self):

        to_remove = [ self.bart_filename,self.avg_filename,self.csm_filename,
                    self.masked_avg_filename,self.masked_csm_filename,
                    self.pics_filename,self.kspace_filename,
                    self.masked_kspace_filename,self.pics_masked_avg_filename ]
        for f in to_remove:
            os.remove('%s.cfl' % f)
            os.remove('%s.hdr' % f)

    def compute_stcr(self):
        pass

if __name__ == '__main__':



    # Initialize Data object
    raw = 'data/test3.dat'
    accel,caldim = 2,32
    d = Data(raw,accel,caldim)

    ## Make sure everything looks good
    # d.show_pics()
    # d.show_csm()
    # d.show_kspace()
    # d.show_coils()
    # d.compare_pics_masked_imspace()

    # d.show_mask()
    # d.show_masked_pics()
    # d.show_masked_csm()
    # d.show_masked_coils()
    # d.show_masked_kspace()
    # d.show_masked_imspace()
    # d.compare_pics()

    ## Now it's time for some comparisons
    # d.show_lowres_kspace()
    # d.show_lowres_imspace()

    # CS priors are the masked_kspace,masked_imspace recons

    # Here are the orderings
    # print(d.true_order_real)
    # print(d.true_order_imag)

    ## Reorder the recons to compare
    # d.show_reordered_pics()
    # d.show_reordered_lowres()
    # d.show_reordered_masked_avg()
    # d.compare_reorderings()
    # d.show_gradients()

    ## Cost function
    # d.compare_costs()

    # NN Stuff
    # input = d.masked_imspace
    # output = Two 2d arrays of indices
    # cost = d.cost(order_real,order_imag)

    ## Example:
    # print(d.cost(d.masked_avg_order_real,d.masked_avg_order_imag))

    ## Cleanup
    # d.remove_files()
