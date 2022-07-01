#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GyotoAnalyser V1.0
Created on Thu May 27 12:08:56 2021
Author: Hadrien Paugnat
"""
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize, curve_fit
from astropy.io import fits
from skimage.transform import radon
from simpson import * # in ~/mypythonlib
from matplotlib.colors import LogNorm
from scipy.signal import find_peaks,peak_widths
import scipy.ndimage
from scipy.interpolate import interp1d
import os


########################
''''Useful functions and constants'''

M_to_muas=3.6212 #1 unit of M in microarcsec for M87* (for the canonical values : M=6.2e9 solar masses, D=16.9 Mpc)
muas_to_rad = math.pi/648000 *1e-6 #1 microarcsec in radians

def circlipse_rotated(phi,R0,R1,R2, phi0):
    return ( R0 + np.sqrt(R1**2 * np.sin(phi-phi0)**2 + R2**2 * np.cos(phi-phi0)**2))

def smooth_connection(x):
    '''Infinitely smooth function, =0 for x<=0'''
    if x<=0:
        return 0
    else:
        return math.exp(-1/x**2)

def smooth_plateau(x):
    '''Infinitely smooth function, =0 for x<=0, =1 for x>=1'''
    return smooth_connection(x)/(smooth_connection(x)+smooth_connection(1-x))

def gaussian(x, mu, sig, amp):
    return amp*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

#u in Glambda, d in microarcsec, visamp in mJy
def expected_visamp(u,alphaL,alphaR, d, zetaw):
    '''Expected visibility amplitude for a thin ring'''
    return (np.exp(-u*zetaw*1e-3) * np.sqrt((alphaL**2 + alphaR**2 + 2*alphaL*alphaR*np.sin(2*np.pi*d*muas_to_rad*u*1e9))/(u*1e9)) * 1e3)

def estimate_visamp_params(u, visamp):
    '''Estimates the parameters alphaL, alphaR, d, zetaw ; assuming the visibility amplitude follows the functional form above'''
    maxima, _ = find_peaks(visamp, width = 10)
    minima, _ = find_peaks(-visamp, width = 10)
    i0 = maxima[0] #first maximum
    u0, v0 = u[i0], visamp[i0]
    i1 = minima[0] #first minimum
    u1, v1 = u[i1], visamp[i1]
    i2 = maxima[-1] #last maximum
    u2, v2 = u[i2], visamp[i2]
    
    period = (u2-u0)/(len(maxima)-1)
    
    if u2 > u0 :
        zetaw = max(0., np.log(v0/v2 * np.sqrt(u0/u2)) / (u2-u0))*1e3
    else:
        zetaw = 0.
    sumalpha = v0*1e-3*np.sqrt(u0*1e9)*np.exp(zetaw*u0*1e-3)
    diffalpha = v1*1e-3*np.sqrt(u1*1e9)*np.exp(zetaw*u1*1e-3)
          
    return np.array([(sumalpha+diffalpha)/2, (sumalpha-diffalpha)/2, 1/(period*muas_to_rad*1e9), zetaw])


def meas4_into_params_circlipse(r0,r1,r2,r3):
    '''From 4 diameter measurement at x, x+pi/4, x+pi/2, x+3pi/4 ; deduce the parameters R0,R1,R2 of the non-rotated circlipse (i.e. with phi0=0) which would have these diameter values at these angles - as well as x (mod pi)'''
    R0 = (r0**2 + r2**2 - r1**2 - r3**2)/(2*(r0+r2-r1-r3))
    sum_square_R1R2 = (r0-R0)**2 + (r2-R0)**2 # R1**2 + R2**2

    #xcalc : calculated angle, supposed to be equal to x mod pi
    #diff_square_R1R2 : R1**2 - R2**2
    if ((r2-R0)**2 - (r0-R0)**2) > 0. :
        xcalc = 0.5*math.atan(((r1-R0)**2 - (r3-R0)**2)/((r2-R0)**2 - (r0-R0)**2))
        diff_square_R1R2 = ((r2-R0)**2 - (r0-R0)**2) / math.cos(2*xcalc)
    elif ((r2-R0)**2 - (r0-R0)**2) < 0. :
        xcalc = np.pi/2 + 0.5*math.atan(((r1-R0)**2 - (r3-R0)**2)/((r2-R0)**2 - (r0-R0)**2))
        diff_square_R1R2 = ((r2-R0)**2 - (r0-R0)**2) / math.cos(2*xcalc)
    else :
        if r1 >= r3 :
            xcalc = math.pi/4
            diff_square_R1R2 = ((r1-R0)**2 - (r3-R0)**2)
        else:
            xcalc = 3*math.pi/4
            diff_square_R1R2 = ((r3-R0)**2 - (r1-R0)**2)
        
    R1 = math.sqrt((sum_square_R1R2 + diff_square_R1R2)/2)
    R2 = math.sqrt((sum_square_R1R2 - diff_square_R1R2)/2)
    
    return [R0,R1,R2,xcalc]

########################

class MyException(Exception):
    pass


class GyotoImage:
    ''' For the analysis of an image calculated with Gyoto'''
    def __init__(self, address, fov, padfact, edge_to_zero):
        self.address = address.replace('\\','/') #address of the fits file
        
        self.image1 = fits.open(address)[0].data[0] 
        self.NN1 = self.image1.shape[0] #resolution
        self.Npad1 = self.NN1*padfact #for a cleaner FT (e.g. padfact = 16 )
        
        #field of view in microarcsec
        if (fov == None):
            self.fov1 = fits.getheader(self.address)['CDELT1']/1e-6*3600.*float(self.NN1)
        else:
            self.fov1 = fov
            
        self.xaxis1 = np.linspace(-self.fov1/2., self.fov1/2, num = self.NN1)
        self.diffaxis = (self.xaxis1[1:]+self.xaxis1[:-1])/2
        
        self.edge_to_zero = edge_to_zero #Using the window function to set the edges of the Radon to zero or not
        self.range_cutoff = 0.25*self.fov1/500 #range where the window function is not 1 (in muas)
        
        self.phimeas = 0. 
        
        
        # Define the Fourier axis
        deltax1=self.xaxis1[1]-self.xaxis1[0]
        self.FTaxis=np.fft.fftshift(np.fft.fftfreq(self.Npad1,d=deltax1)) #re centered frequencies in muas^-1
        self.FTaxis/= 1e-6 * 1./3600. * math.pi/180. # in rad^-1
        self.FTaxis /= 1e9 # in Glambda
        self.indice1=np.where(self.FTaxis>0.)[0] # select only the positive freqs, the FFT is symmetrical anyway
        self.FTaxis = self.FTaxis[self.indice1] # select FT for these freqs
        
        self.intensity = None
        self.Radon = None
        self.visamp = None
        
        self.multiple_diameters = False  

        
    def show_image(self):
        '''Displays the fits file (image)'''
        plt.figure()
        plt.clf()
        plt.ion()
        plt.imshow(self.image1, interpolation='nearest',origin='lower',
                       extent=(0.,self.NN1,0.,self.NN1))
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

    def intensity_slice(self, phimeas):
        '''Computes the specific intensity of a slice at angle phimeas (in °) counterclockwise from the horizontal (spin axis at 90°) - result is in self.intensity'''
        self.phimeas = phimeas
        phimeasinrad = self.phimeas*math.pi/180
        x = (self.xaxis1*np.cos(phimeasinrad)/self.fov1 + 1/2)*self.NN1
        y= (self.xaxis1*np.sin(phimeasinrad)/self.fov1 + 1/2)*self.NN1

        self.intensity = scipy.ndimage.map_coordinates(self.image1, np.vstack((y,x)))
        self.intensity /= np.max(self.intensity) #normalize intensity 
    
    def plot_intensity(self, nbw, curvecolor):
        '''Plots the intensity and its derivative in figure nbw with color curvecolor'''
        fig, axes = plt.subplots(1, 2, figsize=(12,5.5), dpi=80, num=nbw)
        
        axes[0].set_ylabel(r'Specific intensity along direction $\varphi$=%5.1f °'  % (self.phimeas),size=14)
        axes[0].set_xlabel("Distance ($M$)",size=14)
        
        # Plot intensity
        axes[0].plot(self.xaxis1/M_to_muas,self.intensity,color=curvecolor)
    
        #Plot derivative of intensity
        axes[1].set_xlabel("Distance ($M$)", size=14)
        axes[1].set_ylabel('Derivative of the specific intensity')
        axes[1].plot(self.diffaxis/M_to_muas,np.diff(self.intensity),color=curvecolor)
        
    def estimate_diameter_with_intensity(self):
        '''Gives an estimation of the n=2 ring diameter using the maxima of the intensity derivative'''
        dI = np.diff(self.intensity)
        leftpeak = np.argmax(np.abs(dI[self.diffaxis<0]))
        rightpeak = np.argmax(np.abs(dI[self.diffaxis>0]))
        
        return self.diffaxis[self.diffaxis>0.][rightpeak] - self.diffaxis[self.diffaxis<0.][leftpeak]
        
    def Radon_projection(self,phimeas):
        '''Computes the Radon projection along a slice at angle phimeas (in °) counterclockwise from the horizontal (spin at 90°) - result is in self.Radon'''
        self.phimeas = phimeas
        self.Radon = radon(self.image1/1., theta=[self.phimeas], circle=True).flatten() # the division by 1. solves some "Endian" error that I get without it... don't know what's that.
        self.Radon/= np.max(self.Radon) #normalize Radon transform
        
        #If the Fourier transform is dirtied by non-zero values of the Radon edges, this allows to send the edges to zero smoothly.
        if self.edge_to_zero :
            smooth_transition_to_zero = np.vectorize(smooth_plateau) (self.range_cutoff*(self.xaxis1[-1]-self.xaxis1)) * np.vectorize(smooth_plateau) (self.range_cutoff*(self.xaxis1+self.xaxis1[-1]))
            self.Radon *= smooth_transition_to_zero
    
    def FT_Radon(self):
        '''Computes the Fourier Transform of the Radon projection (result is in self.visamp)'''
        radonff = np.fft.fft(self.Radon,self.Npad1) #1D FFT of the projection
        radonff/=radonff.max()
        radonshift=np.fft.fftshift(radonff) # recenter FFT
        radonvisamp=np.abs(radonshift) # this is the visamp
        self.visamp =radonvisamp[self.indice1] 
    
    def plot_Radon_and_FT(self, nbw, curvecolor) :
        '''Plots the Radon projection and its Fourier Transform in figure nbw with color curvecolor'''
        fig, axes = plt.subplots(1, 2, figsize=(12,5.5), dpi=80, num=nbw)
       
        # Plot the Radon transform (ie image projection along phimeas)
        axes[0].set_ylabel(r'Normalized Radon transform along direction $\varphi$=%5.1f °'  % (self.phimeas),size=14)
        axes[0].set_xlabel("Distance ($M$)",size=14)
        axes[1].set_ylabel(r'Normalized visamp along direction $\varphi$=%5.1f °'  % (self.phimeas) ,size=18)
        axes[1].set_xlabel("Baseline (G$\\lambda$)",size=20)
        axes[1].set_xlim(0.,3500.)
        axes[1].set_ylim(1e-7,1)
        axes[1].set_yscale('log')
        axes[1].tick_params(which='both', labelsize=14)
        
        axes[0].plot(self.xaxis1/M_to_muas,self.Radon,color=curvecolor)
        axes[1].plot(self.FTaxis,self.visamp,color=curvecolor,linestyle="solid")
        
    def visamp_fit_from_formula(self, bounds, plot=False):
        '''Computes the best fit of the visibility amplitude for baselines within *bounds*
        with the (ideal) expected visamp functional form described earlier.
        Returns the corresponding diameter in muas (encoding the periodicity), with the uncertainty
        Plots the visamp and its best fit if *plot* is True
        '''
        # Define the Fourier axis within the bounds
        indice2=np.where((self.FTaxis>bounds[0]) & (self.FTaxis<bounds[1]))[0] # select only the freqs within the bounds
        u = self.FTaxis[indice2] # select FT for these freqs
        visamp =self.visamp[indice2] *1e3 #visamp in mJy
        
        pguess = estimate_visamp_params(u,visamp)
        popt, pcov = curve_fit(expected_visamp, u, visamp, p0 = pguess, bounds = (0,np.inf), maxfev=5000)
        
        if plot:
            fig, ax = plt.subplots(1, 1, dpi=80)
            ax.set_ylabel("Normalized visibility amplitude",size=14)
            ax.set_xlabel("Baseline (G$\\lambda$)",size=14)
            ax.plot(u,visamp, 'r' ,linestyle="solid", label = r'Normalized visamp at angle $\varphi$ = %5.1f °' %self.phimeas )
            ax.plot(u, expected_visamp(u,*popt),'k--', label=r'best fit with $\alpha_R$ = %5.3f, $\alpha_L$ = %5.3f, $d_{\varphi}$ = %5.3f $\mu$as, $\zeta$w = %5.3f' %tuple(popt))
            ax.legend()
            
        return [popt[2], pcov[2,2]]
      
    def visamp_fit_with_envelope(self, bounds, plot=False, likelihoods = False):
        '''Computes a maximal and a minimal envelope for the visamp between *bounds*
        then searches the best fit for oscillations between these envelopes
        
        If multiple_diamaters (class attribute) and likelihoods (parameter) are set to False :
            Returns the diameter in muas (encoding the periodicity) corresponding to the best fit, with the uncertainty
            Plots the visamp and its best fit if *plot* is True
            
        If likelihoods (parameter) is set to True :
            Computes the normalized RMS residuals for all the fits, with functions using the computed envelope and
            diameters around an estimated value
            Plots goodness of fit values as a function of the diameter/periodicity
            Returns the diameter in muas (encoding the periodicity) corresponding to the best fir, with the uncertainty
            Plots the visamp, its best fit and the fit minimizing the RMS residuals if *plot* is True
            
        If multiple_diamaters (class attribute) is set to True:
            Computes the normalized RMS residuals for all the fits, with functions using the computed envelope and
            diameters around an estimated value
            Returns all diameters corresponding to local minima of the RMS residuals,
            more precisely to peaks of exp(-RMS residuals), along with the full width at half max. of these peaks
        '''
        
        # Define the Fourier axis within the bounds        
        indice2=np.where((self.FTaxis>bounds[0]) & (self.FTaxis<bounds[1]))[0] # select only the freqs within the bounds
        u = self.FTaxis[indice2] # select FT for these freqs
        visamp =self.visamp[indice2] *1e3 #visamp in mJy
        
        ### 'Manually' determining the envelopes, using a peak identification then interpolating between the peaks
        maxs, _ = find_peaks(visamp)
        mins, _ = find_peaks(-visamp)
        envelopemax = interp1d(u[maxs], visamp[maxs], kind='cubic') 
        envelopemin = interp1d(u[mins], visamp[mins], kind='cubic')
        
        xaxis2 = u[max(maxs[0],mins[0]) : min(maxs[-1], mins[-1])] #range where the min and max envelopes are both well-defined (due to interpolation)
        
        #Estimates of d (in muas), using the mean distance between successive maxima or minima in the visamp
        dmaxs = (1/((u[maxs[-1]]- u[maxs[0]])/(len(maxs)-1)) *1e-9/muas_to_rad)
        dmins = (1/((u[mins[-1]]- u[mins[0]])/(len(mins)-1)) *1e-9/muas_to_rad)
                
        ## Local model for the visamp function considering oscillations between the two envelopes
        ## diameter d (encoding the period) remains a free parameter
        def local_expected_visamp(x,d, phase=0):
            alphaL = (envelopemin(x)+envelopemax(x))/2
            alphaR = (envelopemax(x)-envelopemin(x))/2
            return np.sqrt((alphaL**2 + alphaR**2 + 2*alphaL*alphaR*np.sin(2*np.pi*d*muas_to_rad*x*1e9+phase)))
        
        # Selection of the given visamp within the range where the min and max envelopes are both well-defined
        local_visamp = visamp[max(maxs[0],mins[0]) : min(maxs[-1], mins[-1])]
        
        # Determines best-fit value for the remaining free parameter d
        popt, pcov = curve_fit(lambda u,d : local_expected_visamp(u,d), xaxis2, local_visamp, p0 = [(dmaxs+dmins)/2])
        
        if plot:
            fig, ax = plt.subplots(1, 1, dpi=120)
            ax.set_ylabel("Normalized visibility amplitude",size=14)
            ax.set_xlabel("Baseline (G$\\lambda$)",size=14)
            ax.set_yscale('log')
            ax.plot(u,visamp, 'r' ,linestyle="solid", label = r'Normalized visamp at angle $\varphi$ = %5.1f °' %self.phimeas )
            ax.plot(xaxis2, envelopemax(xaxis2), 'g--', label = r'Superior envelope $e_{max}$')
            ax.plot(xaxis2, envelopemin(xaxis2), 'b--', label = r'Inferior envelope $e_{min}$')
            ax.plot(xaxis2, local_expected_visamp(xaxis2,*popt),'k--', label=r'Visamp with best fit algorithm ($d_{\varphi}$ = %5.3f $\mu$as)' %popt[0])
            ax.legend(fontsize =16)
        
        if likelihoods or self.multiple_diameters:
            diameterscan = np.linspace(min(dmaxs,dmins)-3, max(dmaxs,dmins)+3, 10000) #range of diameters considered for the likelihood estimation
            nrms= []
            
            #Compute the normalized root-mean-squared residual between the given visamp, and the model using oscillations between envelopes with period d
            for d in diameterscan:
                    nrms.append(np.sqrt(np.mean((local_expected_visamp(xaxis2, d)-local_visamp)**2))/np.mean(local_visamp))
                    
            likelihood = np.exp(-np.array(nrms)) #goodness of fit
            
            if likelihoods :
                #Plot the goodness of fit values as a function of the diameter/periodicity
                plt.figure()
                plt.plot(diameterscan, likelihood)
                plt.xlabel(r'Inferred diameter $d_\varphi$ ($\mu$ as)',size=14)
                plt.ylabel(r'Goodness of fit $g(d_\varphi)$ for angle $\varphi$ = %5.1f °' %self.phimeas,size=14)
                plt.tick_params(which='both', labelsize=12)
            
            if self.multiple_diameters : 
                
                ##Find peaks in the goodness-of fit function (likelihood) if they are prominent enough
                wantedprominence = 0.5*(max(likelihood)-min(likelihood))
                possiblediams,properties = find_peaks(likelihood, distance = 10, prominence=wantedprominence, width=1)
                
                if likelihoods :
                    plt.scatter(diameterscan[possiblediams], likelihood[possiblediams], marker = 'o', color = 'r')
                
                ## Determine an uncertainty regarding the location of these peaks
                
                leftpeakbounds = np.int_(np.floor(properties["left_bases"])) #Left and right base of the peaks
                rightpeakbounds = np.int_(np.ceil(properties["right_bases"]))
                errorbars = np.zeros_like(possiblediams, float)
                for j in range(len(possiblediams)):
                    #we take the errorbar as the distance between the two points such that RMS = 2*RMS_peak i.e. for a likelihood g = g_peak²
                    peak = possiblediams[j]
                    g_peak = likelihood[peak]
                    leftsol, rightsol = 0,0
                    for i in range(int(leftpeakbounds[j]), peak+1):
                        #find point left of the peak such that g = g_peak²
                        if likelihood[i]<=g_peak**2 and likelihood[i+1]>g_peak**2:
                            leftsol=i
                            break
                    for i in range(peak, int(rightpeakbounds[j])+1):
                         #find point right of the peak such that g = g_peak²
                        try:
                            if likelihood[i]>=g_peak**2 and likelihood[i+1]<g_peak**2:
                                rightsol=i
                                break
                        except IndexError:
                            rightsol=rightpeakbounds[j]
                    errorbars[j] = min(diameterscan[rightsol]-diameterscan[leftsol], diameterscan[rightpeakbounds[j]]-diameterscan[leftpeakbounds[j]])
                    if likelihoods :
                        #plot the errorbars
                         plt.hlines(g_peak**2, max(diameterscan[leftsol], diameterscan[leftpeakbounds[j]]),min(diameterscan[rightsol], diameterscan[rightpeakbounds[j]]))
                if len(possiblediams)==0:
                    raise MyException
                
                #return the diameter corresponding to the peaks, the height of the peak and the errorbar as defined above
                return [diameterscan[possiblediams], likelihood[possiblediams], errorbars]
            
            #Diameter which maximizes the goodness of fit function
            dopt = diameterscan[np.argmin(nrms)]
            if plot:
                ax.plot(xaxis2, local_expected_visamp(xaxis2, dopt), 'y--', label =r'Visamp with envelope and $argmax[g(d_{\varphi})]$ = %5.3f $\mu$as' %(dopt))
                ax.legend()
            
        return [popt[0], pcov[0,0]]

    
    def sample_diameters(self, nmeas, bounds):
        '''For *nmeas* angles regularly spaced in [0°,180°[,
        computes the diameter(s) & uncertainties retrieved from the visamp at this angle, for baseline between *bounds*
        (multiple values if class attribute multiple_diameters is set to True)
        Then saves the result in a .npy file'''
        measureangles = np.linspace(0., 180., nmeas, endpoint = False)
        
        diameters = []
        for i in range(nmeas):
            print('Program is working on angle: '+ str(measureangles[i]) +'°') #Allows to see the progression
            try:
                self.Radon_projection(measureangles[i])
                self.FT_Radon()
                diameters.append(self.visamp_fit_with_envelope(bounds))
            except (RuntimeError, ValueError) as error:
                diameters.append([np.nan,np.nan])
            except MyException:
                diameters.append([[np.nan],[np.nan],[np.nan]])
                
                
        ##Creates a directory 'Diameter_data' if not already existing
        diam_data_dir = '/'.join(self.address.split("/")[:-1]) + '/Diameter_data'
        if not os.path.exists(diam_data_dir):
            os.makedirs(diam_data_dir)
            
        ## Saves the diameter or multiple_diameter data in a .npy file in a "Diameter_data" folder at the same adress than the original image
        if not(self.multiple_diameters):
            title = self.address.split("/")[-1][:-5] + str(bounds) + '_diameters.npy' #name of the file (we remove the .fits extension of the original image)
            np.save('/'.join(self.address.split("/")[:-1]) + '/Diameter_data/' + title, diameters) #save at the same adress, in "Diameter_data" folder
        else :
            title = self.address.split("/")[-1][:-5] + str(bounds) + '_multiplediameters.npy' #name of the file (we remove the .fits extension of the original image)
            np.save('/'.join(self.address.split("/")[:-1]) + '/Diameter_data/' + title, diameters, allow_pickle = True) #save at the same adress, in "Diameter_data" folder


    def circlipse_fit(self, bounds, removepoints = [], startangle = 0., circlipseguess = None, threshold_jlik = 0.):
        if not(self.multiple_diameters):
            return self.circlipse_fit_single(bounds, removepoints)
        else:
            return self.circlipse_fit_multiple(bounds, circlipseguess , startangle, removepoints, threshold_jlik)
    
    def circlipse_fit_single(self, bounds, removepoints = []):
        '''Loads the result of the diameter calculations for baselines between *bounds* in the corresponding .npy file
        (one diameter for each angle) then fits the values for angles outside *removepoints* (in °) to a circlipse 
        Plots the computed diameter as a function of the angle, as well as the best fit
        Returns the best-fit circlipse parameters, the normalized RMS residual for this best-fit,
         and the associated max & min diameters'''
       
       ## Loads the diameter data from the .npy file
        title = self.address.split("/")[-1][:-5] + str(bounds) + '_diameters.npy'
        diameters = np.load('/'.join(self.address.split("/")[:-1]) + '/Diameter_data/' + title, allow_pickle = True)/M_to_muas
        
        ## Before removal, there should be *nmeas* angles regularly spaced in [0°,180°[
        nmeas = len(diameters[:,0])
        measureangles = np.linspace(0., 180., nmeas, endpoint = False)
        
        ## We restrict ourselves everywhere to indices where the diameter is not NaN and where the angles isn't in the remove list
        notremoved = np.vectorize(lambda x : not(x in removepoints))
        values_indices = np.argwhere(np.logical_and(np.logical_not(np.isnan(diameters[:,0])),notremoved(measureangles)))
        values = np.ndarray.flatten(diameters[:,0][values_indices])
        sampleangles = np.ndarray.flatten(measureangles[values_indices])
        sampleangles*=np.pi/180 #angles in radians
        
        ## Fits the sampled diamaters to a circlipse
        popt, pcov = curve_fit(circlipse_rotated, sampleangles, values, p0= [10.,1.,1.,0.1], bounds = ([0.,0.,0.,-math.pi/4], [np.inf,np.inf,np.inf,math.pi/4]))
        rmsd = math.sqrt( np.sum((values - circlipse_rotated(sampleangles,*popt))**2) / len(values_indices) ) #root mean square residuals
        norm_rmsd = rmsd/np.average(circlipse_rotated(measureangles*np.pi/180,*popt)) # normalized RMS residuals
        dplus, dminus = popt[0]+popt[1], popt[0]+popt[2] #Maximum & minimum diameter
                
        ## Plots the sampled diameters and the best-fit circlipse
        plt.figure()
        plt.errorbar(sampleangles*180/np.pi, values, yerr = np.ndarray.flatten(diameters[:,1][values_indices]), c= 'b', marker= 'o', linestyle = 'None', label = r'Diameters at each angle, retrieved from visamp periodicity')
        plt.xlabel(r'Baseline angle $\varphi$ (°)', fontsize =18)
        plt.ylabel(r'Diameter inferred from periodicity (M)', fontsize =18)
        plt.plot(measureangles, circlipse_rotated(measureangles*np.pi/180,*popt), 'r', label = r'Best fit to a circlipse with $d_{+}$ = %5.3f $M$ and $d_-$ = %5.3f. Normalized RMSD =%5.3f $\times 10^{-5}$' % tuple([dplus, dminus, norm_rmsd*1e5]))
        plt.title(r'Circlipse fit at (%5.0f,%5.0f) G$\lambda$ :' %tuple(bounds))
        plt.legend()
        plt.show()
        
        return popt,norm_rmsd, dplus, dminus
        
    def circlipse_fit_multiple(self, bounds, circlipseguess, startangle = 0., removepoints = [], threshold_jlik = 0.):
        '''Loads the result of the diameter calculations for baselines between *bounds* in the corresponding .npy file
        (several diameter for each angle) 
        For each possible value at *startangle* (in °):
            selects the values at other angles that could belong to the same circlipse using a window of acceptance with a shape described by circlipseguess
            then fits the values for angles outside *removepoints* (in °) to a circlipse 
            Plots the computed diameter as a function of the angle, as well as the best fit - if the joint likelihood of the points exceeds *threshold_jlik*'''
      
       ## Loads the multiple_diameter data from the .npy file
        title = self.address.split("/")[-1][:-5] + str(bounds) + '_multiplediameters.npy'
        diameters = np.load('/'.join(self.address.split("/")[:-1]) + '/Diameter_data/' + title, allow_pickle = True)

        ## Before removal, there should be *nmeas* angles regularly spaced in [0°,180°[
        nmeas= len(diameters)
        measureangles = np.linspace(0., 180., nmeas, endpoint = False)
        
        for i in range(nmeas):
            diameters[i][0]/=  M_to_muas #diameters in units of M
        
        ## We start the selection of points from startangle -> determines the possible diameters at this angle
        indexstartangle = np.argwhere(measureangles==startangle)[0][0]
        startdiams,_,_ = diameters[indexstartangle]
        
        fig, axes = plt.subplots(1, 2, dpi=80)
        
        ## For each possible diameter at startangle, computes a circlipse-shaped acceptance window (with params circlipseguess)
        ## then selects, for each angle, a point in that window
        
        for d0 in startdiams: 
            selectedvalues = np.zeros(nmeas)
            likelihoods = np.zeros(nmeas)
            uncertainties = np.ones(nmeas)                
            for i in range(nmeas): #for each angle
                phi = measureangles[i]*np.pi/180 #in radians
                try :
                    possiblediams, height, errorbars = diameters[i] #the multiple possible diameters at angle phi (with likelihood & errorbars)
                    for j in range(len(possiblediams)):
                        # circlipseguess[:-1] : parameters of the circlipse forming the carrier-envelope of the acceptance window
                        # criclipseguess[-1] : width of the acceptance window (tolerance)
                        if (abs(possiblediams[j] - d0 - circlipse_rotated(phi, *circlipseguess[:-1]) + circlipse_rotated(startangle*np.pi/180, *circlipseguess[:-1])) <= circlipseguess[-1]): #test to see if the diameter is in the acceptance window
                            selectedvalues[i] = possiblediams[j]
                            likelihoods[i] = height[j]
                            uncertainties[i] = errorbars[j]
                            if math.sin(phi)>= math.sin(startangle*np.pi/180):
                                break #we keep the minimal acceptable value if sin(phi)>= sin(startangle);
                                        #the maximal acceptable value otherwise

                    #Plots all the possible diameters for all angles
                    if (np.max(height) - np.min(height)) != 0:
                        enhanced_height = (height - np.min(height))/(np.max(height) - np.min(height)) #rescaled to make nicer plots
                    else:
                        enhanced_height =  height
                    axes[0].scatter(measureangles[i]*np.ones_like(possiblediams), possiblediams, c = enhanced_height, cmap = 'Reds',  vmin=0, vmax=1)
                    
                except (ValueError,TypeError):
                    selectedvalues[i] = np.nan
                    likelihoods[i] = np.nan
                    uncertainties[i] = np.nan
            

            ## Plots the selected set of points (one for each angle) for this particuler choice of startdiam
            axes[0].plot(measureangles, circlipse_rotated(measureangles*np.pi/180,*circlipseguess[:-1]))
            

            ## We restrict ourselves everywhere to indices where the diameter is not NaN and where the angles isn't in the remove list
            notremoved = np.vectorize(lambda x : not(x in removepoints))
            values_indices = np.argwhere(np.logical_and(np.logical_and(np.logical_not(np.isnan(selectedvalues)),selectedvalues != 0),notremoved(measureangles))) #indices where the diameter is not 0 or NaN and not in the remove list
            values = np.ndarray.flatten(selectedvalues[values_indices])
            uncert = np.ndarray.flatten(uncertainties[values_indices])
            sampleangles = np.ndarray.flatten(measureangles[values_indices])
            sampleangles*=np.pi/180 #angles in radians
            
            if len(sampleangles)>=5: 
                try :
                    
                    ##Fits the selection of diameters to a circlipse
                    popt, pcov = curve_fit(circlipse_rotated, sampleangles, values, p0= [10.,1.,1.,0.1], bounds = ([0.,0.,0.,-math.pi/4], [np.inf,np.inf,np.inf,math.pi/4]))
                    print(popt) # prints the params of best-fit circlipse 
                    rmsd = math.sqrt( np.sum((values - circlipse_rotated(sampleangles,*popt))**2) / len(values_indices) ) #root mean square residuals
                    norm_rmsd = rmsd/np.average(circlipse_rotated(measureangles*np.pi/180,*popt)) #normalized RMS residuals 
                    jlik = np.prod(np.ndarray.flatten(likelihoods[values_indices])) #joint likelihood/goodness-of-fit
                    dplus, dminus = popt[0]+popt[1], popt[0]+popt[2] #extremal diameters for this circlipse
                    
                    if jlik > threshold_jlik:
                        print(dplus, dminus) 
                        ##Plots the selection of diameters and the circlipse fit with the same color
                        color = next(axes[1]._get_lines.prop_cycler)['color']
                        axes[1].plot(measureangles, circlipse_rotated(measureangles*np.pi/180,*popt), c=color, label = r'$d_{+}$ = %5.3f $M$ and $d_-$ = %5.3f. Normalized RMSD =%5.3f $\times 10^{-5}$. $\Pi_\varphi g(d_\varphi)$ = %5.3f' % tuple([dplus, dminus, norm_rmsd*1e5, jlik]))
                        axes[1].errorbar(sampleangles*180/np.pi, values, yerr = uncert/2, c = color, marker = 'o', linestyle = 'None')

                except RuntimeError:
                    print('Fit not found for d0=' + str(d0))
                
            else:
                print('Not enough values for d0=' + str(d0))
        
        axes[1].legend(fontsize=13)
        axes[0].tick_params(which='both', labelsize=16)
        axes[1].tick_params(which='both', labelsize=16)
        axes[0].set_xlabel(r'Baseline angle $\varphi$ (°)', size = 20)
        axes[0].set_ylabel(r'Diameter inferred from periodicity (M)',  size = 20)
        axes[1].set_xlabel(r'Baseline angle $\varphi$ (°)',  size = 20)
        axes[1].set_ylabel(r'Diameter inferred from periodicity (M)', size = 20)
    
                    
        
        
        











