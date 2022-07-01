#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GyotoAnalyser V1.0
Created on Thu May 27 12:08:56 2021
Author: Hadrien Paugnat
"""
###########################################################
### WARNING : Only designed for .fits files from GYOTO ####
###########################################################

from main import GyotoImage
import numpy as np
import matplotlib.pyplot as plt


### Give : the adress of the image, the field of view (in muas), a padfactor to improve the FT (e.g. 16), 
###         and a boolean indicating if the edges of the image should be sent to zero before computing visamps (prevents messy visamps)

thindiskimage = GyotoImage("C:/Users/hadri/Documents/Photon ring group (stage LESIA)/GyotoAnalyser/V1/test_image.fits", 
                            181., 16, True)

#thindiskimage = GyotoImage("/home/stage/Stage/GyotoAnalyser/V1/test_image.fits", 
#                            181., 16, True)

# ## Shows the image
# thindiskimage.show_image() 


# ## Computes an intensity slide at a given angle (in degrees)...
# thindiskimage.intensity_slice(0.) 
# ## ... and plots in in a given plot nb, with a given line color - along with its derivative
# thindiskimage.plot_intensity(6,'r')


##### FOURIER SLICE #####

## Choose an angle for a slice
# phimeas = 90
# ## Computes the Radon projection along this angle
# thindiskimage.Radon_projection(phimeas)
# ## Computes the Fourier transform of the Radon (equivalent to a slice at this angle of the 2D FT)
# thindiskimage.FT_Radon()
# ## Plots both 
# thindiskimage.plot_Radon_and_FT(4,'r')



## Choose a baseline domain (in Glambda) for the visamp fit(s) 
## /!\ This choice is critical: make sure that the baseline  domain corresponds to the signature of the ring you want
bounds = (1100,1200)

##### VISAMP FIT #####

## Fits the visamp in order to infer the periodicity (corresponding to the diameter of the ring dominating in this baseline domain)
## Plots this visamp, its minimal/maximal envelopes, the fit 
##      + (if likelihhods is set to True) the goodness-of-fit as a function of diameter (should be a peaked distribution)
# print(phimeas,thindiskimage.visamp_fit_with_envelope(bounds,plot=True, likelihoods = True))


##### SIMPLE CIRCLIPSE FIT #####

## For each one of the nmeas angle, fits the visamp in the given domain -> extracts a angle-dependant diameter 
## !!!! This step is long but can be done once and for all - for these values of bounds (result is saved)
# nmeas = 36
# thindiskimage.sample_diameters(nmeas,bounds)

# ## Fits the angle-dependant diameter to a circlipse
# ## Some angles (in degrees) can be removed from the circlipse fit if added to the list removepoints
# print(thindiskimage.circlipse_fit(bounds, removepoints = [])) ## rotated circlipse best-fit params are printed (R0,R1,R2, phi0)


##### MULTI-FIT TO A CIRCLIPSE #####

thindiskimage.multiple_diameters = True

# ## For each one of the nmeas angle, extracts a peaks in the likelihhod distribution as possible diameters
# ## !!!! This step is long but can be done once and for all - for these values of bounds (result is saved)
# nmeas = 36
# thindiskimage.sample_diameters(nmeas,bounds)

# ## Separates all the possible diameters into circlipse-shaped subsets using the parameters circlipseguess 
# ##            and using startangle as a reference point (make sure that these parameters are well-chosen)
# ## then each subset is fitted to a circlipse (plotted if joint likelihood above threshold_jlik)
# ## Some angles (in degrees) can be removed from the circlipse fit if added to the list removepoints
# print(thindiskimage.circlipse_fit(bounds, startangle = 50., removepoints = [], circlipseguess = [9,   1.4128, 1.2867, 0., 0.015], threshold_jlik = 0.1)) ## rotated circlipse best-fit params are printed (R0,R1,R2, phi0)

