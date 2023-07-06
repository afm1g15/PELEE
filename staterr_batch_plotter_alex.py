#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""@package plotter
Plotter for searchingfornues TTree

This module produces plot from the TTree produced by the
searchingfornues framework (https://github.com/ubneutrinos/searchingfornues)

Example:
    my_plotter = plotter.Plotter(samples, weights)
    fig, ax1, ax2 = my_plotter.plot_variable(
        "reco_e",
        query="selected == 1"
        kind="event_category",
        title="$E_{deposited}$ [GeV]",
        bins=20,
        range=(0, 2)

Attributes:
    category_labels (dict): Description of event categories
    pdg_labels (dict): Labels for PDG codes
    category_colors (dict): Color scheme for event categories
    pdg_colors (dict): Colors scheme for PDG codes
    
    NOTE ALEX: Detsys now dealt with in its own file
"""

import math
import warnings
import bisect

from collections import defaultdict
from collections.abc import Iterable
import scipy.stats
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

import importlib
import Plotter_Functions_Alex
importlib.reload(Plotter_Functions_Alex)
from Plotter_Functions_Alex import plotColourSorting, getWantedLists
importlib.reload(plotColourSorting)
importlib.reload(getWantedLists)
import math as m


category_labels = {
    1: r"$\nu_e$ Other",
    11110: r"$\nu_e$",
    11111: r"$\bar\nu_e$",
    10: r"$\nu_e$ CC0$\pi$0p",
    9: r"$\bar\nu_e$ CC0$\pi$0p",
    12: r"$\bar\nu_e$ CC0$\pi$Np",
    11: r"$\nu_e$ CC0$\pi$Np",
    11357: r"$\nu_e$ CC $\pi^{0}$",
    111: r"MiniBooNE LEE",
    2: r"$\nu_{\mu}$ CC",
    21: r"$\nu_{\mu}$ CC $\pi^{0}$",
    22: r"$\nu_{\mu}$ CC 0p$^+$",
    23: r"$\nu_{\mu}$ CC 1p$^+$",
    24: r"$\nu_{\mu}$ CC 2p$^+$",
    25: r"$\nu_{\mu}$ CC Np$^+$",
    3: r"$\nu$ NC",
    31: r"$\nu$ NC $\pi^{0}$",
    4: r"Cosmic",
    5: r"Out. fid. vol.",
    # eta categories start with 80XX
    801: r"$\eta \rightarrow$ other",
    802: r"$\nu_{\mu} \eta \rightarrow \gamma\gamma$",
    803: r'1 $\pi^0$',
    804: r'2 $\pi^0$',
    805: r'$\nu$ other',
    806: r'out of FV',
    6: r"other",
    0: r"No slice"
}


flux_labels = {
    1: r"$\pi$",
    10: r"K",
    111: r"MiniBooNE LEE",
    0: r"backgrounds"
}

flux_colors = {
    0: "xkcd:cerulean",
    111: "xkcd:goldenrod",
    10: "xkcd:light red",
    12: "xkcd:light red",
    1: "xkcd:purple",
}


pdg_labels = {
    2212: r"$p$",
    13: r"$\mu$",
    11: r"$e$",
    111: r"$\pi^0$",
    -13: r"$\mu$",
    -11: r"$e$",
    211: r"$\pi^{\pm}$",
    -211: r"$\pi$",
    2112: r"$n$",
    22: r"$\gamma$",
    321: r"$K$",
    -321: r"$K$",
    0: "Cosmic"
}

int_labels = {
    0: "QE",
    1: "Resonant",
    2: "DIS",
    3: "Coherent",
    4: "Coherent Elastic",
    5: "Electron scatt.",
    6: "IMDAnnihilation",
    7: r"Inverse $\beta$ decay",
    8: "Glashow resonance",
    9: "AMNuGamma",
    10: "MEC",
    11: "Diffractive",
    12: "EM",
    13: "Weak Mix"
}


int_colors = {
    0: "bisque",
    1: "darkorange",
    2: "goldenrod",
    3: "lightcoral",
    4: "forestgreen",
    5: "turquoise",
    6: "teal",
    7: "deepskyblue",
    80: "steelblue",
    81: "steelblue",
    82: "steelblue",
    9: "royalblue",
    10: "crimson",
    11: "mediumorchid",
    12: "magenta",
    13: "pink",
    111: "black"
}

category_colors = {
    4: "xkcd:light red",
    5: "xkcd:brick",
    8: "xkcd:cerulean",
    2: "xkcd:cyan",
    21: "xkcd:cerulean",
    22: "xkcd:lightblue",
    23: "xkcd:cyan",
    24: "steelblue",
    25: "blue",
    3: "xkcd:cobalt",
    31: "xkcd:sky blue",
    1: "xkcd:moss green",
    10: "xkcd:mint green",
    12: "xkcd:green",
    9 : "xkcd:green",
    11: "xkcd:lime green",
    111: "xkcd:goldenrod",
    6: "xkcd:grey",
    0: "xkcd:black",
    11110:"xkcd:lime green",
    11111:"xkcd:green",
    11357:"xkcd:pink",

    # eta categories
    803: "xkcd:cerulean",
    804: "xkcd:blue",
    801: "xkcd:purple",
    802: "xkcd:lavender",
    806: "xkcd:crimson",
    805: "xkcd:cyan",
}

pdg_colors = {
    2212: "#a6cee3",
    22: "#1f78b4",
    13: "#b2df8a",
    211: "#33a02c",
    111: "#137e6d",
    0: "#e31a1c",
    11: "#ff7f00",
    321: "#fdbf6f",
    2112: "#cab2d6",
}

class Plotter:
    """Main plotter class

    Args:
        samples (dict): Dictionary of pandas dataframes.
            mc`, `nue`, `data`, and `ext` are required. `lee` and `dirt` are optional.
        weights (dict): Dictionary of global dataframes weights.
            One for each entry in the samples dict.
        pot (int): Number of protons-on-target. Defaults is 4.5e19.

    Attributes:
       samples (dict): Dictionary of pandas dataframes.
       weights (dict): Dictionary of global dataframes weights.
       pot (int): Number of protons-on-target.
    """

    def __init__(self, samples, weights, pot=4.5e19):
        self.weights = weights
        self.samples = samples
        self.pot = pot
        self.significance = 0
        self.significance_likelihood = 0
        self.chisqdatamc = 0
        self.sigma_shapeonly = 0
        self.detsys = None
        self.stats = {}
        self.cov = None # covariance matrix from systematics
        self.cov_mc_stat = None
        self.cov_data_stat = None
        self.cov_full = None
        self._ratio_vals = None
        self._ratio_errs = None
        self.data = None # data binned events

        self.nu_pdg = nu_pdg = "~(abs(nu_pdg) == 12 & ccnc == 0)" # query to avoid double-counting events in MC sample with other MC samples

        if ("ccpi0" in self.samples):
            self.nu_pdg = self.nu_pdg+" & ~(mcf_pass_ccpi0==1)"
        if ("ncpi0" in self.samples):
            self.nu_pdg = self.nu_pdg+" & ~(mcf_np0==1 & mcf_nmp==0 & mcf_nmm==0 & mcf_nem==0 & mcf_nep==0)" #note: mcf_pass_ccpi0 is wrong (includes 'mcf_actvol' while sample is in all cryostat)
        if ("ccnopi" in self.samples):
            self.nu_pdg = self.nu_pdg+" & ~(mcf_pass_ccnopi==1 & (nslice==0 | (slnunhits/slnhits)>0.1))"
        if ("cccpi" in self.samples):
            self.nu_pdg = self.nu_pdg+" & ~(mcf_pass_cccpi==1 & (nslice==0 | (slnunhits/slnhits)>0.1))"
        if ("nccpi" in self.samples):
            self.nu_pdg = self.nu_pdg+" & ~(mcf_pass_nccpi==1 & (nslice==0 | (slnunhits/slnhits)>0.1))"
        if ("ncnopi" in self.samples):
            self.nu_pdg = self.nu_pdg+" & ~(mcf_pass_ncnopi==1 & (nslice==0 | (slnunhits/slnhits)>0.1))"

        necessary = ["category"]

        ##OVERLAY/MC
        #nue_missing = np.setdiff1d(necessary, samples["nue_mc"].columns)
        #numu_missing = np.setdiff1d(necessary, samples["numu_mc"].columns)

        #if nue_missing.size > 0 or numu_missing.size > 0:
        #    raise ValueError(
        #        "Missing necessary columns in the DataFrame: ")
##MC /OVERLAY SECTION
    @staticmethod
    def _chisquare(data, overlay, err_mc):
        num = (data - overlay)**2
        den = data+err_mc**2
        if np.count_nonzero(data):
            return sum(num / den) / len(data)
        return np.inf


    @staticmethod
    def _chisq_pearson(data, mc):
        return (data-mc)**2 / mc

    @staticmethod
    def _chisq_neyman(data, mc):
        return (data-mc)**2 / data

    def _chisq_CNP(self,data, mc):
        return np.sum((1/3.) * (self._chisq_neyman(data,mc) + 2 * self._chisq_pearson(data,mc)))/len(data)

    @staticmethod
    def _sigma_calc_likelihood(sig, bkg, err_bkg, scale_factor=1):
        """It calculates the significance with the profile likelihood ratio
        assuming an uncertainity on the background entries.
        Taken from http://www.pp.rhul.ac.uk/~cowan/stat/medsig/medsigNote.pdf
        """
        b = bkg * scale_factor
        if not isinstance(err_bkg, Iterable):
            e = np.array([err_bkg]) * scale_factor
        else:
            e = err_bkg * scale_factor

        s = sig * scale_factor

        p1 = (s+b)*np.log((s+b)*(b+e**2)/(b**2+(s+b)*e**2))

        p2 = -s
        if sum(e) > 0:
            p2 = -b**2/(e**2)*np.log(1+e**2*s/(b*(b+e**2)))
        z = 2*(p1+p2)

        return math.sqrt(sum(z))

    @staticmethod
    def _sigma_calc_matrix(signal, background, scale_factor=1, cov=None):
        """It calculates the significance as the square root of the Δχ2 score

        Args:
            signal (np.array): array of signal histogram
            background (np.array): array of background histogram
            scale_factor (float, optional): signal and background scaling factor.
                Default is 1

        Returns:
            Square root of S•B^(-1)•S^T
        """

        bkg_array = background * scale_factor
        empty_elements = np.where(bkg_array == 0)[0]
        sig_array = signal * scale_factor
        cov = cov * scale_factor * scale_factor
        sig_array = np.delete(sig_array, empty_elements)
        bkg_array = np.delete(bkg_array, empty_elements)
        cov[np.diag_indices_from(cov)] += bkg_array
        emtxinv = np.linalg.inv(cov)
        chisq = float(sig_array.dot(emtxinv).dot(sig_array.T))

        return np.sqrt(chisq)


    def deltachisqfakedata(self, BinMin, BinMax, LEE_v, SM_v, nsample):

        deltachisq_v = []
        deltachisq_SM_v  = []
        deltachisq_LEE_v = []

        #print('deltachisqfakedata!!!!!!')
        
        for n in range(1000):

            SM_obs, LEE_obs = self.genfakedata(BinMin, BinMax, LEE_v, SM_v, nsample)

            #chisq = self._chisq_CNP(SM_obs,LEE_obs)           
            #print ('LEE obs : ',LEE_obs)
            #print ('SM obs : ',SM_obs)
            
            chisq_SM_SM  = self._chisq_CNP(SM_v,SM_obs)
            chisq_LEE_SM = self._chisq_CNP(LEE_v,SM_obs)
            
            chisq_SM_LEE  = self._chisq_CNP(SM_v,LEE_obs)
            chisq_LEE_LEE = self._chisq_CNP(LEE_v,LEE_obs)
            
            deltachisq_SM  = (chisq_SM_SM-chisq_LEE_SM)
            deltachisq_LEE = (chisq_SM_LEE-chisq_LEE_LEE)

            #if (np.isnan(chisq)):
            #    continue

            #deltachisq_v.append(chisq)
            
            if (np.isnan(deltachisq_SM ) or np.isnan(deltachisq_LEE) ):
                continue

            deltachisq_SM_v.append(deltachisq_SM)
            deltachisq_LEE_v.append(deltachisq_LEE)

        #median = np.median(deltachisq_v)
        #dof = len(LEE_v)

        #return median/float(dof)

        #print ('delta SM  : ',deltachisq_SM_v)
        #print ('delta LEE : ',deltachisq_LEE_v)

        deltachisq_SM_v  = np.array(deltachisq_SM_v)
        deltachisq_LEE_v = np.array(deltachisq_LEE_v)

        if (len(deltachisq_SM_v) == 0):
            return 999.
        
        # find median of LEE distribution
        med_LEE = np.median(deltachisq_LEE_v)
        #print ('median LEE is ',med_LEE)
        # how many values in SM are above this value?
        nabove = len( np.where(deltachisq_SM_v > med_LEE)[0] )
        #print ('n above is ',nabove)
        frac = float(nabove) / len(deltachisq_SM_v)

        #print ('deltachisqfakedata!!!!!!')
        
        return math.sqrt(2)*scipy.special.erfinv(1-frac*2)
        
        #return frac

            
    def genfakedata(self, BinMin, BinMax, LEE_v, SM_v, nsample):

        p_LEE = LEE_v / np.sum(LEE_v)
        p_SM  = SM_v / np.sum(SM_v)

        #print ('PDF for LEE : ',p_LEE)
        #print ('PDF for SM  : ',p_SM)

        obs_LEE = np.zeros(len(LEE_v))
        obs_SM  = np.zeros(len(SM_v))

        max_LEE = np.max(p_LEE)
        max_SM  = np.max(p_SM)

        #print ('max of LEE : ',max_LEE)
        #print ('max of SM  : ',max_SM)

        n_sampled_LEE = 0
        n_sampled_SM  = 0

        while (n_sampled_LEE < nsample):

            value = BinMin + (BinMax-BinMin) * np.random.random()

            BinNumber = int((value-BinMin)/(BinMax-BinMin) * len(LEE_v))
            
            prob = np.random.random() * max_LEE
            if (prob < p_LEE[BinNumber]):
                #print ('LEE simulation: prob of %.02f vs. bin prob of %.02f leads to selecting event at bin %i'%(prob,p_LEE[BinNumber],BinNumber))
                obs_LEE[BinNumber] += 1
                n_sampled_LEE += 1

        while (n_sampled_SM < nsample):

            value = BinMin + (BinMax-BinMin) * np.random.random()

            BinNumber = int((value-BinMin)/(BinMax-BinMin) * len(SM_v))
            
            prob = np.random.random() * max_SM
            if (prob < p_SM[BinNumber]):
                obs_SM[BinNumber] += 1
                n_sampled_SM += 1

        return obs_SM, obs_LEE
    
    
    def add_detsys_error(self,sample,mc_entries_v,weight):
        #get empty arrays to fill
        detsys_v  = np.zeros(len(mc_entries_v))
        entries_v = np.zeros(len(mc_entries_v))
        print("LOOK HERE FOR DETSYS")
        print(self.detsys)
        #return if no detsys provided
        if (self.detsys == None): return detsys_v
        #looking for see if the sample (i.e. "nue_nue") is present in the detsys provided
        if sample in self.detsys:
            #if the detsys sample is the same length as the current uncertainties
            if (len(self.detsys[sample]) == len(mc_entries_v)):
                for i,n in enumerate(mc_entries_v):
                    #multiply the sample by the uncertainty value and by the sample weight
                    detsys_v[i] = (self.detsys[sample][i] * n * weight)#**2
                    entries_v[i] = n * weight
            else:
                print ('NO MATCH! len detsys : %i. Len plotting : %i'%(len(self.detsys[sample]),len(mc_entries_v) ))

        return detsys_v

            
            
##MC/OVERLAY CHANGES
    def _chisq_full_covariance(self,data, mc,key, CNP=True,STATONLY=False):

        np.set_printoptions(precision=3)

        dof = len(data)
        if key == "nue": 
            COV = self.nue_cov + self.nue_cov_mc_stat + self.nue_cov_mc_detsys
        elif key == "numu":
            COV = self.numu_cov + self.numu_cov_mc_stat + self.numu_cov_mc_detsys
            
        # remove rows/columns with zero data and MC
        remove_indices_v = []
        for i,d in enumerate(data):
            idx = len(data)-i-1
            if ((data[idx]==0) and (mc[idx] == 0)):
                remove_indices_v.append(idx)

        for idx in remove_indices_v:
            COV = np.delete(COV,idx,0)
            COV = np.delete(COV,idx,1)
            data = np.delete(data,idx,0)
            mc   = np.delete(mc,idx,0)


        COV_STAT = np.zeros([len(data), len(data)])


        ERR_STAT = 3. / ( 1./data + 2./mc )
        
        for i,d in enumerate(data):
            
            if (d == 0):
                ERR_STAT[i] = mc[i]/2.
            if (mc[i] == 0):
                ERR_STAT[i] = d

        if (CNP == False):
            ERR_STAT = data + mc
        

        COV_STAT[np.diag_indices_from(COV_STAT)] = ERR_STAT

        COV += COV_STAT

        if (STATONLY == True):
            COV = COV_STAT

        #print("COV matrix : ",COV)
                
        diff = (data-mc)
        emtxinv = np.linalg.inv(COV)
        chisq = float(diff.dot(emtxinv).dot(diff.T))
        
        covdiag = np.diag(COV)
        chisqsum = 0.
        for i,d in enumerate(diff):
            #print ('bin %i has COV value %.02f'%(i,covdiag[i]))
            chisqsum += ( (d**2) /covdiag[i])

        return chisq, chisqsum, dof

    @staticmethod
    def _data_err(data,doAsym=False):
        obs = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
        low = [0.00,0.17,0.71,1.37,2.09,2.84,3.62,4.42,5.23,6.06,6.89,7.73,8.58,9.44,10.30,11.17,12.04,12.92,13.80,14.68,15.56]
#        hig = [0.38,3.30,4.64,5.92,7.16,8.38,9.58,10.77,11.95,13.11,14.27,15.42,16.56,17.70,18.83,19.96,21.08,22.20,23.32,24.44,25.55]
        hig = [1.15,3.30,4.64,5.92,7.16,8.38,9.58,10.77,11.95,13.11,14.27,15.42,16.56,17.70,18.83,19.96,21.08,22.20,23.32,24.44,25.55]
        if doAsym:
            lb = [i-low[i] if i<=20 else (np.sqrt(i)) for i in data]
            hb = [hig[i]-i if i<=20 else (np.sqrt(i)) for i in data]
            return (lb,hb)
        else: return (np.sqrt(data),np.sqrt(data))


    @staticmethod
    def _ratio_err(num, den, num_err, den_err):
        n, d, n_e, d_e = num, den, num_err, den_err
        n[n == 0] = 0.00001
        #d[d == 0] = 0.00001
        return np.array([
            #n[i] / d[i] * math.sqrt((n_e[i] / n[i])**2 + (d_e[i] / d[i])**2) <= this does not work if n[i]==0
            math.sqrt( ( n_e[i] / d[i] )**2 + ( n[i] * d_e[i] / (d[i]*d[i]) )**2) if d[i]>0 else 0
            for i, k in enumerate(num)
        ])

    @staticmethod
    def _is_fiducial(x, y, z):
        try:
            x_1 = x[:, 0] > 10
            x_2 = x[:, 1] > 10
            y_1 = y[:, 0] > 15
            y_2 = y[:, 1] > 15
            z_1 = z[:, 0] > 10
            z_2 = z[:, 1] > 50

            return x_1 & x_2 & y_1 & y_2 & z_1 & z_2
        except IndexError:
            return True

    def print_stats(self):
        print ('print stats...')
        for key,val in self.stats.items():
            print ('%s : %.02f'%(key,val))


    def _select_showers(self, variable, variable_name, sample, query="selected==1", score=0.5, extra_cut=None):
        variable = variable.ravel()

        if variable.size > 0:
            if isinstance(variable[0], np.ndarray):
                variable = np.hstack(variable)
                if "shr" in variable_name and variable_name != "shr_score_v":
                    shr_score = np.hstack(self._selection(
                        "shr_score_v", sample, query=query, extra_cut=extra_cut))
                    shr_score_id = shr_score < score
                    variable = variable[shr_score_id]
                elif "trk" in variable_name and variable_name != "trk_score_v":
                    trk_score = np.hstack(self._selection(
                        "trk_score_v", sample, query=query, extra_cut=extra_cut))
                    trk_score_id = trk_score >= score
                    variable = variable[trk_score_id]

        return variable


    def _apply_track_cuts(self,df,variable,track_cuts,mask):
        '''
        df is dataframe of the sample of interest
        variable is what values will be in the output
        track_cuts are list of tuples defining track_cuts
        input mask to be built upon

        returns
            Series of values of variable that pass all track_cuts
            boolean mask that represents union of input mask and new cut mask
        '''
        #need to do this fancy business with the apply function to make masks
        #this is because unflattened DataFrames are used
        for (var,op,val) in track_cuts:
            if type(op) == list:
                #this means treat two conditions in an 'or' fashion
                or_mask1 = df[var].apply(lambda x: eval("x{}{}".format(op[0],val[0])))#or condition 1
                or_mask2 = df[var].apply(lambda x: eval("x{}{}".format(op[1],val[1])))#or condition 2
                mask *= (or_mask1 + or_mask2) #just add the booleans for "or"
            else:
                mask *= df[var].apply(lambda x: eval("x{}{}".format(op,val))) #layer on each cut mask
        vars = (df[variable]*mask).apply(lambda x: x[x != False]) #apply mask
        vars = vars[vars.apply(lambda x: len(x) > 0)] #clean up empty slices
        #fix list comprehension issue for non '_v' variables
        if variable[-2:] != "_v":
            vars = vars.apply(lambda x: x[0])
        elif "_v" not in variable:
            print("_v not found in variable, assuming event-level")
            print("not fixing list comprehension bug for this variable")

        return vars, mask

    def _select_longest(self,df, variable, mask):
        '''
        df: dataframe for sample
        variable: Series of values that pass cuts defined by mask
        mask: mask used to find variable

        returns
            list of values of variable corresponding to longest track in each slices
            boolean mask for longest tracks in df
        '''

        #print("selecting longest...")
        #print("mask", mask)
        trk_lens = (df['trk_len_v']*mask).apply(lambda x: x[x != False])#apply mask to track lengths
        trk_lens = trk_lens[trk_lens.apply(lambda x: len(x) > 0)]#clean up slices
        variable = variable.apply(lambda x: x[~np.isnan(x)])#clean up nan vals
        variable = variable[variable.apply(lambda x: len(x) > 0)] #clean up empty slices
        nan_mask = variable.apply(lambda x: np.nan in x or "nan" in x)
        longest_mask = trk_lens.apply(lambda x: x == x[list(x).index(max(x))])#identify longest
        variable = (variable*longest_mask).apply(lambda x: x[x!=False])#apply mask
        if len(variable.iloc[0]) == 1:
            variable = variable.apply(lambda x: x[0] if len(x)>0 else -999)#expect values, not lists, for each event
        else:
            if len(variable.iloc[0]) == 0:
                raise ValueError(
                    "There is no longest track per slice")
            elif len(variable.iloc[0]) > 1:
                #this happens with the reco_nu_e_range_v with unreconstructed values
                print("there are more than one longest slice")
                print(variable.iloc[0])
                try:
                    variable = variable.apply(lambda x: x[0])
                except:
                    raise ValueError(
                        "There is more than one longest track per slice in \n var {} lens {}".format(variable,trk_lens))

        return variable, longest_mask

    def _selection(self, variable, sample, query="selected==1", extra_cut=None, track_cuts=None, select_longest=True):
        '''
        variable,  must be specified
        select_longest, True by default, keeps from multiple tracks of same event making it through
        query must be a string defining event-level cuts
        track_cuts is a list of cuts of which each entry looks like
            (variable_tobe_cut_on, '>'or'<'or'=='etc, cut value )
            or
            (variable, [operator1, operator2], [cutval1, cutval2]) to do an 'or' cut
        track_
        returns an Series of values that pass all track_cuts
        '''
        sel_query = query
        if extra_cut is not None:
            sel_query += "& %s" % extra_cut
        '''
        if ( (track_cuts == None) or (select_longest == False) ):
            return sample.query(sel_query).eval(variable).ravel()
        '''


        '''
        df = sample.query(sel_query)
        #print (df.isna().sum())
        dfna = df.isna()
        for (colname,colvals) in dfna.iteritems():
            if (colvals.sum() != 0):
                print ('name : ',colname)
                print ('nan entries : ',colvals.sum())
        '''  
        df = sample.query(sel_query)
        
        #if (track_cuts != None):
        #    df = sample.query(sel_query).dropna().copy() #don't want to eliminate anything from memory

        #df = sample.query(sel_query).dropna().copy() #don't want to eliminate anything from memory

        track_cuts_mask = None #df['trk_score_v'].apply(lambda x: x == x) #all-True mask, assuming trk_score_v is available
        if track_cuts is not None:
            vars, track_cuts_mask = self._apply_track_cuts(df,variable,track_cuts,track_cuts_mask)
        else:
            vars = df[variable]
        #vars is now a Series object that passes all the cuts
        #select longest of the cut passing tracks
        #assuming all track-level variables end in _v
        if variable[-2:] == "_v" and select_longest:
            vars, longest_mask = self._select_longest(df, vars, track_cuts_mask)
        elif "_v_" in variable:
            print("Variable is being interpretted as event-level, not track_level, despite having _v in name")
            print("the longest track is NOT being selected")
        return vars.ravel()

    def _categorize_entries_pdg(self, sample, variable, query="selected==1", extra_cut=None, track_cuts=None, select_longest=True):

        if "trk" in variable:
            pfp_id_variable = "trk_pfp_id"
            score_v = self._selection("trk_score_v", sample, query=query, extra_cut=extra_cut, track_cuts=track_cuts, select_longest=select_longest)
        else:
            pfp_id_variable = "shr_pfp_id_v"
            score_v = self._selection("shr_score_v", sample, query=query, extra_cut=extra_cut, track_cuts=track_cuts, select_longest=select_longest)


        pfp_id = self._selection(
            pfp_id_variable, sample, query=query, extra_cut=extra_cut, track_cuts=track_cuts, select_longest=select_longest)
        plotted_variable = self._selection(
            variable, sample, query=query, extra_cut=extra_cut, track_cuts=track_cuts, select_longest=select_longest)
        pfp_id = np.subtract(pfp_id, 1)
        backtracked_pdg = np.abs(self._selection(
            "backtracked_pdg", sample, query=query, extra_cut=extra_cut, track_cuts=track_cuts, select_longest=select_longest))

        plotted_variable = self._select_showers(
            plotted_variable, variable, sample, query=query, extra_cut=extra_cut)

        if "trk" in variable:
            pfp_id = np.array([pf_id[score > 0.5] for pf_id, score in zip(pfp_id, score_v)])
        else:
            pfp_id = np.array([pf_id[score <= 0.5] for pf_id, score in zip(pfp_id, score_v)])

        pfp_pdg = np.array([pdg[pf_id]
                            for pdg, pf_id in zip(backtracked_pdg, pfp_id)])
        pfp_pdg = np.hstack(pfp_pdg)
        pfp_pdg = abs(pfp_pdg)

        return pfp_pdg, plotted_variable

    def _categorize_entries_single_pdg(self, sample, variable, query="selection==1", extra_cut=None, track_cuts=None, select_longest=True):
        if "trk" in variable:
            bkt_variable = "trk_bkt_pdg"
        else:
            bkt_variable = "shr_bkt_pdg"

        backtracked_pdg = np.abs(self._selection(
            bkt_variable, sample, query=query, extra_cut=extra_cut, track_cuts=track_cuts, select_longest=select_longest))
        plotted_variable = self._selection(
            variable, sample, query=query, extra_cut=extra_cut, track_cuts=track_cuts, select_longest=select_longest)

        return backtracked_pdg, plotted_variable

    def _categorize_entries(self, sample, variable, query="selected==1", extra_cut=None, track_cuts=None, select_longest=True):
        category = self._selection(
            "category", sample, query=query, extra_cut=extra_cut, track_cuts=track_cuts, select_longest=select_longest)
        plotted_variable = self._selection(
            variable, sample, query=query, extra_cut=extra_cut, track_cuts=track_cuts, select_longest=select_longest)

        if plotted_variable.size > 0:
            if isinstance(plotted_variable[0], np.ndarray):
                if "trk" in variable or select_longest:
                    score = self._selection(
                        "trk_score_v", sample, query=query, extra_cut=extra_cut, track_cuts=track_cuts, select_longest=select_longest)
                    category = np.array([
                        np.array([c] * len(v[s > 0.5])) for c, v, s in zip(category, plotted_variable, score)
                    ])
                else:
                    score = self._selection(
                        "shr_score_v", sample, query=query, extra_cut=extra_cut, track_cuts=track_cuts, select_longest=select_longest)
                    category = np.array([
                        np.array([c] * len(v[s < 0.5])) for c, v, s in zip(category, plotted_variable, score)
                    ])
                category = np.hstack(category)

            plotted_variable = self._select_showers(
                plotted_variable, variable, sample, query=query, extra_cut=extra_cut)

        return category, plotted_variable

    def _categorize_entries_int(self, sample, variable, query="selected==1", extra_cut=None, track_cuts=None, select_longest=True):
        category = self._selection(
            "interaction", sample, query=query, extra_cut=extra_cut, track_cuts=track_cuts, select_longest=select_longest)
        plotted_variable = self._selection(
            variable, sample, query=query, extra_cut=extra_cut, track_cuts=track_cuts, select_longest=select_longest)
        return category, plotted_variable

    def _categorize_entries_flux(self, sample, variable, query="selected==1", extra_cut=None, track_cuts=None, select_longest=True):
        category = self._selection(
            "flux", sample, query=query, extra_cut=extra_cut, track_cuts=track_cuts, select_longest=select_longest)
        plotted_variable = self._selection(
            variable, sample, query=query, extra_cut=extra_cut, track_cuts=track_cuts, select_longest=select_longest)


        if plotted_variable.size > 0:
            if isinstance(plotted_variable[0], np.ndarray):
                if "trk" in variable or select_longest:
                    score = self._selection(
                        "trk_score_v", sample, query=query, extra_cut=extra_cut, track_cuts=track_cuts, select_longest=select_longest)
                    category = np.array([
                        np.array([c] * len(v[s > 0.5])) for c, v, s in zip(category, plotted_variable, score)
                    ])
                else:
                    score = self._selection(
                        "shr_score_v", sample, query=query, extra_cut=extra_cut, track_cuts=track_cuts, select_longest=select_longest)
                    category = np.array([
                        np.array([c] * len(v[s < 0.5])) for c, v, s in zip(category, plotted_variable, score)
                    ])
                category = np.hstack(category)

            plotted_variable = self._select_showers(
                plotted_variable, variable, sample, query=query, extra_cut=extra_cut)

        return category, plotted_variable



    @staticmethod
    def _variable_bin_scaling(bins, bin_width, variable):
        idx = bisect.bisect_left(bins, variable)
        if len(bins) > idx:
            return bin_width/(bins[idx]-bins[idx-1])
        return 0

    def _get_genie_weight(self, sample, variable, query="selected==1", extra_cut=None, track_cuts=None,\
                          select_longest=True, weightvar="weightSplineTimesTuneTimesPPFX",weightsignal=None):

        plotted_variable = self._selection(
            variable, sample, query=query, extra_cut=extra_cut, track_cuts=track_cuts, select_longest=select_longest)
        genie_weights = self._selection(
            weightvar, sample, query=query, extra_cut=extra_cut, track_cuts=track_cuts, select_longest=select_longest)
        if (weightsignal != None):
            genie_weights *= self._selection(
            weightsignal, sample, query=query, extra_cut=extra_cut, track_cuts=track_cuts, select_longest=select_longest)
        if plotted_variable.size > 0:
            if isinstance(plotted_variable[0], np.ndarray):
                if "trk" in variable or select_longest:
                    score = self._selection(
                        "trk_score_v", sample, query=query, extra_cut=extra_cut, track_cuts=track_cuts, select_longest=select_longest)
                else:
                    score = self._selection(
                        "shr_score_v", sample, query=query, extra_cut=extra_cut, track_cuts=track_cuts, select_longest=select_longest)
                genie_weights = np.array([
                    np.array([c] * len(v[s > 0.5])) for c, v, s in zip(genie_weights, plotted_variable, score)
                ])
                genie_weights = np.hstack(genie_weights)
        return genie_weights

    def _get_variable(self, variable, query, track_cuts=None):

        ##MC/OVERLAY CHANGED HERE
        nue_mc_plotted_variable = self._selection(
            variable, self.samples["nue_mc"], query=query, extra_cut=self.nu_pdg, track_cuts=track_cuts)
        nue_mc_plotted_variable = self._select_showers(
            mc_plotted_variable, variable, self.samples["nue_mc"], query=query, extra_cut=self.nu_pdg)
        nue_mc_weight = [self.weights["nue_mc"]] * len(mc_plotted_variable)
        
        numu_mc_plotted_variable = self._selection(
            variable, self.samples["numu_mc"], query=query, extra_cut=self.nu_pdg, track_cuts=track_cuts)
        numu_mc_plotted_variable = self._select_showers(
            mc_plotted_variable, variable, self.samples["numu_mc"], query=query, extra_cut=self.nu_pdg)
        numu_mc_weight = [self.weights["numu_mc"]] * len(mc_plotted_variable)

        nue_nue_plotted_variable = self._selection(
            variable, self.samples["nue_nue"], query=query, track_cuts=track_cuts)
        nue_nue_plotted_variable = self._select_showers(
            nue_plotted_variable, variable, self.samples["nue_nue"], query=query)
        nue_nue_weight = [self.weights["nue_nue"]] * len(nue_plotted_variable)
        
        numu_nue_plotted_variable = self._selection(
            variable, self.samples["numu_nue"], query=query, track_cuts=track_cuts)
        numu_nue_plotted_variable = self._select_showers(
            nue_plotted_variable, variable, self.samples["numu_nue"], query=query)
        numu_nue_weight = [self.weights["numu_nue"]] * len(nue_plotted_variable)

        nue_ext_plotted_variable = self._selection(
            variable, self.samples["nue_ext"], query=query, track_cuts=track_cuts)
        nue_ext_plotted_variable = self._select_showers(
            ext_plotted_variable, variable, self.samples["nue_ext"], query=query)
        nue_ext_weight = [self.weights["nue_ext"]] * len(ext_plotted_variable)
        
        numu_ext_plotted_variable = self._selection(
            variable, self.samples["numu_ext"], query=query, track_cuts=track_cuts)
        numu_ext_plotted_variable = self._select_showers(
            ext_plotted_variable, variable, self.samples["numu_ext"], query=query)
        numu_ext_weight = [self.weights["numu_ext"]] * len(ext_plotted_variable)

        nue_dirt_weight = []
        nue_dirt_plotted_variable = []
        if "nue_dirt" in self.samples:
            nue_dirt_plotted_variable = self._selection(
                variable, self.samples["nue_dirt"], query=query, track_cuts=track_cuts)
            nue_dirt_plotted_variable = self._select_showers(
                dirt_plotted_variable, variable, self.samples["nue_dirt"], query=query)
            nue_dirt_weight = [self.weights["nue_dirt"]] * len(dirt_plotted_variable)
            
        numu_dirt_weight = []
        numu_dirt_plotted_variable = []
        if "numu_dirt" in self.samples:
            numu_dirt_plotted_variable = self._selection(
                variable, self.samples["numu_dirt"], query=query, track_cuts=track_cuts)
            numu_dirt_plotted_variable = self._select_showers(
                dirt_plotted_variable, variable, self.samples["numu_dirt"], query=query)
            numu_dirt_weight = [self.weights["numu_dirt"]] * len(dirt_plotted_variable)

        #DO WE NEED TO DUPLICATE HERE??
        ncpi0_weight = []
        ncpi0_plotted_variable = []
        if "ncpi0" in self.samples:
            ncpi0_plotted_variable = self._selection(
                variable, self.samples["ncpi0"], query=query, track_cuts=track_cuts)
            ncpi0_plotted_variable = self._select_showers(
                ncpi0_plotted_variable, variable, self.samples["ncpi0"], query=query)
            ncpi0_weight = [self.weights["ncpi0"]] * len(ncpi0_plotted_variable)

        ccpi0_weight = []
        ccpi0_plotted_variable = []
        if "ccpi0" in self.samples:
            ccpi0_plotted_variable = self._selection(
                variable, self.samples["ccpi0"], query=query, track_cuts=track_cuts)
            ccpi0_plotted_variable = self._select_showers(
                ccpi0_plotted_variable, variable, self.samples["ccpi0"], query=query)
            ccpi0_weight = [self.weights["ccpi0"]] * len(ccpi0_plotted_variable)

        ccnopi_weight = []
        ccnopi_plotted_variable = []
        if "ccnopi" in self.samples:
            ccnopi_plotted_variable = self._selection(
                variable, self.samples["ccnopi"], query=query, track_cuts=track_cuts)
            ccnopi_plotted_variable = self._select_showers(
                ccnopi_plotted_variable, variable, self.samples["ccnopi"], query=query)
            ccnopi_weight = [self.weights["ccnopi"]] * len(ccnopi_plotted_variable)

        cccpi_weight = []
        cccpi_plotted_variable = []
        if "cccpi" in self.samples:
            cccpi_plotted_variable = self._selection(
                variable, self.samples["cccpi"], query=query, track_cuts=track_cuts)
            cccpi_plotted_variable = self._select_showers(
                cccpi_plotted_variable, variable, self.samples["cccpi"], query=query)
            cccpi_weight = [self.weights["cccpi"]] * len(cccpi_plotted_variable)

        nccpi_weight = []
        nccpi_plotted_variable = []
        if "nccpi" in self.samples:
            nccpi_plotted_variable = self._selection(
                variable, self.samples["nccpi"], query=query, track_cuts=track_cuts)
            nccpi_plotted_variable = self._select_showers(
                nccpi_plotted_variable, variable, self.samples["nccpi"], query=query)
            nccpi_weight = [self.weights["nccpi"]] * len(nccpi_plotted_variable)

        ncnopi_weight = []
        ncnopi_plotted_variable = []
        if "ncnopi" in self.samples:
            ncnopi_plotted_variable = self._selection(
                variable, self.samples["ncnopi"], query=query, track_cuts=track_cuts)
            ncnopi_plotted_variable = self._select_showers(
                ncnopi_plotted_variable, variable, self.samples["ncnopi"], query=query)
            ncnopi_weight = [self.weights["ncnopi"]] * len(ncnopi_plotted_variable)

        lee_weight = []
        lee_plotted_variable = []
        if "lee" in self.samples:
            lee_plotted_variable = self._selection(
                variable, self.samples["lee"], query=query, track_cuts=track_cuts)
            lee_plotted_variable = self._select_showers(
                lee_plotted_variable, variable, self.samples["lee"], query=query)
            lee_weight = self.samples["lee"].query(
                query)["leeweight"] * self.weights["lee"]

        nue_total_weight = np.concatenate((nue_mc_weight, nue_nue_weight, nue_ext_weight, nue_dirt_weight, ncpi0_weight, ccpi0_weight, ccnopi_weight, cccpi_weight, nccpi_weight, ncnopi_weight, lee_weight))
        nuw_total_variable = np.concatenate((nue_mc_plotted_variable, nue_nue_plotted_variable, nue_ext_plotted_variable, nue_dirt_plotted_variable, ncpi0_plotted_variable, ccpi0_plotted_variable, ccnopi_plotted_variable, cccpi_plotted_variable, nccpi_plotted_variable, ncnopi_plotted_variable, lee_plotted_variable))
        numu_total_weight = np.concatenate((numu_mc_weight, numu_nue_weight, numu_ext_weight, numu_dirt_weight, ncpi0_weight, ccpi0_weight, ccnopi_weight, cccpi_weight, nccpi_weight, ncnopi_weight, lee_weight))
        numu_total_variable = np.concatenate((numu_mc_plotted_variable, numu_nue_plotted_variable, numu_ext_plotted_variable, numu_dirt_plotted_variable, ncpi0_plotted_variable, ccpi0_plotted_variable, ccnopi_plotted_variable, cccpi_plotted_variable, nccpi_plotted_variable, ncnopi_plotted_variable, lee_plotted_variable))
        return nue_total_variable, nue_total_weight, numu_total_variable, numu_total_weight


    def plot_variable(self, variable, query="selected==1", currentsample = "nue_nue", title="", kind="event_category", draw_geoSys=False,
                      draw_sys=False, stacksort=0, track_cuts=None, select_longest=False,
                      detsys=None,ratio=True,chisq=False,draw_data=True,asymErrs=False,genieweight="weightSplineTimesTuneTimesPPFX",
                      ncol=2,
                      COVMATRIX='', # path to covariance matrix file
                      DETSYSPATH="", # path where to find detector systematics files
                      **plot_options):
        """It plots the variable from the TTree, after applying an eventual query

        Args:
            variable (str): name of the variable.
            query (str): pandas query. Default is ``selected``.
            title (str, optional): title of the plot. Default is ``variable``.
            kind (str, optional): Categorization of the plot.
                Accepted values are ``event_category``, ``particle_pdg``, and ``sample``
                Default is ``event_category``.
            track_cuts (list of tuples (var, operation, cut val), optional):
                List of cuts ot be made on track-level variables ("_v" in variable name)
                These get applied one at a time in self._selection
            select_longest (bool): if variable is a track-level variable
                setting to True will take the longest track of each slice
                    after QUERY and track_cuts have been applied
                select_longest = False might have some bugs...
            **plot_options: Additional options for matplotlib plot (e.g. range and bins).

        Returns:
            Figure, top subplot, and bottom subplot (ratio)

        """
        #if (detsys != None):
        self.detsys = detsys

        if not title:
            title = variable
        if not query:
            query = "nslice==1"
            
        # pandas bug https://github.com/pandas-dev/pandas/issues/16363
        if plot_options["range"][0] >= 0 and plot_options["range"][1] >= 0 and variable[-2:] != "_v":
            query += "& %s <= %g & %s >= %g" % (
                variable, plot_options["range"][1], variable, plot_options["range"][0])

        #eventually used to subdivide monte-carlo sample
        if kind == "event_category":
            categorization = self._categorize_entries
            cat_labels = category_labels          
        elif kind == "particle_pdg":
            var = self.samples["nue_mc"].query(nue_query).eval(variable)
            if var.dtype == np.float32:
                categorization = self._categorize_entries_single_pdg
            else:
                categorization = self._categorize_entries_pdg
            cat_labels = pdg_labels
        elif kind == "interaction":
            categorization = self._categorize_entries_int
            cat_labels = int_labels
        elif kind == "flux":
            categorization = self._categorize_entries_flux
            cat_labels = flux_labels
        elif kind == "sample":
            return self._plot_variable_samples(variable, query, title, asymErrs, **plot_options)
        else:
            raise ValueError(
                "Unrecognized categorization, valid options are 'sample', 'event_category', and 'particle_pdg'")


        nu_pdg = "~(abs(nu_pdg) == 12 & ccnc == 0)"
        if ("ccpi0" in self.samples):
            nu_pdg = nu_pdg+" & ~(mcf_pass_ccpi0==1)"
        if ("ncpi0" in self.samples):
            nu_pdg = nu_pdg+" & ~(mcf_np0==1 & mcf_nmp==0 & mcf_nmm==0 & mcf_nem==0 & mcf_nep==0)" #note: mcf_pass_ccpi0 is wrong (includes 'mcf_actvol' while sample is in all cryostat)
        if ("ccnopi" in self.samples):
            nu_pdg = nu_pdg+" & ~(mcf_pass_ccnopi==1 & (nslice==0 | (slnunhits/slnhits)>0.1))"
        if ("cccpi" in self.samples):
            nu_pdg = nu_pdg+" & ~(mcf_pass_cccpi==1 & (nslice==0 | (slnunhits/slnhits)>0.1))"
        if ("nccpi" in self.samples):
            nu_pdg = nu_pdg+" & ~(mcf_pass_nccpi==1 & (nslice==0 | (slnunhits/slnhits)>0.1))"
        if ("ncnopi" in self.samples):
            nu_pdg = nu_pdg+" & ~(mcf_pass_ncnopi==1 & (nslice==0 | (slnunhits/slnhits)>0.1))"

        print(query,"\n", self.nu_pdg,"\n",track_cuts,"\n",select_longest)
        
        print("")
        
        ##OVERLAY MC CHANGE HERE
        if (currentsample == "nue_nue"):
            print("current sample is: ", currentsample)
            current_category, current_plotted_variable = categorization(
                self.samples["nue_nue"], variable, query=query, track_cuts=track_cuts, select_longest=select_longest)
        elif (currentsample == "nue_mc"):
            print("current sample is: ", currentsample)
            current_category, current_plotted_variable = categorization(
                self.samples["nue_mc"], variable, query=query, extra_cut=self.nu_pdg, track_cuts=track_cuts, select_longest=select_longest)
        elif (currentsample == "nue_dirt"):
            print("current sample is: ", currentsample)
            current_category, current_plotted_variable = categorization(
                self.samples["nue_dirt"], variable, query=query, track_cuts=track_cuts, select_longest=select_longest) 
        elif (currentsample == "numu_mc"):
            print("current sample is: ", currentsample)
            current_category, current_plotted_variable = categorization(
                self.samples["numu_mc"], variable, query=query, extra_cut=self.nu_pdg, track_cuts=track_cuts, select_longest=select_longest)
        elif (currentsample == "numu_dirt"):
            print("current sample is: ", currentsample)
            current_category, current_plotted_variable = categorization(
                self.samples["numu_dirt"], variable, query=query, track_cuts=track_cuts, select_longest=select_longest)   


        print("")

        bins = np.arange(0, 5.5, 0.5)

        if (currentsample == "nue_nue"):
            current_uncertainties, current_bins = np.histogram(
                current_plotted_variable, **plot_options)
            #print("Rounding to 3dp")
            #current_uncertainties = np.round(current_uncertainties, 3)
            current_err = np.array(
                [n * self.weights["nue_nue"] * self.weights["nue_nue"] for n in current_uncertainties])
            current_detsys = self.add_detsys_error("nue_nue", current_uncertainties, self.weights["nue_nue"])
            print("err nue: ")
            print(current_err)
            print("detsys nue: ")
            print(current_detsys)
        elif (currentsample == "nue_mc"):
            current_uncertainties, current_bins = np.histogram(
                current_plotted_variable, **plot_options)
            #current_uncertainties = np.round(current_uncertainties, 3)
            current_err = np.array(
                [n * self.weights["nue_mc"] * self.weights["nue_mc"] for n in current_uncertainties])
            current_detsys = self.add_detsys_error("nue_mc", current_uncertainties, self.weights["nue_mc"])
            print("err nue mc: ")
            print(current_err)
            print("detsys nue mc: ")
            print(current_detsys)
        elif (currentsample == "nue_dirt"):
            current_uncertainties, current_bins = np.histogram(
                current_plotted_variable, **plot_options)
            #current_uncertainties = np.round(current_uncertainties, 3)
            current_err = np.array(
                [n * self.weights["nue_dirt"] * self.weights["nue_dirt"] for n in current_uncertainties])
            current_detsys = self.add_detsys_error("nue_dirt", current_uncertainties, self.weights["nue_dirt"])
            print("err nue dirt: ")
            print(current_err)
            print("detsys nue dirt: ")
            print(current_detsys)
        elif (currentsample == "numu_mc"):
            current_uncertainties, current_bins = np.histogram(
                current_plotted_variable, **plot_options)
            #current_uncertainties = np.round(current_uncertainties, 3)
            current_err = np.array(
                [n * self.weights["numu_mc"] * self.weights["numu_mc"] for n in current_uncertainties])
            current_detsys = self.add_detsys_error("numu_mc", current_uncertainties, self.weights["numu_mc"])
            print("err numu mc: ")
            print(current_err)
            print("detsys numu mc: ")
            print(current_detsys)
        elif (currentsample == "numu_dirt"):
            current_uncertainties, current_bins = np.histogram(
                current_plotted_variable, **plot_options)
            #current_uncertainties = np.round(current_uncertainties, 3)
            current_err = np.array(
                [n * self.weights["numu_dirt"] * self.weights["numu_dirt"] for n in current_uncertainties])
            current_detsys = self.add_detsys_error("numu_dirt", current_uncertainties, self.weights["numu_dirt"])
            print("err numu dirt: ")
            print(current_err)
            print("detsys numu dirt: ")
            print(current_detsys)
        

        print("")
        x_range=plot_options["range"]
        n_bins=plot_options["bins"] 
        weightVar=genieweight
        
        if (currentsample == "nue_nue"):
            current_tree = self.samples["nue_nue"]
            current_queried_tree = current_tree.query(query)
            variable = current_queried_tree[variable]
            spline_fix_cv  = current_queried_tree[weightVar] * self.weights["nue_nue"]
            current_selected, bins = np.histogram(
                        variable,
                        range=x_range,
                        bins=n_bins,
                        weights=spline_fix_cv)  
            print("selected ", current_selected)
            #print("Rounding to 3dp")
            #current_selected = np.round(current_selected, 3)
            #print("selected ", current_selected)
        elif (currentsample == "nue_mc"):
            current_tree = self.samples["nue_mc"]
            extra_query = "& " + self.nu_pdg
            current_queried_tree = current_tree.query(query+extra_query)
            variable = current_queried_tree[variable]
            spline_fix_cv  = current_queried_tree[weightVar] * self.weights["nue_mc"]
            current_selected, bins = np.histogram(
                        variable,
                        range=x_range,
                        bins=n_bins,
                        weights=spline_fix_cv)  
            print("selected ", current_selected)
            #print("Rounding to 3dp")
            #current_selected = np.round(current_selected, 3)
            #print("selected ", current_selected)
        elif (currentsample == "nue_dirt"):
            current_tree = self.samples["nue_dirt"]
            current_queried_tree = current_tree.query(query)
            variable = current_queried_tree[variable]
            spline_fix_cv  = current_queried_tree[weightVar] * self.weights["nue_dirt"]
            current_selected, bins = np.histogram(
                        variable,
                        range=x_range,
                        bins=n_bins,
                        weights=spline_fix_cv)  
            print("selected ", current_selected)
            #print("Rounding to 3dp")
            #current_selected = np.round(current_selected, 3)
            #print("selected ", current_selected)
        elif (currentsample == "numu_mc"):
            current_tree = self.samples["numu_mc"]
            extra_query = "& " + self.nu_pdg
            current_queried_tree = current_tree.query(query+extra_query)
            variable = current_queried_tree[variable]
            spline_fix_cv  = current_queried_tree[weightVar] * self.weights["numu_mc"]
            current_selected, bins = np.histogram(
                        variable,
                        range=x_range,
                        bins=n_bins,
                        weights=spline_fix_cv)  
            print("selected ", current_selected)
            #print("Rounding to 3dp")
            #current_selected = np.round(current_selected, 3)
            #print("selected ", current_selected)
        elif (currentsample == "numu_dirt"):
            current_tree = self.samples["numu_dirt"]
            current_queried_tree = current_tree.query(query)
            variable = current_queried_tree[variable]
            spline_fix_cv  = current_queried_tree[weightVar] * self.weights["numu_dirt"]
            current_selected, bins = np.histogram(
                        variable,
                        range=x_range,
                        bins=n_bins,
                        weights=spline_fix_cv)  
            print("selected ", current_selected)
            #print("Rounding to 3dp")
            #current_selected = np.round(current_selected, 3)
            #print("selected ", current_selected)
        
        
        
        
        if ratio and draw_data:
            return nue_fig, nue_ax1, nue_ax2, nue_stacked, labels, labels
        elif ratio:
            return nue_fig, nue_ax1, nue_ax2, nue_stacked, labels, numu_fig, numu_ax1, labels
        elif draw_data:
            print("Returning")
            return current_err, current_detsys, current_selected
        else:
            return nue_fig, nue_ax1, nue_stacked, labels, nue_order_var_dict, nue_order_weight_dict

    def _plot_variable_samples(self, variable, query, title, asymErrs, **plot_options):

        if plot_options["range"][0] >= 0 and plot_options["range"][1] >= 0 and variable[-2:] != "_v":
            query += "& %s <= %g & %s >= %g" % (
                variable, plot_options["range"][1], variable, plot_options["range"][0])

            ##OVERLAY/MC
        mc_plotted_variable = self._selection(
            variable, self.samples["mc"], query=query, extra_cut=self.nu_pdg)
        mc_plotted_variable = self._select_showers(
            mc_plotted_variable, variable, self.samples["mc"], query=query, extra_cut=self.nu_pdg)
        mc_weight = [self.weights["mc"]] * len(mc_plotted_variable)

        nue_plotted_variable = self._selection(
            variable, self.samples["nue"], query=query)
        nue_plotted_variable = self._select_showers(
            nue_plotted_variable, variable, self.samples["nue"], query=query)
        nue_weight = [self.weights["nue"]] * len(nue_plotted_variable)

        ext_plotted_variable = self._selection(
            variable, self.samples["ext"], query=query)
        ext_plotted_variable = self._select_showers(
            ext_plotted_variable, variable, self.samples["ext"], query=query)
        ext_weight = [self.weights["ext"]] * len(ext_plotted_variable)
        

        if "dirt" in self.samples:
            dirt_plotted_variable = self._selection(
                variable, self.samples["dirt"], query=query)
            dirt_plotted_variable = self._select_showers(
                dirt_plotted_variable, variable, self.samples["dirt"], query=query)
            dirt_weight = [self.weights["dirt"]] * len(dirt_plotted_variable)

        if "ncpi0" in self.samples:
            ncpi0_plotted_variable = self._selection(
                variable, self.samples["ncpi0"], query=query)
            ncpi0_plotted_variable = self._select_showers(
                ncpi0_plotted_variable, variable, self.samples["ncpi0"], query=query)
            ncpi0_weight = [self.weights["ncpi0"]] * len(ncpi0_plotted_variable)

        if "ccpi0" in self.samples:
            ccpi0_plotted_variable = self._selection(
                variable, self.samples["ccpi0"], query=query)
            ccpi0_plotted_variable = self._select_showers(
                ccpi0_plotted_variable, variable, self.samples["ccpi0"], query=query)
            ccpi0_weight = [self.weights["ccpi0"]] * len(ccpi0_plotted_variable)

        if "ccnopi" in self.samples:
            ccnopi_plotted_variable = self._selection(
                variable, self.samples["ccnopi"], query=query)
            ccnopi_plotted_variable = self._select_showers(
                ccnopi_plotted_variable, variable, self.samples["ccnopi"], query=query)
            ccnopi_weight = [self.weights["ccnopi"]] * len(ccnopi_plotted_variable)

        if "cccpi" in self.samples:
            cccpi_plotted_variable = self._selection(
                variable, self.samples["cccpi"], query=query)
            cccpi_plotted_variable = self._select_showers(
                cccpi_plotted_variable, variable, self.samples["cccpi"], query=query)
            cccpi_weight = [self.weights["cccpi"]] * len(cccpi_plotted_variable)

        if "nccpi" in self.samples:
            nccpi_plotted_variable = self._selection(
                variable, self.samples["nccpi"], query=query)
            nccpi_plotted_variable = self._select_showers(
                nccpi_plotted_variable, variable, self.samples["nccpi"], query=query)
            nccpi_weight = [self.weights["nccpi"]] * len(nccpi_plotted_variable)

        if "ncnopi" in self.samples:
            ncnopi_plotted_variable = self._selection(
                variable, self.samples["ncnopi"], query=query)
            ncnopi_plotted_variable = self._select_showers(
                ncnopi_plotted_variable, variable, self.samples["ncnopi"], query=query)
            ncnopi_weight = [self.weights["ncnopi"]] * len(ncnopi_plotted_variable)

        if "lee" in self.samples:
            lee_plotted_variable = self._selection(
                variable, self.samples["lee"], query=query)
            lee_plotted_variable = self._select_showers(
                lee_plotted_variable, variable, self.samples["lee"], query=query)
            lee_weight = self.samples["lee"].query(query)["leeweight"] * self.weights["lee"]


        data_plotted_variable = self._selection(
            variable, self.samples["data"], query=query)
        data_plotted_variable = self._select_showers(
            data_plotted_variable,
            variable,
            self.samples["data"],
            query=query)

        if "dirt" in self.samples:
            total_variable = np.concatenate(
                [mc_plotted_variable,
                 nue_plotted_variable,
                 ext_plotted_variable,
                 dirt_plotted_variable])
            total_weight = np.concatenate(
                [mc_weight, nue_weight, ext_weight, dirt_weight])
        else:
            total_variable = np.concatenate(
                [mc_plotted_variable, nue_plotted_variable, ext_plotted_variable])
            total_weight = np.concatenate(
                [mc_weight, nue_weight, ext_weight])

        if "lee" in self.samples:
            total_variable = np.concatenate(
                [total_variable,
                 lee_plotted_variable])
            total_weight = np.concatenate(
                [total_weight, lee_weight])

        if "ncpi0" in self.samples:
            total_variable = np.concatenate(
                [total_variable,
                 ncpi0_plotted_variable])
            total_weight = np.concatenate(
                [total_weight, ncpi0_weight])

        if "ccpi0" in self.samples:
            total_variable = np.concatenate(
                [total_variable,
                 ccpi0_plotted_variable])
            total_weight = np.concatenate(
                [total_weight, ccpi0_weight])

        if "ccnopi" in self.samples:
            total_variable = np.concatenate(
                [total_variable,
                 ccnopi_plotted_variable])
            total_weight = np.concatenate(
                [total_weight, ccnopi_weight])

        if "cccpi" in self.samples:
            total_variable = np.concatenate(
                [total_variable,
                 cccpi_plotted_variable])
            total_weight = np.concatenate(
                [total_weight, cccpi_weight])

        if "nccpi" in self.samples:
            total_variable = np.concatenate(
                [total_variable,
                 nccpi_plotted_variable])
            total_weight = np.concatenate(
                [total_weight, nccpi_weight])

        if "ncnopi" in self.samples:
            total_variable = np.concatenate(
                [total_variable,
                 ncnopi_plotted_variable])
            total_weight = np.concatenate(
                [total_weight, ncnopi_weight])


        fig = plt.figure(figsize=(7, 7))
        #fig = plt.figure(figsize=(8, 7))
        gs = gridspec.GridSpec(1, 1)#, height_ratios=[2, 1])
        #gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
        #print (gs[0])

        ax1 = plt.subplot(gs[0])
        #ax2 = plt.subplot(gs[1])

        n_mc, mc_bins, patches = ax1.hist(
            mc_plotted_variable,
            weights=mc_weight,
            label="NuMI overlay: %.1f entries" % sum(mc_weight),
            **plot_options)

        n_nue, nue_bins, patches = ax1.hist(
            nue_plotted_variable,
            bottom=n_mc,
            weights=nue_weight,
            label=r"$\nu_{e}$ overlay: %.1f entries" % sum(nue_weight),
            **plot_options)

        n_dirt = 0
        if "dirt" in self.samples:
            n_dirt, dirt_bins, patches = ax1.hist(
                dirt_plotted_variable,
                bottom=n_mc + n_nue,
                weights=dirt_weight,
                label=r"Dirt: %.1f entries" % sum(dirt_weight),
                **plot_options)

        n_ncpi0 = 0
        if "ncpi0" in self.samples:
            n_ncpi0, ncpi0_bins, patches = ax1.hist(
                ncpi0_plotted_variable,
                bottom=n_mc + n_nue + n_dirt,
                weights=ncpi0_weight,
                label=r"NC$\pi^0$: %.1f entries" % sum(ncpi0_weight),
                **plot_options)

        n_ccpi0 = 0
        if "ccpi0" in self.samples:
            n_ccpi0, ccpi0_bins, patches = ax1.hist(
                ccpi0_plotted_variable,
                bottom=n_mc + n_nue + n_dirt + n_ncpi0,
                weights=ccpi0_weight,
                label=r"CC$\pi^0$: %.1f entries" % sum(ccpi0_weight),
                **plot_options)

        n_ccnopi = 0
        if "ccnopi" in self.samples:
            n_ccnopi, ccnopi_bins, patches = ax1.hist(
                ccnopi_plotted_variable,
                bottom=n_mc + n_nue + n_dirt + n_ncpi0 + n_ccpi0,
                weights=ccnopi_weight,
                label=r"CCNoPi: %.1f entries" % sum(ccnopi_weight),
                **plot_options)

        n_cccpi = 0
        if "cccpi" in self.samples:
            n_cccpi, cccpi_bins, patches = ax1.hist(
                cccpi_plotted_variable,
                bottom=n_mc + n_nue + n_dirt + n_ncpi0 + n_ccpi0 + n_ccnopi,
                weights=cccpi_weight,
                label=r"CCPi+: %.1f entries" % sum(cccpi_weight),
                **plot_options)

        n_nccpi = 0
        if "nccpi" in self.samples:
            n_nccpi, nccpi_bins, patches = ax1.hist(
                nccpi_plotted_variable,
                bottom=n_mc + n_nue + n_dirt + n_ncpi0 + n_ccpi0 + n_ccnopi + n_cccpi,
                weights=nccpi_weight,
                label=r"NCcPi: %.1f entries" % sum(nccpi_weight),
                **plot_options)

        n_ncnopi = 0
        if "ncnopi" in self.samples:
            n_ncnopi, ncnopi_bins, patches = ax1.hist(
                ncnopi_plotted_variable,
                bottom=n_mc + n_nue + n_dirt + n_ncpi0 + n_ccpi0 + n_ccnopi + n_cccpi + n_nccpi,
                weights=ncnopi_weight,
                label=r"Ncnopi: %.1f entries" % sum(ncnopi_weight),
                **plot_options)

        n_lee = 0
        if "lee" in self.samples:
            n_lee, lee_bins, patches = ax1.hist(
                lee_plotted_variable,
                bottom=n_mc + n_nue + n_dirt + n_ncpi0 + n_ccpi0 + n_ccnopi + n_cccpi + n_nccpi + n_ncnopi,
                weights=lee_weight,
                label=r"MiniBooNE LEE: %.1f entries" % sum(lee_weight),
                **plot_options)

        n_ext, ext_bins, patches = ax1.hist(
            ext_plotted_variable,
            bottom=n_mc + n_nue + n_dirt + n_lee + n_ncpi0 + n_ccpi0 + n_ccnopi + n_cccpi + n_nccpi + n_ncnopi,
            weights=ext_weight,
            label="EXT: %.1f entries" % sum(ext_weight),
            hatch="//",
            color="white",
            **plot_options)

        n_tot, tot_bins, patches = ax1.hist(
            total_variable,
            weights=total_weight,
            histtype="step",
            edgecolor="black",
            **plot_options)

        #ERRORS
        mc_uncertainties, bins = np.histogram(
            mc_plotted_variable, **plot_options)
        nue_uncertainties, bins = np.histogram(
            nue_plotted_variable, **plot_options)
        ext_uncertainties, bins = np.histogram(
            ext_plotted_variable, **plot_options)
        err_mc = np.array([n * self.weights["mc"] * self.weights["mc"] for n in mc_uncertainties])
        err_nue = np.array(
            [n * self.weights["nue"] * self.weights["nue"] for n in nue_uncertainties])
        err_ext = np.array(
            [n * self.weights["ext"] * self.weights["ext"] for n in ext_uncertainties])
        err_dirt = np.array([0 for n in n_mc])
        err_lee = np.array([0 for n in n_mc])
            
        if "dirt" in self.samples:
            dirt_uncertainties, bins = np.histogram(dirt_plotted_variable, **plot_options)
            err_dirt = np.array(
                [n * self.weights["dirt"] * self.weights["dirt"] for n in dirt_uncertainties])

        err_ncpi0 = np.array([0 for n in n_mc])
        if "ncpi0" in self.samples:
            ncpi0_uncertainties, bins = np.histogram(ncpi0_plotted_variable, **plot_options)
            err_ncpi0 = np.array(
                [n * self.weights["ncpi0"] * self.weights["ncpi0"] for n in ncpi0_uncertainties])

        err_ccpi0 = np.array([0 for n in n_mc])
        if "ccpi0" in self.samples:
            ccpi0_uncertainties, bins = np.histogram(ccpi0_plotted_variable, **plot_options)
            err_ccpi0 = np.array(
                [n * self.weights["ccpi0"] * self.weights["ccpi0"] for n in ccpi0_uncertainties])

        err_ccnopi = np.array([0 for n in n_mc])
        if "ccnopi" in self.samples:
            ccnopi_uncertainties, bins = np.histogram(ccnopi_plotted_variable, **plot_options)
            err_ccnopi = np.array(
                [n * self.weights["ccnopi"] * self.weights["ccnopi"] for n in ccnopi_uncertainties])

        err_cccpi = np.array([0 for n in n_mc])
        if "cccpi" in self.samples:
            cccpi_uncertainties, bins = np.histogram(cccpi_plotted_variable, **plot_options)
            err_cccpi = np.array(
                [n * self.weights["cccpi"] * self.weights["cccpi"] for n in cccpi_uncertainties])

        err_nccpi = np.array([0 for n in n_mc])
        if "nccpi" in self.samples:
            nccpi_uncertainties, bins = np.histogram(nccpi_plotted_variable, **plot_options)
            err_nccpi = np.array(
                [n * self.weights["nccpi"] * self.weights["nccpi"] for n in nccpi_uncertainties])

        err_ncnopi = np.array([0 for n in n_mc])
        if "ncnopi" in self.samples:
            ncnopi_uncertainties, bins = np.histogram(ncnopi_plotted_variable, **plot_options)
            err_ncnopi = np.array(
                [n * self.weights["ncnopi"] * self.weights["ncnopi"] for n in ncnopi_uncertainties])

        if "lee" in self.samples:
            if isinstance(plot_options["bins"], Iterable):
                lee_bins = plot_options["bins"]
            else:
                bin_size = (
                    plot_options["range"][1] - plot_options["range"][0])/plot_options["bins"]
                lee_bins = [plot_options["range"][0]+n *
                            bin_size for n in range(plot_options["bins"]+1)]

            binned_lee = pd.cut(self.samples["lee"].query(
                query).eval(variable), lee_bins)
            err_lee = self.samples["lee"].query(query).groupby(binned_lee)['leeweight'].agg(
                "sum").values * self.weights["lee"] * self.weights["lee"]

        #Full Error?
        exp_err = np.sqrt(err_mc + err_ext + err_nue + err_dirt + err_lee + err_ncpi0 + err_ccpi0 + err_ccnopi + err_cccpi + err_nccpi + err_ncnopi)
        print("exp_err = ", exp_err)

        bincenters = 0.5 * (tot_bins[1:] + tot_bins[:-1])
        bin_size = [(tot_bins[i + 1] - tot_bins[i]) / 2
                    for i in range(len(tot_bins) - 1)]
        ax1.bar(bincenters, n_tot, width=0, yerr=exp_err)

        n_data, bins = np.histogram(data_plotted_variable, **plot_options)
        data_err = self._data_err(n_data,asymErrs)
        ax1.errorbar(
            bincenters,
            n_data,
            xerr=bin_size,
            yerr=data_err,
            fmt='ko',
            label="NuMI: %i events" % len(data_plotted_variable))

        leg = ax1.legend(
            frameon=False, title=r'MicroBooNE Preliminary %g POT' % self.pot)
        leg._legend_box.align = "left"
        plt.setp(leg.get_title(), fontweight='bold')

        unit = title[title.find("[") + 1:title.find("]")
                     ] if "[" and "]" in title else ""
        xrange = plot_options["range"][1] - plot_options["range"][0]
        if isinstance(bins, Iterable):
            ax1.set_ylabel("N. Entries")
        else:
            ax1.set_ylabel(
                "N. Entries / %g %s" % (xrange / plot_options["bins"], unit))
        #ax1.set_xticks([])
        ax1.set_xlim(plot_options["range"][0], plot_options["range"][1])

        #self._draw_ratio(ax2, bins, n_tot, n_data, exp_err, data_err)

        #ax2.set_xlabel(title)
        ax1.set_xlabel(title)
        #ax2.set_xlim(plot_options["range"][0], plot_options["range"][1])
        fig.tight_layout()
        # fig.savefig("plots/%s_samples.pdf" % variable)
        return fig, ax1#, ax2

    def _draw_ratio(self, ax, bins, n_tot, n_data, tot_err, data_err, draw_data=True):
        bincenters = 0.5 * (bins[1:] + bins[:-1])
        bin_size = [(bins[i + 1] - bins[i]) / 2 for i in range(len(bins) - 1)]
        if draw_data:
            data_err_low = self._ratio_err(n_data, n_tot, data_err[0], np.zeros(len(data_err[0])))
            data_err_high = self._ratio_err(n_data, n_tot, data_err[1], np.zeros(len(data_err[1])))
            ratio_error = (data_err_low,data_err_high)
            ax.errorbar(bincenters, n_data / n_tot,
                    xerr=bin_size, yerr=ratio_error, fmt="ko")

            ratio_error_mc = self._ratio_err(n_tot, n_tot, tot_err, np.zeros(len(n_tot)))
            ratio_error_mc = np.insert(ratio_error_mc, 0, ratio_error_mc[0])
            bins = np.array(bins)
            ratio_error_mc = np.array(ratio_error_mc)
        self._ratio_vals = n_data / n_tot
        self._ratio_errs = ratio_error
        ax.fill_between(
            bins,
            1.0 - ratio_error_mc,
            ratio_error_mc + 1,
            step="pre",
            color="tab:blue",
            alpha=0.5)

        ax.set_ylim(0, 2)
        ax.set_ylabel("NuMI / (MC+EXT)")
        ax.axhline(1, linestyle="--", color="k")
   

    
    def plot_smearing(self, selected, signal, true, reco, bins, norm): 
        fig = plt.figure(figsize=(10, 6))

        smear = plt.hist2d(selected.query(signal)[true],selected.query(signal)[reco],
                       bins, cmin=0.000000001, cmap='OrRd')

        for i in range(len(bins)-1): # reco bins i (y axis) rows
            for j in range(len(bins)-1): # true bins j (x axis) cols
                if smear[0].T[i,j] > 0: 
                    if smear[0].T[i,j]>80: 
                        col='white'
                    else: 
                        col='black'

                    binx_centers = smear[1][j]+(smear[1][j+1]-smear[1][j])/2
                    biny_centers = smear[2][i]+(smear[2][i+1]-smear[2][i])/2

                    plt.text(binx_centers, biny_centers, round(smear[0].T[i,j], 1), 
                        color=col, ha="center", va="center", fontsize=12)

        cbar = plt.colorbar()
        cbar.set_label('Selected Signal Events', fontsize=15)
        print(norm)
        
        if norm: 
            plt.close()

            norm_array = smear[0].T

            # for each truth bin (column): 
            for j in range(len(bins)-1): 

                reco_events_in_column = [ norm_array[i][j] for i in range(len(bins)-1) ]
                tot_reco_events = np.nansum(reco_events_in_column)

                # replace with normalized value 
                for i in range(len(bins)-1): 
                    norm_array[i][j] =  norm_array[i][j] / tot_reco_events

            # now plot
            fig = plt.figure(figsize=(10, 6))
            plt.pcolor(bins, bins, norm_array, cmap='OrRd', vmax=1)

            # Loop over data dimensions and create text annotations.
            for i in range(len(bins)-1): # reco bins (rows)
                for j in range(len(bins)-1): # truth bins (cols)
                    if norm_array[i][j]>0: 

                        if norm_array[i][j]>0.7: 
                            col = 'white'
                        else: 
                            col = 'black'

                        binx_centers = smear[1][j]+(smear[1][j+1]-smear[1][j])/2
                        biny_centers = smear[2][i]+(smear[2][i+1]-smear[2][i])/2

                        plt.text(binx_centers, biny_centers, round(norm_array[i][j], 2), 
                             ha="center", va="center", color=col, fontsize=12)

            cbar = plt.colorbar()
            cbar.set_label('Fraction of Reco Events in True Bin', fontsize=15)

        plt.xlabel('True Nu Energy [GeV]', fontsize=15)
        plt.ylabel('Reco Nu Energy [GeV]', fontsize=15)
        plt.text(0.1, 4.2, r'MicroBooNE Preliminary', fontweight='bold')

        #plt.show()
        return norm_array

    
    def plot_signal_and_eff_and_B(self, selected, df, signal, bins, truth): 

        # generated true signal events per bin 
        gen = plt.hist(df.query(signal)['nu_e'], bins, color='deepskyblue')
        plt.close()
        print("Full numbers = ", gen[0])

        #This should give selected numbers in bin
        #print(selected['neutrino_energy'])
        #This should be the total numbers in bin
        #print(df.query(signal)['neutrino_energy'])

        # plot selected signal events 
        fig, ax1 = plt.subplots(figsize=(4, 5))

        sel = ax1.hist(truth['nu_e'], bins, color='deepskyblue')
        ax1.set_ylabel('Selected Signal Events', fontsize=15)
        ax1.set_xlabel('True Numu Energy [GeV]', fontsize=15)

        # compute efficiency
        sel = ax1.hist(selected['nu_e'], bins, color='white')
        print("Selected numbers = ", sel[0])
        eff = [ a/b for a, b in zip(sel[0], gen[0]) ]
        eff_err = []
        for i in range(len(eff)):
            eff_err.append(math.sqrt( (eff[i]*(1-eff[i]))/gen[0][i] ) )
            print("In bin", i, ", eff = ", eff[i], " with error = ", eff_err[i])

        # compute bin centers 
        bc = 0.5*(sel[1][1:]+sel[1][:-1])
        x_err = []
        for i in range(len(sel[1])-1): 
            x_err.append((sel[1][i+1]-sel[1][i])/2)

        # plot efficiency
        sel = ax1.hist(truth['nu_e'], bins, color='deepskyblue')
        ax1.set_ylim(0, 2000)
        ax2 = ax1.twinx()
        ax2.errorbar(bc, eff, xerr=x_err, yerr=eff_err, fmt='o', color='orangered', ecolor='orangered', markersize=3) 
        ax2.set_ylim(0, 1.00)
        ax2.set_ylabel('Efficiency', fontsize=15)
        ax2.set_title("True Numu Energy and Efficiency") 
        plt.text(0, 0.95, r'MicroBooNE Preliminary', fontweight='bold')
        plt.close()
       

        #plt.show()
        return eff
    
    def div_err(self, res, err1, val1, err2, val2):
        res_err = res * np.sqrt((err1/val1)**2 + (err2/val2)**2)
        return res_err

