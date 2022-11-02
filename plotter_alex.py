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
    )

Attributes:
    category_labels (dict): Description of event categories
    pdg_labels (dict): Labels for PDG codes
    category_colors (dict): Color scheme for event categories
    pdg_colors (dict): Colors scheme for PDG codes
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

        if "nue_dirt" and "numu_dirt" not in samples:
            warnings.warn("Missing dirt sample")

        necessary = ["category"]#, "selected",  # "trk_pfp_id", "shr_pfp_id_v",
                     #"backtracked_pdg", "nu_pdg", "ccnc", "trk_bkt_pdg", "shr_bkt_pdg"]

        ##OVERLAY/MC
        nue_missing = np.setdiff1d(necessary, samples["nue_mc"].columns)
        numu_missing = np.setdiff1d(necessary, samples["numu_mc"].columns)

        if nue_missing.size > 0 or numu_missing.size > 0:
            raise ValueError(
                "Missing necessary columns in the DataFrame: ")
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

        '''
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
        '''

        # if plot_options["range"][0] >= 0 and plot_options["range"][1] >= 0 and variable[-2:] != "_v":
        #     query += "& %s <= %g & %s >= %g" % (
        #         variable, plot_options["range"][1], variable, plot_options["range"][0])

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
        nue_total_variable = np.concatenate((nue_mc_plotted_variable, nue_nue_plotted_variable, nue_ext_plotted_variable, nue_dirt_plotted_variable, ncpi0_plotted_variable, ccpi0_plotted_variable, ccnopi_plotted_variable, cccpi_plotted_variable, nccpi_plotted_variable, ncnopi_plotted_variable, lee_plotted_variable))
        numu_total_weight = np.concatenate((numu_mc_weight, numu_nue_weight, numu_ext_weight, numu_dirt_weight, ncpi0_weight, ccpi0_weight, ccnopi_weight, cccpi_weight, nccpi_weight, ncnopi_weight, lee_weight))
        numu_total_variable = np.concatenate((numu_mc_plotted_variable, numu_nue_plotted_variable, numu_ext_plotted_variable, numu_dirt_plotted_variable, ncpi0_plotted_variable, ccpi0_plotted_variable, ccnopi_plotted_variable, cccpi_plotted_variable, nccpi_plotted_variable, ncnopi_plotted_variable, lee_plotted_variable))
        return nue_total_variable, nue_total_weight, numu_total_variable, numu_total_weight


    def plot_2d(self, variable1_name, variable2_name, query="selected==1", track_cuts=None, **plot_options):
        nue_variable1, nue_weight1, numu_variable1, numu_weight1 = self._get_variable(variable1_name, query, track_cuts=track_cuts)
        nue_variable2, nue_weight2, numu_variable2, numu_weight2 = self._get_variable(variable2_name, query, track_cuts=track_cuts)

        nue_heatmap, nue_xedges, nue_yedges = np.histogram2d(nue_variable1, nue_variable2,
                                                 range=[[plot_options["range_x"][0], plot_options["range_x"][1]], [plot_options["range_y"][0], plot_options["range_y"][1]]],
                                                 bins=[plot_options["bins_x"], plot_options["bins_y"]],
                                                 weights=nue_weight1)

        nue_extent = [nue_xedges[0], nue_xedges[-1], nue_yedges[0], nue_yedges[-1]]
        nue_fig, nue_axes  = plt.subplots(1,3, figsize=(15,5))

        nue_axes[0].imshow(nue_heatmap.T, extent=nue_extent, origin='lower', aspect="auto")
        
        numu_heatmap, numu_xedges, numu_yedges = np.histogram2d(numu_variable1, numu_variable2,
                                                 range=[[plot_options["range_x"][0], plot_options["range_x"][1]], [plot_options["range_y"][0], plot_options["range_y"][1]]],
                                                 bins=[plot_options["bins_x"], plot_options["bins_y"]],
                                                 weights=numu_weight1)

        numu_extent = [numu_xedges[0], numu_xedges[-1], numu_yedges[0], numu_yedges[-1]]
        numu_fig, numu_axes  = plt.subplots(1,3, figsize=(15,5))

        numu_axes[0].imshow(numu_heatmap.T, extent=numu_extent, origin='lower', aspect="auto")

        nue_data_variable1 = self._selection(variable1_name, self.samples["nue_data"], query=query, track_cuts=track_cuts)
        nue_data_variable1 = self._select_showers(nue_data_variable1, variable1_name, self.samples["nue_data"], query=query)

        nue_data_variable2 = self._selection(
            variable2_name, self.samples["nue_data"], query=query, track_cuts=track_cuts)
        nue_data_variable2 = self._select_showers(
            nue_data_variable2, variable2_name, self.samples["nue_data"], query=query)
        
        numu_data_variable1 = self._selection(variable1_name, self.samples["numu_data"], query=query, track_cuts=track_cuts)
        numu_data_variable1 = self._select_showers(numu_data_variable1, variable1_name, self.samples["numu_data"], query=query)

        numu_data_variable2 = self._selection(
            variable2_name, self.samples["numu_data"], query=query, track_cuts=track_cuts)
        numu_data_variable2 = self._select_showers(
            numu_data_variable2, variable2_name, self.samples["numu_data"], query=query)

        nue_heatmap_data, nue_xedges, nue_yedges = np.histogram2d(nue_data_variable1, nue_data_variable2, range=[[plot_options["range_x"][0], plot_options["range_x"][1]], [
                                                      plot_options["range_y"][0], plot_options["range_y"][1]]],
                                                      bins=[plot_options["bins_x"],
                                                      plot_options["bins_y"]])

        nue_axes[1].imshow(nue_heatmap_data.T, extent=nue_extent, origin='lower', aspect="auto")
        
        numu_heatmap_data, numu_xedges, numu_yedges = np.histogram2d(numu_data_variable1, numu_data_variable2, range=[[plot_options["range_x"][0], plot_options["range_x"][1]], [
                                                      plot_options["range_y"][0], plot_options["range_y"][1]]],
                                                      bins=[plot_options["bins_x"],
                                                      plot_options["bins_y"]])

        numu_axes[1].imshow(numu_heatmap_data.T, extent=numu_extent, origin='lower', aspect="auto")

        nue_ratio = nue_heatmap_data/nue_heatmap
        nue_im_ratio = axes[2].imshow(nue_ratio.T, extent=nue_extent, origin='lower', aspect='auto', vmin=0, vmax=2, cmap="coolwarm")
        nue_fig.colorbar(nue_im_ratio)
        
        numu_ratio = numu_heatmap_data/numu_heatmap
        numu_im_ratio = axes[2].imshow(numu_ratio.T, extent=numu_extent, origin='lower', aspect='auto', vmin=0, vmax=2, cmap="coolwarm")
        numu_fig.colorbar(numu_im_ratio)

        nue_axes[0].title.set_text('MC+EXT')
        nue_axes[1].title.set_text('Data')
        nue_axes[2].title.set_text('Data/(MC+EXT)')
        if "title" in plot_options:
            nue_axes[0].set_xlabel(plot_options["title"].split(";")[0])
            nue_axes[0].set_ylabel(plot_options["title"].split(";")[1])
            nue_axes[1].set_xlabel(plot_options["title"].split(";")[0])
            nue_axes[2].set_xlabel(plot_options["title"].split(";")[0])
        else:
            nue_axes[0].set_xlabel(variable1_name)
            nue_axes[0].set_ylabel(variable2_name)
            nue_axes[1].set_xlabel(variable1_name)
            nue_axes[2].set_xlabel(variable1_name)
            
        numu_axes[0].title.set_text('MC+EXT')
        numu_axes[1].title.set_text('Data')
        numu_axes[2].title.set_text('Data/(MC+EXT)')
        if "title" in plot_options:
            numu_axes[0].set_xlabel(plot_options["title"].split(";")[0])
            numu_axes[0].set_ylabel(plot_options["title"].split(";")[1])
            numu_axes[1].set_xlabel(plot_options["title"].split(";")[0])
            numu_axes[2].set_xlabel(plot_options["title"].split(";")[0])
        else:
            numu_axes[0].set_xlabel(variable1_name)
            numu_axes[0].set_ylabel(variable2_name)
            numu_axes[1].set_xlabel(variable1_name)
            numu_axes[2].set_xlabel(variable1_name)   

        return nue_fig, nue_axes, numu_fig, numu_axes

    def plot_2d_oneplot(self, variable1_name, variable2_name, query="selected==1", track_cuts=None, **plot_options):
        nue_variable1, nue_weight1, numu_variable1, numu_weight1 = self._get_variable(variable1_name, query, track_cuts=track_cuts)
        nue_variable2, nue_weight2, numu_variable2, numu_weight2 = self._get_variable(variable2_name, query, track_cuts=track_cuts)

        nue_heatmap, nue_xedges, nue_yedges = np.histogram2d(nue_variable1, nue_variable2,
                                                 range=[[plot_options["range_x"][0], plot_options["range_x"][1]], [plot_options["range_y"][0], plot_options["range_y"][1]]],
                                                 bins=[plot_options["bins_x"], plot_options["bins_y"]],
                                                 weights=nue_weight1)

        nue_extent = [nue_xedges[0], nue_xedges[-1], nue_yedges[0], nue_yedges[-1]]
        
        numu_heatmap, numu_xedges, numu_yedges = np.histogram2d(numu_variable1, numu_variable2,
                                                 range=[[plot_options["range_x"][0], plot_options["range_x"][1]], [plot_options["range_y"][0], plot_options["range_y"][1]]],
                                                 bins=[plot_options["bins_x"], plot_options["bins_y"]],
                                                 weights=numu_weight1)

        numu_extent = [numu_xedges[0], numu_xedges[-1], numu_yedges[0], numu_yedges[-1]]

        #if figure is passed, use that to build plot
        if "figure" in plot_options:
            nue_fig = plot_options["figure"]
            numu_fig = plot_options["figure"]
        else:
            nue_fig = plt.figure(figsize=(6,6))
            numu_fig = plt.figure(figsize=(6,6))
        if "axis" in plot_options:
            nue_axis = plot_options["axis"]
            numu_axis = plot_options["axis"]
        else:
            nue_axis = plt.gca()
            numu_axis = plt.gca()

        if 'range_z' in plot_options:
            nue_image = nue_axis.imshow(nue_heatmap.T, extent=nue_extent, origin='lower', aspect="auto",
                vmin=plot_options['range_z'][0], vmax=plot_options['range_z'][1])
            numu_image = numu_axis.imshow(numu_heatmap.T, extent=numu_extent, origin='lower', aspect="auto",
                vmin=plot_options['range_z'][0], vmax=plot_options['range_z'][1])
        else:
            nue_image = nue_axis.imshow(nue_heatmap.T, extent=nue_extent, origin='lower', aspect="auto")
            numu_image = numu_axis.imshow(numu_heatmap.T, extent=numu_extent, origin='lower', aspect="auto")
        

        return nue_fig, nue_axis, nue_image, numu_fig, numu_axis, numu_image


    def load_detsys_errors(self,var,path,binedges):

        detsys_frac = np.zeros(len(binedges)-1)

        DETSAMPLES = ["X", "YZ", 'aYZ', "aXZ","dEdX","SCE","LYD","LYR","LYA"]

        if os.path.isdir(path) == False:
            #print ('DETSYS. path %s is not valid'%path)
            return detsys_frac

        for varsample in DETSAMPLES:

            filename = var + "_" + varsample + ".txt"

            if (os.path.isfile(path+filename) == False):
                #f ('file-name %s is not valid'%filename)
                continue

            f = open(path+filename,'r')

            for binnumber in range(len(detsys_frac)):

                binmin = binedges[binnumber]
                binmax = binedges[binnumber+1]

                bincenter = 0.5*(binmin+binmax)

                # find uncertainty associated to this bin in the text-file

                f.seek(0,0)
                
                for line in f:

                    words = line.split(",")
                    binrange_v = words[0].split("-")
                    bincenter = 0.5*(float(binrange_v[0])+float(binrange_v[1]))

                    if ( (bincenter > binmin) and (bincenter <= binmax) ):
                    
                        fracerror = float(words[1].split()[0])
                        detsys_frac[binnumber] += fracerror * fracerror

                        break

        detsys_frac = np.sqrt(np.array(detsys_frac))
        print ('detsys diag error terms are ', detsys_frac)

        return detsys_frac


    def add_detsys_error(self,sample,mc_entries_v,weight):
        detsys_v  = np.zeros(len(mc_entries_v))
        entries_v = np.zeros(len(mc_entries_v))
        if (self.detsys == None): return detsys_v
        if sample in self.detsys:
            if (len(self.detsys[sample]) == len(mc_entries_v)):
                for i,n in enumerate(mc_entries_v):
                    detsys_v[i] = (self.detsys[sample][i] * n * weight)#**2
                    entries_v[i] = n * weight
            else:
                print ('NO MATCH! len detsys : %i. Len plotting : %i'%(len(self.detsys[sample]),len(mc_entries_v) ))

        return detsys_v



    def plot_variable(self, variable, nue_query="selected==1", numu_query="selected==1", title="", kind="event_category", draw_geoSys=False,
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
        if not nue_query:
            nue_query = "nslice==1"
        if not numu_query:
            numu_query = "nslice==1"
            
        # pandas bug https://github.com/pandas-dev/pandas/issues/16363
        if plot_options["range"][0] >= 0 and plot_options["range"][1] >= 0 and variable[-2:] != "_v":
            nue_query += "& %s <= %g & %s >= %g" % (
                variable, plot_options["range"][1], variable, plot_options["range"][0])
            numu_query += "& %s <= %g & %s >= %g" % (
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

        print(nue_query,"\n", self.nu_pdg,"\n",track_cuts,"\n",select_longest)
        print(numu_query,"\n", self.nu_pdg,"\n",track_cuts,"\n",select_longest)
        ##OVERLAY MC CHANGE HERE
        nue_category, nue_mc_plotted_variable = categorization(
            self.samples["nue_mc"], variable, query=nue_query, extra_cut=self.nu_pdg, track_cuts=track_cuts, select_longest=select_longest)
        nue_category_mc_unis = nue_category
        print("nue_category_mc_unis", nue_category_mc_unis)
        print("nue_category", nue_category)
        nue_mc_plotted_variable_mc_unis = nue_mc_plotted_variable
        
        numu_category, numu_mc_plotted_variable = categorization(
            self.samples["numu_mc"], variable, query=numu_query, extra_cut=self.nu_pdg, track_cuts=track_cuts, select_longest=select_longest)

        ##OVERLAY/MC CHANGE HERE
        nue_var_dict = defaultdict(list)
        nue_weight_dict = defaultdict(list)
        nue_mc_genie_weights = self._get_genie_weight(
            self.samples["nue_mc"], variable, query=nue_query, extra_cut=self.nu_pdg, track_cuts=track_cuts,select_longest=select_longest, weightvar=genieweight)
        nue_mc_genie_weights_mc_unis = nue_mc_genie_weights
        
        numu_var_dict = defaultdict(list)
        numu_weight_dict = defaultdict(list)
        numu_mc_genie_weights = self._get_genie_weight(
            self.samples["numu_mc"], variable, query=numu_query, extra_cut=self.nu_pdg, track_cuts=track_cuts,select_longest=select_longest, weightvar=genieweight)

        for c, v, w in zip(nue_category, nue_mc_plotted_variable, nue_mc_genie_weights):
            nue_var_dict[c].append(v)
            nue_weight_dict[c].append(self.weights["nue_mc"] * w)

        nue_nue_genie_weights = self._get_genie_weight(
            self.samples["nue_nue"], variable, query=nue_query, track_cuts=track_cuts, select_longest=select_longest, weightvar=genieweight)
        nue_nue_genie_weights_nue_unis = nue_nue_genie_weights

        nue_category, nue_nue_plotted_variable = categorization(
            self.samples["nue_nue"], variable, query=nue_query, track_cuts=track_cuts, select_longest=select_longest)
        nue_category_nue_unis = nue_category
        nue_nue_plotted_variable_nue_unis = nue_nue_plotted_variable
        
        for c, v, w in zip(numu_category, numu_mc_plotted_variable, numu_mc_genie_weights):
            numu_var_dict[c].append(v)
            numu_weight_dict[c].append(self.weights["numu_mc"] * w)

        numu_nue_genie_weights = self._get_genie_weight(
            self.samples["numu_nue"], variable, query=numu_query, track_cuts=track_cuts, select_longest=select_longest, weightvar=genieweight)

        numu_category, numu_nue_plotted_variable = categorization(
            self.samples["numu_nue"], variable, query=numu_query, track_cuts=track_cuts, select_longest=select_longest)

        for c, v, w in zip(nue_category, nue_nue_plotted_variable, nue_nue_genie_weights):
            nue_var_dict[c].append(v)
            nue_weight_dict[c].append(self.weights["nue_nue"] * w)
            
        for c, v, w in zip(numu_category, numu_nue_plotted_variable, numu_nue_genie_weights):
            numu_var_dict[c].append(v)
            numu_weight_dict[c].append(self.weights["numu_nue"] * w)

        if "ncpi0" in self.samples:
            ncpi0_genie_weights = self._get_genie_weight(
                    self.samples["ncpi0"], variable, query=query, track_cuts=track_cuts, select_longest=select_longest, weightvar=genieweight)
            category, ncpi0_plotted_variable = categorization(
                self.samples["ncpi0"], variable, query=query, track_cuts=track_cuts, select_longest=select_longest)

            for c, v, w in zip(category, ncpi0_plotted_variable, ncpi0_genie_weights):
                var_dict[c].append(v)
                weight_dict[c].append(self.weights["ncpi0"] * w)

        if "ccpi0" in self.samples:
            ccpi0_genie_weights = self._get_genie_weight(
                    self.samples["ccpi0"], variable, query=query, track_cuts=track_cuts, select_longest=select_longest, weightvar=genieweight)
            category, ccpi0_plotted_variable = categorization(
                self.samples["ccpi0"], variable, query=query, track_cuts=track_cuts, select_longest=select_longest)

            for c, v, w in zip(category, ccpi0_plotted_variable, ccpi0_genie_weights):
                var_dict[c].append(v)
                weight_dict[c].append(self.weights["ccpi0"] * w)

        if "ccnopi" in self.samples:
            ccnopi_genie_weights = self._get_genie_weight(
                    self.samples["ccnopi"], variable, query=query, track_cuts=track_cuts, select_longest=select_longest, weightvar=genieweight)
            category, ccnopi_plotted_variable = categorization(
                self.samples["ccnopi"], variable, query=query, track_cuts=track_cuts, select_longest=select_longest)

            for c, v, w in zip(category, ccnopi_plotted_variable, ccnopi_genie_weights):
                var_dict[c].append(v)
                weight_dict[c].append(self.weights["ccnopi"] * w)

        if "cccpi" in self.samples:
            cccpi_genie_weights = self._get_genie_weight(
                    self.samples["cccpi"], variable, query=query, track_cuts=track_cuts, select_longest=select_longest, weightvar=genieweight)
            category, cccpi_plotted_variable = categorization(
                self.samples["cccpi"], variable, query=query, track_cuts=track_cuts, select_longest=select_longest)

            for c, v, w in zip(category, cccpi_plotted_variable, cccpi_genie_weights):
                var_dict[c].append(v)
                weight_dict[c].append(self.weights["cccpi"] * w)

        if "nccpi" in self.samples:
            nccpi_genie_weights = self._get_genie_weight(
                    self.samples["nccpi"], variable, query=query, track_cuts=track_cuts, select_longest=select_longest, weightvar=genieweight)
            category, nccpi_plotted_variable = categorization(
                self.samples["nccpi"], variable, query=query, track_cuts=track_cuts, select_longest=select_longest)

            for c, v, w in zip(category, nccpi_plotted_variable, nccpi_genie_weights):
                var_dict[c].append(v)
                weight_dict[c].append(self.weights["nccpi"] * w)

        if "ncnopi" in self.samples:
            ncnopi_genie_weights = self._get_genie_weight(
                    self.samples["ncnopi"], variable, query=query, track_cuts=track_cuts, select_longest=select_longest, weightvar=genieweight)
            category, ncnopi_plotted_variable = categorization(
                self.samples["ncnopi"], variable, query=query, track_cuts=track_cuts, select_longest=select_longest)

            for c, v, w in zip(category, ncnopi_plotted_variable, ncnopi_genie_weights):
                var_dict[c].append(v)
                weight_dict[c].append(self.weights["ncnopi"] * w)

        if "nue_dirt" in self.samples:
            nue_dirt_genie_weights = self._get_genie_weight(
                self.samples["nue_dirt"], variable, query=nue_query, track_cuts=track_cuts, select_longest=select_longest, weightvar=genieweight)
            nue_dirt_genie_weights_dirt_unis = nue_dirt_genie_weights
            nue_category, nue_dirt_plotted_variable = categorization(
                self.samples["nue_dirt"], variable, query=nue_query, track_cuts=track_cuts, select_longest=select_longest)
            nue_category_dirt_unis = nue_category
            nue_dirt_plotted_variable_dirt_unis = nue_dirt_plotted_variable

            for c, v, w in zip(nue_category, nue_dirt_plotted_variable, nue_dirt_genie_weights):
                nue_var_dict[c].append(v)
                nue_weight_dict[c].append(self.weights["nue_dirt"] * w)
                
        if "numu_dirt" in self.samples:
            numu_dirt_genie_weights = self._get_genie_weight(
                self.samples["numu_dirt"], variable, query=numu_query, track_cuts=track_cuts, select_longest=select_longest, weightvar=genieweight)
            numu_category, numu_dirt_plotted_variable = categorization(
                self.samples["numu_dirt"], variable, query=numu_query, track_cuts=track_cuts, select_longest=select_longest)

            for c, v, w in zip(numu_category, numu_dirt_plotted_variable, numu_dirt_genie_weights):
                numu_var_dict[c].append(v)
                numu_weight_dict[c].append(self.weights["numu_dirt"] * w)
                
        

        if "lee" in self.samples:
            category, lee_plotted_variable = categorization(
                self.samples["lee"], variable, query=query, track_cuts=track_cuts, select_longest=select_longest)
            leeweight = self._get_genie_weight(
                self.samples["lee"], variable, query=query, track_cuts=track_cuts, select_longest=select_longest,weightsignal="leeweight", weightvar=genieweight)

            for c, v, w in zip(category, lee_plotted_variable, leeweight):
                var_dict[c].append(v)
                weight_dict[c].append(self.weights["lee"] * w)

            lee_hist, lee_bins = np.histogram(
                var_dict[111],
                bins=plot_options["bins"],
                range=plot_options["range"],
                weights=weight_dict[111])

        if draw_data:
            nue_ext_plotted_variable = self._selection(
                variable, self.samples["nue_ext"], query=nue_query, track_cuts=track_cuts, select_longest=select_longest)
            nue_ext_plotted_variable = self._select_showers(
            nue_ext_plotted_variable, variable, self.samples["nue_ext"], query=nue_query)
            nue_data_plotted_variable = self._selection(
            variable, self.samples["nue_data"], query=nue_query, track_cuts=track_cuts, select_longest=select_longest)
            nue_data_plotted_variable = self._select_showers(nue_data_plotted_variable, variable,
                                                     self.samples["nue_data"], query=nue_query)
            numu_ext_plotted_variable = self._selection(
                variable, self.samples["numu_ext"], query=numu_query, track_cuts=track_cuts, select_longest=select_longest)
            numu_ext_plotted_variable = self._select_showers(
            numu_ext_plotted_variable, variable, self.samples["numu_ext"], query=numu_query)
            numu_data_plotted_variable = self._selection(
            variable, self.samples["numu_data"], query=numu_query, track_cuts=track_cuts, select_longest=select_longest)
            numu_data_plotted_variable = self._select_showers(numu_data_plotted_variable, variable,
                                                     self.samples["numu_data"], query=numu_query)

        if ratio:
            nue_fig = plt.figure(figsize=(8, 7))
            nue_gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
            nue_ax1 = plt.subplot(nue_gs[0])
            nue_ax2 = plt.subplot(nue_gs[1])
            numu_fig = plt.figure(figsize=(8, 7))
            numu_gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
            numu_ax1 = plt.subplot(numu_gs[0])
            numu_ax2 = plt.subplot(numu_gs[1])
            ratio_fig = plt.figure(figsize=(8, 7))
            ratio_gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
            ratio_ax1 = plt.subplot(ratio_gs[0])
            ratio_ax2 = plt.subplot(ratio_gs[1])
        else:
            nue_fig = plt.figure(figsize=(7, 5))
            nue_gs = gridspec.GridSpec(1, 1)#, height_ratios=[2, 1])
            nue_ax1 = plt.subplot(nue_gs[0])
            numu_fig = plt.figure(figsize=(7, 5))
            numu_gs = gridspec.GridSpec(1, 1)#, height_ratios=[2, 1])
            numu_ax1 = plt.subplot(numu_gs[0])
            ratio_fig = plt.figure(figsize=(8, 7))
            ratio_gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
            ratio_ax1 = plt.subplot(ratio_gs[0])

#-------------------------------
        c, nue_order_var_dict, nue_order_weight_dict = Plotter_Functions_Alex.plotColourSorting.sortStackDists(stacksort, nue_var_dict, nue_weight_dict)

        nue_total = sum(sum(nue_order_weight_dict[c]) for c in nue_order_var_dict)
        if draw_data:
            nue_total += sum([self.weights["nue_ext"]] * len(nue_ext_plotted_variable))
            print("nue total ", nue_total)
        labels = [
            "%s: %.1f" % (cat_labels[c], sum(nue_order_weight_dict[c])) \
            if sum(nue_order_weight_dict[c]) else ""
            for c in nue_order_var_dict.keys()
        ]


        if kind == "event_category":
            plot_options["color"] = [category_colors[c]
                                     for c in nue_order_var_dict.keys()]
        elif kind == "particle_pdg":
            plot_options["color"] = [pdg_colors[c]
                                     for c in nue_order_var_dict.keys()]
        elif kind == "flux":
            plot_options["color"] = [flux_colors[c]
                                     for c in nue_order_var_dict.keys()]
        else:
            plot_options["color"] = [int_colors[c]
                                     for c in nue_order_var_dict.keys()]

        nue_stacked = nue_ax1.hist(
            nue_order_var_dict.values(),
            weights=list(nue_order_weight_dict.values()),
            stacked=True,
            label=labels,
            **plot_options)

        nue_total_array = np.concatenate(list(nue_order_var_dict.values()))
        nue_total_weight = np.concatenate(list(nue_order_weight_dict.values()))
        wanted_key = 5 # 7 for full, 5 for truth 

        nue_wanted_list = Plotter_Functions_Alex.getWantedLists.getWantedLists(wanted_key, nue_stacked)
        
        #Remove smearing part
        ###################################################################
        # true nu energy 
        true_var = 'nu_e'
        # reconstructed nu energy 
        reco_var = 'reco_e'
        nue_fiduc_q = "true_nu_vtx_z < 1036.8 and true_nu_vtx_z > 0 and true_nu_vtx_y < 116.5 and true_nu_vtx_y > -116.5 and true_nu_vtx_x < \
        254.8 and true_nu_vtx_x > -1.55 and (nu_pdg == 12 and ccnc == 0) and nproton > 0"
        nue_selected = self.samples["nue_mc"].query(nue_query)
        nue_selected_fid = nue_selected.query(nue_fiduc_q)
        bins = np.arange(0, 5.5, 0.5)
        #smear = plt.hist2d(nue_selected_fid.query(nue_fiduc_q)["nu_e"],nue_selected_fid.query(nue_fiduc_q)["reco_e"],
            #bins, cmin=0.000000001, cmap='OrRd')
        
        norm = True 
        
        nue_norm_array = self.plot_smearing(nue_selected_fid, nue_fiduc_q, true_var, reco_var, bins, norm)
        #print(nue_norm_array)
        #print("")
        
        for i in range(len(bins)-1): # reco bins (rows)
            for j in range(len(bins)-1): # truth bins (cols)
                if nue_norm_array[i][j] > 0:
                    continue
                else:
                    nue_norm_array[i][j] = 0
                    
        #print(nue_norm_array)
        #print("")                    
                    
        nue_smeared_array = np.matmul(np.array(nue_wanted_list), nue_norm_array)
        #print(nue_smeared_array)
        
        nue_wanted_list_smeared = list(nue_smeared_array)
        
        #for i in range(len(bins)-1): # reco bins (rows)
        #    sumCol = 0  
        #    for j in range(len(bins)-1): # truth bins (cols)
        #        if nue_smeared_array[j][i] > 0:
        #            sumCol = sumCol + nue_smeared_array[j][i]  
        #    print("Sum of " + str(i+1) +" column: " + str(sumCol));
        #    nue_wanted_list_smeared.append(sumCol)
            
        #print("")
        #print("nue_wanted_list_smeared = ")
        #print(nue_wanted_list_smeared)
        #print("")
        ##################################################################
        nue_wanted_list_smeared = nue_wanted_list
        nue_smeared_array = np.array(nue_wanted_list)
        
        nue_eff = self.plot_signal_and_eff_and_B(nue_selected_fid, self.samples["nue_mc"], nue_fiduc_q, bins, self.samples["nue_mc"].query(nue_fiduc_q))

        nue_ratio_nums = []

        for i in range(len(nue_wanted_list_smeared)):
            num = nue_wanted_list_smeared[i]*(1/nue_eff[i])
            nue_ratio_nums.append(num)
            
        print("")
        print("nue_ratio_nums:")
        print(nue_ratio_nums)
        print("")

        plot_options.pop('color', None)

        nue_total_hist, nue_total_bins = np.histogram(
            nue_total_array, weights=nue_total_weight,  **plot_options)
        
        #----------------------------------
        c, numu_order_var_dict, numu_order_weight_dict = Plotter_Functions_Alex.plotColourSorting.sortStackDists(stacksort, numu_var_dict, numu_weight_dict)
        numu_total = sum(sum(numu_order_weight_dict[c]) for c in numu_order_var_dict)
        if draw_data:
            numu_total += sum([self.weights["numu_ext"]] * len(numu_ext_plotted_variable))
            print("numu total ", numu_total)
            print("")
        labels = [
            "%s: %.1f" % (cat_labels[c], sum(numu_order_weight_dict[c])) \
            if sum(numu_order_weight_dict[c]) else ""
            for c in numu_order_var_dict.keys()
        ]


        if kind == "event_category":
            plot_options["color"] = [category_colors[c]
                                     for c in numu_order_var_dict.keys()]
        elif kind == "particle_pdg":
            plot_options["color"] = [pdg_colors[c]
                                     for c in numu_order_var_dict.keys()]
        elif kind == "flux":
            plot_options["color"] = [flux_colors[c]
                                     for c in numu_order_var_dict.keys()]
        else:
            plot_options["color"] = [int_colors[c]
                                     for c in numu_order_var_dict.keys()]
        numu_stacked = numu_ax1.hist(
            numu_order_var_dict.values(),
            weights=list(numu_order_weight_dict.values()),
            stacked=True,
            label=labels,
            **plot_options)

        numu_total_array = np.concatenate(list(numu_order_var_dict.values()))
        numu_total_weight = np.concatenate(list(numu_order_weight_dict.values()))
        wanted_key = 3 # 3 for full, 3 for truth
        
        numu_wanted_list = Plotter_Functions_Alex.getWantedLists.getWantedLists(wanted_key, numu_stacked)
        
        ################################################################
        # true nu energy 
        true_var = 'nu_e'
        # reconstructed nu energy 
        reco_var = 'reco_e'
        numu_fiduc_q = "true_nu_vtx_z < 1036.8 and true_nu_vtx_z > 0 and true_nu_vtx_y < 116.5 and true_nu_vtx_y > -116.5 and true_nu_vtx_x < \
        254.8 and true_nu_vtx_x > -1.55 and (nu_pdg == 14 and ccnc == 0) and nproton > 0"
        numu_selected = self.samples["numu_mc"].query(numu_query)
        numu_selected_fid = numu_selected.query(numu_fiduc_q)
        bins = np.arange(0, 5.5, 0.5)
        
        
        norm = True 
        
        numu_norm_array = self.plot_smearing(numu_selected_fid, numu_fiduc_q, true_var, reco_var, bins, norm)
        #print(nue_norm_array)
        #print("")
        
        for i in range(len(bins)-1): # reco bins (rows)
            for j in range(len(bins)-1): # truth bins (cols)
                if numu_norm_array[i][j] > 0:
                    continue
                else:
                    numu_norm_array[i][j] = 0
                    
        #print(nue_norm_array)
        #print("")                    
                    
        
        numu_smeared_array = np.matmul(np.array(numu_wanted_list), numu_norm_array)
        #numu_smeared_array = np.array(numu_wanted_list)
        #print(nue_smeared_array)
        
        numu_wanted_list_smeared = list(numu_smeared_array)
        #print("")
        #print("numu_wanted_list_smeared = ")
        #print(numu_wanted_list_smeared)
        #print("")
        #############################################################
        numu_wanted_list_smeared = numu_wanted_list
        numu_smeared_array = np.array(numu_wanted_list)
        
        numu_eff = self.plot_signal_and_eff_and_B(numu_selected_fid, self.samples["numu_mc"], numu_fiduc_q, bins, self.samples["numu_mc"].query(numu_fiduc_q))
        #numu_eff = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        
        #print("")
        #print(numu_eff)
        #print("")

        numu_ratio_nums = []

        for i in range(len(numu_wanted_list_smeared)):
            num = numu_wanted_list_smeared[i]*(1/numu_eff[i])
            print("numu_wanted_list_smeared[i] ", numu_wanted_list_smeared[i])
            print("1/numu_eff[i] ", 1/numu_eff[i])
            numu_ratio_nums.append(num)
            
        print("")
        print("numu_ratio_nums:")
        print(numu_ratio_nums)
        print("")
        
        ##############################################################
        #Initial plots calculated
        ##############################################################
        
        rbin_ratios = []
        
        
        for i in range(len(numu_ratio_nums)):
            if nue_ratio_nums[i] > 0 and numu_ratio_nums[i] > 0:
                print("nue_ratio_nums[i] ", nue_ratio_nums[i])
                print("numu_ratio_nums[i]", numu_ratio_nums[i])
                rratio = nue_ratio_nums[i]/numu_ratio_nums[i]
                print("rratio ", rratio)
                rbin_ratios.append(rratio)
            else:
                rbin_ratios.append(0)
        
        print("")
        print("bin_ratios:")
        print(rbin_ratios)
        print("")
        
        count = [0.1, 0.6, 1.1, 1.6, 2.1, 2.6, 3.1, 3.6, 4.1, 4.6]
        weights = []
        for i in range(len(count)):
            weight = count[i]*rbin_ratios[i]
            weights.append(weight)
            
        print("")
        print("Making ratio plot.")
        print("count, ", count)
        print("bins, ", bins)
        print("Weights, ", rbin_ratios)
        print("")

        sel = ratio_ax1.hist(count, bins, color='deepskyblue', weights=rbin_ratios)
        ratio_ax1.set_ylabel('Ratio Nue/Numu', fontsize=15)
        ratio_ax1.set_xlabel('True Nu Energy [GeV]', fontsize=15)
        ratio_ax1.set_title("Ratio")

        plot_options.pop('color', None)

        numu_total_hist, numu_total_bins = np.histogram(
            numu_total_array, weights=numu_total_weight,  **plot_options)
        
        #----------------------------------

        if draw_data:
            nue_ext_weight = [self.weights["nue_ext"]] * len(nue_ext_plotted_variable)
            nue_n_ext, nue_ext_bins, nue_patches = nue_ax1.hist(
            nue_ext_plotted_variable,
            weights=nue_ext_weight,
            bottom=nue_total_hist,
            label="EXT: %.1f" % sum(nue_ext_weight) if sum(nue_ext_weight) else "",
            hatch="//",
            color="white",
            **plot_options)
            
            print("")            
            print("nue n_ext ", nue_n_ext)
            
            nue_total_array = np.concatenate([nue_total_array, nue_ext_plotted_variable])
            nue_total_weight = np.concatenate([nue_total_weight, nue_ext_weight])
            
            numu_ext_weight = [self.weights["numu_ext"]] * len(numu_ext_plotted_variable)
            numu_n_ext, numu_ext_bins, numu_patches = numu_ax1.hist(
            numu_ext_plotted_variable,
            weights=numu_ext_weight,
            bottom=numu_total_hist,
            label="EXT: %.1f" % sum(numu_ext_weight) if sum(numu_ext_weight) else "",
            hatch="//",
            color="white",
            **plot_options)
             
            print("numu n_ext ", numu_n_ext)
            print("")
            
            numu_total_array = np.concatenate([numu_total_array, numu_ext_plotted_variable])
            numu_total_weight = np.concatenate([numu_total_weight, numu_ext_weight])
            

        nue_n_tot, nue_bin_edges, nue_patches = nue_ax1.hist(
        nue_total_array,
        weights=nue_total_weight,
        histtype="step",
        edgecolor="black",
        **plot_options)
        
        print("")        
        print("nue n_tot ", nue_n_tot)
        print("total array ", nue_total_array)
        
        numu_n_tot, numu_bin_edges, numu_patches = numu_ax1.hist(
        numu_total_array,
        weights=numu_total_weight,
        histtype="step",
        edgecolor="black",
        **plot_options)
        
        print("numu n_tot ", numu_n_tot)
        print("total array ", numu_total_array)
        print("")
        
        ###########################################################
        #Individual df uncertainties
        ###########################################################
            
        ##MC?OVERLAY CHANGE HERE
        nue_bincenters = 0.5 * (nue_bin_edges[1:] + nue_bin_edges[:-1])
        nue_mc_uncertainties, nue_bins = np.histogram(
            nue_mc_plotted_variable, **plot_options)
        nue_err_mc = np.array(
            [n * self.weights["nue_mc"] * self.weights["nue_mc"] for n in nue_mc_uncertainties])
        nue_sys_mc = self.add_detsys_error("nue_mc",nue_mc_uncertainties,self.weights["nue_mc"])
        print("nue_sys_mc: ")
        print(nue_sys_mc)

        nue_nue_uncertainties, nue_bins = np.histogram(
            nue_nue_plotted_variable, **plot_options)
        nue_err_nue = np.array(
            [n * self.weights["nue_nue"] * self.weights["nue_nue"] for n in nue_nue_uncertainties])
        nue_sys_nue = self.add_detsys_error("nue_nue",nue_nue_uncertainties,self.weights["nue_nue"])
        print("nue_sys nue: ")
        print(nue_sys_nue)
        
        numu_bincenters = 0.5 * (numu_bin_edges[1:] + numu_bin_edges[:-1])
        numu_mc_uncertainties, numu_bins = np.histogram(
            numu_mc_plotted_variable, **plot_options)
        numu_err_mc = np.array(
            [n * self.weights["numu_mc"] * self.weights["numu_mc"] for n in numu_mc_uncertainties])
        numu_sys_mc = self.add_detsys_error("numu_mc",numu_mc_uncertainties,self.weights["numu_mc"])
        print("numu_sys_mc: ")
        print(numu_sys_mc)

        numu_nue_uncertainties, numu_bins = np.histogram(
            numu_nue_plotted_variable, **plot_options)
        numu_err_nue = np.array(
            [n * self.weights["numu_nue"] * self.weights["numu_nue"] for n in numu_nue_uncertainties])
        numu_sys_nue = self.add_detsys_error("numu_nue",numu_nue_uncertainties,self.weights["numu_nue"])
        print("numu_sys_nue: ")
        print(numu_sys_nue)
        print("numu_nue_plotted_variable: ")
        print(numu_nue_plotted_variable)
        print("numu_nue_plotted_variable length: ")
        print(len(numu_nue_plotted_variable))
        
        ratio_bins = np.arange(0, 5.5, 0.5)
        ratio_bincenters = np.arange(0.25, 5.25, 0.5)
        ratio_bin_edges = np.arange(0, 5.5, 0.5)
        #ratio_bin_size = 0.5
        ratio_bin_size = [(ratio_bin_edges[i + 1] - ratio_bin_edges[i]) / 2
                    for i in range(len(ratio_bin_edges) - 1)] 
        print("")
        print("bins ", ratio_bins)
        print("centers ", ratio_bincenters)
        print("edges ", ratio_bin_edges)
        print("size ", ratio_bin_size)

        ratio_err_mc = []
        
        #Get the numbers of that part correctly weighted that were selected
        x_range=plot_options["range"]
        n_bins=plot_options["bins"]  
        weightVar=genieweight
        
        tree_nue_mc = self.samples["nue_mc"]
        extra_query = "& " + self.nu_pdg
        queried_tree_nue_mc = tree_nue_mc.query(nue_query+extra_query)
        variable_nue = queried_tree_nue_mc[variable]
        spline_fix_cv_nue_mc  = queried_tree_nue_mc[weightVar] * self.weights["nue_mc"]
        nue_selected_mc, bins = np.histogram(
                    variable_nue,
                    range=x_range,
                    bins=n_bins,
                    weights=spline_fix_cv_nue_mc)  
        print("nue_selected_mc ", nue_selected_mc)
        
        tree_numu_mc = self.samples["numu_mc"]
        extra_query = "& " + self.nu_pdg
        queried_tree_numu_mc = tree_numu_mc.query(numu_query+extra_query)
        variable_numu = queried_tree_numu_mc[variable]
        spline_fix_cv_numu_mc  = queried_tree_numu_mc[weightVar] * self.weights["numu_mc"]
        numu_selected_mc, bins = np.histogram(
                    variable_numu,
                    range=x_range,
                    bins=n_bins,
                    weights=spline_fix_cv_numu_mc)  
        print("numu_selected_mc ", numu_selected_mc)
        
        for i in range(len(nue_err_mc)):
            dnue = 0
            dnumu = 0
            if nue_selected_mc[i] > 0:
                dnue = (nue_err_mc[i]/nue_selected_mc[i])
            if numu_selected_mc[i] > 0:
                dnumu = (numu_err_mc[i]/numu_selected_mc[i])
            rratio = rbin_ratios[i]*np.sqrt((dnue)**2 + (dnumu)**2)
            ratio_err_mc.append(rratio)
        
        print("")
        print("nue_err_mc: ")
        print(nue_err_mc)
        print("")
        print("numu_err_mc: ")
        print(numu_err_mc)
        print("")
        print("ratio_err_mc: ")
        print(ratio_err_mc)
        print("")
        ratio_err_mc = np.array(ratio_err_mc)

        ratio_err_nue = []
        
        tree_nue_nue = self.samples["nue_nue"]
        queried_tree_nue_nue = tree_nue_nue.query(nue_query)
        variable_nue = queried_tree_nue_nue[variable]
        spline_fix_cv_nue_nue  = queried_tree_nue_nue[weightVar] * self.weights["nue_nue"]
        nue_selected_nue, bins = np.histogram(
                    variable_nue,
                    range=x_range,
                    bins=n_bins,
                    weights=spline_fix_cv_nue_nue)  
        print("nue_selected_nue ", nue_selected_nue)
        
        tree_numu_nue = self.samples["numu_nue"]
        queried_tree_numu_nue = tree_numu_nue.query(numu_query)
        variable_numu = queried_tree_numu_nue[variable]
        spline_fix_cv_numu_nue  = queried_tree_numu_nue[weightVar] * self.weights["numu_nue"]
        numu_selected_nue, bins = np.histogram(
                    variable_numu,
                    range=x_range,
                    bins=n_bins,
                    weights=spline_fix_cv_numu_nue)  
        print("numu_selected_nue ", numu_selected_nue)
        
        for i in range(len(nue_err_mc)):
            dnue = 0
            dnumu = 0
            if nue_selected_nue[i] > 0:
                dnue = (nue_err_nue[i]/nue_selected_nue[i])
            if numu_selected_nue[i] > 0:
                dnumu = (numu_err_nue[i]/numu_selected_nue[i])
            rratio = rbin_ratios[i]*np.sqrt((dnue)**2 + (dnumu)**2)
            ratio_err_nue.append(rratio)
        
        print("")
        print("nue_err_nue: ")
        print(nue_err_nue)
        print("")
        print("numu_err_nue: ")
        print(numu_err_nue)
        print("")
        print("ratio_err_nue: ")
        print(ratio_err_nue)
        print("")
        ratio_err_nue = np.array(ratio_err_nue)
        
        

        nue_err_dirt = np.array([0 for n in nue_mc_uncertainties])        
        if "nue_dirt" in self.samples:
            nue_dirt_uncertainties, bins = np.histogram(
                nue_dirt_plotted_variable, **plot_options)
            nue_err_dirt = np.array(
                [n * self.weights["nue_dirt"] * self.weights["nue_dirt"] for n in nue_dirt_uncertainties])
            nue_sys_dirt = self.add_detsys_error("nue_dirt",nue_dirt_uncertainties,self.weights["nue_dirt"])
            
        numu_err_dirt = np.array([0 for n in numu_mc_uncertainties])        
        if "numu_dirt" in self.samples:
            numu_dirt_uncertainties, bins = np.histogram(
                numu_dirt_plotted_variable, **plot_options)
            numu_err_dirt = np.array(
                [n * self.weights["numu_dirt"] * self.weights["numu_dirt"] for n in numu_dirt_uncertainties])
            numu_sys_dirt = self.add_detsys_error("numu_dirt",numu_dirt_uncertainties,self.weights["numu_dirt"])

        ratio_err_dirt = []
        
        tree_nue_dirt = self.samples["nue_dirt"]
        queried_tree_nue_dirt = tree_nue_dirt.query(nue_query)
        variable_dirt = queried_tree_nue_dirt[variable]
        spline_fix_cv_nue_dirt  = queried_tree_nue_dirt[weightVar] * self.weights["nue_dirt"]
        nue_selected_dirt, bins = np.histogram(
                    variable_dirt,
                    range=x_range,
                    bins=n_bins,
                    weights=spline_fix_cv_nue_dirt)  
        print("nue_selected_dirt ", nue_selected_dirt)
        
        tree_numu_dirt = self.samples["numu_dirt"]
        queried_tree_numu_dirt = tree_numu_dirt.query(numu_query)
        variable_numu = queried_tree_numu_dirt[variable]
        spline_fix_cv_numu_dirt  = queried_tree_numu_dirt[weightVar] * self.weights["numu_dirt"]
        numu_selected_dirt, bins = np.histogram(
                    variable_numu,
                    range=x_range,
                    bins=n_bins,
                    weights=spline_fix_cv_numu_dirt)  
        print("numu_selected_dirt ", numu_selected_dirt)
        
        
        for i in range(len(nue_err_mc)):
            dnue = 0
            dnumu = 0
            if nue_selected_dirt[i] > 0:
                dnue = (nue_err_dirt[i]/nue_selected_dirt[i])
            if numu_selected_dirt[i] > 0:
                dnumu = (numu_err_dirt[i]/numu_selected_dirt[i])
            rratio = rbin_ratios[i]*np.sqrt((dnue)**2 + (dnumu)**2)
            ratio_err_dirt.append(rratio)
         
        print("")
        print("ratio_err_dirt: ")
        print(ratio_err_dirt)
        print("")
        ratio_err_dirt = np.array(ratio_err_dirt)
        
            
#Shouldn't need these errors
        err_lee = np.array([0 for n in nue_mc_uncertainties])
        if "lee" in self.samples:
            if isinstance(plot_options["bins"], Iterable):
                lee_bins = plot_options["bins"]
            else:
                bin_size = (plot_options["range"][1] - plot_options["range"][0])/plot_options["bins"]
                lee_bins = [plot_options["range"][0]+n*bin_size for n in range(plot_options["bins"]+1)]

            if variable[-2:] != "_v":
                binned_lee = pd.cut(self.samples["lee"].query(query).eval(variable), lee_bins)
                err_lee = self.samples["lee"].query(query).groupby(binned_lee)['leeweight'].agg(
                    "sum").values * self.weights["lee"] * self.weights["lee"]

        err_ncpi0 = np.array([0 for n in nue_mc_uncertainties])
        sys_ncpi0 = np.array([0 for n in nue_mc_uncertainties])
        if "ncpi0" in self.samples:
            ncpi0_uncertainties, bins = np.histogram(
                ncpi0_plotted_variable, **plot_options)
            print("ncpi0? ", ncpi0_plotted_variable)
            err_ncpi0 = np.array(
                [n * self.weights["ncpi0"] * self.weights["ncpi0"] for n in ncpi0_uncertainties])
            if ("ncpi0" in self.detsys.keys()):
                self.detsys["ncpi0"] = self.load_detsys_errors(variable,DETSYSPATH,bin_edges)
            sys_ncpi0 = self.add_detsys_error("ncpi0",ncpi0_uncertainties,self.weights["ncpi0"])
            

        err_ccpi0 = np.array([0 for n in nue_mc_uncertainties])
        sys_ccpi0 = np.array([0 for n in nue_mc_uncertainties])
        if "ccpi0" in self.samples:
            ccpi0_uncertainties, bins = np.histogram(
                ccpi0_plotted_variable, **plot_options)
            err_ccpi0 = np.array(
                [n * self.weights["ccpi0"] * self.weights["ccpi0"] for n in ccpi0_uncertainties])
            if ("ccpi0" in self.detsys.keys()):
                self.detsys["ccpi0"] = self.load_detsys_errors(variable,DETSYSPATH,bin_edges)
            sys_ccpi0 = self.add_detsys_error("ccpi0",ccpi0_uncertainties,self.weights["ccpi0"])

        err_ccnopi = np.array([0 for n in nue_mc_uncertainties])
        sys_ccnopi = np.array([0 for n in nue_mc_uncertainties])
        if "ccnopi" in self.samples:
            ccnopi_uncertainties, bins = np.histogram(
                ccnopi_plotted_variable, **plot_options)
            err_ccnopi = np.array(
                [n * self.weights["ccnopi"] * self.weights["ccnopi"] for n in ccnopi_uncertainties])
            if ("ccnopi" in self.detsys.keys()):
                self.detsys["ccnopi"] = self.load_detsys_errors(variable,DETSYSPATH,bin_edges)
            sys_ccnopi = self.add_detsys_error("ccnopi",ccnopi_uncertainties,self.weights["ccnopi"])

        err_cccpi = np.array([0 for n in nue_mc_uncertainties])
        sys_cccpi = np.array([0 for n in nue_mc_uncertainties])
        if "cccpi" in self.samples:
            cccpi_uncertainties, bins = np.histogram(
                cccpi_plotted_variable, **plot_options)
            err_cccpi = np.array(
                [n * self.weights["cccpi"] * self.weights["cccpi"] for n in cccpi_uncertainties])
            if ("cccpi" in self.detsys.keys()):
                self.detsys["cccpi"] = self.load_detsys_errors(variable,DETSYSPATH,bin_edges)
            sys_cccpi = self.add_detsys_error("cccpi",cccpi_uncertainties,self.weights["cccpi"])

        err_nccpi = np.array([0 for n in nue_mc_uncertainties])
        sys_nccpi = np.array([0 for n in nue_mc_uncertainties])
        if "nccpi" in self.samples:
            nccpi_uncertainties, bins = np.histogram(
                nccpi_plotted_variable, **plot_options)
            err_nccpi = np.array(
                [n * self.weights["nccpi"] * self.weights["nccpi"] for n in nccpi_uncertainties])
            if ("nccpi" in self.detsys.keys()):
                self.detsys["nccpi"] = self.load_detsys_errors(variable,DETSYSPATH,bin_edges)
            sys_nccpi = self.add_detsys_error("nccpi",nccpi_uncertainties,self.weights["nccpi"])

        err_ncnopi = np.array([0 for n in nue_mc_uncertainties])
        sys_ncnopi = np.array([0 for n in nue_mc_uncertainties])
        if "ncnopi" in self.samples:
            ncnopi_uncertainties, bins = np.histogram(
                ncnopi_plotted_variable, **plot_options)
            err_ncnopi = np.array(
                [n * self.weights["ncnopi"] * self.weights["ncnopi"] for n in ncnopi_uncertainties])
            if ("ncnopi" in self.detsys.keys()):
                self.detsys["ncnopi"] = self.load_detsys_errors(variable,DETSYSPATH,bin_edges)
            sys_ncnopi = self.add_detsys_error("ncnopi",ncnopi_uncertainties,self.weights["ncnopi"])

            
            
            
        if draw_data:
            nue_err_ext = np.array(
                [n * self.weights["nue_ext"] * self.weights["nue_ext"] for n in nue_n_ext])
        else:
            nue_err_ext = np.zeros(len(nue_err_mc))

        nue_exp_err    = np.sqrt(nue_err_mc + nue_err_ext + nue_err_nue + nue_err_dirt + err_ncpi0 + err_ccpi0 + err_ccnopi + err_cccpi + err_nccpi + err_ncnopi)
        #print("counting_err: {}".format(exp_err))
        if "nue_dirt" in self.samples:
            nue_detsys_err = nue_sys_mc + nue_sys_nue + nue_sys_dirt + sys_ncpi0 + sys_ccpi0 + sys_ccnopi + sys_cccpi + sys_nccpi + sys_ncnopi
        else:
            nue_detsys_err = nue_sys_mc + nue_sys_nue + sys_ncpi0 + sys_ccpi0 + sys_ccnopi + sys_cccpi + sys_nccpi + sys_ncnopi
        #print("detsys_err: {}".format(detsys_err))
        nue_exp_err = np.sqrt(nue_exp_err**2 + nue_detsys_err**2)
        #print ('total exp_err : ', exp_err)

        nue_bin_size = [(nue_bin_edges[i + 1] - nue_bin_edges[i]) / 2
                    for i in range(len(nue_bin_edges) - 1)]
        
        if draw_data:
            numu_err_ext = np.array(
                [n * self.weights["numu_ext"] * self.weights["numu_ext"] for n in numu_n_ext])
        else:
            numu_err_ext = np.zeros(len(numu_err_mc))
        
        numu_exp_err    = np.sqrt(numu_err_mc + numu_err_ext + numu_err_nue + numu_err_dirt + err_ncpi0 + err_ccpi0 + err_ccnopi + err_cccpi + err_nccpi + err_ncnopi)
        #print("counting_err: {}".format(exp_err))
        if "numu_dirt" in self.samples:
            numu_detsys_err = numu_sys_mc + numu_sys_nue + numu_sys_dirt + sys_ncpi0 + sys_ccpi0 + sys_ccnopi + sys_cccpi + sys_nccpi + sys_ncnopi
        else:
            numu_detsys_err = numu_sys_mc + numu_sys_nue + sys_ncpi0 + sys_ccpi0 + sys_ccnopi + sys_cccpi + sys_nccpi + sys_ncnopi
        numu_exp_err = np.sqrt(numu_exp_err**2 + numu_detsys_err**2)

        numu_bin_size = [(numu_bin_edges[i + 1] - numu_bin_edges[i]) / 2
                    for i in range(len(numu_bin_edges) - 1)]     
        
        if draw_data:
            
            ratio_err_ext = []
            #this has no extra genie weight
            
            #for i in range(len(nue_err_ext)):
            #    if nue_err_ext[i] > 0 and numu_err_ext[i] > 0:
            #        rratio = nue_err_ext[i]/numu_err_ext[i]
            #        ratio_err_ext.append(rratio)
            #    else:
            #        ratio_err_ext.append(0)
            
            tree_nue_ext = self.samples["nue_ext"]
            queried_tree_nue_ext = tree_nue_ext.query(nue_query)
            variable_nue = queried_tree_nue_ext[variable]
            #spline_fix_cv_nue_ext  = self.weights["nue_ext"]
            nue_selected_ext, bins = np.histogram(
                        variable_nue,
                        range=x_range,
                        bins=n_bins)  
            print("nue_selected_ext ", nue_selected_ext)

            tree_numu_ext = self.samples["numu_ext"]
            queried_tree_numu_ext = tree_numu_ext.query(numu_query)
            variable_numu = queried_tree_numu_ext[variable]
            #spline_fix_cv_numu_ext  = self.weights["numu_ext"]
            numu_selected_ext, bins = np.histogram(
                        variable_numu,
                        range=x_range,
                        bins=n_bins)  
            print("numu_selected_ext ", numu_selected_ext)
            
            for i in range(len(nue_err_mc)):
                dnue = 0
                dnumu = 0
                if nue_selected_ext[i] > 0:
                    dnue = (nue_err_ext[i]/nue_selected_ext[i])
                if numu_selected_ext[i] > 0:
                    dnumu = (numu_err_ext[i]/numu_selected_ext[i])
                rratio = rbin_ratios[i]*np.sqrt((dnue)**2 + (dnumu)**2)
                ratio_err_ext.append(rratio)
            

            print("")
            print("ratio_err_ext: ")
            print(ratio_err_ext)
            print("")
        else:
            ratio_err_ext = np.zeros(len(numu_err_mc))
            
            
        ratio_err_ext = np.array(ratio_err_ext)
        
        ###########################################################
        #Making cov matricies
        ###########################################################

        self.nue_cov           = np.zeros([len(nue_exp_err), len(nue_exp_err)])
        self.nue_cov_mc_stat   = np.zeros([len(nue_exp_err), len(nue_exp_err)])
        self.nue_cov_mc_detsys = np.zeros([len(nue_exp_err), len(nue_exp_err)])
        self.nue_cov_data_stat = np.zeros([len(nue_exp_err), len(nue_exp_err)])
        
        self.numu_cov           = np.zeros([len(numu_exp_err), len(numu_exp_err)])
        self.numu_cov_mc_stat   = np.zeros([len(numu_exp_err), len(numu_exp_err)])
        self.numu_cov_mc_detsys = np.zeros([len(numu_exp_err), len(numu_exp_err)])
        self.numu_cov_data_stat = np.zeros([len(numu_exp_err), len(numu_exp_err)])
        
        self.ratio_cov           = np.zeros([len(numu_exp_err), len(numu_exp_err)])
        self.ratio_cov_mc_stat   = np.zeros([len(numu_exp_err), len(numu_exp_err)])
        self.ratio_cov_mc_detsys = np.zeros([len(numu_exp_err), len(numu_exp_err)])
        self.ratio_cov_data_stat = np.zeros([len(numu_exp_err), len(numu_exp_err)])        
        
        #COV_MC_STAT
        self.nue_cov_mc_stat[np.diag_indices_from(self.nue_cov_mc_stat)]     = (nue_err_mc + nue_err_ext + nue_err_nue + nue_err_dirt + err_ncpi0 + err_ccpi0 + err_ccnopi + err_cccpi + err_nccpi + err_ncnopi)
        #COV_MC_DETSYS
        if "nue_dirt" in self.samples:
            self.nue_cov_mc_detsys[np.diag_indices_from(self.nue_cov_mc_detsys)] = (nue_sys_mc + nue_sys_nue + nue_sys_dirt + sys_ncpi0 + sys_ccpi0 + sys_ccnopi + sys_cccpi + sys_nccpi + sys_ncnopi)**2
        else:
            self.nue_cov_mc_detsys[np.diag_indices_from(self.nue_cov_mc_detsys)] = (nue_sys_mc + nue_sys_nue + sys_ncpi0 + sys_ccpi0 + sys_ccnopi + sys_cccpi + sys_nccpi + sys_ncnopi)**2
            
            
        #COV_MC_STAT
        self.numu_cov_mc_stat[np.diag_indices_from(self.numu_cov_mc_stat)]     = (numu_err_mc + numu_err_ext + numu_err_nue + numu_err_dirt + err_ncpi0 + err_ccpi0 + err_ccnopi + err_cccpi + err_nccpi + err_ncnopi)
        #COV_MC_DETSYS
        if "numu_dirt" in self.samples:
            self.numu_cov_mc_detsys[np.diag_indices_from(self.numu_cov_mc_detsys)] = (numu_sys_mc + numu_sys_nue + numu_sys_dirt + sys_ncpi0 + sys_ccpi0 + sys_ccnopi + sys_cccpi + sys_nccpi + sys_ncnopi)**2
        else:
            self.numu_cov_mc_detsys[np.diag_indices_from(self.numu_cov_mc_detsys)] = (numu_sys_mc + numu_sys_nue + sys_ncpi0 + sys_ccpi0 + sys_ccnopi + sys_cccpi + sys_nccpi + sys_ncnopi)**2 
            
        
        print("")
        print("Into Cov MC Stat ratio:")
        print("")
        print("ratio_err_mc")
        print(ratio_err_mc)
        print("ratio_err_ext")
        print(ratio_err_ext)
        print("ratio_err_nue")
        print(ratio_err_nue) #<--
        print("ratio_err_nue")
        print(ratio_err_dirt)
        
        self.ratio_cov_mc_stat[np.diag_indices_from(self.ratio_cov_mc_stat)]     = (ratio_err_mc + ratio_err_ext + ratio_err_nue + ratio_err_dirt + err_ncpi0 + err_ccpi0 + err_ccnopi + err_cccpi + err_nccpi + err_ncnopi)
        print("")
        print("self.ratio_cov_mc_stat = ", self.ratio_cov_mc_stat)
        print("")
        #self.ratio_cov_mc_stat  = ratio_exp_err
        #detsys cov will be 0, so leave it
            
            
        
        print("")
        print("Into Draw Sys")
        print("")
        if draw_sys:
            if (COVMATRIX == ""):
                print("")
                print("----------------------------------------------------")
                print("IN COVMATRIX_XS_PPFX (NUE)")
                self.nue_cov = (self.sys_err("weightsPPFX",variable,nue_query,plot_options["range"],plot_options["bins"],genieweight, "nue")+
                            self.sys_err("weightsGenie",variable,nue_query,plot_options["range"],plot_options["bins"],genieweight, "nue")+
                            self.sys_err("weightsReint",variable,nue_query, plot_options["range"],plot_options["bins"],genieweight, "nue"))
                if draw_geoSys :
                    print("Add Drawing Geo Sys (NUE)")
                    self.nue_cov += self.sys_err_NuMIGeo("weightsNuMIGeo",variable,nue_query,plot_options["range"],plot_options["bins"],genieweight, "nue")

            else:
                self.nue_cov = self.nue_get_SBNFit_cov_matrix(COVMATRIX,len(nue_bin_edges)-1)
            
            print("")
            print("self.nue_cov")
            print(self.nue_cov)
            print("")
                
            nue_exp_err = np.sqrt( np.diag((self.nue_cov + self.nue_cov_mc_stat + self.nue_cov_mc_detsys))) # + exp_err*exp_err) # Is this the error line?
            print("nue_exp_err ", nue_exp_err)
            print("")

            np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
            print("drawn Sys (NUE)")
            print("----------------------------------------------------")
            
            #cov = self.sys_err("weightsFlux", variable, query, plot_options["range"], plot_options["bins"], "weightSplineTimesTuneTimesPPFX")
            
        if draw_sys:
            if (COVMATRIX == ""):
                print("")
                print("IN COVMATRIX_XS_PPFX (NUMU)")
                self.numu_cov = (self.sys_err("weightsPPFX",variable,numu_query,plot_options["range"],plot_options["bins"],genieweight, "numu")+
                            self.sys_err("weightsGenie",variable,numu_query,plot_options["range"],plot_options["bins"],genieweight, "numu")+
                            self.sys_err("weightsReint",variable,numu_query, plot_options["range"],plot_options["bins"],genieweight, "numu"))
                if draw_geoSys :
                    print("Add Drawing Geo Sys (numu)")
                    self.numu_cov += self.sys_err_NuMIGeo("weightsNuMIGeo",variable,numu_query,plot_options["range"],plot_options["bins"],genieweight, "numu")

                #self.cov = (self.sys_err("weightsReint",variable,query,plot_options["range"],plot_options["bins"],genieweight))

            else:
                self.numu_cov = self.get_SBNFit_cov_matrix(COVMATRIX,len(numu_bin_edges)-1)
                
            print("")
            print("self.numu_cov")
            print(self.numu_cov)
            print("")
                
            numu_exp_err = np.sqrt( np.diag((self.numu_cov + self.numu_cov_mc_stat + self.numu_cov_mc_detsys))) # + exp_err*exp_err) # Is this the error line?
            print("numu_exp_err ", numu_exp_err)
            print("")

            np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
            print("drawn Sys (NUMU)")
            print("----------------------------------------------------")
            
            
        if draw_sys:
            if (COVMATRIX == ""):
                print("")
                print("IN COVMATRIX_XS_PPFX (RATIO)")
                self.ratio_cov = (self.sys_err_ratio("weightsPPFX",variable,nue_query,numu_query,plot_options["range"],plot_options["bins"],genieweight, nue_var_dict, nue_weight_dict, nue_ext_plotted_variable, nue_category_mc_unis, nue_mc_plotted_variable_mc_unis, nue_mc_genie_weights_mc_unis, nue_category_nue_unis, nue_nue_plotted_variable_nue_unis, nue_nue_genie_weights_nue_unis, nue_category_dirt_unis, nue_dirt_plotted_variable_dirt_unis, nue_dirt_genie_weights_dirt_unis, numu_var_dict, numu_weight_dict, numu_ext_plotted_variable, cat_labels, kind, plot_options)+
                            self.sys_err_ratio("weightsGenie",variable,nue_query,numu_query,plot_options["range"],plot_options["bins"],genieweight, nue_var_dict, nue_weight_dict, nue_ext_plotted_variable, nue_category_mc_unis, nue_mc_plotted_variable_mc_unis, nue_mc_genie_weights_mc_unis, nue_category_nue_unis, nue_nue_plotted_variable_nue_unis, nue_nue_genie_weights_nue_unis, nue_category_dirt_unis, nue_dirt_plotted_variable_dirt_unis, nue_dirt_genie_weights_dirt_unis, numu_var_dict, numu_weight_dict, numu_ext_plotted_variable, cat_labels, kind, plot_options)+
                            self.sys_err_ratio("weightsReint",variable,nue_query,numu_query, plot_options["range"],plot_options["bins"],genieweight, nue_var_dict, nue_weight_dict, nue_ext_plotted_variable, nue_category_mc_unis, nue_mc_plotted_variable_mc_unis, nue_mc_genie_weights_mc_unis, nue_category_nue_unis, nue_nue_plotted_variable_nue_unis, nue_nue_genie_weights_nue_unis, nue_category_dirt_unis, nue_dirt_plotted_variable_dirt_unis, nue_dirt_genie_weights_dirt_unis, numu_var_dict, numu_weight_dict, numu_ext_plotted_variable, cat_labels, kind, plot_options))
                if draw_geoSys:
                    print("Add Drawing Geo Sys (ratio)")
                    self.ratio_cov += self.sys_err_NuMIGeo_ratio("weightsNuMIGeo",variable,nue_query,numu_query,plot_options["range"],plot_options["bins"],genieweight, nue_var_dict, nue_weight_dict, nue_ext_plotted_variable, numu_var_dict, numu_weight_dict, numu_ext_plotted_variable, cat_labels, kind, plot_options)

                #self.cov = (self.sys_err("weightsReint",variable,query,plot_options["range"],plot_options["bins"],genieweight))

            else:
                self.ratio_cov = self.get_SBNFit_cov_matrix(COVMATRIX,len(numu_bin_edges)-1)
                
            ratio_exp_err = np.sqrt( np.diag((self.ratio_cov + self.ratio_cov_mc_stat + self.ratio_cov_mc_detsys)))
            print("self.ratio_cov ",self.ratio_cov ) #~0.2
            print("")
            print("self.ratio_cov_mc_stat ", self.ratio_cov_mc_stat) ## ~0.8
            print("")
            print("self.ratio_cov_mc_detsys ", self.ratio_cov_mc_detsys) # 0
            print("")
            print("ratio_exp_err ", ratio_exp_err)
            print("")

            np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
            print("drawn Sys (RATIO)")      
            print("----------------------------------------------------")
            
            
            
            #cov = self.sys_err("weightsFlux", variable, query, plot_options["range"], plot_options["bins"], "weightSplineTimesTuneTimesPPFX")            

#Don't need this for ratio
        if "lee" in self.samples:
            if kind == "event_category":
                try:
                    self.significance = self._sigma_calc_matrix(
                        lee_hist, n_tot-lee_hist, scale_factor=1.01e21/self.pot, cov=(self.cov+self.cov_mc_stat))
                    self.significance_likelihood = self._sigma_calc_likelihood(
                        lee_hist, n_tot-lee_hist, np.sqrt(err_mc + err_ext + err_nue + err_dirt + err_ncpi0 + err_ccpi0 + err_ccnopi + err_cccpi + err_nccpi + err_ncnopi), scale_factor=1.01e21/self.pot)
                    # area normalized version
                    #normLEE = 68. / np.sum(n_tot)
                    #normSM  = 68. / np.sum(n_tot-lee_hist)
                    #self.significance_likelihood = self._sigma_calc_likelihood(
                    #    lee_hist * normLEE, (n_tot-lee_hist) * normSM, np.sqrt(normSM) * np.sqrt(err_mc + err_ext + err_nue + err_dirt + err_ncpi0 + err_ccpi0 + err_ccnopi + err_cccpi + err_nccpi + err_ncnopi), scale_factor=1.0)
                except (np.linalg.LinAlgError, ValueError) as err:
                    print("Error calculating the significance", err)
                    self.significance = -1
                    self.significance_likelihood = -1
        # old error-bar plotting
        #ax1.bar(bincenters, n_tot, facecolor='none',
        #       edgecolor='none', width=0, yerr=exp_err)
        
        ratio_n_tot = rbin_ratios
        nue_ax1.bar(nue_bincenters, nue_exp_err*2,width=[n*2 for n in nue_bin_size],facecolor='tab:blue',alpha=0.2,bottom=(nue_n_tot-nue_exp_err))
        numu_ax1.bar(numu_bincenters, numu_exp_err*2,width=[n*2 for n in numu_bin_size],facecolor='tab:blue',alpha=0.2,bottom=(numu_n_tot-numu_exp_err))
        ratio_ax1.bar(ratio_bincenters, ratio_exp_err*2,width=[n*2 for n in ratio_bin_size],facecolor='tab:blue',alpha=0.2,bottom=(ratio_n_tot-ratio_exp_err))
        #ax1.errorbar(bincenters,n_tot,yerr=exp_err,fmt='k.',lw=35,alpha=0.2)
        '''
        ax1.fill_between(
            bincenters+(bincenters[1]-bincenters[0])/2.,
            n_tot-exp_err,
            n_tot+exp_err,
            step="pre",
            color="grey",
            alpha=0.5)
        '''

        if draw_data:
            plot_options.pop('color', None)
            nue_n_data, nue_bins = np.histogram(nue_data_plotted_variable, **plot_options)
            self.nue_data = nue_n_data
            nue_data_err = self._data_err(nue_n_data,asymErrs)

            self.nue_cov_data_stat[np.diag_indices_from(self.nue_cov_data_stat)] = nue_n_data
            
            numu_n_data, numu_bins = np.histogram(numu_data_plotted_variable, **plot_options)
            self.numu_data = numu_n_data
            numu_data_err = self._data_err(numu_n_data,asymErrs)

            self.numu_cov_data_stat[np.diag_indices_from(self.numu_cov_data_stat)] = numu_n_data            

        #self.cov_data_stat[np.diag_indices_from(self.cov_data_stat)] = n_data
        # This is a hacky workaround -- I should be ashamed of myself, EG
        else:
            nue_n_data = np.zeros(len(nue_bin_size))
            numu_n_data = np.zeros(len(numu_bin_size))
          

        if sum(nue_n_data) > 0:
            nue_ax1.errorbar(
                nue_bincenters,
                nue_n_data,
                xerr=nue_bin_size,
                yerr=nue_data_err,
                fmt='ko',
                label="NuMI: %i" % len(nue_data_plotted_variable) if len(nue_data_plotted_variable) else "")
            
        if sum(numu_n_data) > 0:
            numu_ax1.errorbar(
                numu_bincenters,
                numu_n_data,
                xerr=numu_bin_size,
                yerr=numu_data_err,
                fmt='ko',
                label="NuMI: %i" % len(numu_data_plotted_variable) if len(numu_data_plotted_variable) else "")            

        #frac = self.deltachisqfakedata(plot_options["range"][0], plot_options["range"][-1], np.array([1,1,1,5,5,5]), np.array([1,1,1,5,5,5]), 70)
        if "lee" in self.samples:
            self.sigma_shapeonly = self.deltachisqfakedata(plot_options["range"][0], plot_options["range"][-1], n_tot, (n_tot-lee_hist), 70)


        nue_chistatonly, nue_aab, nue_aac = self._chisq_full_covariance(nue_n_data,nue_n_tot, key="nue",CNP=True,STATONLY=True)
        self.stats['nue_pvaluestatonly'] = (1 - scipy.stats.chi2.cdf(nue_chistatonly,nue_aac))
        
        numu_chistatonly, numu_aab, numu_aac = self._chisq_full_covariance(numu_n_data,numu_n_tot,key="numu",CNP=True,STATONLY=True)
        self.stats['numu_pvaluestatonly'] = (1 - scipy.stats.chi2.cdf(numu_chistatonly,numu_aac))        
        
        if (draw_sys):
            nue_chisqCNP = self._chisq_CNP(nue_n_data,nue_n_tot)
            nue_chicov, nue_chinocov,nue_dof = self._chisq_full_covariance(nue_n_data,nue_n_tot,key="nue",CNP=True)#,USEFULLCOV=True)
            if "lee" in self.samples:
                chilee, chileenocov,dof = self._chisq_full_covariance(n_tot-lee_hist,n_tot, key="nue",CNP=True)
            self.stats['nue_dof']            = nue_dof
            self.stats['nue_chisqstatonly']  = nue_chistatonly
            self.stats['nue_pvaluediag']     = (1 - scipy.stats.chi2.cdf(nue_chinocov,nue_dof))
            self.stats['nue_chisqdiag']     = nue_chinocov

            self.stats['nue_chisq']          = nue_chicov
            self.stats['nue_pvalue']         = (1 - scipy.stats.chi2.cdf(nue_chicov,nue_dof))
            if "lee" in self.samples:
                self.stats['pvaluelee']         = (1 - scipy.stats.chi2.cdf(chilee,dof))
                
            numu_chisqCNP = self._chisq_CNP(numu_n_data,numu_n_tot)
            numu_chicov, numu_chinocov,numu_dof = self._chisq_full_covariance(numu_n_data,numu_n_tot,key="numu",CNP=True)#,USEFULLCOV=True)
            if "lee" in self.samples:
                chilee, chileenocov,dof = self._chisq_full_covariance(n_tot-lee_hist,n_tot,key="numu",CNP=True)
            self.stats['numu_dof']            = numu_dof
            self.stats['numu_chisqstatonly']  = numu_chistatonly
            self.stats['numu_pvaluediag']     = (1 - scipy.stats.chi2.cdf(numu_chinocov,numu_dof))
            self.stats['numu_chisqdiag']     = numu_chinocov

            self.stats['numu_chisq']          = numu_chicov
            self.stats['numu_pvalue']         = (1 - scipy.stats.chi2.cdf(numu_chicov,numu_dof))
            if "lee" in self.samples:
                self.stats['pvaluelee']         = (1 - scipy.stats.chi2.cdf(chilee,dof))                
               

        if (ncol > 3):
            nue_leg = nue_ax1.legend(
                frameon=False, ncol=4, title=r'MicroBooNE Preliminary %g POT' % self.pot,
                prop={'size': fig.get_figwidth()})
        else:
            nue_leg = nue_ax1.legend(
                frameon=False, ncol=2, title=r'MicroBooNE Preliminary %g POT' % self.pot)
        nue_leg._legend_box.align = "left"
        plt.setp(nue_leg.get_title(), fontweight='bold')
        
        if (ncol > 3):
            numu_leg = numu_ax1.legend(
                frameon=False, ncol=4, title=r'MicroBooNE Preliminary %g POT' % self.pot,
                prop={'size': fig.get_figwidth()})
        else:
            numu_leg = numu_ax1.legend(
                frameon=False, ncol=2, title=r'MicroBooNE Preliminary %g POT' % self.pot)
        numu_leg._legend_box.align = "left"
        plt.setp(numu_leg.get_title(), fontweight='bold')       

        unit = title[title.find("[") +
                     1:title.find("]")] if "[" and "]" in title else ""
        x_range = plot_options["range"][1] - plot_options["range"][0]
        if isinstance(plot_options["bins"], Iterable):
            nue_ax1.set_ylabel("N. Entries",fontsize=16)
            numu_ax1.set_ylabel("N. Entries",fontsize=16)
        else:
            nue_ax1.set_ylabel(
                "N. Entries / %.2g %s" % (round(x_range / plot_options["bins"],2), unit),fontsize=16)
            numu_ax1.set_ylabel(
                "N. Entries / %.2g %s" % (round(x_range / plot_options["bins"],2), unit),fontsize=16)

        if (ratio==True):
            nue_ax1.set_xticks([])
            numu_ax1.set_xticks([])

        nue_ax1.set_xlim(plot_options["range"][0], plot_options["range"][1])
        numu_ax1.set_xlim(plot_options["range"][0], plot_options["range"][1])        

        '''
        ax1.fill_between(
            bincenters+(bincenters[1]-bincenters[0])/2.,
            n_tot - exp_err,
            n_tot + exp_err,
            step="pre",
            color="grey",
            alpha=0.5)
        '''
        print("nue exp_err = ", nue_exp_err)
        print("numu exp_err = ", numu_exp_err)       
        
        if (ratio==True):
            if draw_data == False:
                nue_n_data = np.zeros(len(nue_n_tot))
                nue_data_err = (np.zeros(len(nue_n_tot)),np.zeros(len(nue_n_tot)))
                numu_n_data = np.zeros(len(numu_n_tot))
                numu_data_err = (np.zeros(len(numu_n_tot)),np.zeros(len(numu_n_tot)))
            else:
                self.nue_chisqdatamc = self._chisquare(nue_n_data, nue_n_tot, nue_exp_err)
                self.numu_chisqdatamc = self._chisquare(numu_n_data, numu_n_tot, numu_exp_err)
            self._draw_ratio(nue_ax2, nue_bins, nue_n_tot, nue_n_data, nue_exp_err, nue_data_err)
            self._draw_ratio(numu_ax2, numu_bins, numu_n_tot, numu_n_data, numu_exp_err, numu_data_err)            
            

        if ( (chisq==True) and (ratio==True)):
            if sum(nue_n_data) > 0:
                nue_ax2.text(
                    0.725,
                    0.9,
                    r'$\chi^2 /$n.d.f. = %.2f' % (self.stats['nue_chisq']/self.stats['nue_dof']) +
                             #'K.S. prob. = %.2f' % scipy.stats.ks_2samp(n_data, n_tot)[1],
                             ', p = %.2f' % (1 - scipy.stats.chi2.cdf(self.stats['nue_chisq'],self.stats['nue_dof'])) +
                             ', O/P = %.2f' % (sum(nue_n_data)/sum(nue_n_tot)) +
                             ' $\pm$ %.2f' % (self._data_err([sum(nue_n_data)],asymErrs)[0]/sum(nue_n_tot)),
                    va='center',
                    ha='center',
                    ma='right',
                    fontsize=12,
                    transform=nue_ax2.transAxes)
            if sum(numu_n_data) > 0:
                numu_ax2.text(
                    0.725,
                    0.9,
                    r'$\chi^2 /$n.d.f. = %.2f' % (self.stats['numu_chisq']/self.stats['numu_dof']) +
                             #'K.S. prob. = %.2f' % scipy.stats.ks_2samp(n_data, n_tot)[1],
                             ', p = %.2f' % (1 - scipy.stats.chi2.cdf(self.stats['numu_chisq'],self.stats['numu_dof'])) +
                             ', O/P = %.2f' % (sum(numu_n_data)/sum(numu_n_tot)) +
                             ' $\pm$ %.2f' % (self._data_err([sum(numu_n_data)],asymErrs)[0]/sum(numu_n_tot)),
                    va='center',
                    ha='center',
                    ma='right',
                    fontsize=12,
                    transform=numu_ax2.transAxes)

        if (ratio==True):
            nue_ax2.set_xlabel(title,fontsize=18)
            nue_ax2.set_xlim(plot_options["range"][0], plot_options["range"][1])
            numu_ax2.set_xlabel(title,fontsize=18)
            numu_ax2.set_xlim(plot_options["range"][0], plot_options["range"][1])
        else:
            nue_ax1.set_xlabel(title,fontsize=18)
            numu_ax1.set_xlabel(title,fontsize=18)

        nue_fig.tight_layout()
        if title == variable:
            nue_ax1.set_title(nue_query)
            
        numu_fig.tight_layout()
        if title == variable:
            numu_ax1.set_title(numu_query)            

            
        ratio_fig.savefig("full_ratio_fig.pdf")
        if ratio and draw_data:
            return nue_fig, nue_ax1, nue_ax2, numu_fig, numu_ax1, numu_ax2, nue_stacked, labels, nue_n_ext, numu_stacked, labels, numu_n_ext
        elif ratio:
            return nue_fig, nue_ax1, nue_ax2, nue_stacked, labels, numu_fig, numu_ax1, numu_ax2, numu_stacked, labels
        elif draw_data:
            return nue_fig, nue_ax1, nue_stacked, labels, nue_n_ext, numu_fig, numu_ax1, numu_stacked, labels, numu_n_ext
        else:
            return nue_fig, nue_ax1, nue_stacked, labels, numu_fig, numu_ax1, numu_stacked, labels

    def _plot_variable_samples(self, variable, query, title, asymErrs, **plot_options):

        '''
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
        '''

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

    # NuMI needs to add PPFX workaround for dirt
    def sys_err(self, name, var_name, query, x_range, n_bins, weightVar, key):
        # how many universes?
        Nuniverse = 20 #100 #len(df)
        print("Universes",Nuniverse)

        n_tot = np.empty([Nuniverse, n_bins])
        n_cv_tot = np.empty(n_bins)
        n_tot.fill(0)
        n_cv_tot.fill(0)

        for t in self.samples:
            if t in ["nue_ext", "numu_ext","nue_data", "numu_data", "lee", "data_7e18", "data_1e20","nue_dirt","numu_dirt", "numu_mc", "numu_nue"] and key == "nue": 
                continue
                
            if t in ["nue_ext", "numu_ext","nue_data", "numu_data", "lee", "data_7e18", "data_1e20","nue_dirt","numu_dirt", "nue_mc", "nue_nue"] and key == "numu": 
                continue

            # for pi0 fit only
            #if ((t in ["ncpi0","ccpi0"]) and (name == "weightsGenie") ):
            #    continue
            tree = self.samples[t]

            ##MC/OVERLAY
            extra_query = ""
            if t == ("nue_mc" or "numu_mc"):
                extra_query = "& " + self.nu_pdg # "& ~(abs(nu_pdg) == 12 & ccnc == 0) & ~(npi0 == 1 & category != 5)"

            queried_tree = tree.query(query+extra_query)
            variable = queried_tree[var_name]
            syst_weights = queried_tree[name]
            #print ('N universes is :',len(syst_weights))
            spline_fix_cv  = queried_tree[weightVar] * self.weights[t]
            spline_fix_var = queried_tree[weightVar] * self.weights[t]
            if (name == "weightsGenie"):
                spline_fix_var = queried_tree["weightSpline"] * self.weights[t]

            s = syst_weights
            df = pd.DataFrame(s.values.tolist())
            #print (df)
            #continue

            if var_name[-2:] == "_v":
                #this will break for vector, "_v", entries
                variable = variable.apply(lambda x: x[0])

            n_cv, bins = np.histogram(
                variable,
                range=x_range,
                bins=n_bins,
                weights=spline_fix_cv)
            n_cv_tot += n_cv    #this should run twice

            if not df.empty:  #nue, mc
                for i in range(Nuniverse):
                    weight = df[i].values / 1000.   #why 1000?
                    weight[np.isnan(weight)] = 1
                    weight[weight > 100] = 1
                    weight[weight < 0] = 1
                    weight[weight == np.inf] = 1
                    # n is an array - of full number mc in that df
                    n, bins = np.histogram(
                        variable, weights=weight*spline_fix_var, range=x_range, bins=n_bins)
                    n_tot[i] += n           #will run 500 times

        cov = np.empty([len(n_cv), len(n_cv)])
        cov.fill(0)

        #print("n_tot ", n_tot)
        for n in n_tot:
            for i in range(len(n_cv)):
                for j in range(len(n_cv)):
                    cov[i][j] += (n[i] - n_cv_tot[i]) * (n[j] - n_cv_tot[j])

        cov /= Nuniverse
        print("")
        print("cov sys err: = ", cov)
        print("")

        return cov
    
        # NuMI needs to add PPFX workaround for dirt
    def sys_err_ratio(self, name, var_name, query_nue, query_numu, x_range, n_bins, weightVar, nue_var_dict, nue_weight_dict, nue_ext_plotted_variable, nue_category_mc_unis, nue_mc_plotted_variable_mc_unis, nue_mc_genie_weights_mc_unis, nue_category_nue_unis, nue_nue_plotted_variable_nue_unis, nue_nue_genie_weights_nue_unis, nue_category_dirt_unis, nue_dirt_plotted_variable_dirt_unis, nue_dirt_genie_weights_dirt_unis, numu_var_dict, numu_weight_dict, numu_ext_plotted_variable, cat_labels, kind, plot_options):
        # how many universes?
        Nuniverse = 20 #100 #len(df)
        print("Universes",Nuniverse)

        n_tot = np.empty([Nuniverse, n_bins])
        n_cv_tot = np.empty(n_bins)
        n_tot.fill(0)
        n_cv_tot.fill(0)
        
        keylist = [["nue_nue", "numu_mc"]]

        for t in keylist:
            #print(" KEYS , ", t)
            
            tree_nue = self.samples[t[0]]
            tree_numu = self.samples[t[1]]
            
            ##MC/OVERLAY
            extra_query_nue = ""
            extra_query_numu = ""
            if t[0] == "nue_mc":
                extra_query_nue = "& " + self.nu_pdg
            if t[1] == "numu_mc":
                extra_query_numu = "& " + self.nu_pdg
            if t[0] == "numu_mc":
                extra_query_numu = "& " + self.nu_pdg
            if t[1] == "nue_mc":
                extra_query_nue = "& " + self.nu_pdg

            queried_tree_nue = tree_nue.query(query_nue+extra_query_nue)
            queried_tree_numu = tree_numu.query(query_numu+extra_query_numu)
            variable_nue = queried_tree_nue[var_name]
            variable_numu = queried_tree_numu[var_name]
            syst_weights_nue = queried_tree_nue[name]
            syst_weights_numu = queried_tree_numu[name]
            
            spline_fix_cv_nue  = queried_tree_nue[weightVar] * self.weights[t[0]]
            spline_fix_var_nue = queried_tree_nue[weightVar] * self.weights[t[0]]
            spline_fix_cv_numu  = queried_tree_numu[weightVar] * self.weights[t[1]]
            spline_fix_var_numu = queried_tree_numu[weightVar] * self.weights[t[1]]
            if (name == "weightsGenie"):
                spline_fix_var_nue = queried_tree_nue["weightSpline"] * self.weights[t[0]]
                spline_fix_var_numu = queried_tree_numu["weightSpline"] * self.weights[t[1]]

            sn = syst_weights_nue
            df_n = pd.DataFrame(sn.values.tolist())
            sm = syst_weights_numu
            df_m = pd.DataFrame(sm.values.tolist())

            if var_name[-2:] == "_v":
                #this will break for vector, "_v", entries
                variable_nue = variable_nue.apply(lambda x: x[0])
                variable_numu = variable_numu.apply(lambda x: x[0])
             
            stacksort = 3

            #DO THIS FIRST FOR NUE
            # order stacked distributions
            
            c, nue_order_var_dict, nue_order_weight_dict = Plotter_Functions_Alex.plotColourSorting.sortStackDists(stacksort, nue_var_dict, nue_weight_dict)
            
            nue_total = sum(sum(nue_order_weight_dict[c]) for c in nue_order_var_dict)
            #if draw_data:
            nue_total += sum([self.weights["nue_ext"]] * len(nue_ext_plotted_variable))
            #print("total nue in sys ratio err ", nue_total)
            labels = [
                "%s: %.1f" % (cat_labels[c], sum(nue_order_weight_dict[c])) \
                if sum(nue_order_weight_dict[c]) else ""
                for c in nue_order_var_dict.keys()
            ]

            if kind == "event_category":
                plot_options["color"] = [category_colors[c]
                                         for c in nue_order_var_dict.keys()]
            elif kind == "particle_pdg":
                plot_options["color"] = [pdg_colors[c]
                                         for c in nue_order_var_dict.keys()]
            elif kind == "flux":
                plot_options["color"] = [flux_colors[c]
                                         for c in nue_order_var_dict.keys()]
            else:
                plot_options["color"] = [int_colors[c]
                                         for c in nue_order_var_dict.keys()]
            """    
            nue_fig = plt.figure(figsize=(8, 7))
            nue_gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
            nue_ax1 = plt.subplot(nue_gs[0])
            nue_ax2 = plt.subplot(nue_gs[1])
                
            nue_stacked = nue_ax1.hist(
                nue_order_var_dict.values(),
                weights=list(nue_order_weight_dict.values()),
                stacked=True,
                label=labels,
                **plot_options)
            plt.close(nue_fig)
            
            wanted_key=5
            nue_wanted_list = Plotter_Functions_Alex.getWantedLists.getWantedLists(wanted_key, nue_stacked)
            n_cv_nue = nue_wanted_list
            bins = n_bins
            print(nue_wanted_list)
            """
            #print(" VARIABLE NUE: , ", variable_nue)
            #
            n_cv_nue, bins = np.histogram(
                variable_nue,
                range=x_range,
                bins=n_bins,
                weights=spline_fix_cv_nue)

           # nue_stacked = np.histogram(
           #     nue_order_var_dict.values(),
           #     weights=list(nue_order_weight_dict.values()),
           #     stacked=True,
           #     label=labels,
           #     **plot_options)
            
            
            #NOW DO IT FOR NUMU
            c, numu_order_var_dict, numu_order_weight_dict = Plotter_Functions_Alex.plotColourSorting.sortStackDists(stacksort, numu_var_dict, numu_weight_dict)
            numu_total = sum(sum(numu_order_weight_dict[c]) for c in numu_order_var_dict)
            #if draw_data:
            numu_total += sum([self.weights["numu_ext"]] * len(numu_ext_plotted_variable))
            #print("total numu in sys ratio err ", numu_total)
            labels = [
                "%s: %.1f" % (cat_labels[c], sum(numu_order_weight_dict[c])) \
                if sum(numu_order_weight_dict[c]) else ""
                for c in numu_order_var_dict.keys()
            ]


            if kind == "event_category":
                plot_options["color"] = [category_colors[c]
                                         for c in numu_order_var_dict.keys()]
            elif kind == "particle_pdg":
                plot_options["color"] = [pdg_colors[c]
                                         for c in numu_order_var_dict.keys()]
            elif kind == "flux":
                plot_options["color"] = [flux_colors[c]
                                         for c in numu_order_var_dict.keys()]
            else:
                plot_options["color"] = [int_colors[c]
                                         for c in numu_order_var_dict.keys()]
            """
            numu_fig = plt.figure(figsize=(8, 7))
            numu_gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
            numu_ax1 = plt.subplot(numu_gs[0])
            numu_ax2 = plt.subplot(numu_gs[1])
                
            numu_stacked = numu_ax1.hist(
                numu_order_var_dict.values(),
                weights=list(numu_order_weight_dict.values()),
                stacked=True,
                label=labels,
                **plot_options)
            plt.close(numu_fig)
            
            wanted_key=3
            numu_wanted_list = Plotter_Functions_Alex.getWantedLists.getWantedLists(wanted_key, numu_stacked)
            print(numu_wanted_list)
            """
          #  n_cv_nue, bins = np.histogram(
          #          variable_nue,
          #          range=x_range,
          #          bins=n_bins,
          #          weights=spline_fix_cv_nue)
            
            
            n_cv_numu, bins = np.histogram(
                    variable_numu,
                    range=x_range,
                    bins=n_bins,
                    weights=spline_fix_cv_numu)            
            
            rbin_cv_ratios = []
            #n_cv_numu = numu_wanted_list
            #bins = n_bins
            #print(n_cv_numu)
            
            #print("len(n_cv_numu) ", len(n_cv_numu))

            for i in range(len(n_cv_numu)):
                #print("n_cv_nue[i] ", n_cv_nue[i])
                #print("n_cv_numu[i] ", n_cv_numu[i])
                if n_cv_nue[i] > 0 and n_cv_numu[i] > 0:
                    rratio = n_cv_nue[i]/n_cv_numu[i]
                    rbin_cv_ratios.append(rratio)
                else:
                    rbin_cv_ratios.append(0)
            
            
            n_cv_tot += rbin_cv_ratios    #this should run twice

            
            
            if (not df_n.empty) and (not df_m.empty):  #df is the df of weights here
                for i in range(Nuniverse):
                    weight_n = df_n[i].values / 1000. 
                    weight_n[np.isnan(weight_n)] = 1
                    weight_n[weight_n > 100] = 1
                    weight_n[weight_n < 0] = 1
                    weight_n[weight_n == np.inf] = 1
                    weight_m = df_m[i].values / 1000. 
                    weight_m[np.isnan(weight_m)] = 1
                    weight_m[weight_m > 100] = 1
                    weight_m[weight_m < 0] = 1
                    weight_m[weight_m == np.inf] = 1                    
                    # n is an array - of full number mc in that df
                    n_n, bins = np.histogram(
                            variable_nue, weights=weight_n*spline_fix_var_nue, range=x_range, bins=n_bins)
                    n_m, bins = np.histogram(
                            variable_numu, weights=weight_m*spline_fix_var_numu, range=x_range, bins=n_bins)  
                    """    
                    nue_weight_dict_unis = defaultdict(list)
                    nue_var_dict_unis = defaultdict(list)
                    
                    for c, v, w in zip(nue_category_mc_unis, nue_mc_plotted_variable_mc_unis, nue_mc_genie_weights_mc_unis):
                                nue_var_dict_unis[c].append(v)
                                nue_weight_dict_unis[c].append(self.weights["nue_mc"] * w * weight_n)

                    for c, v, w in zip(nue_category_nue_unis, nue_nue_plotted_variable_nue_unis, nue_nue_genie_weights_nue_unis):
                                nue_var_dict_unis[c].append(v)
                                nue_weight_dict_unis[c].append(self.weights["nue_nue"] * w *  weight_n)

                    for c, v, w in zip(nue_category_dirt_unis, nue_dirt_plotted_variable_dirt_unis, nue_dirt_genie_weights_dirt_unis):
                                nue_var_dict_unis[c].append(v)
                                nue_weight_dict_unis[c].append(self.weights["nue_dirt"] * w * weight_n)
                    
                    print("stacksort = ", stacksort)
                    print(nue_var_dict_unis)
                    print(nue_weight_dict_unis)
                    f, nue_order_var_dict_unis, nue_order_weight_dict_unis= Plotter_Functions_Alex.plotColourSorting.sortStackDists(stacksort, nue_var_dict_unis, nue_weight_dict_unis)
                    nue_total = sum(sum(nue_order_weight_dict_unis[f]) for f in nue_order_var_dict_unis)
                    #if draw_data:
                    nue_total += sum([self.weights["nue_ext"]] * len(nue_ext_plotted_variable))
                    #print("total numu in sys ratio err ", numu_total)
                    labels = [
                        "%s: %.1f" % (cat_labels[f], sum(nue_order_weight_dict_unis[f])) \
                        if sum(nue_order_weight_dict_unis[f]) else ""
                        for c in nue_order_var_dict_unis.keys()
                    ]


                    if kind == "event_category":
                        plot_options["color"] = [category_colors[c]
                                                 for c in nue_order_var_dict_unis.keys()]
                    elif kind == "particle_pdg":
                        plot_options["color"] = [pdg_colors[c]
                                                 for c in nue_order_var_dict_unis.keys()]
                    elif kind == "flux":
                        plot_options["color"] = [flux_colors[c]
                                                 for c in nue_order_var_dict_unis.keys()]
                    else:
                        plot_options["color"] = [int_colors[c]
                                                 for c in nue_order_var_dict_unis.keys()]
                    
                    nue_fig = plt.figure(figsize=(8, 7))
                    nue_gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
                    nue_ax1 = plt.subplot(nue_gs[0])
                    nue_ax2 = plt.subplot(nue_gs[1])

                    print(variable_nue)
                    nue_stacked = nue_ax1.hist(
                        nue_order_var_dict_unis.values(),
                        weights=list(nue_order_weight_dict_unis.values()),
                        stacked=True,
                        label=labels,
                        **plot_options)
                    plt.close(nue_fig)

                    wanted_key=5
                    nue_wanted_list = Plotter_Functions_Alex.getWantedLists.getWantedLists(wanted_key, nue_stacked)
                    n_n = nue_wanted_list
                    bins = n_bins
                    print(n_n)
                    
                    numu_fig = plt.figure(figsize=(8, 7))
                    numu_gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
                    numu_ax1 = plt.subplot(numu_gs[0])
                    numu_ax2 = plt.subplot(numu_gs[1])

                    numu_stacked = numu_ax1.hist(
                        numu_order_var_dict.values(),
                        weights=list(numu_order_weight_dict.values()),
                        stacked=True,
                        label=labels,
                        **plot_options)
                    plt.close(numu_fig)

                    wanted_key=3
                    numu_wanted_list = Plotter_Functions_Alex.getWantedLists.getWantedLists(wanted_key, numu_stacked)
                    n_m= numu_wanted_list
                    bins = n_bins
                    print(n_m)
                    """
                    rbin_ratios_sys = []
        
                    for g in range(len(n_n)):
                        if n_n[g] > 0 and n_m[g] > 0:
                            rratio = n_n[g]/n_m[g]
                            rbin_ratios_sys.append(rratio)
                        else:
                            rbin_ratios_sys.append(0)

                    #print("")
                    #print("bin_ratios:")
                    #print(rbin_ratios_sys)
                    #print("")

                    #n->rbin_ratios_sys
                    n_tot[i] += rbin_ratios_sys          #will run 500 times
                    #print("i ", i)
                    #print("n_tot[i]", n_tot[i])
                    #print("n_tot ", n_tot)

        cov = np.empty([len(rbin_cv_ratios), len(rbin_cv_ratios)])
        cov.fill(0)
        print("n_tot")
        print(len(n_tot))
        print("")
        print("n_cv_tot")
        print(n_cv_tot)

        for n in n_tot:
            for i in range(len(rbin_cv_ratios)):
                for j in range(len(rbin_cv_ratios)):
                    #print("n[i]", n[i])
                    #print("n_cv_tot[i]", n_cv_tot[i])
                    cov[i][j] += (n[i] - n_cv_tot[i]) * (n[j] - n_cv_tot[j])

        cov /= Nuniverse
        print("")
        print("cov of ratio sys error = ", cov)
        print("--------------------------------------------------")
        print("")

        return cov

    def sys_err_NuMIGeoMultiverse(self, name, var_name, query, x_range, n_bins, weightVar, key):
        # how many universes?
        Nuniverse = 20 #100 #len(df)
        print("Universes Geo Multiverses",Nuniverse)

        n_tot = np.empty([Nuniverse, n_bins])
        n_cv_tot = np.empty(n_bins)
        n_tot.fill(0)
        n_cv_tot.fill(0)

        for t in self.samples:
            if t in ["nue_ext", "numu_ext","nue_data", "numu_data", "lee", "data_7e18", "data_1e20","nue_dirt","numu_dirt", "numu_mc", "numu_nue"] and key == "nue": 
                continue
                
            if t in ["nue_ext", "numu_ext","nue_data", "numu_data", "lee", "data_7e18", "data_1e20","nue_dirt","numu_dirt", "nue_mc", "nue_nue"] and key == "numu": 
                continue

            # for pi0 fit only
            #if ((t in ["ncpi0","ccpi0"]) and (name == "weightsGenie") ):
            #    continue

            tree = self.samples[t]


            extra_query = ""
            if t == ("nue_mc" or "numu_mc"):
                extra_query = "& " + self.nu_pdg # "& ~(abs(nu_pdg) == 12 & ccnc == 0) & ~(npi0 == 1 & category != 5)"

            queried_tree = tree.query(query+extra_query)
            variable = queried_tree[var_name]
            syst_weights = queried_tree[name]
            #print ('N universes is :',len(syst_weights))
            spline_fix_cv  = queried_tree[weightVar] * self.weights[t]
            spline_fix_var = queried_tree[weightVar] * self.weights[t]
            if (name != "weightsNuMIGeo"):                     
                    sys.exit(1) 

            s = syst_weights
            df = pd.DataFrame(s.values.tolist())
            #print (df)
            #continue

            if var_name[-2:] == "_v":
                #this will break for vector, "_v", entries
                variable = variable.apply(lambda x: x[0])

            n_cv, bins = np.histogram(
                variable,
                range=x_range,
                bins=n_bins,
                weights=spline_fix_cv)
            n_cv_tot += n_cv

            if not df.empty:
                for i in range(Nuniverse):
                    weight = df[i].values 
                    weight[np.isnan(weight)] = 1
                    weight[weight > 100] = 1
                    weight[weight < 0] = 1
                    weight[weight == np.inf] = 1

                    n, bins = np.histogram(
                        variable, weights=weight*spline_fix_var, range=x_range, bins=n_bins)
                    n_tot[i] += n

        cov = np.empty([len(n_cv), len(n_cv)])
        cov.fill(0)

        for n in n_tot:
            for i in range(len(n_cv)):
                for j in range(len(n_cv)):
                    cov[i][j] += (n[i] - n_cv_tot[i]) * (n[j] - n_cv_tot[j])

        cov /= Nuniverse
        
        

        return cov
    
    def sys_err_NuMIGeo(self, name, var_name, query, x_range, n_bins, weightVar, key):
      # how many universes?
        print("Number of variations Universes",10)
        for variationNumber in [x*2 for x in range(10)]:
            n_tot = np.empty([2, n_bins])
            n_cv_tot = np.empty(n_bins)
            n_tot.fill(0.)
            n_cv_tot.fill(0.)########

            for t in self.samples:
                if t in ["nue_ext", "numu_ext","nue_data", "numu_data", "lee", "data_7e18", "data_1e20","nue_dirt","numu_dirt", "numu_mc", "numu_nue"] and key == "nue": 
                    continue
                
                if t in ["nue_ext", "numu_ext","nue_data", "numu_data", "lee", "data_7e18", "data_1e20","nue_dirt","numu_dirt", "nue_mc", "nue_nue"] and key == "numu": 
                    continue
               
                tree = self.samples[t]
                extra_query = ""
                if t == ("nue_mc" or "numu_mc"):
                    extra_query = "& " + self.nu_pdg # "& ~(abs(nu_pdg) == 12 & ccnc == 0) & ~(npi0 == 1 & category != 5)"

                queried_tree = tree.query(query+extra_query)
                variable = queried_tree[var_name]
                syst_weights = queried_tree[name]
                spline_fix_cv  = queried_tree[weightVar] * self.weights[t]
                spline_fix_var = queried_tree[weightVar] * self.weights[t]
                if (name != "weightsNuMIGeo"):                     
                    sys.exit(1) 
    
                s = syst_weights
                df = pd.DataFrame(s.values.tolist())

                if var_name[-2:] == "_v":
                    #this will break for vector, "_v", entries
                    variable = variable.apply(lambda x: x[0])

                n_cv, bins = np.histogram(
                    variable,
                    range=x_range,
                    bins=n_bins,
                    weights=spline_fix_cv)
                n_cv_tot += n_cv

                if not df.empty:
                    for i in range(2):
                        #print(df.shape)
                        weight = df[i+variationNumber].values
                        weight[np.isnan(weight)] = 1
                        weight[weight > 100] = 1
                        weight[weight < 0] = 1
                        weight[weight == np.inf] = 1

                        n, bins = np.histogram(
                            variable, weights=weight*spline_fix_var, range=x_range, bins=n_bins)
                        #print("i = ", i)
                        n_tot[i] += n

                        
            print("")
            print("variation number ", variationNumber)
            print("n_tot") 
            print(n_tot)
            print("--------##--------")
            print("n_cv_tot") 
            print(n_cv_tot)
            print("------------------")
            tempCov = np.empty([len(n_cv), len(n_cv)])
            tempCov.fill(0)
            for n in n_tot:
                for i in range(len(n_cv)):
                    for j in range(len(n_cv)):
                        tempCov[i][j] += (n[i] - n_cv_tot[i]) * (n[j] - n_cv_tot[j])

            tempCov /= 2
            if variationNumber == 0:
                cov = tempCov
            else:
                cov += tempCov
                
            
        print("")
        print("cov NuMI Geo = ", cov)
        print("")

            
        return cov
    
    def sys_err_NuMIGeo_ratio(self, name, var_name, query_nue, query_numu, x_range, n_bins, weightVar, nue_var_dict, nue_weight_dict, nue_ext_plotted_variable, numu_var_dict, numu_weight_dict, numu_ext_plotted_variable, cat_labels, kind, plot_options):
        # how many universes?
        Nuniverse = 10 #100 
        print("Universes NuMI Geo: ",Nuniverse)

        for variationNumber in [x*2 for x in range(10)]:
            n_tot = np.empty([2, n_bins])
            n_cv_tot = np.empty(n_bins)
            n_tot.fill(0)
            n_cv_tot.fill(0)########
            
            keylist = [["nue_nue", "numu_mc"]]
            

            for t in keylist:

                tree_nue = self.samples[t[0]]
                tree_numu = self.samples[t[1]]

                ##MC/OVERLAY
                extra_query_nue = ""
                extra_query_numu = ""
                if t[0] == "nue_mc":
                    extra_query_nue = "& " + self.nu_pdg
                if t[1] == "numu_mc":
                    extra_query_numu = "& " + self.nu_pdg
                if t[0] == "numu_mc":
                    extra_query_numu = "& " + self.nu_pdg
                if t[1] == "nue_mc":
                    extra_query_nue = "& " + self.nu_pdg

                queried_tree_nue = tree_nue.query(query_nue+extra_query_nue)
                queried_tree_numu = tree_numu.query(query_numu+extra_query_numu)
                variable_nue = queried_tree_nue[var_name]
                variable_numu = queried_tree_numu[var_name]
                syst_weights_nue = queried_tree_nue[name]
                syst_weights_numu = queried_tree_numu[name]

                spline_fix_cv_nue  = queried_tree_nue[weightVar] * self.weights[t[0]]
                spline_fix_var_nue = queried_tree_nue[weightVar] * self.weights[t[0]]
                spline_fix_cv_numu  = queried_tree_numu[weightVar] * self.weights[t[1]]
                spline_fix_var_numu = queried_tree_numu[weightVar] * self.weights[t[1]]
                if (name != "weightsNuMIGeo"):
                    sys.exit(1)

                sn = syst_weights_nue
                df_n = pd.DataFrame(sn.values.tolist())
                sm = syst_weights_numu
                df_m = pd.DataFrame(sm.values.tolist())

                if var_name[-2:] == "_v":
                    #this will break for vector, "_v", entries
                    variable_nue = variable_nue.apply(lambda x: x[0])
                    variable_numu = variable_numu.apply(lambda x: x[0])

                stacksort = 3

                c, nue_order_var_dict, nue_order_weight_dict = Plotter_Functions_Alex.plotColourSorting.sortStackDists(stacksort, nue_var_dict, nue_weight_dict)
                nue_total = sum(sum(nue_order_weight_dict[c]) for c in nue_order_var_dict)
                #if draw_data:
                nue_total += sum([self.weights["nue_ext"]] * len(nue_ext_plotted_variable))
                #print("total ", nue_total)
                labels = [
                    "%s: %.1f" % (cat_labels[c], sum(nue_order_weight_dict[c])) \
                    if sum(nue_order_weight_dict[c]) else ""
                    for c in nue_order_var_dict.keys()
                ]


                if kind == "event_category":
                    plot_options["color"] = [category_colors[c]
                                             for c in nue_order_var_dict.keys()]
                elif kind == "particle_pdg":
                    plot_options["color"] = [pdg_colors[c]
                                             for c in nue_order_var_dict.keys()]
                elif kind == "flux":
                    plot_options["color"] = [flux_colors[c]
                                             for c in nue_order_var_dict.keys()]
                else:
                    plot_options["color"] = [int_colors[c]
                                             for c in nue_order_var_dict.keys()]

               # nue_stacked = np.histogram(
               #     nue_order_var_dict.values(),
               #     weights=list(nue_order_weight_dict.values()),
               #     stacked=True,
               #     label=labels,
               #     **plot_options)

                
                #NOW DO IT FOR NUMU
                # order stacked distributions

                c, numu_order_var_dict, numu_order_weight_dict = Plotter_Functions_Alex.plotColourSorting.sortStackDists(stacksort, numu_var_dict, numu_weight_dict)
                numu_total = sum(sum(numu_order_weight_dict[c]) for c in numu_order_var_dict)
                #if draw_data:
                numu_total += sum([self.weights["numu_ext"]] * len(numu_ext_plotted_variable))
                #print("total ", numu_total)
                labels = [
                    "%s: %.1f" % (cat_labels[c], sum(numu_order_weight_dict[c])) \
                    if sum(numu_order_weight_dict[c]) else ""
                    for c in numu_order_var_dict.keys()
                ]


                if kind == "event_category":
                    plot_options["color"] = [category_colors[c]
                                             for c in numu_order_var_dict.keys()]
                elif kind == "particle_pdg":
                    plot_options["color"] = [pdg_colors[c]
                                             for c in numu_order_var_dict.keys()]
                elif kind == "flux":
                    plot_options["color"] = [flux_colors[c]
                                             for c in numu_order_var_dict.keys()]
                else:
                    plot_options["color"] = [int_colors[c]
                                             for c in numu_order_var_dict.keys()]

               # nue_stacked = np.histogram(
               #     nue_order_var_dict.values(),
               #     weights=list(nue_order_weight_dict.values()),
               #     stacked=True,
               #     label=labels,
               #     **plot_options)

                n_cv_nue, bins = np.histogram(
                        variable_nue,
                        range=x_range,
                        bins=n_bins,
                        weights=spline_fix_cv_nue)

                n_cv_numu, bins = np.histogram(
                        variable_numu,
                        range=x_range,
                        bins=n_bins,
                        weights=spline_fix_cv_numu)            

                rbin_cv_ratios = []

                for i in range(len(n_cv_numu)):
                    if n_cv_nue[i] > 0 and n_cv_numu[i] > 0:
                        rratio = n_cv_nue[i]/n_cv_numu[i]
                        rbin_cv_ratios.append(rratio)
                    else:
                        rbin_cv_ratios.append(0)

                print("")
                print("bin_cv_ratios in geo sys:")
                print(rbin_cv_ratios)
                print("")


                n_cv_tot += rbin_cv_ratios    #this should run twice


                if (not df_n.empty) and (not df_m.empty):  #df is the df of weights here
                    for i in range(2):
                        weight_n = df_n[i+variationNumber].values
                        weight_n[np.isnan(weight_n)] = 1
                        weight_n[weight_n > 100] = 1
                        weight_n[weight_n < 0] = 1
                        weight_n[weight_n == np.inf] = 1
                        weight_m = df_m[i+variationNumber].values
                        weight_m[np.isnan(weight_m)] = 1
                        weight_m[weight_m > 100] = 1
                        weight_m[weight_m < 0] = 1
                        weight_m[weight_m == np.inf] = 1                    
                        # n is an array - of full number mc in that df
                        n_n, bins = np.histogram(
                                variable_nue, weights=weight_n*spline_fix_var_nue, range=x_range, bins=n_bins)
                        n_m, bins = np.histogram(
                                variable_numu, weights=weight_m*spline_fix_var_numu, range=x_range, bins=n_bins)       

                        rbin_ratios_sys = []

                        for j in range(len(n_n)):
                            if n_n[j] > 0 and n_m[j] > 0:
                                rratio = n_n[j]/n_m[j]
                                rbin_ratios_sys.append(rratio)
                            else:
                                rbin_ratios_sys.append(0)

                        #print("")
                        #print("bin_ratios in NuMI Geo Ratio Sys:")
                        #print(rbin_ratios_sys)
                        #print("")
                        
                        #so first set of ratios + second set of ratios
                        #n->rbin_ratios_sys
                        n_tot[i] += rbin_ratios_sys          #will run 500 times

            print("")
            print(len(n_tot))
            print(n_tot)
            print("")
            print("n_tot") 
            print(n_tot)
            print("--------##--------")
            print("n_cv_tot") 
            print(n_cv_tot)
            print("------------------")
            tempCov = np.empty([len(rbin_cv_ratios), len(rbin_cv_ratios)])
            tempCov.fill(0)
            for n in n_tot:
                for i in range(len(rbin_cv_ratios)):
                    for j in range(len(rbin_cv_ratios)):
                        tempCov[i][j] += (n[i] - n_cv_tot[i]) * (n[j] - n_cv_tot[j])


            tempCov /= 2
            if variationNumber == 0:
                cov = tempCov
            else:
                cov += tempCov            

            
        print("")
        print("cov NuMI Geo ratio = ", cov)
        print("")
        print("-------------------------------------------------------")

        return cov

    def get_SBNFit_cov_matrix(self,COVMATRIX,NBINS):

        covmatrix = np.zeros([NBINS,NBINS])
        
        if (os.path.isfile("COV/"+COVMATRIX) == False):
            print ('ERROR : file-path for covariance matrix not valid!')
            return covmatrix

        covmatrixfile = open("COV/"+COVMATRIX,"r")

        NLINES = len(covmatrixfile.readlines())

        print ('file has %i lines and histo has %i bins'%(NLINES,NBINS))

        if NLINES != NBINS:
            print ('ERROR : number of lines in text-file does not match number of bins!')
            return covmatrix

        LINECTR = 0

        covmatrixfile.seek(0,0)
        for line in covmatrixfile:

            words = line.split(",")

            WORDCTR = 0

            if len(words) != NBINS:
                print ('ERROR : number of words in line does not match number of bins!')
                break
                
            for word in words:

                val = float(word)

                covmatrix[LINECTR][WORDCTR] = val

                WORDCTR += 1

            LINECTR += 1

        return covmatrix
    
    
    
    
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
       

        #plt.show()
        return eff
    
    def div_err(self, res, err1, val1, err2, val2):
        res_err = res * np.sqrt((err1/val1)**2 + (err2/val2)**2)
        return res_err

