# ALeRCE ZTF forced photometry toolbox
# Authors: Alejandra Muñoz Arancibia, Lorena Hernández García, Ernesto Camacho, Priyanjali Patel, Andrea Rojas, Kevin Espíndola, Javier Silva Farfan, Priscilla Behar, Santiago Bernal, Patricia Arévalo, Amelia Bayo, Franz Bauer, Francisco Förster, Paula Sánchez
# 
# Based on ZTF forced photometry service documentation, and on
# discussions/notebooks from ALeRCE ZTF forced photometry group
# meetings
# 
# Functions for reading, processing and cleaning data

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
from astroquery.vizier import Vizier
import re

filters = ['g', 'r', 'i']

# Needed for obtaining airmass at ZTF location
Samuel_Oschin = EarthLocation(lat=(33.+(21.+29./60.)/60.)*u.deg,
                              lon=-(116.+(51.+43./60.)/60.)*u.deg,
                              height=1712.*u.m)

def read_ztf_lc(path=None, procstatusnumeric=False, verbose=False):
    if verbose:
        print('Read data from file\n'+path)
    
    df = pd.read_csv(path, sep=' ', comment='#', skip_blank_lines=True,
                     skipinitialspace=True, dtype=str)
    cols = [name.rstrip(',') for name in df.columns.values]
    df.columns = cols
    df.set_index('index', inplace=True)
    cols = df.columns[(df.columns != 'filter') & (df.columns != 'procstatus')]
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
    
    df['mjd'] = df['jd']-2400000.5
    
    if procstatusnumeric:
        df['procstatus'] = df['procstatus'].apply(pd.to_numeric,
                                                  errors='coerce')
    
    return df

def read_radec(path=None):
    ra = re.findall('Requested input R.A. = ([-+]?\d*\.\d+|\d+) degrees',
                    open(path, 'r').read())[0]
    dec = re.findall('Requested input Dec. = ([-+]?\d*\.\d+|\d+) degrees',
                     open(path, 'r').read())[0]
    
    return(ra, dec)

def flag_bad_data(df=None, path=None, zp_fil=None, verbose=False):
    # Flag bad data
    # From https://web.ipac.caltech.edu/staff/fmasci/ztf/ztf_pipelines_
    # deliverables.pdf page 98
    
    if verbose:
        print('Flag bad data')
    
    colnames = ['CCDquadID', 'ZPthres_g', 'ZPthres_r', 'ZPthres_i']
    zp_thres = pd.read_csv(zp_fil, header=None, sep='\s+', comment='#',
                           names=colnames, skip_blank_lines=True)
    
    df = df.copy()
    
    # Obtain airmass (secz) for observations
    ra, dec = read_radec(path=path)
    df['secz'] = SkyCoord(ra, dec, unit='deg').transform_to(AltAz( \
                 obstime=Time(df['jd'], format='jd'),
                 location=Samuel_Oschin)).secz
    
    # Set correct identifier number of the CCD quadrant (0-63)
    # From https://web.ipac.caltech.edu/staff/fmasci/ztf/ztf_pipelines_
    # deliverables.pdf page 11 (note that qid=(0..3))
    df['CCDquadID'] = 4*(df['ccdid']-1)+df['qid']
    
    # Set zero point threshold
    df['ZPthres'] = np.nan
    for filt in filters:
        for cqid in pd.unique(df['CCDquadID']):
            mask = (df['filter'] == 'ZTF_'+filt) & (df['CCDquadID'] == cqid)
            zpval = zp_thres.loc[zp_thres['CCDquadID'] == cqid,
                                 'ZPthres_'+filt]
            df.loc[mask, 'ZPthres'] = float(zpval)
    
    df['crit1'] = np.nan
    df['crit4'] = np.nan
    df['flag_bad'] = 0
    
    for filt in filters:
        mask = df['filter'] == 'ZTF_'+filt
        
        if filt == 'g':
            df.loc[mask, 'crit1'] = 26.7-0.2*df['secz'][mask]
            df.loc[mask, 'crit4'] = df['ZPthres'][mask]-0.2*df['secz'][mask]
        if filt == 'r':
            df.loc[mask, 'crit1'] = 26.65-0.15*df['secz'][mask]
            df.loc[mask, 'crit4'] = df['ZPthres'][mask]-0.15*df['secz'][mask]
        if filt == 'i':
            df.loc[mask, 'crit1'] = 26.-0.07*df['secz'][mask]
            df.loc[mask, 'crit4'] = df['ZPthres'][mask]-0.07*df['secz'][mask]
    
    mask = (df['zpmaginpsci'] > df['crit1'])
    
    mask2 = ((df['filter'] == 'ZTF_g') & (df['zpmaginpscirms'] > 0.06)) | \
            ((df['filter'] == 'ZTF_r') & (df['zpmaginpscirms'] > 0.05)) | \
            ((df['filter'] == 'ZTF_i') & (df['zpmaginpscirms'] > 0.06))
    
    mask3 = ((df['filter'] == 'ZTF_g') & (df['ncalmatches'] < 80)) | \
            ((df['filter'] == 'ZTF_r') & (df['ncalmatches'] < 120)) | \
            ((df['filter'] == 'ZTF_i') & (df['ncalmatches'] < 100))
    
    mask4 = (df['zpmaginpsci'] < df['crit4'])
    
    df.loc[mask | mask2 | mask3 | mask4, 'flag_bad'] = 1
    
    return df

def clean_data_nanflux_infobits_flagbad(df=None, nanflux=True, infobits=True,
                                        flagbad=True, verbose=False):
    # Remove some epochs from data according to forcediffimflux,
    # infobitssci and flag_bad criteria (requires running flag_bad_data
    # first for the last one)
    # By default it applies all criteria - each can be turned off
    # individually
    
    if verbose:
        if nanflux or infobits or flagbad:
            print('Clean data')
        else:
            print('No criteria selected for cleaning data')
            return df
    
    df = df.copy()
    
    if nanflux:
        if verbose:
            print('Keeping only epochs with valid difference fluxes')
        mask = df['forcediffimflux'].notna()
        df = df[mask].copy()
    if infobits:
        if verbose:
            print('Keeping only epochs with infobitsci=0')
        mask = df['infobitssci'] == 0
        df = df[mask].copy()
    if flagbad:
        if 'flag_bad' in df.columns:
            if verbose:
                print('Keeping only epochs with flag_bad=0')
            mask = df['flag_bad'] == 0
            df = df[mask].copy()
        else:
            print('Error: flag_bad column missing, run flag_bad_data first')
            return []
    
    return df

def find_ccdid_maxndet(df=None, show_ccdid=False):
    # Find the number of epochs per ccdid for each filter
    # Create a dataframe storing the ccdid that has the largest number
    # of epochs for each filter
    
    ccdid_ndet = pd.DataFrame(columns=['filter', 'ccdid', 'ndet'])
    
    for filt in filters:
        df_filt = df[df['filter'] == 'ZTF_'+filt]
        ccdid_filt = pd.unique(df_filt['ccdid'])
        nccdid_filt = len(ccdid_filt)
        
        for ccdid in ccdid_filt:
            ndet = len(df_filt[df_filt['ccdid'] == ccdid])
            ccdid_ndet = ccdid_ndet.append({'filter': 'ZTF_'+filt,
                                            'ccdid': ccdid, 'ndet': ndet},
                                           ignore_index=True)
    
    ccdid_maxndet = pd.DataFrame(columns=['filter', 'ccdid', 'ndet'])
    
    for filt in filters:
        ccdid_ndet_filt = ccdid_ndet[ccdid_ndet['filter'] == 'ZTF_'+filt]
        if len(ccdid_ndet_filt) > 0:
            id_maxndet = ccdid_ndet_filt['ndet'].astype(int).idxmax()
            ccdid_maxndet = ccdid_maxndet.append( \
                            ccdid_ndet_filt.loc[id_maxndet],
                            ignore_index=True)
    
    ccdid_maxndet.set_index('filter', inplace=True)
    
    if show_ccdid:
        display(ccdid_ndet)
        display(ccdid_maxndet)
    
    return ccdid_maxndet

def clean_data_ccdid(df=None, show_ccdid=False, verbose=False):
    # Keep data only for the ccdid that has the largest number of
    # epochs for each filter
    
    if verbose:
        print('Clean data keeping only ccdids that have the largest numbers' \
              +' of epochs')
    
    df = df.copy()
    ccdid_maxndet = find_ccdid_maxndet(df=df, show_ccdid=show_ccdid)
    
    for filt in filters:
        if 'ZTF_'+filt in pd.unique(df['filter']):
            mask_keep = (df['filter'] == 'ZTF_'+filt) \
                        & (df['ccdid'] == ccdid_maxndet.loc['ZTF_'+filt,
                                                            'ccdid'])
        else:
            mask_keep = (df['filter'] == 'ZTF_'+filt)
        
        if filt == 'g':
            mask_keep_g = mask_keep
        if filt == 'r':
            mask_keep_r = mask_keep
        if filt == 'i':
            mask_keep_i = mask_keep
        
    df = df[mask_keep_g | mask_keep_r | mask_keep_i]
    
    return df

def add_props(df=None, verbose=False):
    # Compute extra quantities
    # From https://web.ipac.caltech.edu/staff/fmasci/ztf/forcedphot.pdf
    # page 15
    # and https://github.com/alercebroker/alerce_ztf_forced/blob/main/
    # lib/alerceforced.py
    
    if verbose:
        print('Compute more quantities (total fluxes, difference magnitudes' \
              +', etc)')
    
    df = df.copy()
    
    df['nearestrefflux'] = 10.**(0.4*(df['zpdiff']-df['nearestrefmag']))
    df['flux_tot'] =  df['forcediffimflux']+df['nearestrefflux']
    
    flux2uJy = 10.**((8.9-df['zpdiff'])/2.5)*1.e6
    df['flux_diff_uJy'] = df['forcediffimflux']*flux2uJy
    df['sigma_flux_diff_uJy'] = df['forcediffimfluxunc']*flux2uJy
    df['flux_tot_uJy'] = df['flux_tot']*flux2uJy
    
    # Note that this quantity is computed in a different way in
    # https://web.ipac.caltech.edu/staff/fmasci/ztf/forcedphot.pdf page
    # 15, depending on nearestreffluxunc
    df['fluxunc_tot'] = df['forcediffimfluxunc']
    
    df['SNR_tot'] = df['flux_tot']/df['fluxunc_tot']
    df['fluxunc_tot_uJy'] = df['fluxunc_tot']*flux2uJy
    
    # Note that these quantities are computed in a different way in
    # https://web.ipac.caltech.edu/staff/fmasci/ztf/forcedphot.pdf
    # pages 14-15, depending on a S/N threshold
    df['mag_tot'] = df['zpdiff']-2.5*np.log10(df['flux_tot'])
    df['sigma_mag_tot'] = 1.0857/df['SNR_tot']
    
    df['mag_diff'] = df['zpdiff']-2.5*np.log10(df['forcediffimflux'].abs())
    df['sigma_mag_diff'] = 1.0857/(df['forcediffimflux'].abs() \
                           /df['forcediffimfluxunc'])
    df['isdiffpos'] = 1
    df.loc[df['forcediffimflux'] < 0., 'isdiffpos'] = -1
    
    df['flux_tot_ap'] =  df['forcediffimfluxap']+df['nearestrefflux']
    
    df['flux_diff_ap_uJy'] = df['forcediffimfluxap']*flux2uJy
    df['sigma_flux_diff_ap_uJy'] = df['forcediffimfluxuncap']*flux2uJy
    df['flux_tot_ap_uJy'] = df['flux_tot_ap']*flux2uJy
    
    df['fluxunc_tot_ap'] = df['forcediffimfluxuncap']
    df['SNR_tot_ap'] = df['flux_tot_ap']/df['fluxunc_tot_ap']
    df['fluxunc_tot_ap_uJy'] = df['fluxunc_tot_ap']*flux2uJy
    
    df['mag_tot_ap'] = df['zpdiff']-2.5*np.log10(df['flux_tot_ap'])
    df['sigma_mag_tot_ap'] = 1.0857/df['SNR_tot_ap']
    
    df['mag_diff_ap'] = df['zpdiff']-2.5*np.log10(df['forcediffimfluxap'].abs())
    df['sigma_mag_diff_ap'] = 1.0857/(df['forcediffimfluxap'].abs() \
                           /df['forcediffimfluxuncap'])
    df['isdiffpos_ap'] = 1
    df.loc[df['forcediffimfluxap'] < 0., 'isdiffpos_ap'] = -1
    
    filtermapping = {'ZTF_g': int(1), 'ZTF_r': int(2), 'ZTF_i': int(3)}
    df['fid'] = df['filter'].apply(lambda x: filtermapping[x])
    
    return df

def query_ps1(ra_deg=None, dec_deg=None, r_arcsec=None, maxsources=None):
    # Based on https://mommermi.github.io/astronomy/2017/02/14/
    # accessing-the-gaia-and-pan-starrs-catalogs-using-python.html
    
    if maxsources is None:
        Vizier.ROW_LIMIT = -1
    else:
        Vizier.ROW_LIMIT = maxsources
    
    # All columns will be returned - output table will be sorted in
    # increasing distance
    vquery = Vizier(columns=['*', '+_r'])
    
    result = vquery.query_region(SkyCoord(ra=ra_deg, dec=dec_deg,
                                          unit=(u.deg, u.deg), frame='icrs'),
                                 radius=r_arcsec*u.arcsec,
                                 catalog="II/349/ps1")
    
    if len(result) > 0:
        return result[0]
    else:
        return result

def correct_mags_ps1_color(df=None, path=None, r_arcsec=None,
                           use_ps1_table=False, verbose=False):
    # Compute color-corrected apparent magnitudes (using Pan-STARRS1
    # colors)
    # From https://web.ipac.caltech.edu/staff/fmasci/ztf/forcedphot.pdf
    # page 15
    
    if verbose:
        print('Compute apparent magnitudes corrected for Pan-STARRS1 colors')
    
    ra, dec = read_radec(path=path)
    t_obj_ps1 = query_ps1(ra_deg=ra, dec_deg=dec, r_arcsec=r_arcsec)
    #print(len(t_obj_ps1))
    #display(t_obj_ps1)
    
    if len(t_obj_ps1) == 0:
        if verbose:
            print('No PS1 objects found')
        if not use_ps1_table:
            obj_ps1 = pd.DataFrame()
    else:
        obj_ps1 = t_obj_ps1.to_pandas()
        obj_ps1.rename(columns={'_r': 'dist_arcsec'}, inplace=True)
    
    if len(t_obj_ps1) > 1:
        if verbose:
            print(str(len(t_obj_ps1))+' PS1 objects found - will use ' \
                  +'nearest object')
            #display(t_obj_ps1)
    
    if len(t_obj_ps1) >= 1:
        g_ps1 = obj_ps1['gmag'][0]
        r_ps1 = obj_ps1['rmag'][0]
        i_ps1 = obj_ps1['imag'][0]
        g_ps1_err = obj_ps1['e_gmag'][0]
        r_ps1_err = obj_ps1['e_rmag'][0]
        i_ps1_err = obj_ps1['e_imag'][0]
        
        if verbose:
            print('PS1 magnitudes (objID='+str(obj_ps1['objID'][0])+')')
            print('g_ps1='+str(g_ps1)+'+-'+str(g_ps1_err))
            print('r_ps1='+str(r_ps1)+'+-'+str(r_ps1_err))
            print('i_ps1='+str(i_ps1)+'+-'+str(i_ps1_err))
        
        df = df.copy()
        df['mag_tot_clr'] = np.nan
        df['sigma_mag_tot_clr'] = np.nan
        df['mag_tot_ap_clr'] = np.nan
        df['sigma_mag_tot_ap_clr'] = np.nan
        
        # m_{tot,clr,g} = m_{tot,g}+coeff_g*(g_{ps1}-r_{ps1})
        # m_{tot,clr,r} = m_{tot,r}+coeff_r*(g_{ps1}-r_{ps1})
        # m_{tot,clr,i} = m_{tot,i}+coeff_i*(r_{ps1}-i_{ps1})
        
        # (dm_{tot,clr,g})^2 = (dm_{tot,g})^2
        #                      +(g_{ps1}-r_{ps1})^2*(dcoeff_g)^2
        #                      +(coeff_g)^2*((dg_{ps1})^2+(dr_{ps1})^2)
        # (dm_{tot,clr,r})^2 = (dm_{tot,r})^2
        #                      +(g_{ps1}-r_{ps1})^2*(dcoeff_r)^2
        #                      +(coeff_r)^2*((dg_{ps1})^2+(dr_{ps1})^2)
        # (dm_{tot,clr,i})^2 = (dm_{tot,i})^2
        #                      +(r_{ps1}-i_{ps1})^2*(dcoeff_i)^2
        #                      +(coeff_i)^2*((dr_{ps1})^2+(di_{ps1})^2)
        
        mask = (df['filter'] == 'ZTF_g') | (df['filter'] == 'ZTF_r')
        
        aux = df['clrcoeff'][mask]*(g_ps1-r_ps1)
        
        df.loc[mask, 'mag_tot_clr'] = df['mag_tot'][mask]+aux
        df.loc[mask, 'mag_tot_ap_clr'] = df['mag_tot_ap'][mask]+aux
        
        aux = (g_ps1-r_ps1)**2.*(df['clrcoeffunc'][mask])**2. \
              +(df['clrcoeff'][mask])**2.*(g_ps1_err**2.+r_ps1_err**2.)
        
        aux2 = (df['sigma_mag_tot'][mask])**2.+aux
        aux2 = np.sqrt(aux2)
        df.loc[mask, 'sigma_mag_tot_clr'] = aux2
        
        aux2 = (df['sigma_mag_tot_ap'][mask])**2.+aux
        aux2 = np.sqrt(aux2)
        df.loc[mask, 'sigma_mag_tot_ap_clr'] = aux2
        
        mask = df['filter'] == 'ZTF_i'
        
        aux = df['clrcoeff'][mask]*(r_ps1-i_ps1)
        
        df.loc[mask, 'mag_tot_clr'] = df['mag_tot'][mask]+aux
        df.loc[mask, 'mag_tot_ap_clr'] = df['mag_tot_ap'][mask]+aux
        
        aux = (r_ps1-i_ps1)**2.*(df['clrcoeffunc'][mask])**2. \
              +(df['clrcoeff'][mask])**2.*(r_ps1_err**2.+i_ps1_err**2.)
        
        aux2 = (df['sigma_mag_tot'][mask])**2.+aux
        aux2 = np.sqrt(aux2)
        df.loc[mask, 'sigma_mag_tot_clr'] = aux2
        
        aux2 = (df['sigma_mag_tot_ap'][mask])**2.+aux
        aux2 = np.sqrt(aux2)
        df.loc[mask, 'sigma_mag_tot_ap_clr'] = aux2
    
    if use_ps1_table:
        # PS1 data as astropy table (includes column units)
        return df, t_obj_ps1
    else:
        # PS1 data as pandas dataframe
        return df, obj_ps1

def rescale_errors(df=None, verbose=False):
    # Rescale errors using the square root of the mean of the reduced
    # chi-square in PSF-fit
    # From https://web.ipac.caltech.edu/staff/fmasci/ztf/forcedphot.pdf
    # page 13
    # Should be done for forcediffimfluxunc and all quantities
    # depending on it (sigma_flux_diff_uJy, fluxunc_tot, SNR_tot,
    # sigma_mag_tot)
    
    if verbose:
        print('Rescale errors by sqrt(mean(forcediffimchisq)) (per filter ' \
              +'and field)')
    
    df = df.copy()
    df['forcediffimfluxunc_resc'] = np.nan
    df['sigma_flux_diff_uJy_resc'] = np.nan
    df['fluxunc_tot_resc'] = np.nan
    df['fluxunc_tot_uJy_resc'] = np.nan
    df['SNR_tot_resc'] = np.nan
    df['sigma_mag_diff_resc'] = np.nan
    df['sigma_mag_tot_resc'] = np.nan
    
    if 'sigma_mag_tot_clr' in df.columns:
        df['sigma_mag_tot_clr_resc'] = np.nan
    
    sq = pd.DataFrame(columns=['filter', 'field', 'ndet', 'sqrmean_chisq'])
    
    for filt in filters:
        mask = df['filter'] == 'ZTF_'+filt
        field_filt = pd.unique(df['field'][mask])
        
        for field in field_filt:
            mask2 = (df['filter'] == 'ZTF_'+filt) & (df['field'] == field)
            sq2 = np.sqrt(np.mean(df['forcediffimchisq'][mask2]))
            
            sq = sq.append({'filter': 'ZTF_'+filt, 'field': field,
                            'ndet': len(df[mask2]), 'sqrmean_chisq': sq2},
                           ignore_index=True)
            
            df.loc[mask2, 'forcediffimfluxunc_resc'] = \
                df['forcediffimfluxunc'][mask2]*sq2
            df.loc[mask2, 'sigma_flux_diff_uJy_resc'] = \
                df['sigma_flux_diff_uJy'][mask2]*sq2
            
            df.loc[mask2, 'fluxunc_tot_resc'] = \
                df['forcediffimfluxunc_resc'][mask2]
            df.loc[mask2, 'fluxunc_tot_uJy_resc'] = \
                df['fluxunc_tot_uJy'][mask2]*sq2
            
            df.loc[mask2, 'SNR_tot_resc'] = \
                df['flux_tot'][mask2]/df['fluxunc_tot_resc'][mask2]
            df.loc[mask2, 'sigma_mag_diff_resc'] = \
                1.0857/(df['forcediffimflux'][mask2].abs() \
                /df['forcediffimfluxunc_resc'][mask2])
            df.loc[mask2, 'sigma_mag_tot_resc'] = \
                1.0857/df['SNR_tot_resc'][mask2]
            
            # We have
            # m_{tot,clr,r} = m_{tot,r}+coeff_r*(g_{ps1}-r_{ps1})
            # (dm_{tot,clr,g})^2 = (dm_{tot,g})^2
            #                      +(g_{ps1}-r_{ps1})^2*(dcoeff_g)^2
            #                      +(coeff_g)^2*((dg_{ps1})^2+(dr_{ps1})^2)
            #                    = (dm_{tot,g})^2+factor
            # Then
            # (dm_{tot,clr,g,resc})^2 = (dm_{tot,g,resc})^2
            #                           +(g_{ps1}-r_{ps1})^2*(dcoeff_g)^2
            #                           +(coeff_g)^2*((dg_{ps1})^2+(dr_{ps1})^2)
            #                         = (dm_{tot,g,resc})^2+factor
            #                         = (dm_{tot,g,resc})^2
            #                           +(dm_{tot,clr,g})^2-(dm_{tot,g})^2
            # And similarly for r and i
            
            if 'sigma_mag_tot_clr' in df.columns:
                aux = (df['sigma_mag_tot_resc'][mask2])**2. \
                      +(df['sigma_mag_tot_clr'][mask2])**2.-(df['sigma_mag_tot'][mask2])**2.
                aux = np.sqrt(aux)
                df.loc[mask2, 'sigma_mag_tot_clr_resc'] = aux
    
    return df, sq

def print_rows_procstatus(df=None, title=None, subs=None):
    df = df[df['procstatus'].str.contains('|'.join(subs))]
    
    if len(df) > 0:
        print(title)
        display(df)
