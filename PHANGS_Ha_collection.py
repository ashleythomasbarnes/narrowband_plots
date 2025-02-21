# coding: utf-8

import timeit
import sys

import numpy as np
from pathlib import Path

from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import ZScaleInterval
from astropy.table import Table
from astropy.coordinates import SkyCoord, Distance, Angle
import astropy.units as u
from astropy.nddata import Cutout2D

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class PHANGSHa:
    """
    Main class to handle PHANGS H-alpha survey data.
    
    This class stores paths to galaxy images, names, and fits files,
    along with their headers and WCS solutions. It also retrieves
    geometric galaxy properties such as center coordinates, position angle (PA),
    inclination, and R25 (optical 25th isophotal radius).
    """
    
    def __init__(self, path, galname):
        """
        Initialize the PHANGSHa class with the given galaxy path and name.
        
        Parameters:
            path (Path): Path to the galaxy FITS images.
        """
        self.path = path
        self.galname = galname
        self.figures_path = Path('/Users/arazza/DAS/Papers_in_prep/PHANGS/Halpha_Survey/All_plots/Figures')
    
    def set_fits_prop(self, filename):
        """
        Load FITS file properties including data, header, and WCS solution.
        
        Parameters:
            filename (Path): Path to the FITS file.
        """
        self.hdulist = fits.open(filename)
        self.data = self.hdulist[0].data
        self.hdr = self.hdulist[0].header
        self.wcs = WCS(self.hdr)
    
    def set_gal_prop(self, pixel_scale):
        """
        Retrieve galaxy properties from the PHANGS sample table.
        
        Parameters:
            pixel_scale (float): Pixel scale in arcsec/pixel.
        """
        self.phangs_tbl = Table.read('./data_sampletable/phangs_sample_table_v1p6.fits')
        
        pgc = [self.phangs_tbl['pgc'][i] for i, name in enumerate(self.phangs_tbl['name']) if name.strip() == self.galname.lower()][0]
        my_gal = self.phangs_tbl['pgc'] == pgc
        
        self.gal_dist = Distance(self.phangs_tbl['dist'][my_gal][0] * u.Mpc)
        
        self.gal_ctr = SkyCoord(
            ra=self.phangs_tbl['orient_ra'][my_gal][0] * u.deg,
            dec=self.phangs_tbl['orient_dec'][my_gal][0] * u.deg
        )
        
        self.gal_incl = Angle(self.phangs_tbl['orient_incl'][my_gal][0] * u.deg)
        
        if np.isnan(self.phangs_tbl['orient_posang'][my_gal][0]):
            from astroquery.vizier import Vizier
            v = Vizier(columns=['**'])
            v.TIMEOUT = 60
            result = v.query_object(self.galname, catalog='RC3')
            self.gal_PA = Angle(result[0]['PA'][0] * u.deg)
        else:
            self.gal_PA = Angle(self.phangs_tbl['orient_posang'][my_gal][0] * u.deg)
        
        self.gal_r25 = Angle(self.phangs_tbl['size_r25'][my_gal][0] * u.arcsec)
        self.r25_kpc = (self.gal_dist * np.tan(self.gal_r25.radian)).to(u.kpc).value
        self.pixel_scale = pixel_scale * u.arcsec / u.pixel
        self.r25_pix = self.gal_r25 / self.pixel_scale
        self.gal_reff = Angle(self.phangs_tbl['size_reff'][my_gal][0] * u.arcsec)
        self.reff_kpc = (self.gal_dist * np.tan(self.gal_reff.radian)).to(u.kpc).value
        self.reff_pix = self.gal_reff / self.pixel_scale
    
    def trim_data(self):
        """
        Trim data to a square region of 2.3 times R25 around the galaxy center.
        """
        new_dim = (2.3 * self.r25_pix).round()
        # new_dim = (5 * self.reff_pix).round()
        cutout = Cutout2D(self.data, self.gal_ctr, (new_dim, new_dim), wcs=self.wcs)
        self.data_trim = cutout.data
        self.wcs_trim = cutout.wcs
        self.shape_trim = cutout.shape

    def scale_data(self): 

        from scipy.stats import lognorm

        fit_vals = lognorm.fit(self.data[~np.isnan(self.data)])
        median = lognorm.median(*fit_vals)
        std = lognorm.std(*fit_vals)
        flatten_threshold = median + 2 * std

        flat_image = flatten_threshold * np.arctan(self.data / flatten_threshold) 
        self.data_flat = flat_image
