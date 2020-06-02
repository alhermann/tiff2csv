#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 2020

@author: Alexander Hermann
"""

from PIL import Image
import numpy as np
import csv
from progress.bar import Bar

class TiffReader:
    def __init__(self, sFile: str, files: list, dFile: str, imgNum: int, resolution: float):
        """
            Constructor    
        
            Parameters
            ----------
            sFile:      str
                The path to the source image files
            files:      list
                The list of .tif files in source directory
            dFile:      str
                The path to the output image file
            imgNum:     int
                The number of the input images
            resolution: float
                The resolution of the input images
                
            Returns
            -------
            
        """
        self.sFile      = sFile
        self.files      = files
        self.dFile      = dFile
        self.imgNum     = imgNum
        self.resolution = resolution
        self.vox        = np.array([])
        
        # Assuming all tiff files are of the same size
        im = Image.open(self.sFile + self.files[0])
        self.size : tuple = im.size
        
        if self.size == ():
            raise Exception("At least one tiff image file is corrupted. Aborting process.")
            
    def clearVox(self):
        """
            Clears the voxel data.
            
            Parameters
            ----------
            
            Returns
            -------
            
        """
        self.vox = np.array([])
        
    def concatCols(self, im_stack: np.array) -> np.array:
        """
            Concatenate a stack of numpy arrays by the same rows
            
            Parameters
            ----------
            im_stack:   np.array
                Input 3D array
                
            Returns
            -------
            out_array:  np.array
                Output 2D stack
        """
        
        # Output array
        out_array = np.zeros((im_stack.shape[0], im_stack.shape[1] * im_stack.shape[2]))
        
        k: int = 0
        for j in range(im_stack.shape[-2]):
            for i in range(im_stack.shape[-1]):
                out_array[:, k] = im_stack[:, j, i]
                k += 1
                
        return out_array
        
        
    def readImages(self, pooling_img: dict = {}, pooling_img_z: dict = {}) -> bool:
        """
            Reads the image sequence from the source directory and outputs a list of voxel coordinates.
            
            Parameters
            ----------
            pooling_img:    dict
                Perform max/mean pooling on the input images
            pooling_img_z:  dict
                Perform max/mean pooling on vertical image stack
            
            Returns
            -------
            ~:              bool
                Exit status
            
        """
        
        # Initialize 
        self.clearVox()
        
        # Additional variable for vertical stack pooling
        img_z_pool = np.array([])
        
        # Progressbar
        bar = Bar('Processing', max = self.imgNum)
        
        # Iterate all image files
        for i in range(self.imgNum):
        
            # Read image
            im = Image.open(self.sFile + self.files[i])
            
            # Binarize 
            im = im.convert('1')
            
            # Convert to numpy array
            imarray = np.array(im)
            
            if self.size != im.size:
                raise Exception("Image file {0} is not of the same size as the first one. Aborting process.".format(self.files[i]))
                return False
            
            # Optionally perform a pooling operation
            if (pooling_img != {}):
                scaling_xfactor, scaling_yfactor, imarray = self.pooling(imarray, pooling_img)
                imarray = np.array(imarray, dtype = bool)
            else:
                scaling_xfactor, scaling_yfactor = 1.0, 1.0
            
            # Optionally perform a vertical pooling on the image stack
            if (pooling_img_z != {}):
                if (i % pooling_img_z["kernel"][-1]) or (i == 0):
                    if img_z_pool.size == 0:
                        img_z_pool = imarray
                        img_z_pool = img_z_pool.reshape((imarray.shape[0],imarray.shape[1], 1))
                    else:
                        img_z_pool = np.append(img_z_pool, imarray[:,:,None], axis = 2)
                    
                    if (img_z_pool.shape[-1] == pooling_img_z["kernel"][-1]):
                        # Arrange the images columnswise (same columns from all images together)
                        img_z_pool = self.concatCols(img_z_pool)
                        
                        # Perform max pooling 
                        _, __, imarray = self.pooling(img_z_pool, pooling_img_z)
                        imarray = np.array(imarray, dtype = bool)
                        img_z_pool = np.array([])
                    else:
                        continue            
            else:
                scaling_zfactor = 1.0
                
            # Assuming the first image is the bottom with first point being the origin
            # Assuming homogenous resolution            
            j, k = np.where(imarray != True) # find indices where it is nonzero 
            
            if j.size == 0 or k.size == 0:
                continue
            
            # Get coordinates from resolution
            y = j * self.resolution * scaling_yfactor
            x = k * self.resolution * scaling_xfactor
            z = np.ones(x.size) * (i * self.resolution) - self.resolution * (pooling_img_z["kernel"][-1] - 1) / 2 
            
            # Append to voxel list
            if self.vox.size == 0:
                self.vox = np.vstack((x,y,z)).T
            else:
                self.vox = np.vstack((self.vox, np.vstack((x,y,z)).T))
            
            bar.next()
            
        bar.finish()    
        
        return True
    
    def pooling(self, mat: np.array, params: dict = {"kernel": (2,2), "method": "mean", "pad": False}):
        """
            Perform non-overlapping max/min pooling on 2D or 3D matrices.
            
            Parameters
            ----------
            params:           dict
                Dictionary with optional pooling parameters
            
            Returns
            -------
            scaling_xfactor:  float
                Scaling factor in x direction
            scaling_yfactor:  float   
                Scaling factor in y direction
            result:           np.array
                2D numpy array with max/min pooled results
            
        """
        
        m,n = mat.shape[:2]
        ky,kx = params["kernel"]
        
        _ceil = lambda x,y: np.int(np.ceil(x)/np.float(y))
        
        if params["pad"]:
            ny = _ceil(m, ky)
            nx = _ceil(n, kx)
            size = (ny * ky, nx * kx) + mat.shape[2:]
            mat_pad = np.full(size, np.nan)
            mat_pad[:m,:n,...] = mat
        else:
            ny = m // ky
            nx = n // kx
            mat_pad = mat[:ny * ky, :nx * kx, ...]
            
        new_shape = (ny, ky, nx, kx) + mat.shape[2:]
        
        if params["method"] == "max":
            result = np.nanmax(mat_pad.reshape(new_shape), axis = (1, 3))
        elif params["method"] == "mean":
            result = np.nanmean(mat_pad.reshape(new_shape),axis = (1, 3))
        elif params["method"] == "min":
            result = np.nanmin(mat_pad.reshape(new_shape), axis = (1, 3))
        
        scaling_xfactor = mat.shape[0] / result.shape[0]
        scaling_yfactor = mat.shape[1] / result.shape[1]
        
        return scaling_xfactor, scaling_yfactor, result
    

    def write_csv(self, out: str) -> bool:
        """
            Output the voxel data to user specified output directory as a .csv file.
            
            Parameters
            ----------
            out:   str
                Output file name
                
            Returns
            -------
            ~:     bool
                Exit status
            
        """
        with open(self.dFile + out, newline = '\n', mode = 'w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(["X","Y","Z"])
            self.vox.reshape((-1,3))
            np.savetxt(f, self.vox, delimiter=',')
        
            
        