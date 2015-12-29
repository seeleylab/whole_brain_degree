import os
import sys
import numpy as np
import nibabel as nib
import core

def whole_brain_degree(fmri_file,out_file=None,nuisance_file=False,mask_file=False,edge_threshold=0):
    """Compute voxelwise whole brain degree
    Parameters
    ----------
    fmri_file: string
      The path to the fMRI 4D .nii file
    out_file: string, optional
      The path/name of the output .nii file. If not specified, output is 'whole_brain_degree.nii.gz'
    nuisance_file : string, optional
      The path to the nuisance parameters text file. If True regress out nuisance parameters first.
    mask_file: string, optional
      The path to a mask .nii file. If True only calculate whole brain degree amongst voxels in mask.
    edge_threshold: num, optional
      The r-value cutoff for determining whole brain degree; if specified, binary count of edges; otherwise, non-binary sum
    Returns
    -------
    """
    # Determine subset of voxels for which every voxel has nonzero values and optionally is in mask
    input = nib.load(fmri_file)
    input_d = input.get_data()
    data_sum = np.sum(input_d,axis=3)
    data_sum_flat = data_sum.flatten()
    data_nzs = np.nonzero(data_sum_flat)

    if mask_file:
        mask = nib.load(mask_file)
        mask_d = mask.get_data()
        mask_d_flat = mask_d.flatten()
        mask_nzs = np.nonzero(mask_d_flat)
        keep_vox = list(set(data_nzs[0].tolist()) & set(mask_nzs[0].tolist()))
    else:
        keep_vox = data_nzs[0].tolist()
    keep_vox.sort()
    keep_vox_array = np.array(keep_vox)
    n_vox = len(keep_vox)

    dims = input_d.shape
    input_d_flat = np.reshape(input_d, (dims[0]*dims[1]*dims[2],dims[3]))
    input_d_flat_trim = input_d_flat[keep_vox,:]

    del(input,input_d,input_d_flat,mask,mask_d,mask_d_flat)

    # calculate voxelwise correlation matrix
    r_mat = np.zeros((n_vox,n_vox))
    if nuisance_file:
        
        with open(nuisance_file) as infile: 
            with open(os.path.split(nuisance_file)[0]+'/nuisance_regressors_32.txt', 'w') as outfile:
                [outfile.write('   '.join(line.split()[1:])+'\n') for line in infile]
        nuisance_file = os.path.split(nuisance_file)[0]+'/nuisance_regressors_32.txt'
        
        nuis_reg = np.array(core.file_reader(nuisance_file)).T
        input_d_flat_trim_res = np.zeros((input_d_flat_trim.shape))
        for i in range(len(input_d_flat_trim)):
            ts = input_d_flat_trim[i,:]
            reg = np.linalg.lstsq(nuis_reg.T,ts.T) # regress out nuisance parameters
            beta = reg[0]
            input_d_flat_trim_res[i,:] = np.squeeze(ts.T - nuis_reg.T.dot(beta)) # store residuals
                                                    #ts data - #variance in voxel's ts explained by nuisance params = #residuals           
        del(input_d_flat_trim)
        r_mat = np.corrcoef(input_d_flat_trim_res)
    else:
        r_mat = np.corrcoef(input_d_flat_trim)

    r_mat = r_mat[0:len(keep_vox),0:len(keep_vox)]
    r_mat = np.nan_to_num(r_mat)

    # calculate whole brain degree
    if edge_threshold > 0:
        whole_brain_degree_vals = np.sum((r_mat > edge_threshold),axis=0) # binary count
    else:
        whole_brain_degree_vals = np.sum(r_mat,axis=0) # non-binary (weighted) sum

    # get x,y,z values for each index
    l = np.unravel_index(keep_vox_array,(dims[0],dims[1],dims[2]))

    wb_deg_img = np.zeros((dims[0],dims[1],dims[2]))
    wb_deg_img[l[0].tolist(),l[1].tolist(),l[2].tolist()] = whole_brain_degree_vals.tolist()
    mni_img = nib.load('/data/mridata/jbrown/brains/MNI152_T1_4mm_brain.nii.gz')
    img = nib.Nifti1Image(wb_deg_img, mni_img.get_affine())
    if out_file:
        img.to_filename(out_file)
    else:
        img.to_filename("whole_brain_degree.nii.gz")