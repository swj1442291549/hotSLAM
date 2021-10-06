# hotSLAM

SLAM for early-type stars with LAMOST MRS

code for "Exploring the stellar rotation of early-type stars in the LAMOST Medium-Resolution Survey. I. Catalog" (arXiv: 2108.01212), ApJS accepted

## Usage
- `slammer.py`: SLAM
    1. Train ATLAS model to interpolate (`slam0.dump`)
    2. Train rotational model (`slam1.dump`)
    3. Run `slam1` on `dr7_med_clean_sel.fits` 
    4. Run `slam1` on train data with different SNR
- `util/atlasmodel.py`: Model and ModelS class
- `util/mrs.py`: MRS utility
- `util/values`: constant values
- `util/util.py`: general utility
- `util/bctable.py`: BC class
- `util/isoctable.py`: Isoc class
- `util/evotable.py`: Evo class
- `util/deconvolve.py`: deconvolution functions
