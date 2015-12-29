#!/usr/bin/env python
import sys
import wbd_script

def main():
    fmri_file=sys.argv[1]
    out_file=sys.argv[2]
    nuisance_file=sys.argv[3]
    mask_file=sys.argv[4]
    wbd_script.whole_brain_degree(fmri_file,out_file,nuisance_file,mask_file)
if __name__ == "__main__":
    main()
