"""
Construction of the training and validation databases

Copyright (C) 2018, Matias Tassano <matias.tassano@parisdescartes.fr>

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""

from makedataset import *
import argparse

if __name__ == "__main__":
        
    parser = argparse.ArgumentParser(description=\
								  "Building the training patch database")
    
    parser.add_argument("--rgb", action='store_true',default =False,\
						help='prepare RGB database instead of grayscale')
	# Preprocessing parameters
    parser.add_argument("--patch_size", "--p", type=int, default=128, \
					 help="Patch size")
    parser.add_argument("--stride", "--s", type=int, default=64, \
					 help="Size of stride")
    
    args = parser.parse_args()

    if args.rgb:
        TrainSynRGB('./dataset/train/RGB/synthetic',args.patch_size,args.stride)
        #TrainRealRGB('./dataset/train/RGB/real',args.patch_size,args.stride)
    else:
        TrainSynGRAY('./dataset/train/',args.patch_size,args.stride)
        #TrainRealGRAY('./dataset/train/GRAY/real',args.patch_size,args.stride)
    
