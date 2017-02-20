#!/bin/bash
cmake .
make

mkdir devkit/cpp/results/
mkdir devkit/cpp/results/data/
for i in {0..193}
do
    pad='000000'$i
    pad=${pad:(-6)}'_10.png'
    ./SlantedPlane devkit/cpp/data_stereo_flow/training/image_0/$pad input_disp/$pad ./devkit/cpp/results/data/ 
    # 900 500 2000 400 400 2000
done
