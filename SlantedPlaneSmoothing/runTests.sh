#!/bin/bash

#echo "Processing training images"
#echo "Approximated time: 3 hours 15 minutes"
#echo "Calculating left disparity"
#cd sgm_rep/class/Stereo/
#./runStereo.sh
echo "Calculating segmentation, plane-smoothing and boundary labels"
./runSlanted.sh
echo "Calculating result"
cd SlantedPlaneSmoothing/devkit/cpp/
g++ -O3 -DNDEBUG -o evaluate_stereo evaluate_stereo.cpp -lpng
./evaluate_stereo
echo "Result written in SlantedPlaneSmoothing/devkit/cpp/results"
