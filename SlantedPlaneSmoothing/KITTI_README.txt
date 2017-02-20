1. Download KITTI devkit:
http://kitti.is.tue.mpg.de/kitti/devkit_stereo_flow.zip

2. Extract it in SlantedPlaneSmoothing/

3. Replace SlantedPlaneSmoothing/devkit/cpp/evaluate_stereo.cpp with SlantedPlane/evaluate_stereo.cpp. The same with io_disp.h.

4. Put data_stereo_flow folder with the dataset in SlantedPlaneSmoothing/devkit/cpp/data_stereo_flow. (Symbolic link to save time). Put input disparities in SlantedPlaneSmoothing/input_disp/

5a. You can now run all tests by executing runTests.sh. It will calculate slanted plane smoothing and evaluate the result of the 194 training images.

5b. SlantedPlane can be run separately with runSlanted.sh or manually by calling ./SlantedPlane.

5c. Disparities can be evaluated by placing disparity images directly in devkit/cpp/results/data with the name convention 000000_10.png and then compile and run ./evaluate_stereo. It will try to run every image, so modify code to your needs.   
