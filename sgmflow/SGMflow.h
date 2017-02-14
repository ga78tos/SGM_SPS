#include <string>
#include <iostream>
#include <vector>
#include <stdio.h>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/cvaux.h>
#include <opencv2/opencv_modules.hpp>

class SGMflow{
	public:
		SGMflow();
		void compute(const std::string firstImage, const std::string secondimage, float* flowImage);

	private:
		void getRotation(const cv::Mat fun, float* out) const;
		void getTranslation(const float x, const float y, float* out) const;
		void getEpiDerivative(const cv::Mat img, const float* tran, float* out) const;
		void initDisparity(const float* rot, const float x, const float y, float* out) const;
		void costImage(const float* der1, const float* der2, const int* cen1, const int* cen2, const float* disp, const float* tra, const float* rot, const float fac, const int win_size, float* out) const;
		void aggDirection(const int dx, const int dy, const int x, const int y, const float* cost, bool start, float* out) const;
		void allDirSum(float *agg[], const int win_size, const int dirs) const;
		void findMins(const float* agg, const int size, char* out) const;
		void censusTransform(const unsigned char* image, int* censusImage) const;
		void to3c16b(const cv::Mat* den_wps, const float* dir, const float* disp, const float* uw, float* out) const;
		void init(cv::Mat img, int n, int subn, float vmax, int k);

		float* flowImage_;
		int n_;
		int subn_;
		float vmax_;
		int width_;
		int height_;
		int censusWindowRadius_;
		int k_;
};
