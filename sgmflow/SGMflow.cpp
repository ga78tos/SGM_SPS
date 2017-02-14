#include "SGMflow.h"
#include "devkit/cpp/io_flow.h"
#include <cmath>
#include <unistd.h>
#include <opencv2/photo/photo.hpp>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

//default parameters
const int SGMFLOW_DEFAULT_CENSUS_WINDOW_RADIUS = 2;


SGMflow::SGMflow() : censusWindowRadius_(SGMFLOW_DEFAULT_CENSUS_WINDOW_RADIUS) {}

void SGMflow::compute(const string firstImageName, const string secondImageName, float* flowImage){

	//read images in grayscale
	const cv::Mat gray_one = cv::imread(firstImageName, 0);
	const cv::Mat gray_two = cv::imread(secondImageName, 0);

	init(gray_one, 25, 25, 0.20, 5);

	//get SIFT keypoints
	cv::SiftFeatureDetector sift_one;
	std::vector<cv::KeyPoint> kp_one;
	sift_one.detect(gray_one, kp_one);

	cv::SiftFeatureDetector sift_two;
	std::vector<cv::KeyPoint> kp_two;
	sift_two.detect(gray_two, kp_two);

	//get descriptors for the keypoints
	cv::SiftDescriptorExtractor siftDesc;
	
	cv::Mat desc_one;
	siftDesc.compute(gray_one, kp_one, desc_one);
	
	cv::Mat desc_two;
	siftDesc.compute(gray_two, kp_two, desc_two);

	//match keypoints from two images with FLANN matcher
	cv::FlannBasedMatcher m;
	//cv::BruteForceMatcher<cv::L2<float> > m;
	vector<cv::DMatch> matches;
	m.match(desc_one, desc_two, matches);
	cout << matches.size() << endl;

	//extract two point descriptor vectors from descriptor match vector
	vector<int> pdesc_one;
	vector<int> pdesc_two;
	for (vector<cv::DMatch>::const_iterator it = matches.begin(); it != matches.end(); ++it){
		pdesc_one.push_back(it->queryIdx);
		pdesc_two.push_back(it->trainIdx);
	}

	//convert these vectors to POint2f vectors
	vector<cv::Point2f> points_one;
	vector<cv::Point2f> points_two;
	cv::KeyPoint::convert(kp_one, points_one, pdesc_one);
	cv::KeyPoint::convert(kp_two, points_two, pdesc_two);

	//filter point matches
	double max_dist = 0; double min_dist = 100;
	for( int i = 0; i < desc_one.rows; i++ ){
		double dist = matches[i].distance;
		if( dist < min_dist ) min_dist = dist;
		if( dist > max_dist ) max_dist = dist;
	}

	vector<cv::DMatch> ok_matches;

	for( int i = 0; i < desc_one.rows; i++ )
		if( matches[i].distance <= 4*min_dist )
			ok_matches.push_back( matches[i]);

	char* undensify = (char*)calloc((width_/50+1)*(height_/15+1),sizeof(char));
	long num_matches = ok_matches.size();
	vector<cv::Point2f> ok_points_one;
	vector<cv::Point2f> ok_points_two;
	for(int i=0; i < num_matches; i++){
		int idx1 = ok_matches[i].trainIdx;
		int idx2 = ok_matches[i].queryIdx;
		if(undensify[((int)points_one[idx1].y/15)*(width_/50+1)+((int)points_one[idx1].x/50)] == 0){
			undensify[((int)points_one[idx1].y/15)*(width_/50+1)+((int)points_one[idx1].x/50)] = 1;
			ok_points_one.push_back(kp_one[idx2].pt);
			ok_points_two.push_back(kp_two[idx1].pt);
		}
	}

	//calc fundamenta matrix with RANSAC
	cv::Mat mask;
	const cv::Mat fun_mat1 = cv::findFundamentalMat(cv::Mat(ok_points_one), cv::Mat(ok_points_two), CV_FM_RANSAC, 3, 0.99, mask);

	//to get good epipole coords: calc epilines, calc intersection of two epilines
	//extract only inliers for epiline calculation
	vector<cv::Point2f> good_points_one;
	vector<cv::Point2f> good_points_two;
	unsigned int ii;
	for (ii = 0; ii < ok_points_one.size(); ii++)
		if ((unsigned int)mask.at<char>(ii)){
			good_points_one.push_back(ok_points_one.at(ii));
			good_points_two.push_back(ok_points_two.at(ii));
		}

	const cv::Mat fun_mat = cv::findFundamentalMat(cv::Mat(good_points_one), cv::Mat(good_points_two), CV_FM_8POINT, 3, 0.99);
	
	//matrix containing uw vector for each pixel
	//for now just 1D array in format {uw00_x, uw00_y, uw01_x, uw01_y, uw02_x, ...}
	float* uw;
	uw = (float*)malloc(height_*width_*sizeof(float)*2);
	
	getRotation(fun_mat, uw);

	//compute epilines
	vector<cv::Vec3f> lines;
	cv::computeCorrespondEpilines(cv::Mat(good_points_one), 1, fun_mat, lines);

	//compute intersection of two epilines (ideally not two lines next to each other for better accuracy => lines[lines.size()/2])
	cv::Vec3f epole = lines[0].cross(lines[lines.size()/2]);
	if(epole[2] != 0)
		epole = epole * (1.0/epole[2]);

	//matrix containing uv vector for each pixel
	//for now just 1D array in format {uv00_x, uv00_y, uv01_x, uv01_y, uv02_x, ...}
	float* uv;
	uv = (float*)malloc(height_*width_*sizeof(float)*2);

	getTranslation(epole[0], epole[1], uv);

	//calculate directional derivatives in epiline direction
	//1D array in format {der00, der01, der02, ...}
	float* der_one;
	float* der_two;
	der_one = (float*)malloc(height_*width_*sizeof(float));
	der_two = (float*)malloc(height_*width_*sizeof(float));

	getEpiDerivative(gray_one, uv, der_one);
	getEpiDerivative(gray_two, uv, der_two);

	//1D array in the format {d00_0, d00_1, ..., d00_9, d01_0, ...}
	//dxx_y is value of |p + uw(p) + o|*vz/(1-vz) for pixel xx at disparity y
	float* disp;
	disp = (float*)malloc(height_*width_*n_*sizeof(float));

	initDisparity(uw, epole[0], epole[1], disp);

	//1D array in the format {c00, c01, ...}
	//cxx is census
	int* census_one;
	int* census_two;
	census_one = (int*)malloc(width_*height_*sizeof(int));
	census_two = (int*)malloc(width_*height_*sizeof(int));
	
	censusTransform(gray_one.data, census_one);
	censusTransform(gray_two.data, census_two);

	//1D array in the format {c00_0, c00_1, ..., c00_9, c01_0, ...}
	//cxx_y is cost of pixel xx at disparity y
	float* cost;
	cost = (float*)calloc(width_*height_*n_,sizeof(float));

	costImage(der_one, der_two, census_one, census_two, disp, uv, uw, 25, 3, cost);

	//1D array in the format {a00_0, a00_1, ..., a_00_9, a01_0, ...}
	//axx_y is aggregated cost of pixel xx at disparity y (axx_y = L(p, y) = sum L_j(p,y) for each direction j
	//float *agg_r, *agg_l, *agg_u, *agg_d, *agg_ur, *agg_ul, *agg_dr, *agg_dl;
	float *agg[16];

	agg[0] = (float*)malloc(width_*height_*n_*sizeof(float));
	for(int y = 0; y < height_; y++)
		aggDirection(1, 0, 0, y, cost, false, agg[0]);

	agg[1] = (float*)malloc(width_*height_*n_*sizeof(float));
	for(int y = 0; y < height_; y++)
		aggDirection(-1, 0, width_-1, y, cost, false, agg[1]);

	agg[2] = (float*)malloc(width_*height_*n_*sizeof(float));
	for(int x = 0; x < width_; x++)
		aggDirection(0, -1, x, height_-1, cost, false, agg[2]);

	agg[3] = (float*)malloc(width_*height_*n_*sizeof(float));
	for(int x = 0; x < width_; x++)
		aggDirection(0, 1, x, 0, cost, false, agg[3]);

	agg[4] = (float*)malloc(width_*height_*n_*sizeof(float));
	for(int y = 0; y < height_; y++)
		aggDirection(1, -1, 0, y, cost, false, agg[4]);
	for(int x = 0; x < width_; x++)
		aggDirection(1, -1, x, height_-1, cost, false, agg[4]);

	agg[5] = (float*)malloc(width_*height_*n_*sizeof(float));
	for(int y = 0; y < height_; y++)
		aggDirection(-1, -1, width_-1, y, cost, false, agg[5]);
	for(int x = 0; x < width_; x++)
		aggDirection(-1, -1, x, height_-1, cost, false, agg[5]);

	agg[6] = (float*)malloc(width_*height_*n_*sizeof(float));
	for(int y = 0; y < height_; y++)
		aggDirection(1, 1, 0, y, cost, false, agg[6]);
	for(int x = 0; x < width_; x++)
		aggDirection(1, 1, x, 0, cost, false, agg[6]);

	agg[7] = (float*)malloc(width_*height_*n_*sizeof(float));
	for(int y = 0; y < height_; y++)
		aggDirection(-1, 1, width_-1, y, cost, false, agg[7]);
	for(int x = 0; x < width_; x++)
		aggDirection(-1, 1, x, 0, cost, false, agg[7]);

	agg[8] = (float*)malloc(width_*height_*n_*sizeof(float));
	for(int y = 0; y < height_; y++)
		aggDirection(2, -1, 0, y, cost, false, agg[8]);
	for(int x = 0; x < width_; x++)
		aggDirection(2, -1, x, height_-1, cost, false, agg[8]);

	agg[9] = (float*)malloc(width_*height_*n_*sizeof(float));
	for(int y = 0; y < height_; y++)
		aggDirection(1, -2, 0, y, cost, false, agg[9]);
	for(int x = 0; x < width_; x++)
		aggDirection(1, -2, x, height_-1, cost, false, agg[9]);

	agg[10] = (float*)malloc(width_*height_*n_*sizeof(float));
	for(int y = 0; y < height_; y++)
		aggDirection(-2, -1, width_-1, y, cost, false, agg[10]);
	for(int x = 0; x < width_; x++)
		aggDirection(-2, -1, x, height_-1, cost, false, agg[10]);

	agg[11] = (float*)malloc(width_*height_*n_*sizeof(float));
	for(int y = 0; y < height_; y++)
		aggDirection(-1, -2, width_-1, y, cost, false, agg[11]);
	for(int x = 0; x < width_; x++)
		aggDirection(-1, -2, x, height_-1, cost, false, agg[11]);

	agg[12] = (float*)malloc(width_*height_*n_*sizeof(float));
	for(int y = 0; y < height_; y++)
		aggDirection(2, 1, 0, y, cost, false, agg[12]);
	for(int x = 0; x < width_; x++)
		aggDirection(2, 1, x, 0, cost, false, agg[12]);

	agg[13] = (float*)malloc(width_*height_*n_*sizeof(float));
	for(int y = 0; y < height_; y++)
		aggDirection(1, 2, 0, y, cost, false, agg[13]);
	for(int x = 0; x < width_; x++)
		aggDirection(1, 2, x, 0, cost, false, agg[13]);

	agg[14] = (float*)malloc(width_*height_*n_*sizeof(float));
	for(int y = 0; y < height_; y++)
		aggDirection(-2, 1, width_-1, y, cost, false, agg[14]);
	for(int x = 0; x < width_; x++)
		aggDirection(-2, 1, x, 0, cost, false, agg[14]);

	agg[15] = (float*)malloc(width_*height_*n_*sizeof(float));
	for(int y = 0; y < height_; y++)
		aggDirection(-1, 2, width_-1, y, cost, false, agg[15]);
	for(int x = 0; x < width_; x++)
		aggDirection(-1, 2, x, 0, cost, false, agg[15]);

	allDirSum(agg, width_*height_*n_, 16);

	char* min_wp;
	min_wp = (char*)malloc(width_*height_*sizeof(char));
	findMins(agg[0], width_*height_, min_wp);

	
	
	//=================================================================================================
	//FROM HERE ON, ONLY OUTPUT
	//=================================================================================================


	//output matrix/mask data
	cout << "MaskRows: " << mask.rows << endl;
	cout << "MaskCols: " << mask.cols << endl;
	cout << "FunRows: " << fun_mat.rows << endl;
	cout << "FunCols: " << fun_mat.cols << endl;

	for(int i = 0; i < 3; i++){
		for(int j = 0; j < 3; j++)
			cout << fun_mat.at<double>(i,j) << "  ";
		cout << endl;
	}

	//print images with sift keypoints
	cv::Mat out_one;
	cv::drawKeypoints(gray_one, kp_one, out_one);
	cv::imwrite("sift1.png", out_one);

	cv::Mat out_two;
	cv::drawKeypoints(gray_two, kp_two, out_two);
	cv::imwrite("sift2.png", out_two);

	//draw flann matches
	//COPY AND PASTED
	max_dist = 0;
	min_dist = 100;
	for( int i = 0; i < desc_one.rows; i++ ){
		double dist = matches[i].distance;
		if( dist < min_dist ) min_dist = dist;
		if( dist > max_dist ) max_dist = dist;
	}

  	printf("-- Max dist : %f \n", max_dist );
  	printf("-- Min dist : %f \n", min_dist );
  	vector<cv::DMatch> good_matches;

	for( int i = 0; i < desc_one.rows; i++ ){
		if( matches[i].distance <= 20*min_dist ){
			good_matches.push_back( matches[i]);
		}
	}


	//draw good matches
	cv::Mat badmatches;
	cv::cvtColor(gray_one, badmatches, cv::COLOR_GRAY2BGR);
	for( int i = 0; i < (int)matches.size(); i++ )
	{
		//query image is the first frame
		cv::Point2f point_old = kp_one[matches[i].queryIdx].pt;

		//train  image is the next frame that we want to find matched keypoints
		cv::Point2f point_new = kp_two[matches[i].trainIdx].pt;

		//keypoint color for frame 1: RED
		cv::circle(badmatches, point_old, 4, cv::Scalar(0, 0, 255), -1);
		//keypoint color for frame 2: BLUE
		cv::circle(badmatches, point_new, 4, cv::Scalar(0, 0, 255), -1);
		//draw a line between keypoints
		cv::line(badmatches, point_old, point_new, cv::Scalar(0, 255, 0), 2);
	}
	cv::imwrite("badmatches.png",badmatches);

	cv::Mat okmatches;
	cv::cvtColor(gray_one, okmatches, cv::COLOR_GRAY2BGR);
	for( int i = 0; i < (int)ok_matches.size(); i++ )
	{
		//query image is the first frame
		cv::Point2f point_old = kp_one[ok_matches[i].queryIdx].pt;

		//train  image is the next frame that we want to find matched keypoints
		cv::Point2f point_new = kp_two[ok_matches[i].trainIdx].pt;

		//keypoint color for frame 1: RED
		cv::circle(okmatches, point_old, 4, cv::Scalar(0, 0, 255), -1);
		//keypoint color for frame 2: BLUE
		cv::circle(okmatches, point_new, 4, cv::Scalar(0, 0, 255), -1);
		//draw a line between keypoints
		cv::line(okmatches, point_old, point_new, cv::Scalar(0, 255, 0), 2);
	}
	cv::imwrite("okmatches.png",okmatches);

	
	//draw undensified matches
	cv::Mat okp;
	cv::cvtColor(gray_one, okp, cv::COLOR_GRAY2BGR);
	for(int i = 0; i < ok_points_one.size(); i++){
		cv::circle(okp, ok_points_one[i], 4, cv::Scalar(0, 0, 255), -1);
		cv::circle(okp, ok_points_two[i], 4, cv::Scalar(0, 0, 255), -1);
		cv::line(okp, ok_points_one[i], ok_points_two[i], cv::Scalar(0, 255, 0), 2);
	}
	cv::imwrite("okp.png",okp);

	//draw epilines
	cv::Mat gray_lines1;
	cv::cvtColor(gray_one, gray_lines1, cv::COLOR_GRAY2BGR);
	for (vector<cv::Vec3f>::const_iterator it = lines.begin(); it != lines.end(); ++it)
		cv::line(gray_lines1, cv::Point(0, -(*it)[2]/(*it)[1]), cv::Point(width_, -((*it)[2]+(*it)[0]*width_)/(*it)[1]), cv::Scalar(0,255,0));
	cv::imwrite("lines.png", gray_lines1);

	//print rotation vectors
	cv::Mat rot;
	cv::cvtColor(gray_one, rot, cv::COLOR_GRAY2BGR);
	for(int y = 30; y < 350; y += 20)
		for(int x = 70; x < 1190; x += 30){			
			cv::circle(rot, cv::Point2i(x,y), 4, cv::Scalar(0,0,255), -1);
			cv::line(rot, cv::Point2i(x,y), cv::Point2i(x+(int)uw[(width_*y*2)+(x*2)], y+(int)uw[(width_*y*2)+(x*2)+1]), cv::Scalar(0,255,0), 2);
		}
	cv::imwrite("rot.png", rot);

	cv::Mat tra;
	cv::cvtColor(gray_one, tra, cv::COLOR_GRAY2BGR);
	//print translation vectors
	for(int y = 30; y < 350; y += 20)
		for(int x = 70; x < 1190; x += 30){
			cv::circle(tra, cv::Point2i(x,y), 4, cv::Scalar(0,0,255), -1);
			cv::line(tra, cv::Point2i(x,y), cv::Point2i(x+(int)(10*uv[(width_*y*2)+(x*2)]), y+(int)(10*uv[(width_*y*2)+(x*2)+1])), cv::Scalar(0,255,0), 2);
		}
	cv::imwrite("tra.png", tra);

	//print image with derivatives (scale to range 0-255)
	int max = 0;
	for(int i = 0; i < height_*width_; i++)
		if(max < der_one[i]){
			max = der_one[i];
		}
	cout << max << endl;
	cv::Mat der = gray_one.clone();
	for(int y = 0; y < height_; y++){
		for(int x = 0; x < width_; x++){
			der.at<uchar>(y,x) = (int)((der_one[y*width_+x]/max)*255);
		}
	}
	cv::imwrite("der1.png", der);
	for(int y = 0; y < height_; y++){
		for(int x = 0; x < width_; x++){
			der.at<uchar>(y,x) = (int)((der_two[y*width_+x]/max)*255);
		}
	}
	cv::imwrite("der2.png", der);
	for(int offset = 0; offset < n_; offset++){
		max = 0;
		//not correct but good enough
		for (int i = 0; i < width_*height_; i++)
			if(cost[i] > max && cost[i] != FLT_MAX){
				max = cost[i];
			}
		cout << "max cost: " << max << endl;
		cv::Mat c = gray_one.clone();
		for(int y = 0; y < height_; y++){
			for(int x = 0; x < width_; x++){
				c.at<uchar>(y,x) = (int)((cost[(y*width_+x)*n_+offset]/max)*255);
			}
		}
		stringstream sstm;
		sstm << "cost" << offset << ".png";
		cv::imwrite(sstm.str(), c);
	}
	max = 0;
	for(int i = 0; i < width_*height_; i++)
		if(census_one[i] > max)
			max = census_one[i];
	cout << max << endl;
	max = (max/4)*3;
	cv::Mat c = gray_one.clone();
	for(int y = 0; y < height_; y++){
		for(int x = 0; x < width_; x++){
			c.at<uchar>(y,x) = (int)((census_one[y*width_+x]/max)*255);
		}
	}
	cv::imwrite("census.png", c);

	cv::Mat w1 = gray_one.clone();
	for(int y = 0; y < height_; y++){
		for(int x = 0; x < width_; x++){
			w1.at<uchar>(y,x) = (int)(min_wp[y*width_+x]*k_);
		}
	}
	cv::imwrite("wps.png", w1);

	cv::Mat w3 = w1.clone();

	cv::Mat den1;
	cv::fastNlMeansDenoising(w3,den1, 4, 7, 21);
	cv::imwrite("denoise3.png", den1);

	//cv::Mat c_img(height_, width_, CV_16UC3);
	float* c_img;
	c_img = (float*)malloc(width_*height_*3*sizeof(float));
	to3c16b(&den1, uv, disp, uw, c_img);

	FlowImage fi(c_img, width_, height_);
	size_t p = firstImageName.find("0000");
	stringstream sstm1, sstm2;
	sstm1 << "flow1" << firstImageName.substr(p);
	sstm2 << "flow2" << firstImageName.substr(p);
	cout << sstm1.str() << endl;
	fi.write(sstm1.str());
	fi.writeColor(sstm2.str());
}


//calc rotational component for each pixel
void SGMflow::getRotation(const cv::Mat fun, float* out) const {
	cout << "ROT:" << height_*width_*sizeof(float)*2 << endl;
	double cx, cy;
	cx = (double)(width_)/2;
	cy = (double)(height_)/2;
	//to solve Ax=b
	cv::Mat A(width_*height_,5,CV_64FC1,cv::Scalar(0.0));
	cv::Mat b(width_*height_,1,CV_64FC1,cv::Scalar(0.0));
	for(int y = 0; y < height_; y++){
		for(int x = 0; x < width_; x++){
			double x_, y_;
			cv::Mat hom(3,1,CV_64FC1);
			x_ = x - cx;
			y_ = y - cy;
			//generate homogenous coord
			hom.at<double>(0,0) = (double)x;
			hom.at<double>(1,0) = (double)y;
			hom.at<double>(2,0) = 1.0;
			//calc epiline through pixel
			cv::Mat epi = fun*hom;
			//set coefficients in A
			A.at<double>(y*width_+x,0) = epi.at<double>(0,0);
			A.at<double>(y*width_+x,1) = epi.at<double>(1,0);
			A.at<double>(y*width_+x,2) = (epi.at<double>(1,0)*x_)-(epi.at<double>(0,0)*y_);
			A.at<double>(y*width_+x,3) = (epi.at<double>(0,0)*x_*x_)+(epi.at<double>(1,0)*x_*y_);
			A.at<double>(y*width_+x,4) = (epi.at<double>(0,0)*x_*y_)+(epi.at<double>(1,0)*y_*y_);
			//set result in b
			b.at<double>(y*width_+x,0) = -epi.at<double>(2,0)-(epi.at<double>(0,0)*x)-(epi.at<double>(1,0)*y);
		}
	}
	cv::Mat res;
	//solve linear system of equations A*res=b
	//(use DECOMP_SVD or DECOMP_QR because system overdefined, SVD supposedly more accurate but slower, not noticeable for now)
	cv::solve(A,b,res,cv::DECOMP_SVD);

	cout << res << endl;
	for(int y = 0; y < height_; y++){
		for(int x = 0; x < width_; x++){
			float x_, y_;
			x_ = x - cx;
			y_ = y - cy;
			
			out[(y*width_*2)+(x*2)] = (float)(res.at<double>(0,0)-(res.at<double>(2,0)*y_)+(res.at<double>(3,0)*x_*x_)+(res.at<double>(4,0)*x_*y_));
			out[(y*width_*2)+(x*2)+1] = (float)(res.at<double>(1,0)+(res.at<double>(2,0)*x_)+(res.at<double>(3,0)*x_*y_)+(res.at<double>(4,0)*y_*y_));
		}	
	}
	cout << out[180*1226*2+610] << "#" << out[180*1226*2+610] << "#" << out[2] << "#" << out[3] << endl;
}


//calc translational component (only unit vector / no length i.e. disparity), also epipole coords as parameters
void SGMflow::getTranslation(const float epx, const float epy, float* out) const {
	cout << "TRA:" << height_*width_*sizeof(float)*2 << endl;
	for(int y = 0; y < height_; y++)
		for(int x = 0; x < width_; x++){
			float vx, vy, h;
			vx = epx - x;
			vy = epy - y;
			h = sqrt(vx*vx + vy*vy);
			out[(y*width_*2)+(x*2)] = -vx/h;
			out[(y*width_*2)+(x*2)+1] = -vy/h;
		}
}


//calculate directional derivative in direction of epipolar lines for ONE image
void SGMflow::getEpiDerivative(const cv::Mat img, const float* tran, float* out) const {
	//first get x/y derivatives
	cv::Mat gradx, grady;
	cv::Sobel(img, gradx, CV_16S, 1, 0);
	cv::Sobel(img, grady, CV_16S, 0, 1);

	//equation from http://answers.opencv.org/question/387/sobel-derivatives-in-the-45-and-135-degree-direction/
	for(int y = 0; y < height_; y++)
		for(int x = 0; x < width_; x++)
			out[y*width_+x] = sqrt(pow(grady.at<short>(y,x)*tran[(y*width_*2)+(x*2)+1], 2) + pow(gradx.at<short>(y,x)*tran[(y*width_*2)+(x*2)], 2));
}


//census transform
void SGMflow::censusTransform(const unsigned char* image, int* censusImage) const {
	for (int y = 0; y < height_; ++y) {
		for (int x = 0; x < width_; ++x) {
			unsigned char centerValue = image[width_*y + x];

			//for each pixel in range of censuswindowradius (a square around pixel) check if center pixel brighter (higher value) or darker and stor in a bitfield (1 means brighter, 0  darker)
			int censusCode = 0;
			for (int offsetY = -censusWindowRadius_; offsetY <= censusWindowRadius_; ++offsetY) {
				for (int offsetX = -censusWindowRadius_; offsetX <= censusWindowRadius_; ++offsetX) {
					censusCode = censusCode << 1;
					if (y + offsetY >= 0 && y + offsetY < height_
						&& x + offsetX >= 0 && x + offsetX < width_
						&& image[width_*(y + offsetY) + x + offsetX] >= centerValue) censusCode += 1;
				}
			}
			censusImage[width_*y + x] = censusCode;
		}
	}
}


//calculate initial disparity (without vz ratio factor)
void SGMflow::initDisparity(const float* rot, const float epx, const float epy, float* out) const {
	for(int y = 0; y < height_; y++)
		for(int x = 0; x < width_; x++)
			for(int wp = 0; wp < n_; wp++){
				float vz;
				vz = (float)wp/n_*vmax_;
				out[(y*width_+x)*n_+wp] = sqrt(pow(x+rot[(y*width_*2)+(x*2)]-epx, 2)+pow(y+rot[(y*width_*2)+(x*2)+1]-epy, 2))*(vz/(1-vz));
			}
}


//calculates pixel cost according to paper (pixels closer to border than win_size dont have cost)
void SGMflow::costImage(const float* der1, const float* der2, const int* cen1, const int* cen2, const float* disp, const float* tra, const float* rot, const float fac, const int win_size, float* out) const {
	for(int yy = win_size/2; yy < height_-(int)(win_size/2); yy++)
		for(int xx = win_size/2; xx < width_-(int)(win_size/2); xx++) {
			int qx, qy;
			for(int wp = 0; wp < n_; wp++){
				bool never = true;
				//run through window (check paper)
				for(int y = yy-(win_size/2); y < yy+(win_size/2); y++)
					for(int x = xx-(win_size/2); x < xx+(win_size/2); x++) {
						//pos of pixel in second image
						qx = x + rot[(yy*width_*2)+(xx*2)] + tra[(yy*width_*2)+(xx*2)]*disp[((yy*width_)+xx)*n_+wp];
						qy = y + rot[(yy*width_*2)+(xx*2)+1] + tra[(yy*width_*2)+(xx*2)+1]*disp[((yy*width_)+xx)*n_+wp];
						if(qx >= 0 && qx < width_ && qy >= 0 && qy < height_){
							out[((yy*width_)+xx)*n_+wp] += abs(der1[(yy*width_)+xx]-der2[(qy*width_)+qx]) + fac*(_mm_popcnt_u32(static_cast<unsigned int>(cen1[(yy*width_)+xx]^cen2[(qy*width_)+qx])));
							never = false;
						}
					}
				//pixels with no cost at disparity wp have max cost so aggregation does't choose non existent area
				if(never)
					out[((yy*width_)+xx)*n_+wp] = FLT_MAX;
			}
		}
}

//aggregate cost in direction given by dx dy
void SGMflow::aggDirection(const int dx, const int dy, const int x, const int y, const float* cost, bool start, float* out) const {
	if(start && (x+dx < 0 ||  x+dx >= width_ || y+dy < 0 || y+dy >= height_ || cost[(((y+dy)*width_)+(x+dx))*n_+subn_-1] == FLT_MAX)){
		for(int wp = 0; wp < subn_; wp++)
			out[((y*width_)+x)*n_+wp] = cost[((y*width_)+x)*n_+wp];
		return;
	}
	
	if(x+dx >= 0 && x+dx < width_ && y+dy >= 0 && y+dy < height_)
		if(cost[(((y+dy)*width_)+(x+dx))*n_+subn_-1] != FLT_MAX)
			aggDirection(dx, dy, x+dx, y+dy, cost, true, out);
		else
			aggDirection(dx, dy, x+dx, y+dy, cost, false, out);
	else
		return;
	if(start)
		for(int wp = 0; wp < subn_; wp++){
			float min_disp;
			//init min with L_j at wp = 0
			if(wp == 0)
				min_disp = out[(((y+dy)*width_)+(x+dx))*n_];
			else if(wp == 1)
				min_disp = out[(((y+dy)*width_)+(x+dx))*n_]+700;
			else
				min_disp = out[(((y+dy)*width_)+(x+dx))*n_]+10000;
			//check if any L_j smaller than min for wp != 0
			for(int wpp = 1; wpp < subn_; wpp++){
				if(wp == wpp)
					min_disp = min(min_disp, out[(((y+dy)*width_)+(x+dx))*n_+wpp]);
				else if(abs(wp-wpp) == 1)
					min_disp = min(min_disp, out[(((y+dy)*width_)+(x+dx))*n_+wpp]+700);
				else
					min_disp = min(min_disp, out[(((y+dy)*width_)+(x+dx))*n_+wpp]+10000);
			}
			out[((y*width_)+x)*n_+wp] = cost[((y*width_)+x)*n_+wp] + min_disp;
		}
}

void SGMflow::allDirSum(float *agg[], const int win_size, const int dirs) const {
	for(int i = 1; i < dirs; i++)
		for(int j = 0; j < win_size; j++)
			agg[0][j] += agg[i][j];
}

void SGMflow::findMins(const float* agg, const int size, char* out) const {
	float minc = FLT_MAX;
	float maxc = 0;
	for(int i = 0; i < size; i++){
		char wp = 0;
		float agg_cost = agg[i*n_];
		for(int wpp = 1; wpp < subn_; wpp++){
			if(agg[i*n_+wpp] < agg_cost){
				wp = wpp;
				agg_cost = agg[i*n_+wpp];
			}
			if(agg[i*n_+wpp] > maxc)
				maxc = agg[i*n_+wpp];
		}
		out[i] = wp;
		if(agg_cost < minc)
			minc = agg_cost;
	}
	cout << "########################\n" << minc << endl << maxc << endl;
}

//convert to 3 channel float for kitti benchmark
void SGMflow::to3c16b(const cv::Mat* den_wps, const float* dir, const float* disp, const float* uw, float* out) const {
	for(int y = 0; y < height_; y++)
		for(int x = 0; x < width_; x++){
			if((den_wps->at<uchar>(y,x)/k_) != 0){
				out[(y*width_*3)+(x*3)] = dir[(y*width_*2)+(x*2)] * disp[(y*width_*n_)+(x*n_)+(den_wps->at<uchar>(y,x)/k_)] + uw[(y*width_*2)+(x*2)];
				out[(y*width_*3)+(x*3)+1] = dir[(y*width_*2)+(x*2)+1] * disp[(y*width_*n_)+(x*n_)+(den_wps->at<uchar>(y,x)/k_)] + uw[(y*width_*2)+(x*2)+1];
				out[(y*width_*3)+(x*3)+2] = 1.;
			}else{
				out[(y*width_*3)+(x*3)] = dir[(y*width_*2)+(x*2)] * disp[(y*width_*n_)+(x*n_)+n_-1] + uw[(y*width_*2)+(x*2)];
				out[(y*width_*3)+(x*3)+1] = dir[(y*width_*2)+(x*2)+1] * disp[(y*width_*n_)+(x*n_)+n_-1] + uw[(y*width_*2)+(x*2)+1];
				out[(y*width_*3)+(x*3)+2] = 0.;
			}
		}
}

void SGMflow::init(cv::Mat img, int n, int subn, float vmax, int k){
	n_ = n;
	subn_ = subn;
	vmax_ = vmax;
	width_ = (int)img.cols;
	height_ = (int)img.rows;
	censusWindowRadius_ = 2;
	k_ = k;
}
