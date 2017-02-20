#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <stack>
#include <vector>
#include <string.h>
#include "Segment.hpp"
#include "Boundary.hpp"

using namespace cv;

int DEFAULT_SUPERPIXELS = 1000;
int DEFAULT_LAMBDA_POS = 500; 
int DEFAULT_LAMBDA_DISP = 2000;
int DEFAULT_LAMBDA_SMO = 400;
int DEFAULT_LAMBDA_COM = 400;
int DEFAULT_LAMBDA_BOU = 2000;
const int DEFAULT_LAMBDA_D = 9;
const int DEFAULT_LAMBDA_OCC = 15;
const int DEFAULT_LAMBDA_HINGE = 5;
const int DEFAULT_LAMBDA_PEN = 30;
const int DEFAULT_INNER_LOOP = 10;
const int DEFAULT_OUTER_LOOP = 10;
const double DEFAULT_INLIER_THRESHOLD = 3.0;



void initializeSegmentGrid(int superPixels);
void freeArrays();
void markSegmentBoundaries();
void colorSegments();
void markBoundaryLabels();
void colorBoundaryPixel(int x, int y, BoundaryLabel lbl);

void ETPS();
bool validConnectivity(int x, int y);
void getAllBoundaryPixels(std::stack<int>& boundaryPixels);
bool isBoundaryPixel(int x, int y);
bool isOOB(int x, int y);

int getSegmentID(int x, int y);
void setSegmentID(int x, int y, int ID);
int getBoundaryFlag(int x, int y);
void setBoundaryFlag(int x, int y, int flag);
int getOutlierFlag(int x, int y);
void setOutlierFlag(int x, int y, int flag);
int getPixelPos(int x, int y);
int getBoundaryPos(int seg1, int seg2);

double calcPixelEnergy(int x, int y, int segmentID);
double calcPixelDepthEnergy(int x, int y, Point3d plane, int outlierFlag);
double calcTotalEnergy(int x, int y, int segmentID);
double calcSmoothEnergy(int segmentID); 
double calcSmoothEnergy (Boundary bound);

void initiateSegmentPlanes();
void initiateOutlierFlags();
void calcAndSetOutlierFlag(int x, int y, int segmentID);
int computeNrOfSamples(int draws, int inliers, int totalPoints, int currentNrSamples, double p);
Point3d estimatePlaneParameters(Point3d samplePoints[3], double sampleDisparity[3]);
double estimatePixelDisparity(int segmentID, int x, int y);
double estimatePixelDisparity(Point3d plane, int x, int y);

void doIterativeSmoothing();
void initializeBoundaries();
void checkAndSetBoundary(int firstID, int secondID, double x, double y);
void boundarySmoothing();
void segmentSmoothing();

Segment *segmentArray;
Boundary *boundaryArray;
int boundaryTotal;
int *pixelSegmentIDs;
int *pixelBoundaryFlag;
int *pixelOutlierFlag;

int totalSegments;
int totalSegmentX;
int totalSegmentY;
int imageWidth;
int imageHeight;
int imageSize;
double superPixelPixels;

Mat image;
Mat disparityImage;

int fourNeighborOffsetX[4] = {0, -1, 0, 1};
int fourNeighborOffsetY[4] = {-1, 0, 1, 0}; 
int eightNeighborOffsetX[8] = {-1, -1, 0, 1, 1, 1, 0, -1};
int eightNeighborOffsetY[8] = {0, -1, -1, -1, 0, 1, 1, 1};
BoundaryLabel boundaryLabels[4] = {LO, RO, HI, CO};
int boundaryPenalties[4] = {DEFAULT_LAMBDA_OCC, DEFAULT_LAMBDA_OCC, DEFAULT_LAMBDA_HINGE, 0};

int main(int argc, char** argv )
{
  if ( argc != 3 && argc != 4 && argc != 9 && argc !=10 ) {
    printf("usage: DisplayImage.out <Image_Path> <Disparity_Path>\n");
    printf("usage: DisplayImage.out <Image_Path> <Disparity_Path> <Disparity_Output_Path\n");
    return -1;
  }

  if ( argc == 10 || argc == 9) {
    int offset = 0;
    if (argc == 10) offset = 1;
  
    DEFAULT_SUPERPIXELS = atoi(argv[3+offset]);
    DEFAULT_LAMBDA_POS = atoi(argv[4+offset]);
    DEFAULT_LAMBDA_DISP = atoi(argv[5+offset]);
    DEFAULT_LAMBDA_SMO = atoi(argv[6+offset]);
    DEFAULT_LAMBDA_COM = atoi(argv[7+offset]);
    DEFAULT_LAMBDA_BOU = atoi(argv[8+offset]);
  }

  image = imread( argv[1], 1 );
  disparityImage = imread( argv[2], CV_LOAD_IMAGE_GRAYSCALE );
  
  if ( !image.data ) {
    printf("No image data \n");
    return -1;
  }
  if ( !disparityImage.data ) {
    printf("No disparityImage data \n");
    return -1;
  }

  std::string path = "output_disp/";
  if (argc == 4 || argc == 10) {
    path = argv[3];
  }
  
  //Init segmentation to a regular grid and compute mean position and color.
  initializeSegmentGrid(DEFAULT_SUPERPIXELS);
  //Init assignments by running TPS.
  ETPS();
  //Init segment planes by using RANSAC
  initiateSegmentPlanes();
  
  //Iterative smoothing with ETPS, boundary and segment estimation.
  doIterativeSmoothing();

  //Create output images
  markSegmentBoundaries();
  markBoundaryLabels();

  //Modify output disparity to fit KITTI tests
  std::string path2 = "output/";
  std::string argstring = argv[1];
  std::size_t found = argstring.find_last_of("/\\");
  std::string filename = argstring.substr(found+1);
  std::string fullpath = path+filename;
  std::cout << "Processed: " << filename << "\n";
  system(("mkdir " + path).c_str());
  system(("mkdir " + path2).c_str());
  
  cv::imwrite("output/"+filename, image);
  
  colorSegments();
  std::vector<Mat> rgbChannels(3);
  split(image, rgbChannels);
  rgbChannels[0].convertTo(rgbChannels[0], CV_16U);
  
  Mat kitti_disp = Mat::eye(imageHeight, imageWidth, CV_16U); 
  MatIterator_<ushort> it, end;
  int x, y;
  for( it = kitti_disp.begin<ushort>(), end = kitti_disp.end<ushort>(); it != end; ++it) { 
    x = it.pos().x;
    y = it.pos().y;
    *it = (ushort)((float)image.at<Vec3b>(y,x)[0]/256*65535);
  }
  
  cv::imwrite(fullpath, kitti_disp);
  std::cout << fullpath << " and " << "output/"+filename << " written" << std::endl;
  
  freeArrays();
  return 0;
}

void initializeSegmentGrid(int superPixels){
  //initialize global variables
  Size s = image.size();
  imageWidth = s.width;
  imageHeight = s.height;
  imageSize = imageWidth*imageHeight;

  pixelSegmentIDs = new int[imageSize];
  pixelBoundaryFlag = new int[imageSize];
  pixelOutlierFlag = new int[imageSize];
  
  double gridSize = sqrt(imageSize/superPixels);
  totalSegmentX = ceil(imageWidth/gridSize);
  totalSegmentY = ceil(imageHeight/gridSize);
  totalSegments = totalSegmentX * totalSegmentY;
  superPixelPixels = imageSize/totalSegments;
  segmentArray = new Segment[totalSegments];
  boundaryArray = new Boundary[totalSegments*totalSegments];
  
  for (int i = 0; i < totalSegments; i++) {
    segmentArray[i] = Segment();
  }
  
  // accept only char type matrices and 3 channel images
  CV_Assert(image.depth() == CV_8U);
  CV_Assert(image.channels() == 3);

  //Optimization: iterate with pointers.
  MatIterator_<Vec3b> it, end;
  int x, y, segmentID;
  for( it = image.begin<Vec3b>(), end = image.end<Vec3b>(); it != end; ++it) { //BGR
    x = it.pos().x;
    y = it.pos().y;
	    
    segmentID = floor(x/gridSize) + floor(y/gridSize)*totalSegmentX;
    segmentArray[segmentID].addPixel(*it, x, y);
    setSegmentID(x, y, segmentID);
  }
}

//colors disparity according to plane disparity estimates
void colorSegments() {
  MatIterator_<Vec3b> it, end;
  int x, y;
  for( it = image.begin<Vec3b>(), end = image.end<Vec3b>(); it != end; ++it) { //BGR
    x = it.pos().x;
    y = it.pos().y;

    int currSegmentID = pixelSegmentIDs[getPixelPos(x,y)];
    Segment currSegment = segmentArray[currSegmentID];
    Point3d plane = currSegment.getPlaneParams();
    //int posx = currSegment.position(0);
    //int posy = currSegment.position(1);

    double colorb = currSegment.color(0);
    double colorg = currSegment.color(1);
    double colorr = currSegment.color(2);
    
    //std::cout << "posx:" << posx << "posy:" << posy << "\n";
    int depth = estimatePixelDisparity(currSegmentID, x, y);
    if (depth < 0 || depth > 255) {
      depth = 0;
    }
    /*if (x < 10 && y < 10) {
      std::cout << depth << " " << x << " " <<  y << "\n";
      }*/
    
    (*it)[0] = depth;
    (*it)[1] = depth;
    (*it)[2] = depth;
  }
}

//for visualizing the boundaries
void markSegmentBoundaries(){
  MatIterator_<Vec3b> it, end;
  int x, y;
  for( it = image.begin<Vec3b>(), end = image.end<Vec3b>(); it != end; ++it) {
    //BGR
    x = it.pos().x;
    y = it.pos().y;
	    
    int currSegment = pixelSegmentIDs[getPixelPos(x,y)];
    int neighborX, neighborY, neighborSegment;    

    if (x < imageWidth - 1) { //check segment to right
      neighborSegment = getSegmentID(x+1,y);
      if (neighborSegment != currSegment) {
	(*it)[0] = 0;
	(*it)[1] = 0;
	(*it)[2] = 0;
      }
    }
    if (y < imageHeight - 1) { //check segment to down
      neighborSegment = getSegmentID(x,y+1);
      if (neighborSegment != currSegment) {
	(*it)[0] = 0;
	(*it)[1] = 0;
	(*it)[2] = 0;
      }
    }
  }
}

void markBoundaryLabels() {
  MatIterator_<Boundary> it, end;
  Boundary *bound;
  BoundaryLabel lbl;
  int i = 0;
  
  int x,y;
  for (int i = 0; i < totalSegments*totalSegments; i++) {
    bound = &boundaryArray[i];
    std::vector<double> pixelsX = (*bound).getBoundaryPixelsX();
    std::vector<double> pixelsY = (*bound).getBoundaryPixelsY();
    lbl = (*bound).getBoundaryLabel();

    int length = pixelsX.size();
    for (int j = 0; j < length; j++) {
      x = pixelsX.back();
      y = pixelsY.back();
      pixelsY.pop_back();
      pixelsX.pop_back();
      if (lbl == LO) { 
	//seg1 before seg2
	int seg1 = (*bound).getSegment1();
	int seg2 = (*bound).getSegment2();
	int x1 = x-0.5;
	int x2 = x+0.5;
	if (getSegmentID(x1, y) == seg1) {
	  colorBoundaryPixel(x1,y,RO);
	  colorBoundaryPixel(x2,y,LO);
	} else if (getSegmentID(x1, y) == seg2) {
	  colorBoundaryPixel(x2,y,RO);
	  colorBoundaryPixel(x1,y,LO);
	}
	int y1 = y-0.5;
	int y2 = y+0.5;
	if (getSegmentID(x, y1) == seg1) {
	  colorBoundaryPixel(x,y1,RO);
	  colorBoundaryPixel(x,y2,LO);
	} else if (getSegmentID(x, y1) == seg2) {
	  colorBoundaryPixel(x,y2,RO);
	  colorBoundaryPixel(x,y1,LO);
	}
	
      } else if (lbl == RO) {
	//seg2 before seg1
	int seg1 = (*bound).getSegment1();
	int seg2 = (*bound).getSegment2();
	int x1 = x-0.5;
	int x2 = x+0.5;
	if (getSegmentID(x1, y) == seg1) {
	  colorBoundaryPixel(x2,y,RO);
	  colorBoundaryPixel(x1,y,LO);
	} else if (getSegmentID(x1, y) == seg2) {
	  colorBoundaryPixel(x1,y,RO);
	  colorBoundaryPixel(x2,y,LO);
	}

	int y1 = y-0.5;
	int y2 = y+0.5;
	if (getSegmentID(x, y1) == seg1) {
	  colorBoundaryPixel(x,y2,RO);
	  colorBoundaryPixel(x,y1,LO);
	} else if (getSegmentID(x, y1) == seg2) {
	  colorBoundaryPixel(x,y1,RO);
	  colorBoundaryPixel(x,y2,LO);
	}
	
      } else {
	colorBoundaryPixel(x,y,lbl);
      }
    }
  }
}

void colorBoundaryPixel(int x, int y, BoundaryLabel lbl) {
  Vec3b *pixel = &image.at<Vec3b>(y,x);
  switch (lbl) {
    case LO: {
      (*pixel)[0] = 255;
      (*pixel)[1] = 0;
      (*pixel)[2] = 0;
      break;
    }
    case RO: {
      (*pixel)[0] = 0;
      (*pixel)[1] = 0;
      (*pixel)[2] = 255;
      break;
    }
    case HI: {
      (*pixel)[0] = 0;
      (*pixel)[1] = 255;
      (*pixel)[2] = 0;
      break;
    }
    case CO: {
      break;
      (*pixel)[0] = 0;
      (*pixel)[1] = 255;
      (*pixel)[2] = 0;
      break;
    }
  default:
      break;
    }
}

//algorithm 2 & 3, will run as TPS when boundary and plane are not initiated.
void ETPS(){
  std::stack<int> boundaryPixels;
  getAllBoundaryPixels(boundaryPixels);
  while (!boundaryPixels.empty()) {
    int x,y,id = boundaryPixels.top(); 
    boundaryPixels.pop();
    x = id%imageWidth;
    y = id/imageWidth;
    setBoundaryFlag(x, y, 0);
    if (!validConnectivity(x, y)) continue; //speed? look into
    
    //Assign pixel to the boundary segment with best fit.
    double bestEnergy = calcTotalEnergy(x, y, getSegmentID(x, y));
    
    int neighborX, neighborY, neighborSegment, bestSegment, currentSegment;
    currentSegment = getSegmentID(x,y);
    bestSegment = currentSegment;
    
    double newEnergy;
    std::vector<int> checkedIDs;
    checkedIDs.push_back(currentSegment);
    
    for (int i = 0; i < 4; i++) {
      neighborX = x + fourNeighborOffsetX[i];
      neighborY = y + fourNeighborOffsetY[i];
      if (isOOB(neighborX, neighborY)) continue;
      
      neighborSegment = getSegmentID(neighborX, neighborY);

      if (neighborSegment == currentSegment) continue;
      
      //only calculates energy if neccesary.
      if (std::find(checkedIDs.begin(), checkedIDs.end(), neighborSegment) == checkedIDs.end()) {
	cv::Vec3b colors = image.at<Vec3b>(y,x);
	segmentArray[neighborSegment].addPixel(colors, x, y); 
	calcAndSetOutlierFlag(x, y, neighborSegment);
	newEnergy = calcTotalEnergy(x, y, neighborSegment);
	segmentArray[neighborSegment].removePixel(colors, x, y);
	calcAndSetOutlierFlag(x, y, currentSegment);
	
	if (newEnergy < bestEnergy) {
	  bestSegment = neighborSegment;
	}
	checkedIDs.push_back(neighborSegment);
      }
    }

    if (bestSegment != currentSegment) {
      //change segment and update color and pos.
      setSegmentID(x, y, bestSegment);
      cv::Vec3b colors = image.at<Vec3b>(y,x);  
      segmentArray[currentSegment].removePixel(colors, x, y);
      segmentArray[bestSegment].addPixel(colors, x, y); 

      calcAndSetOutlierFlag(x, y, bestSegment);
      
      //push neighbors to stack
      for (int i = 0; i < 4; i++) {
	neighborX = x + fourNeighborOffsetX[i];
	neighborY = y + fourNeighborOffsetY[i];
	if (isOOB(neighborX, neighborY)) continue;

	if (getBoundaryFlag(neighborX, neighborY) == 0) {
	  boundaryPixels.push(getPixelPos(neighborX, neighborY));
	  setBoundaryFlag(neighborX, neighborY, 1);
	}
      }
    }
  }
}

double calcPixelEnergy(int x, int y, int segmentID){
  uchar *p = image.ptr<uchar>(y);

  double colorEnergy = (p[x*3]-segmentArray[segmentID].color(0))*(p[x*3]-segmentArray[segmentID].color(0))+
    (p[x*3+1]-segmentArray[segmentID].color(1))*(p[x*3+1]-segmentArray[segmentID].color(1))+
    (p[x*3+2]-segmentArray[segmentID].color(2))*(p[x*3+2]-segmentArray[segmentID].color(2));
  colorEnergy /= 3;
  double positionEnergy = (x-segmentArray[segmentID].position(0))*(x-segmentArray[segmentID].position(0))+
    (y-segmentArray[segmentID].position(1))*(y-segmentArray[segmentID].position(1));
  //positionEnergy /= 2;
  positionEnergy *= (DEFAULT_LAMBDA_POS/superPixelPixels);
  
  double boundaryEnergy = 0;
  
  for (int i = 0; i < 8; i++) {
    int neighborX = x + eightNeighborOffsetX[i];
    int neighborY = y + eightNeighborOffsetY[i];
    if (isOOB(neighborX, neighborY)) continue;
  
    if (segmentID != getSegmentID(neighborX, neighborY)) {
	boundaryEnergy++;
    }
  } 
  boundaryEnergy = boundaryEnergy*DEFAULT_LAMBDA_BOU;

  //printf("color: %f pos: %f bound: %f\n", colorEnergy, positionEnergy, boundaryEnergy);
  return colorEnergy + positionEnergy + boundaryEnergy;
}

void setSegmentID(int x, int y, int ID){
  pixelSegmentIDs[x + y*imageWidth] = ID;
}

int getSegmentID(int x, int y){
  return pixelSegmentIDs[x + y*imageWidth];
}

void setBoundaryFlag(int x, int y, int flag){
  pixelBoundaryFlag[x + y*imageWidth] = flag;
}

int getBoundaryFlag(int x, int y){
  return pixelBoundaryFlag[x + y*imageWidth];
}

int getOutlierFlag(int x, int y) {
  return pixelOutlierFlag[x + y*imageWidth];
}


void setOutlierFlag(int x, int y, int flag) {
  pixelOutlierFlag[x + y*imageWidth] = flag;
}

int getPixelPos(int x, int y){
  return x + y*imageWidth;
}

int getBoundaryPos(int seg1, int seg2){
  return seg1 + seg2*totalSegments;
}

bool validConnectivity(int x, int y){
  int neighborX, neighborY;
  Mat neighbors = Mat::zeros(3, 3, CV_8U);
  Mat connectedMat = Mat::zeros(3, 3, CV_16U);
  int currSegment = pixelSegmentIDs[getPixelPos(x, y)];
  
  for (int i = 0; i < 8; i++) {
    neighborX = x + eightNeighborOffsetX[i];
    neighborY = y + eightNeighborOffsetY[i];
    if (isOOB(neighborX, neighborY)) {
      continue;
    }
    int neighborSegment = pixelSegmentIDs[getPixelPos(neighborX, neighborY)];
    if (neighborSegment == currSegment) {
      neighbors.at<uchar>(eightNeighborOffsetX[i]+1, eightNeighborOffsetY[i]+1) = 1;
    } else {
      neighbors.at<uchar>(eightNeighborOffsetX[i]+1, eightNeighborOffsetY[i]+1) = 0;
    }
  }
  int conCompNoP = connectedComponents(neighbors, connectedMat, 8, CV_16U);
  neighbors.at<uchar>(1,1) = 1;
  int conComp = connectedComponents(neighbors, connectedMat, 8, CV_16U);
  
  if (conCompNoP > conComp) {
    return false;
  } else {
    return true;
  }
}

//is pixel x,y out of bounds?
bool isOOB(int x, int y){
  if (x >= imageWidth || x < 0 || y < 0 || y >= imageHeight) {
    return true;
  }
  return false;
}

void getAllBoundaryPixels(std::stack<int>& boundaryPixels){
  for (int i = 0; i < imageSize; i++) {
    int x = i%imageWidth;
    int y = i/imageWidth;
    //check if boundary pixels are same segment or not.
    if (isBoundaryPixel(x,y)) {
      setBoundaryFlag(x, y, 1);
      boundaryPixels.push(getPixelPos(x, y));
    } else {
      setBoundaryFlag(x, y, 0);
    }
  }  
}

bool isBoundaryPixel(int x, int y){
  int currSegment = pixelSegmentIDs[getPixelPos(x, y)];
  int neighborX, neighborY, neighborSegment;
	    
  for (int i = 0; i < 4; i++) {
    neighborX = x + fourNeighborOffsetX[i];
    neighborY = y + fourNeighborOffsetY[i];
    if (isOOB(neighborX, neighborY)) {	
      continue;
    }
    neighborSegment = pixelSegmentIDs[getPixelPos(neighborX, neighborY)];
	      
    if (neighborSegment != currSegment) {
      return true;
    }
  }
  return false;
}

//RANSAC
void initiateSegmentPlanes(){
  int x, y, d, segmentID;
  double confidence = 0.99;
  std::vector< std::vector<int> > segmentPixelsX(totalSegments);
  std::vector< std::vector<int> > segmentPixelsY(totalSegments);
  std::vector< std::vector<int> > segmentPixelsD(totalSegments);  
  
  //collect all disparitypixels for each segment. 
  for (int i = 0; i < disparityImage.rows; ++i) {
    for (int j = 0; j < disparityImage.cols; ++j) {
      uchar disparity = disparityImage.at<uchar>(i,j);
      if (disparity == 0) {
	//disparityImage.at<uchar>(i,j) = 255;
	//std::cout << "no disparity pixel" << std::endl;
	continue;
      }
      
      x = j;
      y = i;
      segmentID = getSegmentID(x, y);
      segmentPixelsX[segmentID].push_back(x);
      segmentPixelsY[segmentID].push_back(y);
      segmentPixelsD[segmentID].push_back(disparity);
    }
  }
    
  for (int i = 0; i < totalSegments; i++) {
    //Get three random pixels to base plane on.
    int vectorSize = segmentPixelsX[i].size();
    if (vectorSize < 3) {
      //std::cout << "Segment has less than 3 disparitypixels" << std::endl;
      continue;
    }

    Point3d samplePoints[3];
    double sampleDisparity[3];

    std::vector<int> consensusIndices;
    std::vector<int> bestConsensusIndices;
    int totalInliers = 0;
    int bestTotalInliers = 0;
    Point3d bestPlane;
    double inlierThreshhold = 1.0;
    int totalSamples = vectorSize;

    for (int k = 0; k < totalSamples; k++) {     
      int randIDs[3]; 
      randIDs[0] = rand()%vectorSize;
      while ((randIDs[1] = rand()%vectorSize) == randIDs[0]);
      while ((randIDs[2] = rand()%vectorSize) == randIDs[0] && randIDs[2] == randIDs[1]);
    
      for (int j = 0; j < 3; j++) {
	samplePoints[j].x = (segmentPixelsX[i])[randIDs[j]];
	samplePoints[j].y = (segmentPixelsY[i])[randIDs[j]];
	samplePoints[j].z = 1;
	sampleDisparity[j] = (segmentPixelsD[i])[randIDs[j]];
      }
    
      //Fit a plane model according to the points.
      Point3d planeParams = estimatePlaneParameters(samplePoints, sampleDisparity);

      //Iterate over all pixels and calculate if they're inliers
      for (int j = 0; j < vectorSize; j++) {
	double energy = calcPixelDepthEnergy((segmentPixelsX[i])[j], (segmentPixelsY[i])[j], planeParams, 0);
	if (energy <= inlierThreshhold) {
	  consensusIndices.push_back(j);
	  totalInliers++;
	} 
      }
      if (totalInliers > bestTotalInliers) {
	bestTotalInliers = totalInliers;
	bestConsensusIndices = consensusIndices;
	bestPlane = planeParams;
	totalSamples = computeNrOfSamples(3, bestTotalInliers, vectorSize, totalSamples, confidence); 
      }
      totalInliers = 0;
      consensusIndices.clear();
    }
    //we now have set of best inliers. Solve best solution from these.
    std::vector<int> segPixelsX = segmentPixelsX[i];
    std::vector<int> segPixelsY = segmentPixelsY[i];
    std::vector<int> segPixelsD = segmentPixelsD[i];
    double sumXsqr = 0, sumYsqr = 0, sumXY = 0, sumX = 0,
      sumY = 0, sumXD = 0, sumYD = 0, sumD = 0;
    for (int j = 0; j < bestTotalInliers; j++) {
      x = segPixelsX[bestConsensusIndices[j]];
      y = segPixelsY[bestConsensusIndices[j]];
      d = segPixelsD[bestConsensusIndices[j]];
      sumXsqr += x*x;
      sumYsqr += y*y;
      sumXY += x*y;
      sumX += x;
      sumY += y;
      sumXD += x*d;
      sumYD += y*d;
      sumD += d;
    }
    samplePoints[0] = Point3d(sumXsqr,sumXY,sumX);
    samplePoints[1] = Point3d(sumXY,sumYsqr, sumY);
    samplePoints[2] = Point3d(sumX, sumY, bestTotalInliers);
    sampleDisparity[0] = sumXD;
    sampleDisparity[1] = sumYD;
    sampleDisparity[2] = sumD;
    
    Point3d planeParams = estimatePlaneParameters(samplePoints, sampleDisparity);

    segmentArray[i].setPlaneParams(planeParams);
  }
  initiateOutlierFlags();
}

void initiateOutlierFlags () {
  int x,y,segmentID;
  for (int i = 0; i < disparityImage.rows; ++i) {
    for (int j = 0; j < disparityImage.cols; ++j) {
      uchar disparity = disparityImage.at<uchar>(i,j);
      /*if (disparity == 0) {
	disparityImage.at<uchar>(i,j) = 255;
	//std::cout << "no disparity pixel" << std::endl;
	continue;
	}*/
      x = j;
      y = i;
      segmentID = getSegmentID(x, y);
      calcAndSetOutlierFlag(x, y, segmentID);      
    }
  }
}

void calcAndSetOutlierFlag(int x, int y, int segmentID) {
  double disparity = disparityImage.at<uchar>(y,x);
  if (disparity == 0) {
    //disparityImage.at<uchar>(y,x) = 255;
    //std::cout << "no disparity pixel" << std::endl;
    setOutlierFlag(x, y, 1);
    return;
  }
  double est = estimatePixelDisparity(segmentID, x, y);
  if (abs(est - disparity) >= DEFAULT_INLIER_THRESHOLD) {
    setOutlierFlag(x, y, 1);
  } else {
    setOutlierFlag(x, y, 0);
  }
}

int computeNrOfSamples(int draws, int inliers, int totalPoints, int currentNrSamples, double p) {
  //prob of observing outlier
  double v = 1-(double)inliers/(double)totalPoints;
  if (v == 1.0) {
    v = 0.5;
  }
  int nrOfSamples = log(1-p)/log(1-pow(1-v, draws))+0.5; 
  if (nrOfSamples < currentNrSamples) {
    return nrOfSamples;
  } else {
    return currentNrSamples;
  }
}

double calcPixelDepthEnergy(int x, int y, Point3d plane, int outlierFlag) {
  if (plane.z == -1) {
    return 0;
  }

  if (outlierFlag == 0) {
    double est = estimatePixelDisparity(plane, x, y);
    double truth = disparityImage.at<uchar>(y,x);
    return (truth-est)*(truth-est);
  } else {
    return DEFAULT_LAMBDA_D;
  }
}

//Takes a lot of time!!!
double calcTotalEnergy(int x, int y, int segmentID) {
  double pixelEnergy = calcPixelEnergy(x, y, segmentID); //colorEnergy + positionEnergy + boundaryEnergy
  pixelEnergy += DEFAULT_LAMBDA_DISP*calcPixelDepthEnergy(x, y, segmentArray[segmentID].getPlaneParams(), getOutlierFlag(x, y)); //depth
  return pixelEnergy;
}

double estimatePixelDisparity(int segmentID, int x, int y){
  Point3d plane = segmentArray[segmentID].getPlaneParams();
  return plane.x*x+plane.y*y+plane.z;
}      

double estimatePixelDisparity(Point3d plane, int x, int y){
  return plane.x*x+plane.y*y+plane.z;
}

//Solves Ax = b, A being points and b being disparity. TODO: normalize?
Point3d estimatePlaneParameters(Point3d samplePoints[3], double sampleDisparity[3]){

  Mat A = (Mat_<double>(3,3) <<
	       samplePoints[0].x, samplePoints[0].y, samplePoints[0].z,
	       samplePoints[1].x, samplePoints[1].y, samplePoints[1].z,
	       samplePoints[2].x, samplePoints[2].y, samplePoints[2].z);

  Mat B = (Mat_<double>(3,1) <<
	       sampleDisparity[0],
	       sampleDisparity[1],
	       sampleDisparity[2]);
	       
  Mat x;

  solve(A, B, x, DECOMP_SVD);

  double a = x.at<double>(0);
  double b = x.at<double>(1);
  double c = x.at<double>(2);

  Point3d plane;
  plane.x = a;
  plane.y = b;
  plane.z = c;
  
  return plane;
}

void doIterativeSmoothing() {
  
  for (int i = 0; i < DEFAULT_OUTER_LOOP; i++) {
    std::cout << "ETPS " << i+1 << "\n";
    ETPS();
    initializeBoundaries();
    for (int j = 0; j < DEFAULT_INNER_LOOP; j++) {
      boundarySmoothing();      
      segmentSmoothing();
    }
  }
}

void initializeBoundaries() {
  boundaryTotal = 1;
  for (int i = 0; i < totalSegments*totalSegments; i++) {
    boundaryArray[i].clearBoundary();
  }
  for (int i = 0; i < totalSegments; i++) {
    segmentArray[i].clearSums();
    segmentArray[i].clearBounds();
  }
  
  MatIterator_<Vec3b> it, end;
  int x, y, segmentID, boundSegmentID, boundID, firstID, secondID;
  for( it = image.begin<Vec3b>(), end = image.end<Vec3b>(); it != end; ++it) { 
    x = it.pos().x;
    y = it.pos().y;
	    
    segmentID = getSegmentID(x,y);
    segmentArray[segmentID].addSumPixel(x, y);
    if (!getOutlierFlag(x, y)) {
      segmentArray[segmentID].addSumDPixel(x, y, disparityImage.at<uchar>(y,x));
    }
    
    if (x < imageWidth - 2) { //check segment to right
      boundSegmentID = getSegmentID(x+1,y);
      checkAndSetBoundary(segmentID, boundSegmentID, x+0.5, y);
    }
    if (y < imageHeight - 2) { //check segment below
      boundSegmentID = getSegmentID(x,y+1);
      checkAndSetBoundary(segmentID, boundSegmentID, x, y+0.5);
    }
  }
}

void checkAndSetBoundary(int segmentID, int boundSegmentID, double x, double y) {
  int firstID, secondID;
  Boundary *currBoundary;
  int boundID;

  if (segmentID < boundSegmentID) {
    firstID = segmentID;
    secondID = boundSegmentID;
  } else {
    firstID = boundSegmentID;
    secondID = segmentID;
  }

  currBoundary = &boundaryArray[getBoundaryPos(firstID, secondID)]; 
  boundID = (*currBoundary).getBoundaryIndex();
  if (firstID != secondID) { 
    (*currBoundary).addBoundaryPixel(x, y);
    if ((*currBoundary).getBoundaryIndex() == 0) {
      (*currBoundary).setBoundaryIndex(boundaryTotal);
      (*currBoundary).setSegments(firstID, secondID);
      segmentArray[firstID].addBoundary(currBoundary);
      segmentArray[secondID].addBoundary(currBoundary);
      boundaryTotal++;
    }
  }
}
  
void boundarySmoothing() {
  MatIterator_<Boundary> it, end;
  Boundary *bound;
  BoundaryLabel bestLbl;
  double currEnergy, bestEnergy;
  double energies[4];
  
  for (int j = 0; j < totalSegments*totalSegments; j++) {
    bound = &boundaryArray[j];
    if ((*bound).boundaryIndex != 0) {
      bestEnergy = DBL_MAX;
      bestLbl = NONE;

      for (int i = 0; i < 4; i++) {
	(*bound).setBoundaryLabel(boundaryLabels[i]);
	//currEnergy = DEFAULT_LAMBDA_SMO * calcSmoothEnergy((*bound));
	//currEnergy += DEFAULT_LAMBDA_COM * boundaryPenalties[i];
	currEnergy = calcSmoothEnergy((*bound));
	currEnergy += boundaryPenalties[i];
	
	energies[i] = currEnergy;
	if (currEnergy < bestEnergy) {
	  bestEnergy = currEnergy;
	  bestLbl = boundaryLabels[i];
	}
      }
      (*bound).setBoundaryLabel(bestLbl);
    }
  }  
}

void segmentSmoothing() {
  for (int i = 0; i < totalSegments; i++) {
    Segment currSeg = segmentArray[i];
    double *sumsD = currSeg.getSumsD();
    double *sums = currSeg.getSums();
    Point3d samplePoints[3];
    double sampleDisparity[3];
    double sumXsqr = 0, sumYsqr = 0, sumXY = 0, sumX = 0,
      sumY = 0, sumXD = 0, sumYD = 0, sumD = 0, sumTot = 0;

    sumXsqr += sumsD[0];
    sumYsqr += sumsD[1];
    sumXY += sumsD[2];
    sumX += sumsD[3];
    sumY += sumsD[4];
    sumXD += sumsD[5];
    sumYD += sumsD[6];
    sumD += sumsD[7];
    sumTot += sumsD[8];
    
    std::vector<Boundary*> boundaries = currSeg.segmentBoundaries;
    for (std::vector<Boundary*>::iterator it = boundaries.begin() ; it != boundaries.end(); ++it) {
      Boundary currBound = *(*it);
      BoundaryLabel lbl = currBound.getBoundaryLabel();
      int neighborIndex = currBound.getSegment1();
      if (neighborIndex == i) neighborIndex = currBound.getSegment2();
      Segment neighborSegment = segmentArray[neighborIndex];
      Point3d nPlane = neighborSegment.getPlaneParams();
      
      if (lbl == HI) { //sum over boundary
	double *boundSums = currBound.getSums();
	double weight = (double)DEFAULT_LAMBDA_SMO/(double)DEFAULT_LAMBDA_DISP/(double)boundSums[5]*superPixelPixels;
	sumXsqr += weight*boundSums[0];
	sumYsqr += weight*boundSums[1];
	sumXY += weight*boundSums[2];
	sumX += weight*boundSums[3];
	sumY += weight*boundSums[4];

	sumXD += weight*(nPlane.x*boundSums[0]+nPlane.y*boundSums[2]+nPlane.z*boundSums[3]);
	sumYD += weight*(nPlane.x*boundSums[2]+nPlane.y*boundSums[1]+nPlane.z*boundSums[4]);
	sumD += weight*(nPlane.x*boundSums[3]+nPlane.y*boundSums[4]+nPlane.z*boundSums[5]);
	sumTot += weight*boundSums[5];
	
      } else if (lbl == CO) {
	double weight = (double)DEFAULT_LAMBDA_SMO/(double)DEFAULT_LAMBDA_DISP/(double)(neighborSegment.pixelTotal+currSeg.pixelTotal)*superPixelPixels;
	sumXsqr += weight*sums[0];
	sumYsqr += weight*sums[1];
	sumXY += weight*sums[2];
	sumX += weight*sums[3];
	sumY += weight*sums[4];

	sumXD += weight*(nPlane.x*sums[0]+nPlane.y*sums[2]+nPlane.z*sums[3]);
	sumYD += weight*(nPlane.x*sums[2]+nPlane.y*sums[1]+nPlane.z*sums[4]);
	sumD += weight*(nPlane.x*sums[3]+nPlane.y*sums[4]+nPlane.z*sums[5]);
	sumTot += weight*sums[5];
      } 
    }
    
    samplePoints[0] = Point3d(sumXsqr,sumXY,sumX);
    samplePoints[1] = Point3d(sumXY,sumYsqr, sumY);
    samplePoints[2] = Point3d(sumX, sumY, sumTot);
    sampleDisparity[0] = sumXD;
    sampleDisparity[1] = sumYD;
    sampleDisparity[2] = sumD;
    
    Point3d planeParams = estimatePlaneParameters(samplePoints, sampleDisparity);
    segmentArray[i].setPlaneParams(planeParams);    
  }
}

double calcSmoothEnergy (Boundary bound){
  //get all pixels in the boundary, segments, their planes. 
  int segID1, segID2;
  double energy, xSum, ySum;
  Point3d plane1, plane2;
  segID1 = bound.getSegment1();
  segID2 = bound.getSegment2();
  std::vector<double> pixelsX = bound.getBoundaryPixelsX();
  std::vector<double> pixelsY = bound.getBoundaryPixelsY();
  BoundaryLabel lbl = bound.getBoundaryLabel();
  plane1 = segmentArray[segID1].getPlaneParams();
  plane2 = segmentArray[segID2].getPlaneParams();
  double *sums = bound.getSums();
  double *seg1Sums = segmentArray[segID1].getSums();
  double *seg2Sums = segmentArray[segID2].getSums();
  
  //calculate energy based on label
  switch (lbl) {
  case LO: {
    energy = (plane1.x-plane2.x)*sums[3]+(plane1.y-plane2.y)*sums[4]+(plane1.z-plane2.z)*sums[5];
    if (energy < 0) {
      energy = DEFAULT_LAMBDA_PEN;
    } else {
      energy = 0;
    }
    
    break;
  }
  case RO: {
    energy = (plane1.x-plane2.x)*sums[3]+(plane1.y-plane2.y)*sums[4]+(plane1.z-plane2.z)*sums[5];
    if (energy < 0) {
      energy = 0;
    } else {
      energy = DEFAULT_LAMBDA_PEN;
    }
    break;
  }

  case HI: {
    energy = 0;
    double A = plane1.x-plane2.x;
    double B = plane1.y-plane2.y;
    double C = plane1.z-plane2.z;
    energy = A*A*sums[0]+B*B*sums[1]+2*A*B*sums[2]+2*A*C*sums[3]+2*B*C*sums[4]+C*C*sums[5];
    energy /= sums[5];
    break;
  }

  case CO: {
    energy = 0;
    double A = plane1.x-plane2.x;
    double B = plane1.y-plane2.y;
    double C = plane1.z-plane2.z;
    energy = A*A*(seg1Sums[0]+seg2Sums[0])+B*B*(seg1Sums[1]+seg2Sums[1])+2*A*B*(seg1Sums[2]+seg2Sums[2])+2*A*C*(seg1Sums[3]+seg2Sums[3])+2*B*C*(seg1Sums[4]+seg2Sums[4])+C*C*(seg1Sums[5]+seg2Sums[5]);
    energy /= (seg1Sums[5]+seg2Sums[5]);         
    break;
  }
  default:
    energy = 0;
    break;
  }

  return energy;
}

void freeArrays(){
  delete[] segmentArray;
  delete[] pixelSegmentIDs;
  delete[] boundaryArray;

  delete[] pixelBoundaryFlag;
  delete[] pixelOutlierFlag;
}


