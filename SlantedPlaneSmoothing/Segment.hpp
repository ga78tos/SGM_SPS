#ifndef __SEGMENT_H_INCLUDED__
#define __SEGMENT_H_INCLUDED__

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "Boundary.hpp"

class Segment {
public:
  Segment() {
    pixelTotal = 0;
    colorSum[0] = 0;
    colorSum[1] = 1;
    colorSum[2] = 2;
    positionSum[0] = 0;
    positionSum[1] = 0;
    disparityPlane.x = 0;
    disparityPlane.y = 0;
    disparityPlane.z = -1;
    clearSums();
    clearBounds();
  }
  
  int pixelTotal;
  double colorSum[3];
  double positionSum[2];
  cv::Point3d disparityPlane;
  //sumXsqr, sumYsqr, sumXY, sumX, sumY, sumTot;
  double sums[6];
  //sumXsqr, sumYsqr, sumXY, sumX, sumY, sumDX, sumDY, sumD, sumTot;
  double sumsD[9];
  std::vector<Boundary*> segmentBoundaries;
  
  void addPixel(cv::Vec3b colors, int x, int y);
  void removePixel(cv::Vec3b colors, int x, int y);
  cv::Point3d getPlaneParams();
  void setPlaneParams(cv::Point3d);
  double color(int id);
  double position(int id);

  void addSumPixel(int x, int y);
  void addSumDPixel(int x, int y, int d);
  double *getSums();
  double *getSumsD();
  void clearSums();
  void addBoundary(Boundary *bound);
  void clearBounds();
};

void Segment::addPixel(cv::Vec3b colors, int x, int y){
  colorSum[0] += colors[0]; 
  colorSum[1] += colors[1];
  colorSum[2] += colors[2];
  
  positionSum[0] += x;
  positionSum[1] += y;
  pixelTotal++;
}

void Segment::removePixel(cv::Vec3b colors, int x, int y){
  colorSum[0] -= colors[0]; 
  colorSum[1] -= colors[1];
  colorSum[2] -= colors[2];
  
  positionSum[0] -= x;
  positionSum[1] -= y;
  pixelTotal--;
}

cv::Point3d Segment::getPlaneParams(){
  return disparityPlane;
}

void Segment::setPlaneParams(cv::Point3d p){
  disparityPlane.x = p.x;
  disparityPlane.y = p.y;
  disparityPlane.z = p.z;
}

double Segment::color(int id){
  return colorSum[id]/pixelTotal;
}

double Segment::position(int id){
  return positionSum[id]/pixelTotal;
}

std::ostream &operator<<(std::ostream &os, Segment const &m) { 
  return os << m.pixelTotal << " " << m.colorSum[0] << " " << m.colorSum[1] << " " << m.colorSum[2] << " " << m.positionSum[0] << " " << m.positionSum[1];
}

void Segment::addSumPixel(int x, int y) {
  sums[0] += x*x;
  sums[1] += y*y;
  sums[2] += x*y;
  sums[3] += x;
  sums[4] += y;
  sums[5] += 1;
}

double *Segment::getSums(){
  return sums;
}

void Segment::clearSums() {
  for (int i = 0; i < 6; i++) {
    sums[i] = 0;
  }
  for (int i = 0; i < 9; i++) {
    sumsD[i] = 0;
  }
}

//sumXsqr, sumYsqr, sumXY, sumX, sumY, sumDX, sumDY, sumD, sumTot;  
void Segment::addSumDPixel(int x, int y, int d) {
  sumsD[0] += x*x;  
  sumsD[1] += y*y;
  sumsD[2] += x*y;
  sumsD[3] += x;
  sumsD[4] += y;
  sumsD[5] += d*x;
  sumsD[6] += d*y;
  sumsD[7] += d;
  sumsD[8] += 1;
} 

double *Segment::getSumsD() {
  return sumsD;
}

void Segment::addBoundary(Boundary *bound) {
  segmentBoundaries.push_back(bound);
}

void Segment::clearBounds() {
  segmentBoundaries.clear();
}

#endif
