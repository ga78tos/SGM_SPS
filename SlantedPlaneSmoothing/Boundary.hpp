#ifndef __BOUNDARY_H_INCLUDED__
#define __BOUNDARY_H_INCLUDED__

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "Segment.hpp"
#include "Definitions.hpp"

class Boundary {
public:
  Boundary() {
    boundaryIndex = 0;
    label = NONE;
  }

  Boundary(int seg1, int seg2) {
    boundaryIndex = 0;
    segments[0] = seg1;
    segments[1] = seg2;
    label = NONE;
    clearSums();
  }

  int segments[2];
  BoundaryLabel label;
  int boundaryIndex;
  std::vector<double> boundaryPixelsX;
  std::vector<double> boundaryPixelsY;
  //sumXsqr, sumYsqr, sumXY, sumX, sumY, sumTot;
  double sums[6];

  void setSegments(int segmentID1, int segmentID2);
  int getSegment1();
  int getSegment2();
  bool containsSegments(int segmentID);
  void setBoundaryLabel(BoundaryLabel lbl);
  BoundaryLabel getBoundaryLabel();
  void setBoundaryIndex(int index);
  int getBoundaryIndex();
  void addBoundaryPixel(double x, double y);
  std::vector<double> getBoundaryPixelsX();
  std::vector<double> getBoundaryPixelsY();

  void clearSums();
  double *getSums();
  void clearBoundary();
};

void Boundary::setSegments(int segmentID1, int segmentID2) {
  if (segmentID1 < segmentID2) {
    segments[0]=segmentID1;
    segments[1]=segmentID2;
  } else {
    segments[0]=segmentID2;
    segments[1]=segmentID1;
  }
}

int Boundary::getSegment1() {
  return segments[0];
}
int Boundary::getSegment2() {
  return segments[1];
}

bool Boundary::containsSegments(int segmentID){
  if (segmentID == segments[0] || segmentID == segments[1]) {
    return true;
  }
  return false;
}

void Boundary::setBoundaryLabel(BoundaryLabel lbl){
  label = lbl;
}

BoundaryLabel Boundary::getBoundaryLabel(){
  return label;
}

void Boundary::setBoundaryIndex(int index) {
  boundaryIndex = index;
}

int Boundary::getBoundaryIndex() {
  return boundaryIndex;
}

void Boundary::addBoundaryPixel(double x, double y) {
  boundaryPixelsX.push_back(x);
  boundaryPixelsY.push_back(y);
  sums[0] += x*x;
  sums[1] += y*y;
  sums[2] += x*y;
  sums[3] += x;
  sums[4] += y;
  sums[5] += 1;
}

std::vector<double> Boundary::getBoundaryPixelsX() {
  return boundaryPixelsX;
}
std::vector<double> Boundary::getBoundaryPixelsY() {
  return boundaryPixelsY;
}

void Boundary::clearSums() {
  for (int i = 0; i < 6; i++) {
    sums[i] = 0;
  } 
}

double *Boundary::getSums() {
  return sums;
}

void Boundary::clearBoundary() {
  boundaryIndex = 0;
  segments[0] = 0;
  segments[1] = 0;
  label = NONE;
  clearSums();
  boundaryPixelsX.clear();
  boundaryPixelsY.clear();
}

#endif
