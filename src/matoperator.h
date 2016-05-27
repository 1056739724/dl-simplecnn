/*
 * matoperator.h
 *
 *  Created on: Apr 24, 2016
 *      Author: dingling
 */

#ifndef MATOPERATOR_H_
#define MATOPERATOR_H_
#include <vector>
#include<iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
using namespace cv;
using namespace std;


Mat concatenateMat(vector<Mat> &);

Mat rot90(Mat &M, int k);

Mat concatenatemat(vector<vector<Mat> > &vec);

#endif /* MATOPERATOR_H_ */
