/*
 * supportOperator.h
 *
 *  Created on: Apr 26, 2016
 *      Author: dingling
 */

#ifndef SUPPORTOPERATOR_H_
#define SUPPORTOPERATOR_H_

#include<cmath>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;

Mat ReLU(Mat& mat);
Mat dReLU(Mat& mat);
Mat Sigmoid(Mat& mat);
Mat dSigmoid(Mat& mat);


#endif /* SUPPORTOPERATOR_H_ */
