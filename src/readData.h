/*
 * readData.h
 *
 *  Created on: Apr 23, 2016
 *      Author: dingling
 */

#ifndef READDATA_H_
#define READDATA_H_

#include<vector>
#include"fstream"
#include<iostream>
//下面这四句加上 Mat才可以用
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
using namespace cv;
using namespace std;
//注意这个string 是在std命名空间中
 void readData(vector<Mat>&,Mat&,string,string);

#endif /* READDATA_H_ */
