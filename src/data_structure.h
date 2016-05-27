/*
 * data_structure.h
 *
 *  Created on: Apr 24, 2016
 *      Author: dingling
 */

#ifndef DATA_STRUCTURE_H_
#define DATA_STRUCTURE_H_

#include <vector>
#include<iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
using namespace cv;
using namespace std;

//卷积核结构体
struct ConvKernel
{
	//四个参数的意义？？？？
	Mat W;
	double b;
	Mat Wgrad;
	double bgrad;
};


struct ConvLayer
{
	//卷积层中有多个卷积核
    vector<ConvKernel> layer;
    int kernelAmount;//卷积核的数量
    ConvLayer(int num):kernelAmount(num){}
};


struct Networklayer//隐藏层
{

    Mat W;//权重
    Mat b;//偏置
    Mat Wgrad;
    Mat bgrad;
};


struct Softmaxlayer
{
    Mat Weight;//权重
    Mat Wgrad;//权重的偏导数
    Mat b;//偏置
    Mat bgrad;//偏置项的偏导数
    double cost;
};



#endif /* DATA_STRUCTURE_H_ */
