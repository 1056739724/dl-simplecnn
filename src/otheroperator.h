/*
 * otheroperator.h
 *
 *  Created on: Apr 24, 2016
 *      Author: dingling
 */

#ifndef OTHEROPERATOR_H_
#define OTHEROPERATOR_H_

#include<vector>
#include"data_structure.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "matoperator.h"
#include"supportOperator.h"


// Conv2 parameter
#define CONV_FULL 0
#define CONV_SAME 1
#define CONV_VALID 2

// Pooling methods
#define POOL_MAX 0
#define POOL_MEAN 1
#define POOL_STOCHASTIC 2

using namespace cv;

double getLearningRate(Mat &data);

void getNetworkCost(vector<Mat> &x,//600个随机样本
		Mat &y,//600个样本对应的标签
		ConvLayer &cvl,//卷积层
		vector<Networklayer> &hLayers,//隐藏层
		int&,
		Softmaxlayer &smr,//softmax层
		double& lambda,
		int& nclasses);

Mat conv2(Mat &img,//某一个样本
		Mat &kernel,//卷积核的卷积矩阵旋转180度
		int convtype); //2


//磁化
Mat Pooling(Mat &M, //卷积之后经过线性修正后的矩阵16*16
		int pVert, //4
		int pHori, //4
		int poolingMethod,//POOL_STOCHASTIC 2
		vector<Point> &locat,
		bool isTest);//false


Mat resultProdict(vector<Mat> &x, //x：10000个samples.每个28×*28的矩阵，测试集
		ConvLayer &cvl, //卷积层
		vector<Networklayer> &hLayers,
		int& NumHiddenLayers,
		Softmaxlayer &smr,
		double lambda,//3e-3
		int& nclasses);

#endif /* OTHEROPERATOR_H_ */
