/*
 * trainNet.h
 *
 *  Created on: Apr 25, 2016
 *      Author: dingling
 */

#ifndef TRAINNET_H_
#define TRAINNET_H_

#include<iostream>
#include<vector>
#include"data_structure.h"
#include"otheroperator.h"

void trainNetwork(vector<Mat> &x,
		Mat &y,
		ConvLayer &cvl,
		vector<Networklayer> &HiddenLayers,
		int&,
		Softmaxlayer &smr,
		double lambda,
		int MaxIter,
		double lrate,
		int& batch,
		int& nclasses);


#endif /* TRAINNET_H_ */
