/*
 * weight_init.h
 *
 *  Created on: Apr 24, 2016
 *      Author: dingling
 */

#ifndef WEIGHT_INIT_H_
#define WEIGHT_INIT_H_


#include"data_structure.h"

//int KernelAmount = 8;//卷积层卷积核的数量

void ConvNetInitPrarms(ConvLayer &,int&,int&,
		vector<Networklayer>  &,int&,int&,
		Softmaxlayer &,int&,int&);


#endif /* WEIGHT_INIT_H_ */
