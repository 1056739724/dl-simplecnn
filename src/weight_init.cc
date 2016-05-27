#include"weight_init.h"

//init conv layer
void  weightRandomInit(ConvKernel& tmpConvK,int& KernelSize)
{
	tmpConvK.W=Mat::ones(KernelSize,KernelSize,CV_64FC1);
	double epsilon = 0.1;
	double *pData;
	for(int i=0;i<KernelSize;i++)
	{
		pData=tmpConvK.W.ptr<double>(i);
		for(int j=0;j<KernelSize;j++)
		{
			pData[j]=randu<double>();
		}
	}
	tmpConvK.W=tmpConvK.W*(2*epsilon)-epsilon;
	tmpConvK.b=0;
	tmpConvK.bgrad=0;
	tmpConvK.Wgrad=Mat::zeros(KernelSize,KernelSize,CV_64FC1);
}

//init hidden layer
void weightRandomInit(Networklayer& hidlayer,int& inputsize, int& hiddensize)
{
	double epsilon = sqrt(6) / sqrt(hiddensize + inputsize + 1);//sqrt开平方
	hidlayer.W=Mat::ones(hiddensize,inputsize,CV_64FC1);
	double *pData;
	for(int i=0;i<hiddensize;i++)//行数
	{
		//Mat最直接的访问方法是通过.ptr<>函数得到一行的指针，并用[]操作符访问某一列的像素值
		 pData = hidlayer.W.ptr<double>(i);
		 for(int j=0; j<inputsize; j++)
		 {
		     pData[j] = randu<double>();
		 }
	}
	hidlayer.W=hidlayer.W*(2*epsilon)-epsilon;
	hidlayer.b=Mat::zeros(hiddensize,1,CV_64FC1);//200*1，那就是每一个神经元一个偏置
	hidlayer.Wgrad=Mat::zeros(hiddensize,inputsize,CV_64FC1);
	hidlayer.bgrad=Mat::zeros(hiddensize,1,CV_64FC1);
}

//init softmax layer
void weightRandomInit(Softmaxlayer& smr,int&  nclasses,int& NumHiddenNeurons)
{
	double epsilon = 0.01;
	smr.Weight=Mat::ones(nclasses,NumHiddenNeurons,CV_64FC1);
	double *pData;
	for(int i=0;i<nclasses;i++)
	{
		pData = smr.Weight.ptr<double>(i);
		for(int j=0;j<NumHiddenNeurons;j++)
		{
		   pData[j] = randu<double>();
		}
	}
	smr.Weight=smr.Weight*(2*epsilon)-epsilon;
	smr.b=Mat::zeros(nclasses,1,CV_64FC1);
	smr.Wgrad=Mat::zeros(nclasses,NumHiddenNeurons,CV_64FC1);
	smr.bgrad=Mat::zeros(nclasses,1,CV_64FC1);
	smr.cost=0.0;
}


//ConvLayer:conv layer  vector<Networklayer>:hidden layer  Softmaxlayer imgDim：图片的长宽  nsamples：训练集样本数
void ConvNetInitPrarms(ConvLayer &cvl,int& KernelSize,int& PoolingDim,
		vector<Networklayer> &HiddenLayers,int& NumHiddenLayers,int& NumHiddenNeurons,
		Softmaxlayer &smr,
		int& nclasses,//10
		int& imgDim)//28
{
	cout<<"KernelSize: "<<KernelSize<<endl;//KernelSize: 13
	cout<<"PoolingDim: "<<PoolingDim<<endl;//PoolingDim: 4
	cout<<"NumHiddenNeurons: "<<NumHiddenNeurons<<endl;//NumHiddenNeurons: 200
	cout<<"imgDim: "<<imgDim<<endl;//imgDim: 28
	cout<<"nclasses: "<<nclasses<<endl;//nclasses: 10
	// Init Conv layers
	for(int i=0;i<cvl.kernelAmount;i++)//Conv kernel number,8
	{
		ConvKernel tmpConvK;
	    weightRandomInit(tmpConvK, KernelSize);//KernelSize:13
		cvl.layer.push_back(tmpConvK);
	 }

	int outDim = imgDim - KernelSize + 1; //卷积后矩阵大小
	outDim = outDim / PoolingDim;//磁化后矩阵大小
	//磁化后原本一张图片产生的特征数量，4*4*8=128
	int hiddenfeatures = pow(outDim, 2) * cvl.kernelAmount;//特征数

	// Init Hidden layers
	Networklayer hidlayer;
	//hiddenfeatures: 128   //NumHiddenNeurons:200 number of hideen layer
	//nsamples: 60000  //hidden weight:200*128  has 200 neurel
	weightRandomInit(hidlayer, hiddenfeatures, NumHiddenNeurons);
	HiddenLayers.push_back(hidlayer);
	//in total 2 hidden layer
	for(int i=1;i<NumHiddenLayers;i++)
	{
		Networklayer tempHiddenlayer;
		weightRandomInit(tempHiddenlayer, NumHiddenNeurons, NumHiddenNeurons);
		HiddenLayers.push_back(tempHiddenlayer);
	}

	//Init Softmax layer
    weightRandomInit(smr, nclasses, NumHiddenNeurons);
}
