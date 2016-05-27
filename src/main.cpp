#include<iostream>
#include<vector>
#include"readData.h"
#include"data_structure.h"
#include"weight_init.h"
#include"matoperator.h"
#include"otheroperator.h"
#include"trainNet.h"
//下面这四句加上 Mat才可以用
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
using namespace cv;

using namespace std;


int main()
{
	float startTime=clock();
    //读数据集:训练集和测试集
    vector<Mat> trainx;
    vector<Mat> testx;
    Mat trainy;
    Mat testy;
    readData(trainx,trainy,"mnist/train-images.idx3-ubyte", "mnist/train-labels.idx1-ubyte");
    readData(testx,testy,"mnist/t10k-images.idx3-ubyte", "mnist/t10k-labels.idx1-ubyte");

    // trainX.size(): 60000 样本数量训练集，样本特征28*28，trainY:标签数也是60000个，标签1*60000的矩阵，
    //测试集testX：10000个samples.每个28×*28的矩阵   testY：标签1*10000的矩阵，
    cout<<"Read trainX successfully, including "<<trainx[0].cols * trainx[0].rows<<" features and "<<trainx.size()<<" samples."<<endl;
    cout<<"Read trainY successfully, including "<<trainy.cols<<" samples"<<endl;
    cout<<"Read testX successfully, including "<<testx[0].cols * testx[0].rows<<" features and "<<testx.size()<<" samples."<<endl;
    cout<<"Read testY successfully, including "<<testy.cols<<" samples"<<endl;

    int nfeatures = trainx[0].rows * trainx[0].cols;//28*28=784，一张图片的特征
    int imgDim = trainx[0].rows;//28,一张图片的行数高度
    int nsamples = trainx.size();//60000,训练集的数量

    //卷积层，隐藏层，softmax层
    ConvLayer cvlayer(8);//卷积层卷积核数量8
    int KernelSize=13;//卷积核大小
    int PoolingDim=4;//对卷积结果，进行4*4磁化
    vector<Networklayer> HiddenLayers;//隐藏层可能不止一层
    int NumHiddenNeurons=200;//隐藏层神经元数量
    Softmaxlayer smr;
    int nclasses=10;//10个分类
    int NumHiddenLayers=2;
    //初始化每一层的参数

    ConvNetInitPrarms(cvlayer,KernelSize,PoolingDim, HiddenLayers,NumHiddenLayers,NumHiddenNeurons, smr, nclasses,imgDim);

    // Train network using Back Propogation（反向传播）
    int batch = nsamples / 100;//batch:600，每一次训练用600个样本

    //原始每个图片28*28也就是784，每个图片一列，共60000张图片，这些特征整合成一个矩阵,目的仅仅是下面的得到学习速率
    Mat tpX = concatenateMat(trainx);//784×60000
    //得到学习速率
    double lrate = getLearningRate(tpX);

    //训练网络
    trainNetwork(trainx, trainy, cvlayer, HiddenLayers,NumHiddenLayers, smr, 3e-3, 100000, lrate,batch,nclasses);

    // Test use test set
    Mat result = resultProdict(testx, cvlayer, HiddenLayers, NumHiddenLayers,smr, 3e-3,nclasses);//网络训练得到的结果
    Mat err(testy);//标签
    //err.rows 1
    //err.cols 10000
    err -= result;
    int correct = err.cols;//总共就一行
    for(int i=0; i<err.cols; i++)
    {
       if(err.at<double>(0, i) != 0)
    	   --correct;
    }
    cout<<"correct: "<<correct<<", total: "<<err.cols<<", accuracy: "<<double(correct) / (double)(err.cols)<<endl;
    float endTime=clock();
	cout<<"The program Totally used time: "<<(endTime-startTime)/CLOCKS_PER_SEC<<" second"<<endl;
	return 0;
}
