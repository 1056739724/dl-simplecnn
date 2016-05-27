#include"trainNet.h"



void trainNetwork(vector<Mat> &x,//数据集
		Mat &y,//1*60000 label标签
		ConvLayer &cvl,//卷积层
		vector<Networklayer> &HiddenLayers,//隐藏层
		int& NumHiddenLayers,
		Softmaxlayer &smr,//softmax层
		double lambda,//3e-3
		int MaxIter,//200000，训练的迭代次数
		double lrate,//学习率,第一次是0.0236276
        int& batch,//600
        int& nclasses)
{
	int converge = 0;//记录已经跑了多少次
	double lastcost = 0.0;//上一次的代价，初始是0
	cout<<"Network Learning, trained learning rate: "<<lrate<<endl;
	while(converge<MaxIter)//每一层while执行完，converge加上1，MaxIter是训练的迭代次数
	{
		//batch: 600  下面几行相当于随机从60000个样本中随机选取600个样本    x.size()：60000个样本
		int randNum=((long)rand()+(long)rand())%(x.size()-batch);
		vector<Mat> batchx;//存放本次训练的训练集
		for(int i=0;i<batch;i++)
		{
			batchx.push_back(x[randNum+i]);
		}
        //取对应的label
		Rect roi=Rect(randNum,0,batch,y.rows);////roi:高度1，宽度600
		Mat batchy=y(roi);////y：1*60000 原始样本的标签
		//开始训练，batchx：随机选取的600个样本作为本次训练的样本
		getNetworkCost(batchx, batchy, cvl, HiddenLayers,NumHiddenLayers, smr, lambda,nclasses);//lambda：权重衰减项中会用
		cout<<"learning step: "<<converge<<", Cost function value = "<<smr.cost<<endl;
		//训练完成，现在开始更新权重
		if(fabs((smr.cost - lastcost) / smr.cost) <= 1e-7 && converge > 0)
		{
		    break;//while循环结束
		}
		if(smr.cost <= 0)
		{
		    break;
		}
		lastcost = smr.cost;
		smr.Weight -= lrate * smr.Wgrad;//更新softmax层权重
		smr.b -= lrate * smr.bgrad;//更新softmax层偏重
		//HiddenLayers.size(): 2
		for(int i=0; i<HiddenLayers.size(); i++)
		{
		     HiddenLayers[i].W -= lrate * HiddenLayers[i].Wgrad;
		     HiddenLayers[i].b -= lrate * HiddenLayers[i].bgrad;
		}

		for(int i=0; i<cvl.kernelAmount; i++)//8
		{
		     cvl.layer[i].W -= lrate * cvl.layer[i].Wgrad;
		     cvl.layer[i].b -= lrate * cvl.layer[i].bgrad;
		}
		++ converge;
	}
}


