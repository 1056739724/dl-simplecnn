#include "matoperator.h"

//把样本连城一个矩阵，一个样本一列
Mat concatenateMat(vector<Mat> &vecmat)
{
   int rows=vecmat[0].rows;
   int cols=vecmat[0].cols;
   Mat mat=Mat::zeros(rows*cols,vecmat.size(),CV_64FC1);
   for(int i=0;i<vecmat.size();i++)//60000
   {
	   Mat temp=vecmat[i];//28*28
	   // reshape(int cn, int rows=0), cn is num of channels.
	   //在无需复制数据的前提下改变2D矩阵的形状和通道数或其中之一。
	   //ptmat: 784*1  将28*28的矩阵转成784*1
	   Mat ptmat=temp.reshape(0, rows * cols);
	   Rect rect=cv::Rect(i,0,ptmat.cols,ptmat.rows);//roi: 784*1
	   Mat subView = mat(rect);
	   //subView: 784*1
	    ptmat.copyTo(subView);
   }
   return mat;
}

//把向量中矩阵放入一个矩阵中
Mat concatenatemat(vector<vector<Mat> > &vec)
{
    //vec是磁化后的矩阵，vec中有600个vector<Mat>，每个vector<Mat>中有8个Mat
    int subFeatures = vec[0][0].rows * vec[0][0].cols;//4*4=16
    int height = vec[0].size() * subFeatures;//8*16=128
    int width = vec.size();//600
    Mat res = Mat::zeros(height, width, CV_64FC1);//128*600
    for(int i=0; i<vec.size(); i++)//600
    {
        for(int j=0; j<vec[i].size(); j++)//8
        {
            Rect roi = Rect(i, j * subFeatures, 1, subFeatures);
            Mat subView = res(roi);
            Mat ptmat = vec[i][j].reshape(0, subFeatures);
            ptmat.copyTo(subView);
        }
    }
    return res;
}

Mat rot90(Mat &M, int k)//k=2
{
    Mat res;
    if(k == 0)
    {
    	return M;
    }
    else if(k == 1)
    {
    	//M.t() 是实现矩阵的转置
    	//每执行一次，逆时针旋转90度
    	////0为上下翻转，>0为左右翻转，<0同时上下左右翻转
        flip(M.t(), res, 0);//第三个参数: 整数，水平发转；0垂直反转；负数，水平垂直均反转。
    }
    else
    {
        flip(rot90(M, k - 1).t(), res, 0);
    }
    return res;
}



