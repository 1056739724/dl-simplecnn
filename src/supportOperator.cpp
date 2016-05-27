#include"supportOperator.h"

Mat ReLU(Mat& mat)
{
	Mat res(mat);

   for(int i=0;i<res.rows;i++)
   {
	   for(int j=0;j<res.cols;j++)
	   {
		   if(res.at<double>(i,j)<0)
			   res.at<double>(i,j)=0.0;
	   }
   }
   return res;
}

Mat dReLU(Mat& mat)
{
	Mat res = Mat::zeros(mat.rows, mat.cols, CV_64FC1);
	for(int i=0;i<res.rows;i++)
	{
		for(int j=0;j<res.cols;j++)
		{
			if(mat.at<double>(i,j)>0.0)
				res.at<double>(i,j)=1.0;
		 }
	 }

    return res;
}

Mat Sigmoid(Mat& mat)
{
	Mat res;
	exp(-mat, res);
    return 1.0 / (res + 1.0);
}


Mat dSigmoid(Mat& mat)
{
	Mat res = 1.0 - mat;
    res = res.mul(mat);
	return res;

}
