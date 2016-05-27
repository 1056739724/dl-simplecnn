
#include"otheroperator.h"


double getLearningRate(Mat &data)//784×60000
{
    // see Yann LeCun's Efficient BackProp, 5.1 A Little Theory
    int nfeatures = data.rows;//784
    int nsamples = data.cols;//60000
    //covariance:协方差
    //covariance matrix = x * x' ./ nfeatures;
    Mat Sigma = data * data.t() / nsamples;//784×60000*60000×784
    //SVD:奇异值分解
    SVD uwvT = SVD(Sigma);
    return 0.9 / uwvT.w.at<double>(0, 0);
}

//返回经过卷积核之后的特征
Mat conv2(Mat &img,//原本某一张图片
		Mat &kernel,//卷积核的卷积矩阵旋转180度
		int convtype) //2
{
	Mat dest;//目的矩阵
	Mat source=img;//原矩阵
   if(CONV_FULL==convtype)////CONV_FULL:0   convtype：2
   {
	   source = Mat();
	   int additionalRows=kernel.rows-1;
	   int additionalCols=kernel.cols-1;
	   //copyMakeBorder 填充边界像素 copyMakeBorder( src, dst, top, bottom, left, right, borderType, value );
	   //接受参数:img: 原图像  source: 目标图像
	  // top, bottom, left, right: 各边界的宽度，此处定义近为原图像尺寸的一半。
	   //borderType: 边界类型，此处是常数边界
	  // value: 如果 borderType 类型是 BORDER_CONSTANT, 该值用来填充边界像素。
	   copyMakeBorder(img, source, (additionalRows+1)/2, additionalRows/2, (additionalCols+1)/2, additionalCols/2, BORDER_CONSTANT, Scalar(0));
   }
   //kernel.cols:13   kernel.rows:13   定义一个点，点坐标6,6,下标从0开始的
   Point anchor(kernel.cols-kernel.cols/2-1,kernel.rows-kernel.rows/2-1);
   int borderMode = BORDER_CONSTANT;//0  边界类型
   Mat fkernal;//fkernal就是kernel转了180度
   flip(kernel,fkernal,-1);//第三个参数: 整数，水平发转；0垂直反转；负数，水平垂直均反转。
   //利用filter2D函数进行图像卷积（原图，目标图，图像深度，核函数，模板中心点，delta，边界类型）
   filter2D(source, dest, img.depth(), fkernal, anchor, 0, borderMode);
   //dest: 28*28
   //不取上6，下6，左6右6，故得到矩阵dest：16*16
   if(CONV_VALID==convtype)
   {
	   //Mat.rowRange（int x，int y）和Mat.rowRange(range(int x,int y)得到的结果一样，
	   //函数取的实际行数y-x，只取到范围的右边界，而不取左边界，Mat.colRange(int x,int y)有类似的用法。
	   //dest.colRange(6,22).rowRange(6,22)   取从第6行到第21行，第6列到21列
	   dest = dest.colRange((kernel.cols-1)/2, dest.cols - kernel.cols/2).rowRange((kernel.rows-1)/2, dest.rows - kernel.rows/2);
   }
   return dest;
}

//找到越接近于给定值的点，返回坐标
Point findLoc(double val, Mat &M)
{
    Point res = Point(0, 0);
    double minDiff = 1e8;
    for(int i=0; i<M.rows; i++)
    {
        for(int j=0; j<M.cols; j++)
        {
            if(val >= M.at<double>(i, j) && (val - M.at<double>(i, j) < minDiff))
            {
                minDiff = val - M.at<double>(i, j);
                res.x = j;
                res.y = i;
            }
        }
    }
    return res;
}


//磁化
Mat Pooling(Mat &M, //卷积之后经过线性修正后的矩阵16*16
		int pVert, //4
		int pHori, //4
		int poolingMethod,//POOL_STOCHASTIC 2
		vector<Point> &locat,
		bool isTest)//false
{
	 int remX = M.cols % pHori;//16%4=0
	 int remY = M.rows % pVert;//16%4=0
	 Mat newM;
	 if(remX == 0 && remY == 0)
	 {
	    M.copyTo(newM);//不用剔除，直接拷贝
	 }
	 else
	 {
	    //创建一个矩形对象，通过使用四个整数来初始化矩形左上角的横坐标、纵坐标以及矩形的宽度、高度（不要弄反）
	    //假如假如M是15*15，则remX，remY为3    左边上面踢掉之后newM是12*12   这边目的是为了磁化前的矩阵大小能够被4*4整除
	    Rect roi = Rect(remX, remY, M.cols - remX, M.rows - remY);//左边上边都剔除了
	    M(roi).copyTo(newM);
	 }
	 Mat res = Mat::zeros(newM.rows / pVert, newM.cols / pHori, CV_64FC1);//磁化时不重叠  4*4
	 for(int i=0; i<res.rows; i++)//4
	 {
	     for(int j=0; j<res.cols; j++)//4
	     {
	    	 Mat temp;
	    	 Rect roi = Rect(j * pHori, i * pVert, pHori, pVert);
	    	 newM(roi).copyTo(temp);
	    	 double val;
	    	 // for Max Pooling
	    	 if(POOL_MAX == poolingMethod)//poolingMethod=2   磁化方法是找指定范围中最大值
	    	 {
	    		 //把最大值最小值以及相应的位置保存下来
	    		 double minVal;
	    		 double maxVal;
	    		 Point minLoc;
	    		 Point maxLoc;
	    		 //temp: 匹配矩阵  &minVal 和 &maxVal: 在矩阵result中的最小值和最大值
	    		 //&minLoc 和 &maxLoc: 在结果矩阵中最小值和最大值的坐标.
	    		 minMaxLoc( temp, &minVal, &maxVal, &minLoc, &maxLoc );//库函数
	    		 val = maxVal;//保存最大值
	    		 locat.push_back(Point(maxLoc.x + j * pHori, maxLoc.y + i * pVert));//保存位置
	    	 }
	    	 else if(POOL_MEAN == poolingMethod)   //磁化方法是找指定范围内的平均值
	    	 {
	    		 // Mean Pooling
	    		 val = sum(temp)[0] / (pVert * pHori);
	    	 }
	    	 else if(POOL_STOCHASTIC == poolingMethod)//磁化方法是 以概率的方式
	    	 {
	    		 // Stochastic(随机) Pooling
	    		 double sumval = sum(temp)[0];
	    		 Mat prob = temp / sumval;//概率
	    		 if(isTest)
	    		 {
	    		     val = sum(prob.mul(temp))[0];//求加权平均值
	    		 }
	    		 else
	    		 {
	    			 RNG rng;//使用随机数发生器类(RNG) 并得到均匀分布的随机数
	    			 //这指定了一个在0 和1之间的均匀分布(包含0,但不含1)
	    			 double ran = rng.uniform((double)0, (double)1);//0到1之间均匀分布采样  取一个值
	    			 double minVal=0.0;//最小概率，因为prob是概率矩阵
	    			 double maxVal=0.0;//最大概率
	    			 Point minLoc; //最小值点
	    			 Point maxLoc;//最大值点
	    			 minMaxLoc(prob, &minVal, &maxVal, &minLoc, &maxLoc );
	    			 ran*= maxVal;//乘上最大概率
	    			 Point loc = findLoc(ran, prob);
	    			 val = temp.at<double>(loc.y, loc.x);//temp 4*4
	    			 locat.push_back(Point(loc.x + j * pHori, loc.y + i * pVert));
	    		 }
	    	 }
	    	 res.at<double>(i, j) = val;
	     }
	 }
	 return res;
}


//把每个样本产生的残差分离出来
void unconcatenateMat(Mat &M,//隐藏层第一层的残差
		vector<vector<Mat> > &vec,
		int vsize)//8
{
    int sqDim = M.rows / vsize;//128/8=16
    int Dim = sqrt(sqDim);//4
    for(int i=0; i<M.cols; i++)//600
    {
        vector<Mat> oneColumn;
        for(int j=0; j<vsize; j++)//8
        {
            Rect roi = Rect(i, j * sqDim, 1, sqDim);//roi.height 16   roi.width 1
            Mat temp;
            M(roi).copyTo(temp);
            Mat img = temp.reshape(0, Dim);
            //img.rows<<endl 4
            //img.cols<<endl 4
            oneColumn.push_back(img);
        }
        vec.push_back(oneColumn);
    }
}

//a：磁化后每个核的残差   b:4*4矩阵 元素全是1
Mat kron(Mat &a, Mat &b)
{
    Mat res = Mat::zeros(a.rows * b.rows, a.cols * b.cols, CV_64FC1);//16*16
    for(int i=0; i<a.rows; i++)//4
    {
        for(int j=0; j<a.cols; j++)//4
        {
            Rect roi = Rect(j * b.cols, i * b.rows, b.cols, b.rows);
            Mat temp = res(roi);
            Mat c = b.mul(a.at<double>(i, j));
            c.copyTo(temp);
        }
    }
    return res;
}


Mat UnPooling(Mat &M,//一张图经过一个卷积核后再磁化，得到的残差  4*4
		int pVert,//4
		int pHori,//4
		int poolingMethod,//2
		vector<Point> &locat)//存放磁化后的点在原卷积后的矩阵中的位置
{
    Mat res;
    if(POOL_MEAN == poolingMethod)//平均值方法磁化
    {
        Mat one = Mat::ones(pVert, pHori, CV_64FC1);
        res = kron(M, one) / (pVert * pHori);
    }
    else if(POOL_MAX == poolingMethod || POOL_STOCHASTIC == poolingMethod)//执行这个分支
    {
        res = Mat::zeros(M.rows * pVert, M.cols * pHori, CV_64FC1);//卷积后的16*16矩阵
        for(int i=0; i<M.rows; i++)//4
        {
            for(int j=0; j<M.cols; j++)//4
            {
            	//原本取值的位置赋值原来的值，其他位置都是0
                res.at<double>(locat[i * M.cols + j].y, locat[i * M.cols + j].x) = M.at<double>(i, j);
            }
        }
    }
    return res;
}


void getNetworkCost(vector<Mat> &x,//600个随机样本
		Mat &y,//600个样本对应的标签
		ConvLayer &cvl,//卷积层
		vector<Networklayer> &hLayers,//隐藏层
		int& NumHiddenLayers,
		Softmaxlayer& smr,//softmax层
		double& lambda,//权重衰减的那个lambda
		int& nclasses)
{
	//求每一张图片通过卷积层8个核，通过磁化层后的结果
	int nsamples=x.size();//600
	// Conv & Pooling
	vector<vector<Mat> > Conv1st;
	vector<vector<Mat> > Pool1st;
	vector<vector<vector<Point> > > PoolLoc;//记录磁化后矩阵元素在原（卷积后）矩阵中的位置
    for(int k=0;k<nsamples;k++)//600
    {
    	 vector<Mat> tpConv1st;//卷积后的特征
    	 vector<Mat> tpPool1st;//磁化后的特征
    	 vector<vector<Point> > PLperSample;//存放磁化时取得值的位置
    	 for(int i=0; i<cvl.kernelAmount; i++)//卷积核数量8
    	 {
    		 vector<Point> PLperKernel;
    		 //cvl.layer[i].W:  13*13
    		 Mat temp = rot90(cvl.layer[i].W, 2);//temp:13*13 //temp是cvl.layer[i].W转了180度
    		 //原本图片28*28,现在卷积之后图片     CONV_VALID：2
    		 Mat tmpconv = conv2(x[k], temp, CONV_VALID);//tmpconv.rows 16*16
    		 tmpconv += cvl.layer[i].b;//加上偏置
    		 tmpconv = ReLU(tmpconv);//线性修正激活函数
    		 tpConv1st.push_back(tmpconv);
    		 // 磁化 tmpconv:卷积后的特征
    		 int PoolingDim=4;
    		 int Pooling_Methed=POOL_STOCHASTIC;
    		 tmpconv = Pooling(tmpconv, PoolingDim, PoolingDim, Pooling_Methed, PLperKernel, false);
    		 // tmpconv:4*4

    		 PLperSample.push_back(PLperKernel);//PLperSample:8项
    		 tpPool1st.push_back(tmpconv);
    	 }
    	 //vector中有600项
    	 PoolLoc.push_back(PLperSample);
    	 Conv1st.push_back(tpConv1st);
    	 Pool1st.push_back(tpPool1st);
    }
    //convolvedX:128*600 把卷积后得到的16*16矩阵，磁化后得到4*4矩阵，经过8个卷积核后，一张图片得到128个特征，共600张图片
    Mat convolvedX = concatenatemat(Pool1st);
    // full connected layers，隐藏层
    vector<Mat> acti;//隐藏层的输入
    acti.push_back(convolvedX);
    for(int i=0;i<NumHiddenLayers;i++)//2
    {
    	//* 是矩阵相乘 mul是对应元素相乘  hLayers[i].b:  200*1  向x方向重复
    	//repeat(hLayers[i - 1].b, 1, convolvedX.cols)得到200*600的矩阵   矩阵拷贝的时候指定按x/y方向重复
    	Mat tmpacti =hLayers[i].W*acti[i]+repeat(hLayers[i].b, 1, convolvedX.cols);//convolvedX.cols:600
    	acti.push_back(Sigmoid(tmpacti));
    }
    //acti[0]存放卷积磁化后拼接好的矩阵   acti[1]中存放第一层隐藏层的输出    acti[2]中存放第二层隐藏层的输出
    //M:10*600,每一行对应属于某个类别的概率 10*200 *200*600         smr.b:10*1
    Mat M = smr.Weight * acti[acti.size() - 1] + repeat(smr.b, 1, nsamples);//softmax层
    //Softmax 回归有一个不寻常的特点：它有一个“冗余”的参数集。为了便于阐述这一特点，假设我们从参数向量
    //M中减去了向量tmp，这时，每一个M都变成了 M - tmp。完全不影响假设函数的预测结果！
    //这表明前面的 softmax 回归模型中存在冗余的参数。更正式一点来说， Softmax 模型被过度参数化了。
    Mat tmp;
    reduce(M, tmp, 0, CV_REDUCE_MAX);//得到M矩阵每列的最大值，并存于tmp矩阵中, tmp: 1*600
    M -= repeat(tmp, M.rows, 1);//y方向重复
    Mat p;
    exp(M, p);//以e为底计算每一个矩阵元素的e的次密，结果存于p中
    reduce(p, tmp, 0, CV_REDUCE_SUM);//得到M矩阵每列之和，存于tmp中
    divide(p, repeat(tmp, p.rows, 1), p);// p/tmp结果存在p中，这就是求概率
    // softmax regression   groundTruth：真实label
    Mat groundTruth = Mat::zeros(nclasses, nsamples, CV_64FC1);
    for(int i=0; i<nsamples; i++)
    {
    	groundTruth.at<double>(y.at<double>(0,i), i) = 1.0;
    }
    //求网络代价
    Mat mat;
    log(p,mat);
    mat=mat.mul(groundTruth);//每个元素对应运算
    smr.cost=-sum(mat)[0]/nsamples;
    //再加上权重衰减项
    pow(smr.Weight, 2.0, tmp);
    smr.cost+=sum(tmp)[0]*lambda/2;
    for(int i=0;i<hLayers.size();i++)
    {
    	pow(hLayers[i].W,2.0,tmp);
    	smr.cost+=sum(tmp)[0]*lambda/2;
    }
    for(int j=0;j<cvl.kernelAmount;j++)
    {
    	pow(cvl.layer[j].W,2.0,tmp);
    	smr.cost+=sum(tmp)[0]*lambda/2;
    }
 //   cout<<"smr.cost: "<<smr.cost<<endl;
    //上面的代价我理解是给我看得，实际上计算代价函数的偏导数，就用softmax回归上面的一个公式
    // bp - softmax
    smr.Wgrad= (groundTruth-p)*acti[acti.size()-1].t()/(-nsamples)+lambda*smr.Weight;
    // Reduce简化一个矩阵成为一个向量
    //cvReduce( const CvArr* src, CvArr* dst, int dim, int op=CV_REDUCE_SUM);
    //src 输入矩阵   dst 输出的通过处理输入矩阵的所有行/列而得到的单行/列向量
    //dim 矩阵被简化后的维数索引.0意味着矩阵被处理成一行,1意味着矩阵被处理成为一列,-1时维数将根据输出向量的大小自动选择.
    //op 简化操作的方式,可以有以下几种取值: CV_REDUCE_SUM-输出是矩阵的所有行/列的和.
    //CV_REDUCE_AVG-输出是矩阵的所有行/列的平均向量.   CV_REDUCE_MAX-输出是矩阵的所有行/列的最大值.
    //CV_REDUCE_MIN-输出是矩阵的所有行/列的最小值.
    reduce((groundTruth - p), tmp, 1, CV_REDUCE_SUM);//处理成一列，tmp：10*1
    smr.bgrad = tmp / -nsamples;//softmax层偏置项的偏导数

    // bp - full connected
    //acti[0]存放卷积磁化后拼接好的矩阵   acti[1]中存放第一层隐藏层的输出    acti[2]中存放第二层隐藏层的输出,输入softmax层
    //delta[2]、delta[1]、delta[0]是存放从后向前每一层传过来的残差
    vector<Mat> delta(acti.size());//acti.size()：3
    //计算softmax层残差存于delta[2]中,这边通常都乘以softmax的参数smr.Weight.t()
    delta[delta.size() -1] = -smr.Weight.t() * (groundTruth - p);//delta[2]
    //(dsigmoid(acti[acti.size() - 1]))：softmax的输出经过了激活函数
    //乘以上一层的输入经过上一层激活函数的导数
    //delta[delta.size() -1]=delta[delta.size() -1].mul(dSigmoid(acti[acti.size()-1]));
    delta[delta.size() -1]=delta[delta.size() -1].mul(dSigmoid(acti[acti.size()-1]));  //200*600
    //计算隐藏层的残差
    for(int i = delta.size() - 2; i >= 0; i--)//delta.size():3     1 0
    {
        delta[i] = hLayers[i].W.t() * delta[i + 1];//delta[i + 1]：前面一层传过来的残差
        if(i > 0)
        {
        	//delta[i] = delta[i].mul(dSigmoid(acti[i]));
           	delta[i] = delta[i].mul(dSigmoid(acti[i]));//这地方感觉写错了，应该是没有经过激活函数的，带验证
        }
    }

    //计算隐藏层偏导数
    for(int i=NumHiddenLayers - 1; i >=0; i--)//NumHiddenLayers:2    1  0
    {
        //delta.size() 3
        hLayers[i].Wgrad = delta[i + 1] * acti[i].t();//delta[i + 1]:上一层的残差
        hLayers[i].Wgrad /= nsamples;
        reduce(delta[i + 1], tmp, 1, CV_REDUCE_SUM);
        hLayers[i].bgrad = tmp / nsamples;
    }
    //bp - Conv layer
    int PoolingDim=4;
    int Pooling_Methed=POOL_STOCHASTIC;
    Mat one = Mat::ones(PoolingDim, PoolingDim, CV_64FC1);//PoolingDim：4
    vector<vector<Mat> > Delta;//残差
    vector<vector<Mat> > convDelta;
    // delta[0]:128*600
    //把每张图经过每个卷积核再磁化后得到的矩阵的残差分开
    unconcatenateMat(delta[0], Delta, cvl.kernelAmount);//cvl.kernelAmount:8
    //Delta.size() 600
    //Delta[0].size() 8,8个卷积核
    //Delta[0][0].rows  4
    //Delta[0][0].cols  4
    for(int k=0; k<Delta.size(); k++)//600
    {
        vector<Mat> tmp;
        for(int i=0; i<Delta[k].size(); i++)//8
        {
            //Pooling_Methed 2
            Mat upDelta = UnPooling(Delta[k][i], PoolingDim, PoolingDim, Pooling_Methed, PoolLoc[k][i]);
            //upDelta = upDelta.mul(dsigmoid(Conv1st[k][i]));
            //upDelta: 16*16
            //Conv1st：一张图片 经过8个卷积核之后的16*16矩阵的存放位置
            upDelta = upDelta.mul(dReLU(Conv1st[k][i]));//原本卷积用的就是ReLU激活函数
            tmp.push_back(upDelta);
         }
         convDelta.push_back(tmp);
    }
    int KernelSize=13;
    for(int j=0; j<cvl.kernelAmount; j++)//8
    {
    	Mat tpgradW = Mat::zeros(KernelSize, KernelSize, CV_64FC1);//13*13
        double tpgradb = 0.0;
        for(int i=0; i<nsamples; i++)//600
        {
        	Mat temp = rot90(convDelta[i][j], 2);//16*16
            tpgradW += conv2(x[i], temp, CONV_VALID);
            tpgradb += sum(convDelta[i][j])[0];
        }
        cvl.layer[j].Wgrad = tpgradW / nsamples + lambda * cvl.layer[j].W;
        cvl.layer[j].bgrad = tpgradb / nsamples;
     }

     // deconstruct
     for(int i=0; i<Conv1st.size(); i++)//600
     {
         Conv1st[i].clear();
         Pool1st[i].clear();
     }
     Conv1st.clear();
     Pool1st.clear();
     for(int i=0; i<PoolLoc.size(); i++)
     {
    	 for(int j=0; j<PoolLoc[i].size(); j++)
         {
    		 PoolLoc[i][j].clear();
         }
         PoolLoc[i].clear();
      }
      PoolLoc.clear();
      acti.clear();
      delta.clear();
}

//把向量中矩阵放入一个矩阵中
Mat concatenateMat(vector<vector<Mat> > &vec)
{
    //vec是磁化后的矩阵存放位置，vec中有600个vector<Mat>，每个vector<Mat>中有8个Mat
    int subFeatures = vec[0][0].rows * vec[0][0].cols;//4*4=16
    int height = vec[0].size() * subFeatures;//8*16=128
    int width = vec.size();//600
    Mat res = Mat::zeros(height, width, CV_64FC1);//128*600
    for(int i=0; i<vec.size(); i++)//600
    {
        for(int j=0; j<vec[i].size(); j++)//128
        {
            Rect roi = Rect(i, j * subFeatures, 1, subFeatures);
            Mat subView = res(roi);
            Mat ptmat = vec[i][j].reshape(0, subFeatures);
            ptmat.copyTo(subView);
        }
    }
    return res;
}

//测试，预测结果
Mat resultProdict(vector<Mat> &x, //x：10000个samples.每个28×*28的矩阵，测试集
		ConvLayer &cvl, //卷积层
		vector<Networklayer> &hLayers,
		int& NumHiddenLayers,
		Softmaxlayer &smr,
		double lambda,//3e-3
		int& nclasses)
{
	int nsamples = x.size();//10000
	vector<vector<Mat> > Conv1st;//存放卷积之后的结果
    vector<vector<Mat> > Pool1st;//存放磁化之后的结果
	vector<Point> PLperKernel;//这边怎么不是向量中嵌套向量，估计是后面不在使用，那么直接省略怎么样
	for(int k=0; k<nsamples; k++)//10000
	{
		vector<Mat> tpConv1st;
		vector<Mat> tpPool1st;
		int PoolingDim=4;
		int Pooling_Methed=POOL_STOCHASTIC;
		for(int i=0; i<cvl.kernelAmount; i++)//8
		{
			Mat temp = rot90(cvl.layer[i].W, 2);//卷积核矩阵旋转180度
			Mat tmpconv = conv2(x[k], temp, CONV_VALID);
			tmpconv += cvl.layer[i].b;
			tmpconv = ReLU(tmpconv);
			tpConv1st.push_back(tmpconv);
			tmpconv = Pooling(tmpconv, PoolingDim, PoolingDim, Pooling_Methed, PLperKernel, true);
			tpPool1st.push_back(tmpconv);
		}
		Conv1st.push_back(tpConv1st);
		Pool1st.push_back(tpPool1st);
	}

	Mat convolvedX = concatenateMat(Pool1st);//输入到隐藏层的特征
	vector<Mat> acti;
	acti.push_back(convolvedX);
	for(int i=1; i<=NumHiddenLayers; i++)//2，隐藏层2层
	{
		Mat tmpacti = hLayers[i - 1].W * acti[i - 1] + repeat(hLayers[i - 1].b, 1, convolvedX.cols);
		acti.push_back(Sigmoid(tmpacti));
	}
    //经过softmax层
	Mat M = smr.Weight * acti[acti.size() - 1] + repeat(smr.b, 1, nsamples);
	Mat tmp;
    reduce(M, tmp, 0, CV_REDUCE_MAX);//得到1*1000的矩阵
	M -= repeat(tmp, M.rows, 1);
	Mat p;
	exp(M, p);
	reduce(p, tmp, 0, CV_REDUCE_SUM);
	divide(p, repeat(tmp, p.rows, 1), p);
	log(p, tmp);

	Mat result = Mat::ones(1, tmp.cols, CV_64FC1);
	for(int i=0; i<tmp.cols; i++)
	{
		double maxele = tmp.at<double>(0, i);
		int which = 0;//记录最大概率的出现索引，对应某种分类，也就是预测的最可能的结果
		for(int j=1; j<tmp.rows; j++)
		{
			if(tmp.at<double>(j, i) > maxele)
			{
				maxele = tmp.at<double>(j, i);
				which = j;
			}
		}
		result.at<double>(0, i) = which;
	}

	// deconstruct
	    for(int i=0; i<Conv1st.size(); i++)
	    {
	        Conv1st[i].clear();
	        Pool1st[i].clear();
	    }
	    Conv1st.clear();
	    Pool1st.clear();
	    acti.clear();
	    return result;
}
