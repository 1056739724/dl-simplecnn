#include"readData.h"


using namespace std;

int reverseInt(int number)
{
	unsigned char ch1,ch2,ch3,ch4;
	ch1=number & 255;//取最低位
	ch2=(number>>8) & 255;
	ch3=(number>>16) & 255;
	ch4=(number>>24) & 255;//取最高位
	//真实的数字是
	return ((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}

void read_Mnist(vector<Mat>& x,string xPath)
{
	ifstream file(xPath,ios::binary);
	if(!file.is_open())//文件打开失败
	{
	    cout<<"file open fail!!";
	    return;
	}
	//文件格式，第一个不知道是什么，后面依次是图片数量，行数，列数
	int magic_number=0;
	file.read((char*)&magic_number,sizeof(magic_number));//为什么读出来的整形数要转成char*
	//字节反转，转成要读出来的数
	magic_number=reverseInt(magic_number);
	int pic_number=0;
	file.read((char*)&pic_number,sizeof(pic_number));
	pic_number=reverseInt(pic_number);
	int rows=0;
	file.read((char*)&rows,sizeof(rows));
	rows=reverseInt(rows);
	int cols=0;
	file.read((char*)&cols,sizeof(cols));
	cols=reverseInt(cols);
	//表示了矩阵中元素的类型以及矩阵的通道个数，它是一系列的预定义的常量，
	//其命名规则为CV_(位数）+（数据类型）+（通道数）
	//CV_8UC1,这里U（unsigned integer）表示的是无符号整数
    //例如：CV_8UC1，表示的是元素类型是一个8位的无符号整数，通道为1
	for(int i = 0; i < pic_number; ++i)//60000
    {
	    Mat tempMat=Mat::zeros(rows,cols,CV_8UC1);
		for(int r=0;r<rows;r++)
		{
			for(int c=0;c<cols;c++)
			{
				unsigned char temp=0;
				file.read((char*)&temp,sizeof(temp));
				//由于计算机里数值都是以补码形式存在，所以这个时候转化为int的Buffer值应该为
				//0xFFFFFFF0（将-16以补码形式的表现），然后在做右移两位操作；
				//而如果是unsigned char buffer = 0xF0的话转化为Int还是0xF0，所以不会有影
				tempMat.at<uchar>(r,c)=(int)temp;
			}
	    }
		x.push_back(tempMat);
	 }
}

void read_Mnist_label(Mat& y,string yPath)
{
	ifstream file(yPath,ios::binary);
	if(!file.is_open())
	{
		cout<<"File open fail "<<endl;
		return;
	}
	int magic_number=0;
	int pic_number=0;
	//magic_number: 2049？？？不知道干什么的
   //pic_number: 60000
	file.read((char*)&magic_number,sizeof(magic_number));
	magic_number=reverseInt(magic_number);
	file.read((char*)&pic_number,sizeof(pic_number));
	pic_number=reverseInt(pic_number);
    for(int i=0;i<pic_number;i++)
    {
    	unsigned char temp=0;
    	file.read((char*)&temp,sizeof(temp));
    	y.at<double>(0,i)=(double)temp;
    }
}


 /*trainx:存放训练集图片    trainy：训练集对应的标签
  trainxPath：训练集对应的路径   trainyPath:训练集标签对应的路径    trainNum：训练集数量
 */
 void readData(vector<Mat>& x,Mat& y,string xPath,string yPath)
 {
	read_Mnist(x,xPath);
    for(int i=0;i<x.size();i++)
    {
    	//Mat::convertTo在缩放或不缩放的情况下转换为另一种数据类型。
    	//void Mat::convertTo(OutputArray m,int rtype,double alpha=1,double beta=0)const
    	//参数：m – 目标矩阵。如果它的尺寸和类型不正确，在操作之前会重新分配。
    	//rtype – 要求是目标矩阵的类型，或者在当前通道数与源矩阵通道数相同的情况下的depth。
    	//如果rtype 为负，目标矩阵与源矩阵类型相同。
        //beta – 可选的delta加到缩放值中去。该方法将源像素值转化为目标类型saturate_cast<> 要放在最后以避免溢出
    	///CV_64FC1表示64位浮点型,通道为1
    	x[i].convertTo(x[i], CV_64FC1, 1.0/255, 0);//x[i]矩阵中每个元素都除以255
    }
    y=Mat::zeros(1,x.size(),CV_64FC1);
    read_Mnist_label(y,yPath);
 }

