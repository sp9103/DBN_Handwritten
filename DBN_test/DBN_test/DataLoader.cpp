#include "DataLoader.h"


DataLoader::DataLoader(void)
{
	magic_number=0; number_of_images=0;
	n_rows=0; n_cols=0;
}


DataLoader::~DataLoader(void)
{
}

int DataLoader::reverseInt(int i) {
	unsigned char c1, c2, c3, c4;
	c1 = i & 255;
	c2 = (i >> 8) & 255;
	c3 = (i >> 16) & 255;
	c4 = (i >> 24) & 255;
	return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void DataLoader::create_image(IplImage **dst, CvSize size, int channels, unsigned char data[28][28], int imagenumber) {
	string imgname; ostringstream imgstrm;string fullpath;
	imgstrm << imagenumber;
	imgname=imgstrm.str();

	/*if(*dst != NULL)
	cvReleaseImage(&(*dst));*/
	*dst=cvCreateImageHeader(size, IPL_DEPTH_8U, channels);
	cvSetData(*dst, data, size.width);
}

void DataLoader::ImageDataLoad(int batchSize, cv::Mat *dataMat){

	if (m_file.is_open())
	{
		CvSize size;
		unsigned char temp=0;

		dataMat->create(batchSize, n_cols*n_rows, CV_32FC1);		//28*28

		//Image Read
		unsigned char arr[28][28];
		IplImage *dataImg = NULL;
		for(int i = 0; i < batchSize; i++){

			for(int r = 0;r<n_rows;++r)
			{
				for(int c = 0;c<n_cols;++c)
				{                 
					m_file.read((char*)&temp,sizeof(temp));
					//temp = ((temp > 123) ? 255 : 0);							//binary image로 변환
					arr[r][c]= temp;
				}           
			}
			size.height=n_rows;
			size.width=n_cols;
			create_image(&dataImg,size,1,arr, i);
			/*cvShowImage("test", dataImg);
			cvWaitKey(0);*/

			//Data matrix 만들기
			m_process.ImageToDataMat(dataImg, dataMat, i);

		}
	}else{
		printf("File Not opened!\n");
	}
}

void DataLoader::FileOpen(char *fileName){
	m_file.open(fileName, ios::binary);

	//File Header read
	m_file.read((char*)&magic_number,sizeof(magic_number)); 
	magic_number= reverseInt(magic_number);

	m_file.read((char*)&number_of_images,sizeof(number_of_images));
	number_of_images= reverseInt(number_of_images);

	m_file.read((char*)&n_rows,sizeof(n_rows));
	n_rows= reverseInt(n_rows);
	m_file.read((char*)&n_cols,sizeof(n_cols));
	n_cols= reverseInt(n_cols);
}

void DataLoader::FileClose(){
	m_file.close();
}

int DataLoader::getDataCount(){
	return number_of_images;
}