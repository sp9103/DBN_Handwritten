#include "LabelLoader.h"


LabelLoader::LabelLoader(void)
{
}


LabelLoader::~LabelLoader(void)
{
}

int LabelLoader::reverseInt(int i) {
	unsigned char c1, c2, c3, c4;
	c1 = i & 255;
	c2 = (i >> 8) & 255;
	c3 = (i >> 16) & 255;
	c4 = (i >> 24) & 255;
	return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void LabelLoader::FileOpen(char *fileName){
	m_file.open(fileName, ios::binary);

	//File Header read
	m_file.read((char*)&magic_number,sizeof(magic_number)); 
	magic_number= reverseInt(magic_number);

	m_file.read((char*)&number_of_images,sizeof(number_of_images));
	number_of_images= reverseInt(number_of_images);
}

void LabelLoader::FileClose(){
	m_file.close();
}

void LabelLoader::LabelDataLoad(int batchSize, cv::Mat *dataMat){

	if (m_file.is_open())
	{
		unsigned char temp=0;

		dataMat->create(batchSize, 9, CV_32FC1);

		//Label Read
		for(int i = 0; i < batchSize; i++){
			m_file.read((char*)&temp,sizeof(temp));

			for(int j = 0; j < dataMat->cols; j++)
				if(j == (int)temp)
					dataMat->at<float>(i,j) = 1.0f;
				else
					dataMat->at<float>(i,j) = 0.0f;
		}
	}else{
		printf("File Not opened!\n");
	}
}

int LabelLoader::getDataCount(){
	return number_of_images;
}