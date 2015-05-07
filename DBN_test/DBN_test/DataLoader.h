#pragma once
#include "common.h"
#include "preProcessor.h"

class DataLoader
{
public:
	DataLoader(void);
	~DataLoader(void);

	void ImageDataLoad(int batchSize, cv::Mat *dataMat);

	void FileOpen(char *fileName);
	void FileClose();
	
	int getDataCount();

private:
	ifstream m_file;
	preProcessor m_process;

	int magic_number;
	int number_of_images;
	int n_rows, n_cols;

	int reverseInt(int i);
	void create_image(IplImage **dst, CvSize size, int channels, unsigned char data[28][28], int imagenumber);
};

