#pragma once
#include "common.h"

class preProcessor
{
public:
	preProcessor(void);
	~preProcessor(void);

	void ImageToDataMat(IplImage *src, cv::Mat *dst, int row);
	void Mopology(IplImage *src, IplImage *dst, int iter);
	void ThresholdBin(IplImage *input, IplImage *bin, int threshold);					//binary image·Î ¹Ù²Þ
	void ResizeNMakeMat(IplImage *src, cv::Mat *dst);

private:

};

