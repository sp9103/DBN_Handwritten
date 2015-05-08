#include "preProcessor.h"


preProcessor::preProcessor(void)
{
}


preProcessor::~preProcessor(void)
{
}

void preProcessor::ImageToDataMat(IplImage *src, cv::Mat *dst, int row){
	int i;
	for(i = 0; i < src->height * src->width; i++){
		dst->at<float>(row, i) = ((uchar)src->imageData[i] > 123) ? 1.0f : 0.0f;
	}
}