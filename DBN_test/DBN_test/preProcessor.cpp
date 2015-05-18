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


void preProcessor::ThresholdBin(IplImage *input, IplImage *bin, int threshold){
	//Gray scale
	if(input->nChannels != 1 || input->width != bin->width || input->height != input->height){
		printf("input error!");
		return;
	}

	for(int i = 0; i < input->height; i++){
		for(int j = 0; j < input->width; j++){
			uchar val = (uchar)input->imageData[i*input->widthStep + j];
			bin->imageData[i*input->widthStep + j] = (uchar)(val > threshold ? 255 : 0);
		}
	}
}

void preProcessor::Mopology(IplImage *src, IplImage *dst, int iter){
	cvErode(src, dst, 0, iter);
	cvDilate(src, dst, 0, iter);

	cvDilate(src, dst, 0, iter);
	cvErode(src, dst, 0, iter);
}