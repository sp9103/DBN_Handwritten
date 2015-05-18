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
		uchar val = ((uchar)src->imageData[i] > 123) ? 1.0f : 0.0f;
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
			bin->imageData[i*input->widthStep + j] = (uchar)(val > threshold ? 0 : 255);
		}
	}
}

void preProcessor::Mopology(IplImage *src, IplImage *dst, int iter){
	cvErode(src, dst, 0, iter);
	cvDilate(src, dst, 0, iter);

	cvDilate(src, dst, 0, iter);
	cvErode(src, dst, 0, iter);
}

void preProcessor::ResizeNMakeMat(IplImage *src, cv::Mat *dst){
	IplImage *reInput, *tsrc;
	reInput = cvCreateImage(cvSize(28,28), IPL_DEPTH_8U, 1);

	//resize
	int tw, th;
	tw = src->width;
	th = src->height;

	//추후 수정
	int tsize;
	tsize = (tw > th ? tw : th) + 2;

	tsrc = cvCreateImage(cvSize(tsize, tsize), IPL_DEPTH_8U, 1);
	cvZero(tsrc);

	CvRect tROI;
	tROI.width = src->width;
	tROI.height = src->height;
	tROI.x = (tsrc->width/2) - (src->width/2);
	tROI.y = (tsrc->height/2) - (src->height/2);

	cvSetImageROI(tsrc, tROI);
	cvCopy(src, tsrc);
	cvResetImageROI(tsrc);

	cvResize(tsrc, reInput);
	cvShowImage("testset", reInput);
	cvWaitKey(0);

	//IplImage to Mat
	ImageToDataMat(reInput, dst, 0);

	cvReleaseImage(&reInput);
	cvReleaseImage(&tsrc);
}