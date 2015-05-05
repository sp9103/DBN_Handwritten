#include "common.h"

class preProcessor
{
public:
	preProcessor(void);
	~preProcessor(void);

	void ImageToDataMat(IplImage *src, cv::Mat *dst, int row);

private:

};

