
#include "common.h"
#include "BlobLabeling.h"
#include "preProcessor.h"

class Application
{
public:
	Application(void);
	~Application(void);

	IplImage* getImagePatch(cv::Point2d ClickPos);										//Ŭ�� ��ġ�� �̹����� ������. 28*28 ������� ���̳ʸ���
	void getLabel(IplImage *src, std::vector<cv::Rect> *LabelVec, int threshold);		//�̹����� �־��ָ� Label pos ����

private:
	void ThresholdBin(IplImage *input, IplImage *bin, int threshold);					//binary image�� �ٲ�

	preProcessor m_preProcess;
};

