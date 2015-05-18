
#include "common.h"
#include "BlobLabeling.h"
#include "preProcessor.h"

class Application
{
public:
	Application(void);
	~Application(void);

	void Run();

private:
	IplImage *m_ori;

	IplImage* getImagePatch(cv::Point2d ClickPos);										//Ŭ�� ��ġ�� �̹����� ������. 28*28 ������� ���̳ʸ���
	void getLabel(IplImage *src, std::vector<cv::Rect> *LabelVec, int threshold);		//�̹����� �־��ָ� Label pos ����

	preProcessor m_preProcess;
	CBlobLabeling m_bloblabeling;
};

