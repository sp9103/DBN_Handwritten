
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
	IplImage *m_gray;

	preProcessor m_preProcess;
	CBlobLabeling m_bloblabeling;

	IplImage* getImagePatch(cv::Point2d ClickPos);										//Ŭ�� ��ġ�� �̹����� ������. 28*28 ������� ���̳ʸ���
	void getLabel(IplImage *src, std::vector<cv::Rect> *LabelVec, int threshold);		//�̹����� �־��ָ� Label pos ����

	//mouse call back
	static void mouseCallback(int event, int x, int y, int flags, void *param);
	void DomouseCallback(int event, int x, int y, int flags);

};

