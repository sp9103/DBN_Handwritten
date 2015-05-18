
#include "common.h"
#include "BlobLabeling.h"
#include "preProcessor.h"
#include "DBN.h"

class Application
{
public:
	Application(void);
	~Application(void);

	void Run();

private:
	IplImage *m_ori;
	IplImage *m_gray;
	IplImage *m_board;
	IplImage *m_process;

	preProcessor m_preProcess;
	CBlobLabeling m_bloblabeling;
	DBN	m_DBN;

	IplImage* getImagePatch(cv::Point2d ClickPos);										//Ŭ�� ��ġ�� �̹����� ������. 28*28 ������� ���̳ʸ���
	void getLabel(IplImage *src, std::vector<cv::Rect> *LabelVec, int threshold);		//�̹����� �־��ָ� Label pos ����

	//mouse call back
	static void mouseCallback(int event, int x, int y, int flags, void *param);
	void DomouseCallback(int event, int x, int y, int flags);

};

