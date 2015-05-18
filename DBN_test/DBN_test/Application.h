
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

	IplImage* getImagePatch(cv::Point2d ClickPos);										//클릭 위치의 이미지를 가져옴. 28*28 사이즈로 바이너리로
	void getLabel(IplImage *src, std::vector<cv::Rect> *LabelVec, int threshold);		//이미지를 넣어주면 Label pos 나옴

	//mouse call back
	static void mouseCallback(int event, int x, int y, int flags, void *param);
	void DomouseCallback(int event, int x, int y, int flags);

};

