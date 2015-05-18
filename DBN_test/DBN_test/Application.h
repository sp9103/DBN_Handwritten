
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

	IplImage* getImagePatch(cv::Point2d ClickPos);										//클릭 위치의 이미지를 가져옴. 28*28 사이즈로 바이너리로
	void getLabel(IplImage *src, std::vector<cv::Rect> *LabelVec, int threshold);		//이미지를 넣어주면 Label pos 나옴

	preProcessor m_preProcess;
	CBlobLabeling m_bloblabeling;
};

