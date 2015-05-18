#include "Application.h"


Application::Application(void)
{

}


Application::~Application(void)
{
}

void Application::Run(){
	char buf[256];
	IplImage *tboard, *tprocess;
	int prevTrackVal = -1, TrackVal = 190;

	printf("Insert File Name : ");
	scanf("%s", buf);
	if(strlen(buf) < 2)
		sprintf(buf, "sample1.jpg");

	m_ori = cvLoadImage(buf, CV_LOAD_IMAGE_GRAYSCALE);
	tboard = cvCloneImage(m_ori);
	tprocess = cvCloneImage(m_ori);

	cvNamedWindow("Input Image");					//���� �̹��� �ε�
	cvNamedWindow("Image processing");				//Ʈ���� ���̰� ������ ���͸� �� �̹���

	cv::createTrackbar("Thre", "Image processing", &TrackVal, 255);

	while(1){

		if(TrackVal != prevTrackVal){
			//��ó��
			m_preProcess.ThresholdBin(m_ori, tprocess, TrackVal);
			m_preProcess.Mopology(tprocess, tprocess, 1);

			m_bloblabeling.SetParam(tprocess, 100);
			m_bloblabeling.DoLabeling();

			prevTrackVal = TrackVal;
		}

		cvShowImage("Input Image", tboard);
		cvShowImage("Image processing", tprocess);

		if(cv::waitKey(10) == 27)
			break;
	}

	cvDestroyAllWindows();

	cvReleaseImage(&m_ori);
	cvReleaseImage(&tprocess);
	cvReleaseImage(&tboard);

}