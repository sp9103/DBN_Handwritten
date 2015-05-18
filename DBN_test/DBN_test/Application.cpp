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

	m_ori = cvLoadImage(buf);
	m_gray = cvCreateImage(cvGetSize(m_ori), IPL_DEPTH_8U, 1);
	cvCvtColor(m_ori, m_gray, CV_BGR2GRAY);
	tboard = cvCloneImage(m_ori);
	tprocess = cvCloneImage(m_gray);

	cvNamedWindow("Input Image");					//원본 이미지 로드
	cvNamedWindow("Image processing");				//트랙바 붙이고 노이즈 필터링 된 이미지

	cv::createTrackbar("Thre", "Image processing", &TrackVal, 255);
	cv::setMouseCallback("Input Image", mouseCallback, this);

	while(1){

		if(TrackVal != prevTrackVal){
			cvCopy(m_ori, tboard);

			//전처리
			m_preProcess.ThresholdBin(m_gray, tprocess, TrackVal);
			m_preProcess.Mopology(tprocess, tprocess, 1);

			m_bloblabeling.SetParam(tprocess, 100);
			m_bloblabeling.DoLabeling();
			m_bloblabeling.DrawBlob(tboard, cvScalar(0, 0, 255));

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
	cvReleaseImage(&m_gray);

}

void Application::mouseCallback(int event, int x, int y, int flags, void *param){
	Application *self = static_cast<Application*>(param);
	self->DomouseCallback(event, x, y, flags);
}

void Application::DomouseCallback(int event, int x, int y, int flags){
	CvRect BlobInfo;

	switch(event){
	case CV_EVENT_LBUTTONDOWN:
		m_bloblabeling.GetLabel(cvPoint(x,y), &BlobInfo);
		break;

	}
}