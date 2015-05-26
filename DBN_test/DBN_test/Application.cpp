#include "Application.h"


Application::Application(void)
{
	//m_DBN.NetLoad("FullNetworkData.bin");
}


Application::~Application(void)
{
}

void Application::Run(){
	char buf[256];
	int prevTrackVal = -1, TrackVal = 190;

	m_DBN.InitNetwork();
	m_DBN.NetLoad("FullNetworkData.bin");
	printf("Insert File Name : ");
	scanf("%s", buf);
	if(strlen(buf) < 2)
		sprintf(buf, "sample1.jpg");

	m_ori = cvLoadImage(buf);
	m_gray = cvCreateImage(cvGetSize(m_ori), IPL_DEPTH_8U, 1);
	cvCvtColor(m_ori, m_gray, CV_BGR2GRAY);
	m_board = cvCloneImage(m_ori);
	m_process = cvCloneImage(m_gray);

	cvNamedWindow("Input Image");					//원본 이미지 로드
	cvNamedWindow("Image processing");				//트랙바 붙이고 노이즈 필터링 된 이미지

	cv::createTrackbar("Thre", "Image processing", &TrackVal, 255);
	cv::setMouseCallback("Input Image", mouseCallback, this);

	while(1){

		if(TrackVal != prevTrackVal){
			cvCopy(m_ori, m_board);

			//전처리
			m_preProcess.ThresholdBin(m_gray, m_process, TrackVal);
			m_preProcess.Mopology(m_process, m_process, 1);

			m_bloblabeling.SetParam(m_process, 100);
			m_bloblabeling.DoLabeling();
			m_bloblabeling.DrawBlob(m_board, cvScalar(0, 0, 255));

			prevTrackVal = TrackVal;
		}

		cvShowImage("Input Image", m_board);
		cvShowImage("Image processing", m_process);

		if(cv::waitKey(10) == 27){
			printf("Exit this program...\n");
			break;
		}
	}

	cvDestroyAllWindows();

	cvReleaseImage(&m_ori);
	cvReleaseImage(&m_process);
	cvReleaseImage(&m_board);
	cvReleaseImage(&m_gray);

}

void Application::mouseCallback(int event, int x, int y, int flags, void *param){
	Application *self = static_cast<Application*>(param);
	self->DomouseCallback(event, x, y, flags);
}

void Application::DomouseCallback(int event, int x, int y, int flags){

	switch(event){
	case CV_EVENT_LBUTTONDOWN:
		{
			CvRect BlobInfo;
			IplImage *temp;
			cv::Mat tPatch;

			m_bloblabeling.GetLabel(cvPoint(x,y), &BlobInfo);
			//Patch extraction
			if(BlobInfo.width != -1){
				temp = cvCreateImage(cvSize(BlobInfo.width, BlobInfo.height), IPL_DEPTH_8U, 1);
				cvSetImageROI(m_process, BlobInfo);
				cvCopy(m_process, temp);
				cvResetImageROI(m_process);

				//28*28 scale로 맞추기
				tPatch.create(1, 28*28, CV_32FC1);
				m_preProcess.ResizeNMakeMat(temp, &tPatch);

				//DBN query
				int result = m_DBN.DBNquery(tPatch);

				//Draw result
				printf("result : %d\n", result);

				cvReleaseImage(&temp);
			}
			break;

		}
	}
}