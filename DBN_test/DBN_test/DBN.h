#include "common.h"
#include "DataLoader.h"
#include "LabelLoader.h"
#include "Layer.h"

class DBN
{
public:
	DBN(void);
	~DBN(void);

	void Training();
	void Testing();

	void save(char *fileName);
	void Load(char *fileName);

	void InitNetwork();

private:
	DataLoader m_Dataloader;
	LabelLoader m_Labelloader;

	Layer visible;
	Layer hidden[LAYERHEIGHT-1];

	float RBMupdata(cv::Mat x1, float e, Layer *layer, int step);

	void BatchLoad(cv::Mat *batch, cv::Mat *Label, char* DataName, char* LabelName);

	/*이미 할당된 메트릭스 원소를 모두 0으로 초기화*/
	void MatZeros(cv::Mat *target);

	/*bias b calculation*/
	cv::Mat calcB(cv::Mat x1, cv::Mat xk);
	/*bias c calculation*/
	cv::Mat calcC(cv::Mat h1, cv::Mat prob);
	/*weight w calculation*/
	cv::Mat calcW(cv::Mat h1, cv::Mat x1, cv::Mat prob, cv::Mat xk);

	/*Pick Matrix Maximum element*/
	float MatMaxEle(cv::Mat src);

	//Debug용 함수
	void DataVis(cv::Mat data);
};

