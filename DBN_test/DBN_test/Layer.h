#include "common.h"

class Layer
{
public:
	Layer(void);
	~Layer(void);

	void Init(int unitCount);

	float sampling(float prob);

	void setDataDirect(cv::Mat src);				//Visible node에서만
	int getUnitNum();
	void setLayerRelation(Layer *prev, Layer *post);

	/*RBM 학습을 위한*/

	/*현재 레이어 아웃풋을 계산 - hidden layer의 값*/
	void processData(cv::Mat *dst, cv::Mat data);

	/*바로 이전 데이터를 넣고 현재 레이어 출력값을냄(현재 레이어가 hidden)*/
	void processTempData(cv::Mat *dst, cv::Mat input);
	/*hidden Layer의 값을 넣고 back process 수행*/
	void processTempBack(cv::Mat *dst, cv::Mat input, cv::Mat *firstRow);

	void ApplyGrad(cv::Mat wGrad, cv::Mat bGrad, cv::Mat cGrad);

	cv::Mat calcProbH(cv::Mat x);

	//matrix copy
	void MatCopy(cv::Mat src, cv::Mat *dst);

	Layer *m_prevLayer, *m_postLayer;

	cv::Mat m_weight;								//bias 미포함
	cv::Mat m_b;									//bias - visible
	cv::Mat m_c;									//bias - hidden

	void WeightVis();

private:
	int n_units;

	float sigmoid(float src);
};
