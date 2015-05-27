#include "common.h"

class Layer
{
public:
	Layer(void);
	~Layer(void);

	void Init(int unitCount);

	float sampling(float prob);
	
	int getUnitNum();
	void setLayerRelation(Layer *prev, Layer *post);

	/*RBM 학습을 위*/

	/*현재 레이어 아웃풋을 계산 - hidden layer의 값*/
	void processData(cv::Mat *dst, cv::Mat data);

	/*바로 이전 데이터를 넣고 현재 레이어 출력값을냄(현재 레이어가 hidden)*/
	void processTempData(cv::Mat *dst, cv::Mat input);
	/*hidden Layer의 값을 넣고 back process 수행*/
	void processTempBack(cv::Mat *dst, cv::Mat input, cv::Mat *firstRow);
	/*SoftMax function통과 시킨 결과 (not sigmoid)*/
	void processTempSoft(cv::Mat *dst, cv::Mat input);

	void ApplyGrad(cv::Mat wGrad, cv::Mat bGrad, cv::Mat cGrad);

	cv::Mat calcProbH(cv::Mat x);

	//matrix copy
	void MatCopy(cv::Mat src, cv::Mat *dst);

	Layer *m_prevLayer, *m_postLayer;

	cv::Mat m_weight;								//bias 미포함
	cv::Mat m_b;									//bias - visible
	cv::Mat m_c;									//bias - hidden

	void WeightVis();

	void processPresData(cv::Mat *dst, cv::Mat data);		//sampling 없이 플로팅 포인트로 결과를 산출함

private:
	int n_units;

	float sigmoid(float src);
};
