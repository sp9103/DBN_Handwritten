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

	/*RBM �н��� ��*/

	/*���� ���̾� �ƿ�ǲ�� ��� - hidden layer�� ��*/
	void processData(cv::Mat *dst, cv::Mat data);

	/*�ٷ� ���� �����͸� �ְ� ���� ���̾� ��°�����(���� ���̾ hidden)*/
	void processTempData(cv::Mat *dst, cv::Mat input);
	/*hidden Layer�� ���� �ְ� back process ����*/
	void processTempBack(cv::Mat *dst, cv::Mat input, cv::Mat *firstRow);
	/*SoftMax function��� ��Ų ��� (not sigmoid)*/
	void processTempSoft(cv::Mat *dst, cv::Mat input);

	void ApplyGrad(cv::Mat wGrad, cv::Mat bGrad, cv::Mat cGrad);

	cv::Mat calcProbH(cv::Mat x);

	//matrix copy
	void MatCopy(cv::Mat src, cv::Mat *dst);

	Layer *m_prevLayer, *m_postLayer;

	cv::Mat m_weight;								//bias ������
	cv::Mat m_b;									//bias - visible
	cv::Mat m_c;									//bias - hidden

	void WeightVis();

	void processPresData(cv::Mat *dst, cv::Mat data);		//sampling ���� �÷��� ����Ʈ�� ����� ������

private:
	int n_units;

	float sigmoid(float src);
};
