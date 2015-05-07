#include "common.h"

class Layer
{
public:
	Layer(void);
	~Layer(void);

	void Init(int unitCount);

	float sampling(float prob);

	void setDataDirect(cv::Mat src);				//Visible node������
	int getUnitNum();
	void setLayerRelation(Layer *prev, Layer *post);

	/*RBM �н��� ����*/

	/*���� ���̾� �ƿ�ǲ�� ��� - hidden layer�� ��*/
	void processData(cv::Mat *dst, cv::Mat data);

	/*�ٷ� ���� �����͸� �ְ� ���� ���̾� ��°�����(���� ���̾ hidden)*/
	void processTempData(cv::Mat *dst, cv::Mat input);
	/*hidden Layer�� ���� �ְ� back process ����*/
	void processTempBack(cv::Mat *dst, cv::Mat input);

	void ApplyGrad(cv::Mat wGrad, cv::Mat bGrad, cv::Mat cGrad);

	Layer *m_prevLayer, *m_postLayer;

	cv::Mat m_weight;								//bias ������
	cv::Mat m_b;									//bias - visible
	cv::Mat m_c;									//bias - hidden

private:
	int n_units;

	float sigmoid(float src);
};
