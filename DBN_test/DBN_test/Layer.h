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
	void processData(cv::Mat *dst, cv::Mat data);								//���� ���̾� �ƿ�ǲ�� ���
	void processTempData(cv::Mat *dst, cv::Mat input);

	Layer *m_prevLayer, *m_postLayer;

private:
	int n_units;
	cv::Mat m_weight;								//bias ������
	cv::Mat m_b;									//bias - visible
	cv::Mat m_c;									//bias - hidden

	float sigmoid(float src);
};
