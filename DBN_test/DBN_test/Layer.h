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

	void processData(cv::Mat *dst, cv::Mat data);								//현재 레이어 아웃풋을 계산
	void processTempData(cv::Mat *dst, cv::Mat input);

	Layer *m_prevLayer, *m_postLayer;

private:
	int n_units;
	cv::Mat m_weight;								//bias 포함

	float sigmoid(float src);
};
