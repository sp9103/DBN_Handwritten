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
	void processData(cv::Mat *dst, cv::Mat data);								//현재 레이어 아웃풋을 계산
	void processTempData(cv::Mat *dst, cv::Mat input);

	Layer *m_prevLayer, *m_postLayer;

private:
	int n_units;
	cv::Mat m_weight;								//bias 미포함
	cv::Mat m_b;									//bias - visible
	cv::Mat m_c;									//bias - hidden

	float sigmoid(float src);
};
