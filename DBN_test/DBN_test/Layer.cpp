#include "Layer.h"

Layer::Layer(void)
{
	m_postLayer = m_prevLayer = NULL;
}


Layer::~Layer(void)
{
}

void Layer::Init(int unitCount){
	n_units = unitCount;
}

int Layer::getUnitNum(){
	return n_units;
}

float Layer::sigmoid(float src){
	return (1.0f / (1.0f + exp(-src)));
}

float Layer::sampling(float prob){
	random_device rd;
	mt19937_64 rng(rd());

	uniform_real_distribution<float> dist1(0.0f, 1.0f);

	float ran = dist1(rng);

	return (ran > prob) ? 0.0f : 1.0f;
}

void Layer::setLayerRelation(Layer *prev, Layer *post){
	m_prevLayer = prev;
	m_postLayer = post;

	int twidth;
	if(prev == NULL)
		twidth = 28*28+1;
	else
		twidth = prev->getUnitNum()+1;
	m_weight.create(twidth, n_units, CV_32FC1);

	for(int i = 0; i < m_weight.rows; i++)
		for(int j = 0; j < m_weight.cols; j++){
			m_weight.at<float>(i,j) = 1.0f;
		}
}

void Layer::processData(cv::Mat *dst, cv::Mat data){
	Layer *visible = this;
	cv::Mat input;
	dst->create(1, n_units, CV_32FC1);
	//������ ���̾� �˻�
	while(1){
		if(visible->m_prevLayer == NULL)
			break;
		visible = visible->m_prevLayer;
	}

	input = data.clone();
	while(1){
		visible->m_postLayer->processTempData(dst, input);
		visible = visible->m_postLayer;
		input = dst->clone();

		if(visible == this)
			break;

		//sampling ����� ������
		for(int i = 0; i < input.cols; i++)
			input.at<float>(0,i) = sampling(input.at<float>(0,i));
	}
}

void Layer::processTempData(cv::Mat *dst, cv::Mat input){
	cv::Mat tInput;
	int i;

	tInput.create(1, input.cols+1, CV_32FC1);
	for(i = 0; i < input.cols; i++)
		tInput.at<float>(0,i) = (float)input.at<float>(0,i);
	tInput.at<float>(0,i) = 1.0f;

	dst->create(1, n_units, CV_32FC1);
	*dst = tInput * m_weight;
	
	for(i = 0; i < dst->cols; i++)
		dst->at<float>(0,i) = sigmoid(dst->at<float>(0,i));
}