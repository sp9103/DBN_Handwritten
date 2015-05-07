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

	//W create
	int twidth;
	if(prev == NULL)
		twidth = 28*28;
	else
		twidth = prev->getUnitNum();
	m_weight.create(twidth, n_units, CV_32FC1);

	for(int i = 0; i < m_weight.rows; i++){
		for(int j = 0; j < m_weight.cols; j++){
			m_weight.at<float>(i,j) = 0.0f;
		}
	}

	//bias b create
	if(prev != NULL){
		m_b.create(1,prev->getUnitNum(), CV_32FC1);
		for(int i = 0; i < m_b.rows; i++){
			for(int j = 0; j < m_b.cols; j++)
				m_b.at<float>(i,j) = 0.0f;
		}
	}

	//bias c create
	m_c.create(1,getUnitNum(), CV_32FC1);
	for(int i = 0; i < m_c.rows; i++){
		for(int j = 0; j < m_c.cols; j++)
			m_c.at<float>(i,j) = 0.0f;
	}
}

//이전 레이어부터 현재 레이어까지 쭉 pulling
void Layer::processData(cv::Mat *dst, cv::Mat data){
	Layer *visible = this;
	cv::Mat input;
	dst->create(data.rows, n_units, CV_32FC1);
	//최하위 레이어 검색
	while(1){
		if(visible->m_prevLayer == NULL)
			break;
		visible = visible->m_prevLayer;
	}

	input = data.clone();
	*dst = data.clone();
	while(1){
		if(visible == this)
			break;

		visible->m_postLayer->processTempData(dst, input);
		visible = visible->m_postLayer;

		//sampling 결과를 가져옴
		for(int i = 0; i < dst->rows; i++){
			for(int j = 0; j < dst->cols; j++)
				dst->at<float>(i,j) = sampling(dst->at<float>(i,j));
		}
		input = dst->clone();
	}
}

//바로 이전 레이어에서만 pulling
void Layer::processTempData(cv::Mat *dst, cv::Mat input){
	cv::Mat tInput;
	int j;

	tInput.create(input.rows, input.cols+1, CV_32FC1);
	for(int i = 0; i < input.rows; i++){
		for(j = 0; j < input.cols; j++)
			tInput.at<float>(i,j) = (float)input.at<float>(i,j);
		tInput.at<float>(i,j) = 1.0f;
	}

	dst->create(input.rows, n_units, CV_32FC1);
	cv::Mat tW;
	tW.create(m_weight.rows+1, m_weight.cols, CV_32FC1);
	for(int i = 0; i < m_weight.rows; i++){
		for(int j = 0; j < m_weight.cols; j++){
			if(i == m_weight.rows)
				tW.at<float>(i,j) = m_c.at<float>(0,j);

			else
				tW.at<float>(i,j) = m_weight.at<float>(i,j);
		}
	}

	*dst = tInput * tW;

	for(int i = 0; i < dst->rows; i++){
		for(j = 0; j < dst->cols; j++)
			dst->at<float>(i,j) = sigmoid(dst->at<float>(i,j));
	}
}

void Layer::processTempBack(cv::Mat *dst, cv::Mat input){
	cv::Mat tInput;
	int i;

	tInput.create(input.rows, input.cols+1, CV_32FC1);
	for(i = 0; i < input.cols; i++)
		tInput.at<float>(0,i) = (float)input.at<float>(0,i);
	tInput.at<float>(0,i) = 1.0f;

	dst->create(input.rows, m_prevLayer->n_units, CV_32FC1);
	cv::Mat tW;
	tW.create(m_weight.cols+1, m_weight.rows, CV_32FC1);
	for(int i = 0; i < tW.rows; i++){
		for(int j = 0; j < tW.cols; j++){
			if(i == m_weight.rows)
				tW.at<float>(i,j) = m_b.at<float>(0,j);

			else
				tW.at<float>(i,j) = m_weight.at<float>(j,i);
		}
	}

	*dst = tInput * tW;

	for(i = 0; i < dst->cols; i++)
		dst->at<float>(0,i) = sigmoid(dst->at<float>(0,i));
}

void Layer::ApplyGrad(cv::Mat wGrad, cv::Mat bGrad, cv::Mat cGrad){
	m_weight = m_weight + wGrad;
	m_b = m_b + bGrad;
	m_c = m_c + cGrad;
}