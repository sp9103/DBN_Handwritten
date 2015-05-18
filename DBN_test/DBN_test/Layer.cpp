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

	MatCopy(data, &input);
	MatCopy(data, dst);
	while(1){
		if(visible == this)
			break;

		visible->m_postLayer->processTempData(dst, input);
		visible = visible->m_postLayer;

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
	for(int i = 0; i < tW.rows; i++){
		for(j = 0; j < tW.cols; j++){
			if(i == tW.rows-1)
				tW.at<float>(i,j) = m_c.at<float>(0,j);

			else
				tW.at<float>(i,j) = m_weight.at<float>(i,j);
		}
	}

	*dst = tInput * tW;

	///////////////AVG
	//cv::Mat AvgInput;

	//cv::reduce(tInput, AvgInput, 0, CV_REDUCE_AVG);
	//*dst = AvgInput * tW;

	for(int i = 0; i < dst->rows; i++){
		for(j = 0; j < dst->cols; j++)
			dst->at<float>(i,j) = sampling(sigmoid(dst->at<float>(i,j)));
	}
}

void Layer::processTempBack(cv::Mat *dst, cv::Mat input, cv::Mat *firstRow){
	cv::Mat tInput;
	int j;

	tInput.create(input.rows, input.cols+1, CV_32FC1);

	for(int i = 0; i < input.rows; i++){
		for(j = 0; j < input.cols; j++)
			tInput.at<float>(i,j) = (float)input.at<float>(i,j);
		tInput.at<float>(i,j) = 1.0f;
	}

	dst->create(input.rows, m_prevLayer->n_units, CV_32FC1);
	//firstRow->create(1, m_prevLayer->n_units, CV_32FC1);
	
	cv::Mat tW;
	tW.create(m_weight.cols+1, m_weight.rows, CV_32FC1);
	for(int i = 0; i < tW.rows; i++){
		for(j = 0; j < tW.cols; j++){
			if(i == tW.rows-1)
				tW.at<float>(i,j) = m_b.at<float>(0,j);

			else
				tW.at<float>(i,j) = m_weight.at<float>(j,i);
		}
	}

	if(firstRow != NULL)		*firstRow = tInput.row(0) * tW;

	///////////////AVG
	/*cv::Mat AvgInput;

	cv::reduce(tInput, AvgInput, 0, CV_REDUCE_AVG);*/
	
	*dst = tInput * tW;
	//*dst = AvgInput * tW;

	for(int i = 0; i < dst->rows; i++){
		for(j = 0; j < dst->cols; j++)
			dst->at<float>(i,j) = sampling(sigmoid(dst->at<float>(i,j)));
	}

	//for(int i = 0; i < firstRow->cols; i++)
	//	firstRow->at<float>(0,i) = sampling(sigmoid(firstRow->at<float>(0,i)));
}

void Layer::ApplyGrad(cv::Mat wGrad, cv::Mat bGrad, cv::Mat cGrad){
	m_weight = m_weight + wGrad;
	m_b = m_b + bGrad;
	m_c = m_c + cGrad;
}

cv::Mat Layer::calcProbH(cv::Mat x){
	cv::Mat result, tInput;
	int j;

	result.create(x.rows, n_units, CV_32FC1);

	tInput.create(x.rows, x.cols+1, CV_32FC1);
	for(int i = 0; i < x.rows; i++){
		for(j = 0; j < x.cols; j++)
			tInput.at<float>(i,j) = (float)x.at<float>(i,j);
		tInput.at<float>(i,j) = 1.0f;
	}

	cv::Mat tW;
	tW.create(m_weight.rows+1, m_weight.cols, CV_32FC1);
	for(int i = 0; i < tW.rows; i++){
		for(j = 0; j < tW.cols; j++){
			if(i == tW.rows-1)
				tW.at<float>(i,j) = m_c.at<float>(0,j);

			else
				tW.at<float>(i,j) = m_weight.at<float>(i,j);
		}
	}

	result = tInput * tW;

	for(int i = 0; i < result.rows; i++){
		for(j = 0; j < result.cols; j++)
			result.at<float>(i,j) = sigmoid(result.at<float>(i,j));
	}

	return result.clone();
}

void Layer::MatCopy(cv::Mat src, cv::Mat *dst){
	dst->create(src.rows, src.cols, CV_32FC1);

	for(int i = 0; i < src.rows; i++){
		for(int j = 0; j <src.cols; j++){
			dst->at<float>(i,j) = src.at<float>(i,j);
		}
	}
}

void Layer::WeightVis(){
	cv::Mat tboard, tpatch, normpatch;
	double tmin, tmax;

	tboard.create(28*20, 28*25, CV_32FC1);
	tpatch.create(28,28, CV_32FC1);
	normpatch.create(28,28, CV_32FC1);

	for(int i = 0; i < m_weight.cols; i++){
		//create patch
		for(int j = 0; j < m_weight.rows; j++){
			tpatch.at<float>(j/28,j%28) = m_weight.at<float>(j,i);
		}

		//normalize
		cv::minMaxLoc(tpatch, &tmin);
		tpatch = tpatch - tmin;
		cv::minMaxLoc(tpatch, &tmin, &tmax);
		tpatch = tpatch / tmax;
		cv::minMaxLoc(tpatch, &tmin, &tmax);

		//attach board
		cv::Point startP;
		startP.x = (i % 25) * 28;
		startP.y = (i / 25) * 28;

		for(int j = 0; j < tpatch.rows; j++){
			for(int k = 0; k < tpatch.cols; k++){
				float val = tpatch.at<float>(j,k);
				tboard.at<float>(startP.y+j,startP.x+k) = val;
			}
		}
	}
	
	cv::imshow("Weight vis", tboard);
	cv::waitKey(0);
}

void Layer::processPresData(cv::Mat *dst, cv::Mat data){
	cv::Mat tInput;
	int j;

	tInput.create(data.rows, data.cols+1, CV_32FC1);
	for(int i = 0; i < data.rows; i++){
		for(j = 0; j < data.cols; j++)
			tInput.at<float>(i,j) = (float)data.at<float>(i,j);
		tInput.at<float>(i,j) = 1.0f;
	}

	dst->create(data.rows, n_units, CV_32FC1);
	cv::Mat tW;
	tW.create(m_weight.rows+1, m_weight.cols, CV_32FC1);
	for(int i = 0; i < tW.rows; i++){
		for(j = 0; j < tW.cols; j++){
			if(i == tW.rows-1)
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