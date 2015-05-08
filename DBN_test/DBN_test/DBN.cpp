#include "DBN.h"

DBN::DBN(void)
{
}


DBN::~DBN(void)
{
}

void DBN::InitNetwork(){
	visible.Init(NVISIBLE);
	hidden[0].Init(NHIDDEN1);
	hidden[1].Init(NHIDDEN2);
	hidden[2].Init(NHIDDEN3);

	visible.setLayerRelation(NULL, &hidden[0]);
	hidden[0].setLayerRelation(&visible, &hidden[1]);
	hidden[1].setLayerRelation(&hidden[0], &hidden[2]);
	hidden[2].setLayerRelation(&hidden[1], NULL);
}

void DBN::save(char *fileName){
	printf("Training result writing..\n");
	FILE *fp = fopen(fileName, "wb");
	//SaveFile

	fclose(fp);
	printf("Writing complete!\n");
}

void DBN::Load(char *fileName){
	FILE *fp = fopen(fileName, "rb");
	//LoadFile

	fclose(fp);
}

void DBN::Training(){
	float tgrad = 0.0f;
	int _start, _super, _phase;

	InitNetwork();

	//unsupervised training - �� RBM �н�
	printf("Unsupervised Training phase start\n");
	_start = clock();
	for(int i = 0; i < LAYERHEIGHT-1; i++){
		printf("[%d] RBM Training...\n", i+1);
		_phase = clock();

		while(1){
			cv::Mat miniBatch;
			BatchLoad(&miniBatch, NULL, "Data\\train-images.idx3-ubyte", "");

			//Bottom Layer training
			if(i == 0){
				//Input�� �̹��� �״�� ��
				tgrad = RBMupdata(miniBatch, EPSILON, &hidden[i], CDStep);
			}
			//others
			else{
				//input�� ���� ���̾��� �ƿ�ǲ. - ( ���������� )
				cv::Mat tdata;
				hidden[i].processData(&tdata, miniBatch);
				tgrad = RBMupdata(tdata, EPSILON, &hidden[i], CDStep);
			}

			//Termination condition
			if(GRADTHRESHOLD > tgrad)
				break;
		}

		printf("Complete! (%dms)\n", clock() - _phase);
	}
	printf("Unsupervised Training complete (%dms)\n", clock() - _start);

	//supervised training - full optimization MLP backpropagation
	printf("Supervised Training phase start\n");
	_super = clock();


	printf("Supervised Training phase complete (%dms)\n", clock() - _super);
}

void DBN::Testing(){
	m_Dataloader.FileOpen("t10k-images.idx3-ubyte");
}

float DBN::RBMupdata(cv::Mat minibatch, float e, Layer *layer, int step){
	float tgrad = 0.0f;
	cv::Mat wGrad, bGrad, cGrad;
	cv::Mat x1, xk, h1, hk;

	x1.create(minibatch.rows, minibatch.cols, CV_32FC1);
	xk.create(minibatch.rows, minibatch.cols, CV_32FC1);
	h1.create(minibatch.rows, layer->getUnitNum(), CV_32FC1);
	hk.create(minibatch.rows, layer->getUnitNum(), CV_32FC1);

	//Matrix Allocation
	wGrad.create(layer->m_weight.rows, layer->m_weight.cols, CV_32FC1);
	bGrad.create(1, layer->m_b.cols, CV_32FC1);
	cGrad.create(1, layer->m_c.cols, CV_32FC1);

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//k-step Contrast Divergence
	layer->m_prevLayer->processData(&x1, minibatch);
	xk = x1.clone();
	layer->processData(&h1, x1);
	hk = h1.clone();
	for(int k = 1; k < step; k++){
		layer->processTempBack(&xk, h1);
		layer->processTempData(&hk, xk);
	}

	//gradient ���
	cv::Mat tprob;
	wGrad = calcW(h1, x1, tprob, xk);
	bGrad = calcB(x1, xk);
	cGrad = calcC(h1, tprob);

	//��� �ݿ�
	layer->ApplyGrad(wGrad, bGrad, cGrad);
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////

	return tgrad;
}

//Label�� NULL �϶��� �ε����.
//Label != NULL�̸� supervised learning�� ���ؼ� �ε�
void DBN::BatchLoad(cv::Mat *batch, cv::Mat *Label, char* DataName, char* LabelName){
	//���������� ���� (���� ���� ��ġ�� ����)
	static int tCount = 0;

	if(tCount == 0){
		m_Dataloader.FileOpen(DataName);

		if(Label != NULL)
			m_Labelloader.FileOpen(LabelName);
	}

	m_Dataloader.ImageDataLoad(BATCHSIZE, batch);

	if(Label != NULL)
		m_Labelloader.LabelDataLoad(BATCHSIZE, Label);
	tCount += BATCHSIZE;

	if(tCount == m_Dataloader.getDataCount()){
		m_Dataloader.FileClose();

		if(Label != NULL)
			m_Labelloader.FileClose();
		tCount = 0;
	}
}

void DBN::MatZeros(cv::Mat *target){
	for(int i = 0; i < target->rows; i++){
		for(int j = 0; j < target->cols; j++){
			target->at<float>(i,j) = 0.0f;
		}
	}
}

cv::Mat DBN::calcB(cv::Mat x1, cv::Mat xk){
	cv::Mat result;

	result.create(1, x1.cols, CV_32FC1);
	MatZeros(&result);

	for(int i = 0; i < x1.rows; i++){
		for(int j = 0; j < x1.cols; j++){
			result.at<float>(0,j) += EPSILON * (x1.at<float>(i,j) - xk.at<float>(i,j)) / (float)x1.rows;
		}
	}

	return result.clone();
}

cv::Mat DBN::calcC(cv::Mat h1, cv::Mat prob){
	cv::Mat result;

	result.create(1, h1.cols, CV_32FC1);

	for(int i = 0; i < h1.rows; i++){
		for(int j = 0; j < h1.cols; j++){
			result.at<float>(0,j) += EPSILON * (h1.at<float>(i,j) - prob.at<float>(0,j)) / (float)h1.rows;
		}

	}

	return result.clone();
}

cv::Mat DBN::calcW(cv::Mat h1, cv::Mat x1, cv::Mat prob, cv::Mat xk){
	cv::Mat result;

	result.create(x1.cols, h1.cols, CV_32FC1);



	return result.clone();
}