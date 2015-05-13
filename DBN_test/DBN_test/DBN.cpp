#include "DBN.h"

DBN::DBN(void)
{
	srand(time(NULL));
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
	int batchCount = 0;

	InitNetwork();

	//RBMLayerload("Layer1Data.bin", &hidden[0]);

	//unsupervised training - �� RBM �н�
	printf("Unsupervised Training phase start\n");
	BatchOpen("Data\\train-images.idx3-ubyte", "");

	_start = clock();
	for(int i = 0; i < LAYERHEIGHT-1; i++){
		printf("[%d] RBM Training...\n", i+1);
		batchCount = 0;
		_phase = clock();

		while(1){
			cv::Mat miniBatch;
			/*BatchLoad(&miniBatch, NULL, "Data\\train-images.idx3-ubyte", "");*/
			BatchRandLoad(&miniBatch, NULL);

#ifdef DEBUG_VISIBLE
			//Debug visualization
			DataSingleVis(miniBatch, "Original");
#endif

			//Bottom Layer training
			if(i == 0){
				//Input�� �̹��� �״�� ��
				tgrad = RBMupdata(miniBatch, EPSILON, &hidden[i], CDStep);
			}
			//others
			else{
				//input�� ���� ���̾��� �ƿ�ǲ. - ( ���������� )
				cv::Mat tdata;
				hidden[i-1].processData(&tdata, miniBatch);
				tgrad = RBMupdata(tdata, EPSILON, &hidden[i], CDStep);
			}

			//Termination condition
			printf("Layer[%d] - [%d] Batch calculated, Max gradient : %f\n", i+1, ++batchCount, tgrad);
			if(GRADTHRESHOLD > tgrad){
				//�� RBM Training Data�� ������
				char tbuf[256];
				sprintf(tbuf, "Layer%dData.bin", i+1);
				RBMLayersave(tbuf, hidden[i]);
				break;
			}
		}

		printf("Complete! (%dms)\n", clock() - _phase);
	}
	printf("Unsupervised Training complete (%dms)\n", clock() - _start);
	BatchClose();

	RBMsave("RBMDATA.bin");

	//supervised training - full optimization MLP backpropagation
	printf("\nSupervised Training phase start\n");
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

	h1.create(minibatch.rows, layer->getUnitNum(), CV_32FC1);
	hk.create(minibatch.rows, layer->getUnitNum(), CV_32FC1);

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//k-step Contrast Divergence
	MatCopy(minibatch, &x1);
	MatCopy(minibatch, &xk);
	layer->processTempData(&h1, x1);
	MatCopy(h1, &hk);
	for(int k = 1; k < step; k++){
		layer->processTempBack(&xk, h1);
		layer->processTempData(&hk, xk);
	}

#ifdef DEBUG_VISIBLE
	//�� ���� �϶���
	if(layer->m_prevLayer->m_prevLayer == NULL){
		DataSingleVis(xk, "Reconstruct");
		cv::waitKey(1);
	}

	//�ι�° ���̾��϶�
	if(layer->m_prevLayer->m_prevLayer->m_prevLayer == NULL){
		cv::Mat debugXk;
		layer->m_prevLayer->processTempBack(&debugXk, xk);
		DataSingleVis(debugXk, "Reconstruct");
		cv::waitKey(1);
	}
#endif

	//gradient ���
	cv::Mat tprob = layer->calcProbH(xk);

	wGrad = calcW(h1, x1, tprob, xk);
	bGrad = calcB(x1, xk);
	cGrad = calcC(h1, tprob);

	//��� �ݿ�
	layer->ApplyGrad(wGrad, bGrad, cGrad);
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////

	tgrad = MatMaxEle(wGrad) / EPSILON;

	return tgrad;
}

//Label�� NULL �϶��� �ε����.
//Label != NULL�̸� supervised learning�� ���ؼ� �ε�
void DBN::BatchLoad(cv::Mat *batch, cv::Mat *Label, char* DataName, char* LabelName){
	//���������� ���� (���� ���� ��ġ�� ����)
	static int tCount = 0;

	if(tCount <= 0){
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
	MatZeros(&result);

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
	MatZeros(&result);

	result = x1.t()*h1 - xk.t()*prob;

	result = EPSILON * result / h1.rows;

	return result.clone();
}

void DBN::DataVis(cv::Mat data1, cv::Mat data2){
	cv::Mat fic, fic2;
	fic.create(28, 28, CV_8UC1);
	fic2.create(28, 28, CV_8UC1);

	for(int i = 0; i < data1.cols; i++){
		fic.at<uchar>(i/28,i%28) = (data1.at<float>(0,i) > 0.0f) ? 255 : 0;
		fic2.at<uchar>(i/28,i%28) = (data2.at<float>(0,i) > 0.0f) ? 255 : 0;
	}

	cv::imshow("ttt", fic);
	cv::imshow("eee", fic2);
	cv::waitKey(1);
}

float DBN::MatMaxEle(cv::Mat src){
	float tmax = 0.0f;

	for(int i = 0; i < src.rows; i++){
		for(int j = 0; j < src.cols; j++){
			float tele = abs(src.at<float>(i,j));

			if(tele > tmax)
				tmax = tele;
		}
	}

	return tmax;
}

void DBN::MatCopy(cv::Mat src, cv::Mat *dst){
	dst->create(src.rows, src.cols, CV_32FC1);

	for(int i = 0; i < src.rows; i++){
		for(int j = 0; j <src.cols; j++){
			dst->at<float>(i,j) = src.at<float>(i,j);
		}
	}
}

void DBN::RBMsave(char *fileName){
	printf("Training result writing..\n");
	FILE *fp = fopen(fileName, "wb");
	//SaveFile
	int temp = LAYERHEIGHT;
	fwrite(&temp,sizeof(int),1, fp);			//Layer Height write

	//Layer unit information write
	temp = NHIDDEN1;
	fwrite(&temp, sizeof(int), 1, fp);
	temp = NHIDDEN2;
	fwrite(&temp, sizeof(int), 1, fp);
	temp = NHIDDEN3;
	fwrite(&temp, sizeof(int), 1, fp);

	//Weight & biase write
	for(int i = 0; i < LAYERHEIGHT-1; i++){
		//weight
		for(int j = 0; j < hidden[i].m_weight.rows; j++){
			for(int k = 0; k < hidden[i].m_weight.cols; k++){
				float tf = hidden[i].m_weight.at<float>(j,k);
				fwrite(&tf, sizeof(float), 1, fp);
			}
		}

		//b
		for(int j = 0; j < hidden[i].m_b.rows; j++){
			for(int k = 0; k < hidden[i].m_b.cols; k++){
				float tf = hidden[i].m_b.at<float>(j,k);
				fwrite(&tf, sizeof(float), 1, fp);
			}
		}

		//c
		for(int j = 0; j < hidden[i].m_c.rows; j++){
			for(int k = 0; k < hidden[i].m_c.cols; k++){
				float tf = hidden[i].m_c.at<float>(j,k);
				fwrite(&tf, sizeof(float), 1, fp);
			}
		}
	}

	fclose(fp);
	printf("Writing complete!\n");
}

void DBN::RBMLoad(char *fileName){
	printf("Training result Loading..\n");
	FILE *fp = fopen(fileName, "rb");
	//SaveFile
	int temp;
	fread(&temp,sizeof(int),1, fp);			//Layer Height write

	//Layer unit information write
	fread(&temp, sizeof(int), 1, fp);
	fread(&temp, sizeof(int), 1, fp);
	fread(&temp, sizeof(int), 1, fp);

	//Weight & biase write
	for(int i = 0; i < LAYERHEIGHT-1; i++){
		//weight
		for(int j = 0; j < hidden[i].m_weight.rows; j++){
			for(int k = 0; k < hidden[i].m_weight.cols; k++){
				float tf;
				fread(&tf, sizeof(float), 1, fp);
				hidden[i].m_weight.at<float>(j,k) = tf;
			}
		}

		//b
		for(int j = 0; j < hidden[i].m_b.rows; j++){
			for(int k = 0; k < hidden[i].m_b.cols; k++){
				float tf;
				fread(&tf, sizeof(float), 1, fp);
				hidden[i].m_b.at<float>(j,k) = tf;
			}
		}

		//c
		for(int j = 0; j < hidden[i].m_c.rows; j++){
			for(int k = 0; k < hidden[i].m_c.cols; k++){
				float tf;
				fread(&tf, sizeof(float), 1, fp);
				hidden[i].m_c.at<float>(j,k) = tf;
			}
		}
	}

	fclose(fp);
	printf("Writing complete!\n");
}

void DBN::RBMLayersave(char *fileName, Layer src){
	printf("Training result writing..\n");
	FILE *fp = fopen(fileName, "wb");

	//Layer unit information write
	int temp = src.getUnitNum();
	fwrite(&temp, sizeof(int), 1, fp);

	//Weight & biase write

	//weight
	for(int j = 0; j < src.m_weight.rows; j++){
		for(int k = 0; k < src.m_weight.cols; k++){
			float tf = src.m_weight.at<float>(j,k);
			fwrite(&tf, sizeof(float), 1, fp);
		}
	}

	//b
	for(int j = 0; j < src.m_b.rows; j++){
		for(int k = 0; k < src.m_b.cols; k++){
			float tf = src.m_b.at<float>(j,k);
			fwrite(&tf, sizeof(float), 1, fp);
		}
	}

	//c
	for(int j = 0; j < src.m_c.rows; j++){
		for(int k = 0; k < src.m_c.cols; k++){
			float tf = src.m_c.at<float>(j,k);
			fwrite(&tf, sizeof(float), 1, fp);
		}
	}

	fclose(fp);
	printf("Writing complete!\n");
}

void DBN::RBMLayerload(char *fileName, Layer *dst){
	printf("Training result Loading..\n");
	FILE *fp = fopen(fileName, "rb");


	//Layer unit information write
	int temp;
	fread(&temp, sizeof(int), 1, fp);

	//Weight & biase write
	//weight
	for(int j = 0; j < dst->m_weight.rows; j++){
		for(int k = 0; k < dst->m_weight.cols; k++){
			float tf;
			fread(&tf, sizeof(float), 1, fp);
			dst->m_weight.at<float>(j,k) = tf;
		}
	}

	//b
	for(int j = 0; j < dst->m_b.rows; j++){
		for(int k = 0; k < dst->m_b.cols; k++){
			float tf;
			fread(&tf, sizeof(float), 1, fp);
			dst->m_b.at<float>(j,k) = tf;
		}
	}

	//c
	for(int j = 0; j < dst->m_c.rows; j++){
		for(int k = 0; k < dst->m_c.cols; k++){
			float tf;
			fread(&tf, sizeof(float), 1, fp);
			dst->m_c.at<float>(j,k) = tf;
		}
	}

	fclose(fp);
	printf("Load complete!\n");
}

void DBN::DataSingleVis(cv::Mat data, char *windowName){
	cv::Mat fic;
	fic.create(28, 28, CV_8UC1);

	for(int i = 0; i < data.cols; i++){
		fic.at<uchar>(i/28,i%28) = (data.at<float>(0,i) > 0.0f) ? 255 : 0;
	}

	cv::imshow(windowName, fic);
}

void DBN::BatchOpen(char *DataName, char* LabelName){

	m_Dataloader.FileOpen(DataName);
	m_Dataloader.ImageDataLoad(m_Dataloader.getDataCount(), &m_DataSet);

	if(strlen(LabelName) > 2){
		m_Labelloader.FileOpen(LabelName);
		m_Labelloader.LabelDataLoad(m_Labelloader.getDataCount(), &m_LabelSet);
	}

	m_box = (int *)malloc(sizeof(int)*m_Dataloader.getDataCount());
	for(int i = 0; i < m_Dataloader.getDataCount(); i++)
		m_box[i] = i;

}
void DBN::BatchRandLoad(cv::Mat *batch, cv::Mat *Label){
	static int count = 0;
	
	if(count < 1){
		//Random box mixing
		for(int i = 0; i < m_Dataloader.getDataCount(); i++){
			int temp;
			int trand = rand() % m_Dataloader.getDataCount();
			SWAP(m_box[i], m_box[trand], temp);
		}
	}

	if(batch != NULL){
		batch->create(BATCHSIZE, 28*28, CV_32FC1);
		for(int i = 0; i < BATCHSIZE; i++){
			m_DataSet.row(m_box[count + i]).copyTo(batch->row(i));
		}
	}
	if(Label != NULL){
		Label->create(BATCHSIZE, 28*28, CV_32FC1);
		for(int i = 0; i < BATCHSIZE; i++){
			m_LabelSet.row(m_box[count + i]).copyTo(Label->row(i));
		}
	}
	count += BATCHSIZE;
	if(count >= m_Dataloader.getDataCount()){
		count = 0;
	}

}

void DBN::BatchClose(){
	m_Dataloader.FileClose();
	m_Labelloader.FileClose();
}