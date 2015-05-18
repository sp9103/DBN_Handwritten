#include "DBN.h"

DBN::DBN(void)
{
	srand(time(NULL));

	m_NEpoch = 0;
}


DBN::~DBN(void)
{
}

void DBN::InitNetwork(){
	visible.Init(NVISIBLE);
	hidden[0].Init(NHIDDEN1);
	hidden[1].Init(NHIDDEN2);
	hidden[2].Init(NHIDDEN3);
	classLayer.Init(NOUTPUT);

	visible.setLayerRelation(NULL, &hidden[0]);
	hidden[0].setLayerRelation(&visible, &hidden[1]);
	hidden[1].setLayerRelation(&hidden[0], &hidden[2]);
	hidden[2].setLayerRelation(&hidden[1], &classLayer);
	classLayer.setLayerRelation(&hidden[2], NULL);
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
	int prevEpoch = 0;

	InitNetwork();

	RBMLayerload("Layer1Info.bin", &hidden[0]);
	RBMLayerload("Layer2Info.bin", &hidden[1]);
	RBMLayerload("Layer3Info.bin", &hidden[2]);
	hidden[0].WeightVis();

	//unsupervised training - 각 RBM 학습
#ifdef RBM_TRAINING
	_start = clock();
	printf("Unsupervised Training phase start\n");
	printf("\nData load from file....\n");
	BatchOpen("Data\\train-images.idx3-ubyte", "");
	printf("Data load Complete! (%dms)\n", clock() - _start);

	for(int i = 2; i < LAYERHEIGHT-1; i++){
		printf("[%d] RBM Training...\n", i+1);
		batchCount = 0;
		_phase = clock();
		m_NEpoch = 0;

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
				//Input이 이미지 그대로 들어감
				tgrad = RBMupdata(miniBatch, EPSILON, &hidden[i], CDStep);
			}
			//others
			else{
				//input은 이전 레이어의 아웃풋. - ( 계산해줘야함 )
				cv::Mat tdata;
				hidden[i-1].processData(&tdata, miniBatch);
				tgrad = RBMupdata(tdata, EPSILON, &hidden[i], CDStep);
			}

			//Termination condition
			printf("gradient MAx : %f\n", tgrad);
			printf("[%d] Batch train Complete!\n", ++batchCount);
			if(m_NEpoch > NEPOCH){
				printf("Layer[%d] train Complete!\n", i+1);
				batchCount = 0;

				char buf[256];
				sprintf(buf, "Layer%dInfo.bin", i+1);
				RBMLayersave(buf, hidden[i]);
				break;
			}

			prevEpoch = m_NEpoch;
		}

		printf("Complete! (%dms)\n", clock() - _phase);
	}
	printf("Unsupervised Training complete (%dms)\n", clock() - _start);
	BatchClose();
#endif

	//supervised training - full optimization MLP backpropagation
	printf("\nSupervised Training phase start\n");
	_super = clock();
	FullBackpropagation();
	Netsave("FullNetworkData.bin");

	printf("Supervised Training phase complete (%dms)\n", clock() - _super);
}

void DBN::Testing(){
	m_Dataloader.FileOpen("t10k-images.idx3-ubyte");
}

float DBN::RBMupdata(cv::Mat minibatch, float e, Layer *layer, int step){
	float tgrad = 0.0f;
	cv::Mat wGrad, bGrad, cGrad;
	cv::Mat x1, xk, h1, hk;
	cv::Mat xkF;						//디버깅용 첫째 로우만 관리
	int _start;
	static int VisFrequency = 0;

	_start = clock();
	h1.create(minibatch.rows, layer->getUnitNum(), CV_32FC1);
	hk.create(minibatch.rows, layer->getUnitNum(), CV_32FC1);

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//k-step Contrast Divergence
	MatCopy(minibatch, &x1);
	MatCopy(minibatch, &xk);
	printf("RBM update Init (%dms)\n", clock() - _start);
	_start = clock();
	layer->processTempData(&h1, x1);
	MatCopy(h1, &hk);
	for(int k = 1; k < step; k++){
		layer->processTempBack(&xk, h1, &xkF);
	}
	printf("RBM CD (%dms)\n", clock() - _start);
	_start = clock();

#ifdef DEBUG_VISIBLE
	//맨 하위 일때만
	if(layer->m_prevLayer->m_prevLayer == NULL){
		DataSingleVis(xkF, "Reconstruct");
		cv::waitKey(1);
	}else if(layer->m_prevLayer->m_prevLayer->m_prevLayer == NULL){								//두번째
		//if(VisFrequency == 0){
		cv::Mat debugXk;
		layer->m_prevLayer->processTempBack(&debugXk, xk, NULL);
		DataSingleVis(debugXk, "Reconstruct");
		cv::waitKey(1);
		//}

		//VisFrequency = (VisFrequency+1) % 10;
	}else{
		cv::Mat debugXk;
		layer->m_prevLayer->processTempBack(&debugXk, xk, NULL);
		layer->m_prevLayer->m_prevLayer->processTempBack(&debugXk, debugXk, NULL);
		DataSingleVis(debugXk, "Reconstruct");
		cv::waitKey(1);
	}
	printf("Visualize (%dms)\n", clock() - _start);
	_start = clock();
#endif

	//cv::Mat AVx1, AVxk, AVh1;

	//cv::reduce(x1, AVx1, 0, CV_REDUCE_AVG);
	//cv::reduce(xk, AVxk, 0, CV_REDUCE_AVG);
	//cv::reduce(h1, AVh1, 0, CV_REDUCE_AVG);

	//gradient 계산
	cv::Mat tprob = layer->calcProbH(x1);
	wGrad = calcW(h1, x1, tprob, xk);
	bGrad = calcB(x1, xk);
	cGrad = calcC(h1, tprob);
	//printf("RBM update calculate (%dms)\n", clock() - _start);
	//_start = clock();

	//결과 반영
	layer->ApplyGrad(wGrad, bGrad, cGrad);
	//printf("RBM update apply (%dms)\n\n", clock() - _start);
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////

	tgrad = MatMaxEle(wGrad) / EPSILON;

	return tgrad;
}

//Label이 NULL 일때는 로드안함.
//Label != NULL이면 supervised learning을 위해서 로드
void DBN::BatchLoad(cv::Mat *batch, cv::Mat *Label, char* DataName, char* LabelName){
	//순차적으로 뽑음 (추후 랜덤 배치로 구현)
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
	cv::Mat tAvg;

	result.create(1, h1.cols, CV_32FC1);
	MatZeros(&result);

	for(int i = 0; i < h1.rows; i++){
		for(int j = 0; j < h1.cols; j++){
			result.at<float>(0,j) += EPSILON * (h1.at<float>(i,j) - prob.at<float>(0,j)) / (float)h1.rows;
		}
	}

	/*cv::reduce((h1 - prob), result, 0, CV_REDUCE_AVG);
	result = EPSILON * result;*/

	return result.clone();
}

cv::Mat DBN::calcW(cv::Mat h1, cv::Mat x1, cv::Mat prob, cv::Mat xk){
	cv::Mat result;

	result.create(x1.cols, h1.cols, CV_32FC1);

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

void DBN::Netsave(char *fileName){
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
	temp = NOUTPUT;
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

	for(int j = 0; j < classLayer.m_weight.rows; j++){
		for(int k = 0; k < classLayer.m_weight.cols; k++){
			float tf = classLayer.m_weight.at<float>(j,k);
			fwrite(&tf, sizeof(float), 1, fp);
		}
	}

	//b
	for(int j = 0; j < classLayer.m_b.rows; j++){
		for(int k = 0; k < classLayer.m_b.cols; k++){
			float tf = classLayer.m_b.at<float>(j,k);
			fwrite(&tf, sizeof(float), 1, fp);
		}
	}

	//c
	for(int j = 0; j < classLayer.m_c.rows; j++){
		for(int k = 0; k < classLayer.m_c.cols; k++){
			float tf = classLayer.m_c.at<float>(j,k);
			fwrite(&tf, sizeof(float), 1, fp);
		}
	}

	fclose(fp);
	printf("Writing complete!\n");
}

void DBN::NetLoad(char *fileName){
	printf("Training result Loading..\n");
	FILE *fp = fopen(fileName, "rb");
	//SaveFile
	int temp;
	fread(&temp,sizeof(int),1, fp);			//Layer Height write

	//Layer unit information write
	fread(&temp, sizeof(int), 1, fp);
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

	///////////////
	for(int j = 0; j < classLayer.m_weight.rows; j++){
		for(int k = 0; k < classLayer.m_weight.cols; k++){
			float tf;
			fread(&tf, sizeof(float), 1, fp);
			classLayer.m_weight.at<float>(j,k) = tf;
		}
	}

	//b
	for(int j = 0; j < classLayer.m_b.rows; j++){
		for(int k = 0; k < classLayer.m_b.cols; k++){
			float tf;
			fread(&tf, sizeof(float), 1, fp);
			classLayer.m_b.at<float>(j,k) = tf;
		}
	}

	//c
	for(int j = 0; j < classLayer.m_c.rows; j++){
		for(int k = 0; k < classLayer.m_c.cols; k++){
			float tf;
			fread(&tf, sizeof(float), 1, fp);
			classLayer.m_c.at<float>(j,k) = tf;
		}
	}

	fclose(fp);
	printf("Loading complete!\n");
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
int DBN::BatchRandLoad(cv::Mat *batch, cv::Mat *Label){
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
		m_NEpoch++;
		return 1;
	}

	return 0;
}

void DBN::BatchClose(){
	m_Dataloader.FileClose();
	m_Labelloader.FileClose();
}

void DBN::FullBackpropagation(){
	cv::Mat miniBatch, BatchLabel;

	BatchOpen("Data\\train-images.idx3-ubyte", "Data\\train-labels.idx3-ubyte");

	m_NEpoch = 0;

	//정해진 Epoch만큼 루프
	while(1){
		BatchRandLoad(&miniBatch, &BatchLabel);


		if(m_NEpoch > NEPOCH){
			break;
		}

	}
	BatchClose();
}

int DBN::DBNquery(cv::Mat src){
	int retVal = -1;
	cv::Mat tInput, tOutput;

	MatCopy(src, &tInput);

	for(int i = 0; i < LAYERHEIGHT-1; i++){
		hidden[i].processPresData(&tOutput, tInput);
		MatCopy(tOutput, &tInput);
	}

	classLayer.processPresData(&tOutput, tInput);

	//결과중 가장 큰 놈 산출
	float tmax = -99999.0f;
	for(int i = 0; i < tOutput.cols; i++){
		float temp = tOutput.at<float>(0, i);
		if(tmax < temp){
			tmax = temp;
			retVal = i;
		}
	}

	return retVal;
}