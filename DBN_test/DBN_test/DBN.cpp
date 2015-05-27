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

	/*visible.setLayerRelation(NULL, &hidden[0]);
	hidden[0].setLayerRelation(&visible, &classLayer);
	classLayer.setLayerRelation(&hidden[0], NULL);*/
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
			BatchRandLoad(&miniBatch, NULL, BATCHSIZE);

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

	//supervised training
#ifdef BP_TRAINING
	//full optimization MLP backpropagation
	printf("\nSupervised Training phase start\n");
	_super = clock();
	FullBackpropagation();
	Netsave("FullNetworkData.bin");

	printf("Supervised Training phase complete (%dms)\n", clock() - _super);
#elif defined(SOFTMAX)
	//Logistic regresion
	printf("\nSupervised Training phase  start - Logistic regression Layer train\n");
	_super = clock();
	LogisticTraining();
	Netsave("LogisticNetData.bin");

	printf("Supervised Trainin complete (%dms)\n", clock() - _super);
#endif
}

void DBN::Testing(){
	cv::Mat miniBatch, BatchLabel;
	int Ncorrect = 0, total = 0;

	m_NEpoch = 0;

	printf("DBN Testing phase\n");
	printf("\nNetwork Initialzie from file....\n");
	InitNetwork();
	NetLoad("FullNetworkData.bin");
	printf("Initialize complete!\n");

	hidden[0].WeightVis();

	//Test data open
	printf("\nTest data set open....\n");
	//BatchOpen("Data\\t10k-images.idx3-ubyte", "Data\\t10k-labels.idx1-ubyte");
	BatchOpen("Data\\train-images.idx3-ubyte", "Data\\train-labels.idx1-ubyte");
	printf("\nTest data set open complete!\n");

	while(1){
		BatchRandLoad(&miniBatch, &BatchLabel, BATCHSIZE);

		for(int i = 0; i < BATCHSIZE; i++){
			int ans = DBNquery(miniBatch.row(i));
			int gtrue = FindMaxIdx(BatchLabel.row(i));

			total++;
			if(ans == gtrue)	Ncorrect++;

			float CorrectRatio = (float)Ncorrect/(float)total * 100.0f;
			printf("[%d] true : %d, ans : %d  (%f%%)\n", total, gtrue, ans, CorrectRatio);

			/*if(ans != gtrue){
				DataSingleVis(miniBatch.row(i), "Error");
				printf("false!\n");
				cv::waitKey(0);
			}*/
		}

		if(m_NEpoch > 0){
			printf("Test Complete!\n");
			break;
		}
	}

	float CorrectRatio = (float)Ncorrect/(float)total * 100.0f;
	printf("\n===================================================\n");
	printf("Number of Test data : %d\n", total);
	printf("Correct ratio  : %f%%\n", CorrectRatio);
	printf("Error ratio  : %f%%\n", 100.0f - CorrectRatio);
	printf("\n===================================================\n");
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
	layer->processTempBack(&xk, hk, &xkF);

	printf("RBM CD (%dms)\n", clock() - _start);
	_start = clock();

#ifdef DEBUG_VISIBLE
	//맨 하위 일때만
	if(layer->m_prevLayer->m_prevLayer == NULL){
		DataSingleVis(xk, "Reconstruct");
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
	cv::Mat tprob = layer->calcProbH(xk);
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
			result.at<float>(0,j) += EPSILON * (h1.at<float>(i,j) - prob.at<float>(i,j)) / (float)h1.rows;
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

			//if(tele > tmax)
				tmax += tele;
		}
	}

	return tmax / (src.rows * src.cols);
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
	fic.create(28, 28, CV_32FC1);

	for(int i = 0; i < data.cols; i++){
		/*fic.at<uchar>(i/28,i%28) = (data.at<float>(0,i) > 0.0f) ? 255 : 0;*/
		float tData = data.at<float>(0,i);

		fic.at<float>(i/28,i%28) = tData;
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
int DBN::BatchRandLoad(cv::Mat *batch, cv::Mat *Label, int nBatch){
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
		batch->create(nBatch, 28*28, CV_32FC1);
		for(int i = 0; i < nBatch; i++){
			m_DataSet.row(m_box[count + i]).copyTo(batch->row(i));
		}
	}
	if(Label != NULL){
		Label->create(nBatch, 10, CV_32FC1);
		for(int i = 0; i < nBatch; i++){
			m_LabelSet.row(m_box[count + i]).copyTo(Label->row(i));
		}
	}
	count += nBatch;
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
	cv::Mat Ok[LAYERHEIGHT];
	cv::Mat delta[LAYERHEIGHT];
	cv::Mat wGrad, cGrad;
	int _start, batchCount = 0;

	BatchOpen("Data\\train-images.idx3-ubyte", "Data\\train-labels.idx1-ubyte");
	printf("Batch Load complete!\n");

	m_NEpoch = 0;

	//정해진 Epoch만큼 루프
	while(1){
		BatchRandLoad(&miniBatch, &BatchLabel, BATCHSIZE);

		//Forward process
		_start = clock();
		BPForward(miniBatch, Ok);
		printf("BP Forward process (%dms)\n", clock() - _start);
		_start = clock();

		//Backward process
		for(int i = LAYERHEIGHT-1; i > -1; i--){
			//output layer
			if(i == LAYERHEIGHT-1){
				delta[i].create(BATCHSIZE, classLayer.getUnitNum(), CV_32FC1);

				//delta calculation
				for(int j = 0; j < delta[i].rows; j++){
					for(int k = 0; k < delta[i].cols; k++){
						delta[i].at<float>(j,k) = Ok[i].at<float>(j,k) * (1 - Ok[i].at<float>(j,k)) * (BatchLabel.at<float>(j,k) - Ok[i].at<float>(j,k));
					}
				}
			}
			//hidden layer
			else{
				delta[i].create(BATCHSIZE, hidden[i].getUnitNum(), CV_32FC1);

				//delta calculation - hidden
				for(int j = 0; j < delta[i].rows; j++){
					for(int k = 0; k < delta[i].cols; k++){
						float wd;
						if(i+1 == LAYERHEIGHT-1)
							wd = BPMulWDelta( delta[i+1], classLayer.m_weight, j, k);
						else
							wd = BPMulWDelta( delta[i+1], hidden[i].m_weight, j, k);

						delta[i].at<float>(j,k) = Ok[i].at<float>(j,k) * (1 - Ok[i].at<float>(j,k)) * wd;
					}
				}
			}

			//gradient calculate - TO-DO
			if(i > 0)
				BPgradCalc(delta[i], Ok[i-1], &wGrad, &cGrad);
			else
				BPgradCalc(delta[i], miniBatch, &wGrad, &cGrad);

			//Average Gradient
			cv::Mat Avg, cAvg;
			cv::reduce(wGrad, Avg, 0, CV_REDUCE_AVG);
			cv::reduce(Avg, Avg, 1, CV_REDUCE_AVG);
			cv::reduce(cGrad, cAvg, 0, CV_REDUCE_AVG);
			cv::reduce(cAvg, cAvg, 1, CV_REDUCE_AVG);
			printf("[%d] gradient : %f, bias : %f\n", i, Avg.at<float>(0,0)*100000.0f, cAvg.at<float>(0,0)*100000.0f);

			//gradient apply
			BPgradApply(wGrad, cGrad, i);
		}

		printf("\n[%d] BP Loop process (%dms)\n\n", ++batchCount, clock() - _start);
		_start = clock();

		if(m_NEpoch > NEPOCH){
			break;
		}

	}
	BatchClose();
}

float DBN::BPMulWDelta(cv::Mat Deltak, cv::Mat Wkh, int row, int Didx){
	float retVal = 0.0f;

	for(int i = 0; i < Deltak.cols; i++){
		retVal += Deltak.at<float>(row, i) * Wkh.at<float>(Didx, i);
	}

	return retVal;
}

void DBN::BPgradCalc(cv::Mat delta, cv::Mat x, cv::Mat *wGrad, cv::Mat *cGrad){
	cv::Mat tx, twGrad;
	tx.create(x.rows, x.cols + 1, CV_32FC1);

	for(int i = 0; i < tx.rows; i++){
		for(int j = 0; j < tx.cols; j++){
			if(j == tx.cols - 1)
				tx.at<float>(i,j) = 1.0f;
			else
				tx.at<float>(i,j) = x.at<float>(i,j);
		}
	}

	//calculate
	float tsum;
	twGrad.create( x.cols+1, delta.cols, CV_32FC1 );
	for(int i = 0; i < twGrad.rows; i++){
		for(int j = 0; j < twGrad.cols; j++){

			tsum = 0.0f;
			for(int k = 0; k < BATCHSIZE; k++){
				tsum += delta.at<float>(k,j) * tx.at<float>(k,i);
			}
			twGrad.at<float>(i,j) = EPSILON * tsum / BATCHSIZE;

		}
	}

	//분해
	wGrad->create(twGrad.rows - 1, twGrad.cols, CV_32FC1);
	for(int i = 0; i < twGrad.rows - 1; i++){
		for(int j = 0; j < twGrad.cols; j++){
			wGrad->at<float>(i,j) = twGrad.at<float>(i,j);
		}
	}
	MatCopy(twGrad.row(twGrad.rows-1), cGrad );


}

void DBN::BPForward(cv::Mat batch, cv::Mat *Ok){
	cv::Mat tInput;
	MatCopy(batch, &tInput);
	for(int i = 0; i < LAYERHEIGHT-1; i++){
		hidden[i].processPresData(&Ok[i], tInput);
		MatCopy(Ok[i], &tInput);
	}
	classLayer.processPresData(&Ok[LAYERHEIGHT-1], tInput);
}

void DBN::BPgradApply(cv::Mat wGrad, cv::Mat cGrad, int idx){
	//classification layer
	if(idx == LAYERHEIGHT - 1){
		classLayer.m_weight = classLayer.m_weight + wGrad;
		classLayer.m_c = classLayer.m_c + cGrad;
	}
	//hidden layer
	else{
		hidden[idx].m_weight = hidden[idx].m_weight + wGrad;
		hidden[idx].m_c = hidden[idx].m_c + cGrad;
	}
}

int DBN::DBNquery(cv::Mat src){
	int retVal = -1;
	cv::Mat tInput, tOutput;

	ForwardProcess(src, &tOutput);

	//결과중 가장 큰 놈 산출
	for(int i = 0; i < tOutput.cols; i++)
		printf("%f\t", tOutput.at<float>(0,i));
	printf("\n");
	retVal = FindMaxIdx(tOutput);

	return retVal;
}

int DBN::FindMaxIdx(cv::Mat src){
	float tmax = -99999.0f;
	int retVal = -1;

	for(int i = 0; i < src.cols; i++){
		float temp = src.at<float>(0, i);
		if(tmax < temp){
			tmax = temp;
			retVal = i;
		}
	}

	return retVal;
}

void DBN::ForwardProcess(cv::Mat src, cv::Mat *dst){
	cv::Mat tInput, tOutput;

	MatCopy(src, &tInput);

	for(int i = 0; i < LAYERHEIGHT-1; i++){
		hidden[i].processPresData(&tOutput, tInput);
		MatCopy(tOutput, &tInput);
	}

	classLayer.processPresData(&tOutput, tInput);

	MatCopy(tOutput, dst);
}

void DBN::PrintMat(cv::Mat src){
	printf("=================================================================\n");
	for(int i = 0; i < src.rows; i++){
		for(int j = 0; j < src.cols; j++){
			printf("%f\t", src.at<float>(i,j));
		}
		printf("\n");
	}
	printf("=================================================================\n");
}

void DBN::LogisticTraining(){

	printf("Batch Loading....\n");
	BatchOpen("Data\\train-images.idx3-ubyte", "Data\\train-labels.idx1-ubyte");
	printf("Batch Load complete!\n");
}