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

	//unsupervised training - 각 RBM 학습
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
				//Input이 이미지 그대로 들어감
				tgrad = RBMupdata(miniBatch, EPSILON, &hidden[i]);
			}
			//others
			else{
				//input은 이전 레이어의 아웃풋. - ( 계산해줘야함 )
				cv::Mat tdata;
				hidden[i].processData(&tdata, miniBatch);
				tgrad = RBMupdata(tdata, EPSILON, &hidden[i]);
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

float DBN::RBMupdata(cv::Mat minibatch, float e, Layer *layer){
	float tgrad = 0.0f;

	for(int i = 0; i < layer->getUnitNum(); i++){
	}

	return tgrad;
}

//Label이 NULL 일때는 로드안함.
//Label != NULL이면 supervised learning을 위해서 로드
void DBN::BatchLoad(cv::Mat *batch, cv::Mat *Label, char* DataName, char* LabelName){
	//순차적으로 뽑음 (추후 랜덤 배치로 구현)
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