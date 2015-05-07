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

	InitNetwork();

	//unsupervised training - �� RBM �н�
	printf("Unsupervised Training phase start\n");
	for(int i = 0; i < LAYERHEIGHT-1; i++){
		printf("[%d] RBM Training...\n", i+1);

		while(1){
			cv::Mat miniBatch;
			BatchLoad(&miniBatch, NULL, "Data\\train-images.idx3-ubyte", "");

			//Bottom Layer training
			if(i == 0){
				//Input�� �̹��� �״�� ��

			}
			//others
			else{
				//input�� ���� ���̾��� �ƿ�ǲ. - ( ���������� )
				cv::Mat tdata;
				hidden[i].processData(&tdata, miniBatch);
			}

			//Termination condition
			if(GRADTHRESHOLD > tgrad)
				break;
		}

		printf("Complete!\n");
	}
	printf("Unsupervised Training complete\n");

	//supervised training - full optimization MLP backpropagation
	printf("Supervised Training phase start\n");


	printf("Supervised Training phase complete\n");
}

void DBN::Testing(){
	m_Dataloader.FileOpen("t10k-images.idx3-ubyte");
}

float DBN::RBMupdata(cv::Mat minibatch, float e, cv::Mat *W, cv::Mat *b, cv::Mat *c){
	return 0.0f;
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