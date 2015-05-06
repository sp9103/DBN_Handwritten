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
	FILE *fp = fopen(fileName, "wb");
	//SaveFile

	fclose(fp);
}

void DBN::Load(char *fileName){
	FILE *fp = fopen(fileName, "rb");
	//LoadFile

	fclose(fp);
}

void DBN::Training(){
	InitNetwork();

	//unsupervised training - 각 RBM 학습
	for(int i = 0; i < LAYERHEIGHT-1; i++){
		
		//Bottom Layer training
		if(i == 0){
			//Input이 이미지 그대로 들어감
		}
		//others
		else{
			//input은 이전 레이어의 아웃풋. - ( 계산해줘야함 )
		}

		////Batch training
		//cv::Mat BatchData;
		//m_Dataloader.FileOpen("train-images.idx3-ubyte");
		//m_Dataloader.ImageDataLoad(1, &BatchData);
		//m_Dataloader.FileClose();
	}

	//supervised training - classifier

}

void DBN::Testing(){
	m_Dataloader.FileOpen("t10k-images.idx3-ubyte");
}