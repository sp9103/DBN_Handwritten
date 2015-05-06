#include "DBN.h"

DBN::DBN(void)
{
}


DBN::~DBN(void)
{
}

void DBN::InitNetwork(){
	visible.Init(NVISIBLE);
	hidden1.Init(NHIDDEN1);
	hidden2.Init(NHIDDEN2);
	hidden3.Init(NHIDDEN3);

	visible.setLayerRelation(NULL, &hidden1);
	hidden1.setLayerRelation(&visible, &hidden2);
	hidden2.setLayerRelation(&hidden1, &hidden3);
	hidden3.setLayerRelation(&hidden2, NULL);
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

	//unsupervised training - °¢ RBM ÇÐ½À
	for(int i = 0; i < LAYERHEIGHT-1; i++){
		
		//Batch training
		cv::Mat BatchData;
		m_Dataloader.FileOpen("train-images.idx3-ubyte");
		m_Dataloader.ImageDataLoad(1, &BatchData);
		m_Dataloader.FileClose();
	}

	//supervised training - classifier

}

void DBN::Testing(){
	m_Dataloader.FileOpen("t10k-images.idx3-ubyte");
}