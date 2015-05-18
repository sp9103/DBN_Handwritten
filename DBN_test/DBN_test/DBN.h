#include "common.h"
#include "DataLoader.h"
#include "LabelLoader.h"
#include "Layer.h"

#define DEBUG_VISIBLE
#define RBM_TRAINING

class DBN
{
public:
	DBN(void);
	~DBN(void);

	void Training();
	void Testing();

	void save(char *fileName);
	void Load(char *fileName);

	void Netsave(char *fileName);
	void NetLoad(char *fileName);

	void InitNetwork();

	//28*28 �̹��� ������ ������ classification ����� ����
	int DBNquery(cv::Mat src);

private:
	DataLoader m_Dataloader;
	LabelLoader m_Labelloader;

	Layer visible;
	Layer hidden[LAYERHEIGHT-1];
	Layer classLayer;																//������ Classification�� ���� ���̾�

	float RBMupdata(cv::Mat x1, float e, Layer *layer, int step);

	void BatchLoad(cv::Mat *batch, cv::Mat *Label, char* DataName, char* LabelName);

	//����
	void BatchOpen(char *DataName, char* LabelName);
	int BatchRandLoad(cv::Mat *batch, cv::Mat *Label);
	void BatchClose();
	int *m_box;
	cv::Mat m_DataSet, m_LabelSet;
	int m_NEpoch;

	/*�̹� �Ҵ�� ��Ʈ���� ���Ҹ� ��� 0���� �ʱ�ȭ*/
	void MatZeros(cv::Mat *target);

	/*bias b calculation*/
	cv::Mat calcB(cv::Mat x1, cv::Mat xk);
	/*bias c calculation*/
	cv::Mat calcC(cv::Mat h1, cv::Mat prob);
	/*weight w calculation*/
	cv::Mat calcW(cv::Mat h1, cv::Mat x1, cv::Mat prob, cv::Mat xk);

	/*Pick Matrix Maximum element*/
	float MatMaxEle(cv::Mat src);

	//matrix copy
	void MatCopy(cv::Mat src, cv::Mat *dst);

	//Debug�� �Լ�
	void DataVis(cv::Mat data, cv::Mat data2);
	void DataSingleVis(cv::Mat data, char *windowName);

	void RBMLayersave(char *fileName, Layer src);
	void RBMLayerload(char *fileName, Layer *dst);

	//full optimization
	void FullBackpropagation();

	void ForwardProcess(cv::Mat src, cv::Mat *dst);
	int FindMaxIdx(cv::Mat src);
};

