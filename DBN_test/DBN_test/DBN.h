#include "common.h"
#include "DataLoader.h"
#include "LabelLoader.h"
#include "Layer.h"

class DBN
{
public:
	DBN(void);
	~DBN(void);

	void Training();
	void Testing();

	void save(char *fileName);
	void Load(char *fileName);

	void InitNetwork();

private:
	DataLoader m_Dataloader;
	LabelLoader m_Labelloader;

	Layer visible;
	Layer hidden[LAYERHEIGHT-1];

	float RBMupdata(cv::Mat x1, float e, Layer *layer, int step);

	void BatchLoad(cv::Mat *batch, cv::Mat *Label, char* DataName, char* LabelName);

	/*�̹� �Ҵ�� ��Ʈ���� ���Ҹ� ��� 0���� �ʱ�ȭ*/
	void MatZeros(cv::Mat *target);
};

