#include "common.h"

class LabelLoader
{
public:
	LabelLoader(void);
	~LabelLoader(void);

	void FileOpen(char *fileName);
	void FileClose();

	void LabelDataLoad(int batchSize, cv::Mat *dataMat);

	int getDataCount();

private:
	ifstream m_file;

	int magic_number;
	int number_of_images;

	int reverseInt(int i);
};

