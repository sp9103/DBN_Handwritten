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
	Layer hidden1, hidden2, hidden3;

};

