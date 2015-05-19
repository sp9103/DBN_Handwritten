#include "common.h"
//#include "DBN.h"
#include "Application.h"

int main(){
	DBN dbn;
	Application app;

	printf("DBN Handwritten Recognition\n\n");

	//Training phase
	dbn.Training();

	//Testing phase
	/*dbn.Load("DBN_Data.bin");
	dbn.Testing();*/

	//application
	//app.Run();

	return 0;
}