#include "common.h"
#include "DBN.h"
#include "Application.h"

int main(){
	DBN dbn;

	printf("DBN Handwritten Recognition\n\n");

	//Training phase
	dbn.Training();
	dbn.save("Data\\DBN_Data.bin");
	

	//Testing phase
	/*dbn.Load("DBN_Data.bin");
	dbn.Testing();*/

	return 0;
}