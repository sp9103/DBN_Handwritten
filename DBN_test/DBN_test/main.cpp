#include "common.h"
#include "DBN.h"
#include "Application.h"

int main(){
	DBN dbn;

	//Training phase
	dbn.Training();
	dbn.save("DBN_Data.bin");
	

	//Testing phase
	/*dbn.Load("DBN_Data.bin");
	dbn.Testing();*/

	return 0;
}