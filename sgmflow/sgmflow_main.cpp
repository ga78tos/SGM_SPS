#include <iostream>
#include "SGMflow.h"

using namespace std;

int main (int argc, char* argv[]){

	if (argc < 3) {
	cerr << "usage: sgmstereo first second" << endl;
		exit(1);
	}

	string leftImageFilename = argv[1];
	string rightImageFilename = argv[2];

	float* flowImage;

	SGMflow sgmf;

	sgmf.compute(leftImageFilename, rightImageFilename, flowImage);

	cout << "ok" << endl;

	return 0;
}
