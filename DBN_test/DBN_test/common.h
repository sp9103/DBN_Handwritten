#include <stdio.h>
#include <opencv.hpp>
#include <iostream>
#include <fstream>
#include <random>
#include <time.h>

using namespace std;

#define BATCHSIZE 100
#define GRADTHRESHOLD 0.1
#define EPSILON		0.01
#define CDStep		2

//Deep belif network - Layer information
#define LAYERHEIGHT 4
#define NVISIBLE	28*28
#define NHIDDEN1	500
#define NHIDDEN2	400
#define NHIDDEN3	200