#include <stdio.h>
#include <opencv.hpp>
#include <iostream>
#include <fstream>
#include <random>
#include <time.h>

using namespace std;

#define BATCHSIZE 100
#define GRADTHRESHOLD 0.12
#define EPSILON		0.02
#define CDStep		2
#define NEPOCH		10

//Deep belif network - Layer information
#define LAYERHEIGHT 4
#define NVISIBLE	28*28
#define NHIDDEN1	500
#define NHIDDEN2	400
#define NHIDDEN3	200

#define SWAP(x,y,t) ((t)=(x), (x)=(y), (y)=(t))