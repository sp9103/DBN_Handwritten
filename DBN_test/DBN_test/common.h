#include <stdio.h>
#include <opencv.hpp>
#include <iostream>
#include <fstream>
#include <random>
#include <time.h>

using namespace std;

#define BATCHSIZE 1
#define GRADTHRESHOLD 0.12
#define EPSILON		0.005
#define CDStep		2
#define NEPOCH		19

//Deep belif network - Layer information
#define LAYERHEIGHT 4
#define NVISIBLE	28*28
#define NHIDDEN1	500
#define NHIDDEN2	400
#define NHIDDEN3	200
#define NOUTPUT		10

#define SWAP(x,y,t) ((t)=(x), (x)=(y), (y)=(t))