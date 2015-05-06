#include <stdio.h>
#include <opencv.hpp>
#include <iostream>
#include <fstream>
#include <random>

using namespace std;

#define BATCHSIZE 1000

//Deep belif network - Layer information
#define LAYERHEIGHT 4
#define NVISIBLE	28*28
#define NHIDDEN1	500
#define NHIDDEN2	400
#define NHIDDEN3	200