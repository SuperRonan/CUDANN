
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include "Buffer.cuh"
#include "NeuralNetwork.cuh"


int main()
{
	cudann::NeuralNetwork<float> nn({ 16, 8, 4, 3 });
	nn.init_host();
	nn.fill_random_weights();
	nn.init_device();
	nn.print_info();
	nn.print_weights();
}