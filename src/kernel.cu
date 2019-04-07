
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include "Buffer.cuh"
#include "NeuralNetwork.cuh"


template <class floot=float>
void test_nn()
{
	cudann::NeuralNetwork<floot> nn({ 16, 8, 4, 3 });
	nn.init_host();
	nn.fill_random_weights();
	nn.init_device();
	nn.print_info();
	nn.print_weights();


	cudann::Buffer<floot> input(16);
	input.malloc_host();
	input.fill_host({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});

	const cudann::Buffer<floot> & res = nn.predict_host(input);

	utils::print_collection(std::cout, res.host_compact());
}

int main()
{
	test_nn();
}