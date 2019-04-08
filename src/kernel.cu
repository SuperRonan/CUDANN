
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include "Buffer.cuh"
#include "NeuralNetwork.cuh"





template <class floot = float>
void test_xor()
{
	cudann::NeuralNetwork<floot> xornn({ 2, 2, 1 });
	xornn.init_host();
	xornn.fill_random_weights();
	xornn.print_info();
	xornn.print_weights();

	std::vector<std::pair<cudann::Buffer<floot>, cudann::Buffer<floot>>> training_set;
	training_set.reserve(4);
	for (char x = 0; x < 2; ++x)			


	{
		for (char y = 0; y < 2; ++y)
		{

			training_set.emplace_back(2, 1);
			cudann::Buffer<floot> & example = training_set.back().first;
			cudann::Buffer<floot> & truth = training_set.back().second;
			
			example.malloc_host();
			truth.malloc_host();

			example.fill_host({ floot(x), floot(y) });
			truth.fill_host({ floot(x ^ y) });
		}
	}

	xornn.fit_host(training_set, 100);

	xornn.print_info();
	xornn.print_weights();

}

template <class floot=float>
void test_nn()
{
	const unsigned int input_size = 3;
	cudann::NeuralNetwork<floot> nn({ input_size, 2, 1 });
	nn.init_host();
	nn.fill_random_weights();
	nn.init_device();
	nn.print_info();
	nn.print_weights();


	cudann::Buffer<floot> input(input_size);
	input.malloc_host();
	input.fill_host({-1, 1, 2});

	const cudann::Buffer<floot> & res = nn.predict_host(input);

	utils::print_collection(std::cout, res.host_compact());
}

int main()
{
	test_nn();
	test_xor();
}