#pragma once

#include <iostream>
#include "Buffer.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <device_launch_parameters.h>
#include <cassert>
#include <vector>
#include "utils.h"
#include <iostream>

namespace cudann
{
	/////////////////////////////////////////////////////
	// This class is for representing the weights of a NN
	/////////////////////////////////////////////////////
	template <class floot, class uint = unsigned int, bool MUTE = false>
	class NeuralNetwork
	{
	public:
		using Bufferf = Buffer<floot, uint, MUTE>;
		using Compactf = Compact<floot, uint>;

	protected:

		class Layer
		{
		public:

			//the size of each buffer this vector is the number of neurons + 1 (bias)
			//each buffer represents the weight of the inputs of the neurons
			std::vector<Bufferf> m_weights;
			mutable Bufferf m_result;
			Bufferf m_error;

		public:

			Layer(uint number_of_neurons, uint prev_layer_size):
				m_result(number_of_neurons),
				m_error(number_of_neurons)
			{
				m_weights.reserve(number_of_neurons);
				for (uint i = 0; i < number_of_neurons; ++i)
				{
					m_weights.emplace_back(prev_layer_size + 1);
					//m_weights.push_back(Bufferf(prev_layer_size + 1));
				}
			}

			void init_host()
			{
				for (Bufferf & buffer : m_weights)
				{
					buffer.malloc_host();
				}
				m_result.malloc_host();
				m_error.malloc_host();
			}

			void init_device()
			{
				for (Bufferf & buffer : m_weights)
				{
					buffer.malloc_device();
				}
				m_result.malloc_device();
				m_error.malloc_device();
			}

			//implicitly on the host
			void fill_random_weights(floot scale)
			{
				for (Bufferf & buffer : m_weights)
				{
					assert(buffer.host_loaded());
					auto compact = buffer.host_compact();
					for (uint i = 0; i < compact.size(); ++i)
					{
						compact[i] = utils::random(-scale, scale);
					}
				}
			}

			void send_host_to_device()
			{
				for (Bufferf & buffer : m_weights)
				{
					assert(buffer.device_loaded());
					assert(buffer.host_loaded());
					buffer.send_host_to_device();
				}
				m_result.send_host_to_device();
				m_error.send_host_to_device();
			}

			void print_weights(bool host = true, std::ostream & out=std::cout)
			{
				if (host)
				{
					out << "[\n";
					uint i = 0;
					for (Bufferf & buffer : m_weights)
					{
						out << "--[" << i << "]--";
						assert(buffer.host_loaded());
						utils::print_collection(out, buffer.host_compact());
						out << "\n";
						++i;
					}
					out << "]\n";
				}
			}

			void compute_host(const Bufferf * input)const
			{
				const Compactf compact_in = input->host_compact();
				Compactf cmpct = m_result.host_compact();
				//for each neuron (TODO multi thread)
				for (uint i = 0; i < cmpct.size(); ++i)
				{
					const Bufferf & weigths = m_weights[i];
					const Compactf weight_compact = weigths.host_compact();

					assert(weigths.size() == input->size() + 1);
					
					floot res = 0;
					uint j = 0;
					for (; j < compact_in.size(); ++j)
					{
						res += weight_compact[i] * compact_in[j];
					}
					assert(j < weight_compact.size());
					res += weight_compact[j];
					
					//apply the sigmoid
					res = floot(1) / (floot(1) + exp(-res));

					cmpct[i] = res;
				}
			}
				
		};

		
		std::vector<uint> m_struct;
		std::vector<Layer> m_layers;

		
		

	public:

		bool m_host_init = false, m_device_init = false;
		bool m_host_loaded = false, m_device_loaded = false;

		//////////////////////////////////////////////////////////
		// structure represent the general structure of the NN
		// {12, 7, 5, 3} means:
		//	- 12 perceptrons
		//	- two hidden layers: one of 7 neurons and one of 5
		//	- the output layer of 3 neurons
		//
		//	So it means there is only 3 layers of weights to learn
		//////////////////////////////////////////////////////////
		NeuralNetwork(std::vector<uint> const& structure):
			m_struct(structure)
		{
			m_layers.reserve(m_struct.size() - 1);
			for (uint i = 1; i < m_struct.size() ; ++i)
			{
				m_layers.emplace_back(m_struct[i], m_struct[i-1]);
			}
		}


		void init_host()
		{
			for (Layer & layer : m_layers)
			{
				layer.init_host();
			}
			m_host_init = true;
		}

		void init_device()
		{
			for (Layer & layer : m_layers)
			{
				layer.init_device();
			}
			m_device_init = true;
		}

		//fill the weights in U[-scale, scale]
		void fill_random_weights(floot scale=1)
		{
			for (Layer & layer : m_layers)
			{
				layer.fill_random_weights(scale);
			}
			m_host_loaded = true;
		}

		void send_host_to_device()
		{
			assert(m_device_init);
			for (Layer & layer : m_layers)
			{
				layer.send_host_to_device();
			}
			m_device_loaded = true;
		}

		void print_info(std::ostream & out=std::cout)
		{
			out << "Neural network: " << std::endl;
			out << "Structure: "; 
			utils::print_collection(std::cout, m_struct);
			std::cout << std::endl;
		}


		void print_weights(bool host = true, std::ostream & out = std::cout)
		{
			if (host)
			{
				assert(m_host_loaded);
				int i = 1;
				for (Layer & layer : m_layers)
				{
					out << "layer[" << i << "]: size: " << m_struct[i] << std::endl;
					layer.print_weights(host, out);
					++i;
				}
			}
			else
			{
				//TODO
			}
		}


		const Bufferf & predict_host(Bufferf const& input)const
		{
			assert(input.size() == m_struct.front());
			const Bufferf * layer_input = &input;
			for (const Layer & layer : m_layers)
			{
				layer.compute_host(layer_input);
				layer_input = &layer.m_result;
			}

			return m_layers.back().m_result;
		}


		void fit_host(std::vector<std::pair<Bufferf, Bufferf>> const& training_set, floot alpha=0.5, const uint number_of_pass=1)
		{
			for (uint pass = 0; pass < number_of_pass; ++pass)
			{
				for (std::pair<Bufferf, Bufferf> const& p : training_set)
				{
					Bufferf const& example = p.first;
					Bufferf const& truth = p.second;
					predict_host(example);

					Layer * output = &m_layers.back();

					//the small delta
					Bufferf * error = &output.m_error;

					//error->apply_function_host(truth, output.m_result, std::minus<floot>());

					for (int layer_id = m_layers.size() - 1; layer_id >= 0; --layer_id)
					{
						Layer & layer = m_layers[layer_id];
						if(layer_id == m_layers.size() - 1)
						{
							//output layer
							for (uint neuron_id = 0; neuron_id < m_struct[layer_id + 1]; ++neuron_id)
							{
								floot u = truth.host_compact()[neuron_id];
								floot y = layer.m_result.host_compact()[neuron_id];
								floot small_delta = (u - y) * y * (1 - y);
							}
						}
						else
						{
							for (uint neuron_id = 0; neuron_id < m_struct[layer_id + 1]; ++neuron_id)
							{

							}
						}
					}
				}
			}
		}

	};
}