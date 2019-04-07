#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <device_launch_parameters.h>
#include <cassert>
#include <vector>

namespace cudann
{

	template <class T, class uint = unsigned int>
	struct Compact
	{
		uint m_size;
		T * const data;

		__device__ __host__ T & operator[](uint i)
		{
			return data[i];
		}

		__device__ __host__ const T & operator[](uint i)const
		{
			return data[i];
		}

		__device__ __host__ uint size()const
		{
			return m_size;
		}

		__device__ __host__ T * begin()
		{
			return data;
		}

		__device__ __host__ const T * begin()const
		{
			return data;
		}

		__device__ __host__ const T * cbegin()const
		{
			return data;
		}

		__device__ __host__ T * end()
		{
			return data + m_size;
		}

		__device__ __host__ const T * end()const
		{
			return data + m_size;
		}

		__device__ __host__ const T * cend()const
		{
			return data + m_size;
		}

	};


	/////////////////////////////////////////////////////////
	// A fixed size buffer
	// living on the device | the host memory
	/////////////////////////////////////////////////////////
	template <class T, class uint = unsigned int, bool MUTE=false>
	class Buffer
	{
	protected:

		uint m_size;
		
		T * h_data;
		T * d_data;

		



	public:

		Buffer(Buffer const& other):
			m_size(other.m_size)
		{
			std::cerr << "copy!" << std::endl;
		}


		Buffer(uint size=0):
			m_size(size),
			h_data(nullptr),
			d_data(nullptr)
		{}

		~Buffer()
		{
			if (device_loaded())
			{
				cudaError_t error = cudaFree(d_data);
				if (error != cudaSuccess)
				{
					std::cerr << "Error, could not free the buffer from the device memory! this: " << this << std::endl;
					std::cerr << error << std::endl;
				}
			}
			if (host_loaded())
			{
				delete[] h_data;
			}
		}

		void set_size(uint size)
		{
			if (size == 0)
			{
				m_size = size;
			}
			else
			{
				std::cerr << "set size, Not yes implemented!" << std::endl;
			}
		}

		uint size()const
		{
			return m_size;
		}

		T * host_data()
		{
			return h_data;
		}

		const T * host_data()const
		{
			return h_data;
		}

		T * device_data()
		{
			return d_data;
		}

		const Compact<T, uint> host_compact()const
		{
			return { m_size, h_data };
		}

		const Compact<T, uint> device_compact()const
		{
			return { m_size, d_data };
		}


		const T * device_data()const
		{
			return d_data;
		}

		bool host_loaded()const
		{
			return h_data != nullptr;
		}

		bool device_loaded()const
		{
			return d_data != nullptr;
		}


		bool malloc_host()
		{
			if (!host_loaded())
			{
				h_data = new T[m_size];
				assert(h_data != nullptr);
				return true;
			}
			else
			{
				if (!MUTE)
				{
					std::cout << "Warning, buffer alreary mallocated on host memory! this: "<< this << std::endl;
				}
				return false;
			}
		}

		bool malloc_device()
		{
			if (!device_loaded())
			{
				cudaError_t error = cudaMalloc((void**)&d_data, sizeof(T) * m_size);
				if (error != cudaSuccess)
				{
					if (!MUTE)
					{
						std::cerr << "Error, Could not malloc the buffer on the device memory! this: " << this << std::endl;
						std::cerr << error << std::endl;
					}
					d_data = nullptr;
					return false;
				}
				return true;
			}
			else
			{
				if  (!MUTE)
				{
					std::cout << "Warning, buffer alreary mallocated on device memory! this: " << this << std::endl;
				}
				return false;
			}
		}

		bool free_host()
		{
			if (host_loaded())
			{
				assert(m_size > 0);
				delete[] h_data;
				h_data = nullptr;
				return true;
			}
			else
			{
				if (!MUTE)
				{
					std::cout << "Warnong, Trying to free an unloaded buffer from host memory! this: " << this << std::endl;
				}
				return false;
			}
		}

		bool free_device()
		{
			if (device_loaded())
			{
				assert(m_size > 0);
				cudaError_t error = cudaFree((void*)d_data);
				if (error != cudaSuccess)
				{
					if (!MUTE)
					{
						std::cerr << "Error, could not free the device buffer! this: " << this << std::endl;
						std::cerr << error << std::endl;
					}
					return false;
				}
				d_data = nullptr;
				return true;
			}
			else
			{
				if (!MUTE)
				{
					std::cout << "Warnong, Trying to free an unloaded buffer from device memory! this: " << this << std::endl;
				}
				return false;
			}
		}

		
		bool send_host_to_device()
		{
			assert(host_loaded());
			assert(device_loaded());
			
			cudaError_t error = cudaMemcpy(d_data, h_data, sizeof(T)*m_size, cudaMemcpyHostToDevice);
			if (error != cudaSuccess)
			{
				if (!MUTE)
				{
					std::cerr << "Error, could not send the host buffer to the device! this: " << this << std::endl;
					std::cerr << error << std::endl;
				}
				return false;
			}
			return true;
		}
		
		bool send_device_to_host()
		{
			assert(host_loaded());
			assert(device_loaded());

			cudaError_t error = cudaMemcpy(h_data, d_data, sizeof(T)*m_size, cudaMemcpyDeviceToHost);
			if (error != cudaSuccess)
			{
				if (!MUTE)
				{
					std::cerr << "Error, could not send the device buffer to the host! this: " << this << std::endl;
					std::cerr << error << std::endl;
				}
				return false;
			}
			return true;
		}


		template <class begin_it, class end_it>
		void fill_host(begin_it it, end_it const& end)
		{
			assert(host_loaded());
			uint i = 0;
			while (it != end && i < m_size)
			{
				h_data[i] = *it;
				++it;
				++i;
			}

			if (!MUTE && it != end)
			{
				std::cout << "Warning, The collection size to fill the host buffer exeeds the size of the buffer, discarding elements!, this: " << this << std::endl;
			}
		}

		void fill_host(std::vector<T> const& vec)
		{
			fill_host(vec.cbegin(), vec.cend());
		}
		
	};
}