#pragma once

#include <random>

namespace utils
{

	template <class floot>
	floot random(floot min, floot max)
	{
		floot res = floot(rand()) / floot(RAND_MAX);
		return  res * (max - min) + min;
	}



	template <class out_t, class Collection>
	out_t & print_collection(out_t & out, Collection const& col)
	{
		out << "[";
		size_t i = 0;
		for (auto const& elem : col)
		{
			out << elem;
			++i;
			if (i != col.size())
			{
				out << ", ";
			}
		}
		out << "]";
		return out;
	}
	
}


//template <class out_t, class Collection>
//out_t & operator<<(out_t & out, Collection const& col)
//{
//	return utils::print(out, col);
//}