#pragma once

#include <random>
#include "mdvector.h"
#include "softmax.h"


struct VectorModel
{
	std::vector<size_t> frequencies;
	mdvector<int8_t, 2> code;
	mdvector<int32_t, 2> path;
	mdvector<float, 3> in;
	mdvector<float, 2> out;
	float alpha;
	float d;
	mdvector<float, 2> counts;

	VectorModel(size_t max_length, size_t V, size_t M, size_t T = 1,
		float _alpha = 1e-2, float _d = 0)
		: path({ max_length, V }), code({ max_length, V }),
		in({ M, T, V }), out({ M, V }), counts({ T, V }), frequencies(V),
		alpha(_alpha), d(_d)
	{
		std::fill(code.begin(), code.end(), -1);
	}

	VectorModel(const std::vector<size_t>& _freqs, size_t M, size_t T = 1,
		float _alpha = 1e-2, float _d = 0)
		: in({ M, T, _freqs.size() }), out({ M, _freqs.size() }), counts({ T, _freqs.size() }), frequencies(_freqs),
		alpha(_alpha), d(_d)
	{
		auto V = _freqs.size();
		auto nodes = build_huffman_tree(_freqs);
		auto outputs = convert_huffman_tree(nodes, V);
		auto max_length = max_element(outputs.begin(), outputs.end(), [](const HierarchicalOuput& a, const HierarchicalOuput& b)
		{
			return a.code.size() < b.code.size();
		})->code.size();

		std::mt19937_64 rg;
		std::uniform_real_distribution<float> urd{-.5f / M, .5f / M};
		for (auto& r : in) r = urd(rg);
		for (auto& r : out) r = urd(rg);

		path.resize({ max_length, V });
		code.resize({ max_length, V });

		for (size_t v = 0; v < V; ++v)
		{
			std::fill(&code[{0, v}], &code[{0, v}] + max_length, -1);
			for (size_t i = 0; i < outputs[v].length(); ++i)
			{
				code[{i, v}] = outputs[v].code[i];
				path[{i, v}] = outputs[v].path[i];
			}
		}
	}

	size_t M() const // dimensionality of word vectors
	{
		return in.size()[0];
	}

	size_t T() const // number of meanings
	{
		return in.size()[1];
	}

	size_t V() const // number of words
	{
		return in.size()[2];
	}
};
