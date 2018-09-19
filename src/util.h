#pragma once

#include <numeric>
#include "VectorModel.h"
#include "dictionary.h"
#include "stick_breaking.h"
#include "gradient.h"

std::vector<std::tuple<std::string, size_t, float>> 
	nearest_neighbors(const VectorModel& vm, const WordDictionary<>& dict,
	const std::vector<float>& word, size_t K = 10, float min_count = 1.f)
{
	std::vector<std::tuple<std::string, size_t, float>> top;
	mdvector<float, 2> sim({ vm.T(), vm.V() });
	for (size_t v = 0; v < vm.V(); ++v)
	{
		for (size_t s = 0; s < vm.T(); ++s)
		{
			if (vm.counts[{s, v}] < min_count)
			{
				sim[{s, v}] = -INFINITY;
				continue;
			}
			sim[{s, v}] = std::inner_product(word.begin(), word.end(), &vm.in[{0, s, v}], 0.f) 
				/ sqrt(std::inner_product(&vm.in[{0, s, v}], &vm.in[{0, s + 1, v}], &vm.in[{0, s, v}], 0.f));
		}
	}

	for (size_t k = 0; k < K; ++k)
	{
		size_t idx = std::max_element(sim.begin(), sim.end()) - sim.begin();
		top.emplace_back(dict.getStr(idx / vm.T()), idx % vm.T(), sim.begin()[idx]);
		sim.begin()[idx] = -INFINITY;
	}

	return top;
}

std::vector<std::tuple<std::string, size_t, float>> 
	nearest_neighbors(const VectorModel& vm, const WordDictionary<>& dict,
	const std::string& word, size_t s, size_t K = 10, float min_count = 1.f)
{
	size_t v = dict.get(word);
	if (v == (size_t)-1) return {};
	std::vector<float> vec{ &vm.in[{0, s, v}], &vm.in[{0, s + 1, v}] };
	float norm = sqrt(std::inner_product(vec.begin(), vec.end(), vec.begin(), 0.f));
	for (auto& x : vec) x /= norm;
	return nearest_neighbors(vm, dict, vec, K, min_count);
}


std::vector<double> disambiguate(const VectorModel& vm, size_t x, const std::vector<size_t>& context,
	bool use_prior = true, float min_prob = 1e-3)
{
	std::vector<double> z(vm.T());
	if (use_prior)
	{
		expected_pi(z, vm, x);
		for (size_t k = 0; k < vm.T(); ++k)
		{
			if (z[k] < min_prob) z[k] = 0;
			z[k] = log(z[k]);
		}
	}

	for (auto& y : context)
	{
		var_update_z(vm, x, y, z, vm.T());
	}

	exp_normalize(z);
	return z;
}

std::vector<double> disambiguate(const VectorModel& vm, const WordDictionary<>& dict, const std::string& x, const std::vector<std::string>& context,
	bool use_prior = true, float min_prob = 1e-3)
{
	size_t xId = dict.get(x);
	if (xId == (size_t)-1) return {};
	std::vector<size_t> contextId;
	for (auto& y : context)
	{
		size_t yId = dict.get(y);
		if (yId == (size_t)-1) continue;
		contextId.emplace_back(yId);
	}
	return disambiguate(vm, xId, contextId, use_prior, min_prob);
}
