#pragma once

#include "mathUtils.h"
#include "VectorModel.h"

template<class _Tw>
int expected_logpi(std::vector<double>& pi, const VectorModel& vm, _Tw w, float min_prob = 1e-3)
{
	float r = 0.f, x = 1.f;
	int senses = 0;
	std::copy(&vm.counts[{0, w}], &vm.counts[{0, w + 1}], pi.begin());
	float ts = std::accumulate(pi.begin(), pi.end(), 0.f);
	for (size_t k = 0; k < vm.T() - 1; ++k)
	{
		ts = std::max(ts - pi[k], 0.f);
		float a = 1 + pi[k] - vm.d, b = vm.alpha + (k + 1) * vm.d + ts;
		pi[k] = meanlog_beta(a, b) + r;
		r += meanlog_mirror(a, b);

		float pi_k = mean_beta(a, b) * x;
		x = std::max(x - pi_k, 0.f);
		if (pi_k >= min_prob) senses++;
	}
	pi[vm.T() - 1] = r;
	if (x >= min_prob) senses++;
	return senses;
}

template<class _Tw>
int expected_pi(std::vector<double>& pi, const VectorModel& vm, _Tw w, float min_prob = 1e-3)
{
	float r = 1.f;
	int senses = 0;
	float ts = std::accumulate(&vm.counts[{0, w}], &vm.counts[{0, w + 1}], 0.f);
	for (size_t k = 0; k < vm.T() - 1; ++k)
	{
		ts = std::max(ts - vm.counts[{k, w}], 0.f);
		float a = 1 + vm.counts[{k, w}] - vm.d, b = vm.alpha + (k + 1) * vm.d + ts;
		pi[k] = mean_beta(a, b) + r;
		if (pi[k] >= min_prob) senses++;
		r = std::max(r - pi[k], 0.f);
	}
	pi[vm.T() - 1] = r;
	if (x >= min_prob) senses++;
	return senses;
}

template<class _Tw>
std::vector<double>& expected_pi(const VectorModel& vm, _Tw w, float min_prob = 1e-3)
{
	std::vector<double> pi(vm.T());
	expected_pi(pi, vm, w, min_prob);
	return pi;
}