#pragma once

#include "VectorModel.h"
#include "stick_breaking.h"
#include "skip_gram.h"
#include "kahan.h"

std::pair<double, int64_t> likelihood(const VectorModel& vm, const std::vector<int>& doc, 
	int window_length, float min_prob = 1e-5)
{
	const size_t N = doc.size();
	if (N <= 1) return {0., 0};
	std::vector<double> z(vm.T());
	MeanCounter<double> m;
	for (size_t i = 0; i < N; ++i)
	{
		const auto& x = doc[i];
		auto window = window_length;
		std::fill(z.begin(), z.end(), 0.);
		expected_pi(z, vm, x);

		size_t jBegin = 0, jEnd = N;
		if (i > window) jBegin = i - window;
		if (i + window < N) jEnd = i + window;
		for (auto j = jBegin; j < jEnd; ++j)
		{
			if (i == j) continue;
			const auto& y = doc[j];

			Kahan<double> local_ll;
			for (size_t s = 0; s < vm.T(); ++s)
			{
				if (z[s] < min_prob) continue;
				local_ll.add(z[s] * exp(log_skip_gram(vm, x, s, y)));
			}
			m.add(log(local_ll.sum));
		}
	}
	return { m.mean.sum, m.n };
}