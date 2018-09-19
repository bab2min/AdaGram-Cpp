#pragma once

#include <cmath>

inline float sigmoid(float x)
{
	return 1 / (1 + exp(-x));
}

inline float logsigmoid(float x)
{
	return -log(1 + exp(-x));
}

inline float digamma(float x)
{
	if(x < 100) return log(x + 2) - 0.5 / (x + 2) - 1 / 12.0 / pow(x + 2, 2) - 1 / (x + 1) - 1 / x;
	return log(x) - 0.5 / x - 1 / 12.0 / pow(x, 2);
}

inline float mean_beta(float a, float b)
{
	return a / (a + b);
}

inline float meanlog_beta(float a, float b)
{
	return digamma(a) - digamma(b);
}

inline float mean_mirror(float a, float b)
{
	return mean_beta(b, a);
}

inline float meanlog_mirror(float a, float b)
{
	return meanlog_beta(b, a);
}

template<class _Tf>
void exp_normalize(std::vector<_Tf>& x)
{
	auto max_x = *std::max_element(x.begin(), x.end());
	_Tf sum_x = 0;
	for (auto& xi : x)
	{
		xi = exp(xi - max_x);
		sum_x += xi;
	}
	for (auto& xi : x)
	{
		xi /= sum_x;
	}
}