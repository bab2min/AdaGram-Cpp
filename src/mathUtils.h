#pragma once
#include <array>
#include "gamma.h"

inline float sigmoid(float x)
{
	return 1 / (1 + exp(-x));
}

struct F_logsigmoid
{
	double operator()(double x) { return -log(1 + exp(-x)); }
	double forLarge(double x) { return -0.f; }
};

template<class _Func, size_t N, size_t S>
class SimpleLUT
{
protected:
	std::array<float, N> points;
	static constexpr float P = 1.f / S;
	SimpleLUT()
	{
		_Func fun;
		for (size_t i = 0; i < N; i++)
		{
			points[i] = fun(i * P);
		}
	}

	float _get(float x) const
	{
		size_t idx = (size_t)(x * S);
		if (idx >= N) return _Func{}.forLarge(x);
		return points[idx];
	}
public:
	static const SimpleLUT& getInst()
	{
		static SimpleLUT lg;
		return lg;
	}

	static float get(float x)
	{
		return getInst()._get(x);
	}
};

inline float logsigmoid(float x)
{
	if (x >= 0) return SimpleLUT<F_logsigmoid, 32 * 1024, 1024>::get(x);
	return SimpleLUT<F_logsigmoid, 32 * 1024, 1024>::get(-x) + x;
}

inline float mean_beta(float a, float b)
{
	return a / (a + b);
}

inline float meanlog_beta(float a, float b)
{
	return DIGAMMA(a) - DIGAMMA(a + b);
}

inline float mean_mirror(float a, float b)
{
	return mean_beta(b, a);
}

inline float meanlog_mirror(float a, float b)
{
	return meanlog_beta(b, a);
}
